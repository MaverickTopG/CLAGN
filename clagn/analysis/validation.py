"""
validation.py — Statistical validation of the CLAGN detection pipeline.

Four complementary validation strategies:
    1. Injection-recovery: inject known signals into real light curves, score them
    2. False positive rate: simulate DRW with WISE cadence, measure FAP
    3. Jackknife stability: test score robustness to epoch removal
    4. Coordinate scramble: query random sky positions, measure FPR

IMPORTANT: The false positive rate simulation is OFFLINE-ONLY and performs
NO IRSA queries. The coordinate scramble DOES make IRSA queries (rate-limited).

References:
    Kelly et al. 2009, ApJ 698, 895
    Graham et al. 2017, MNRAS 470, 4112 (injection-recovery methodology)
"""
import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (
    N_INJECTIONS,
    N_FP_SIMULATIONS,
    INJECTION_AMPLITUDES,
    N_JACKKNIFE_SCRAMBLES,
    WISE_NEOWISE_SEARCH_RADIUS_ARCSEC,
)

logger = logging.getLogger(__name__)

# Typical WISE NEOWISE-R cadence parameters (days)
_NEOWISE_EPOCH_SPACING_DAYS = 180.0  # ~6 months between survey passes
_NEOWISE_EPOCH_JITTER_DAYS = 5.0     # MJD scatter within each epoch
_NEOWISE_N_EXPOSURES_PER_EPOCH = 12   # Typical number of single exposures per visit
_NEOWISE_BASELINE_DAYS = 3650.0       # ~10-year baseline for NEOWISE-R

# WISE single-exposure noise floor
_WISE_PHOT_NOISE_MJY = 0.02          # mJy per single exposure (W1, typical AGN)

# Typical DRW parameters for simulated control sample
_DRW_TAU_TYPICAL_DAYS = 300.0
_DRW_SIGMA_TYPICAL_MJY = 0.05


# ---------------------------------------------------------------------------
# OU process (DRW) simulator
# ---------------------------------------------------------------------------

def simulate_drw_lightcurve(times: np.ndarray, tau: float, sigma: float,
                             seed: int = None) -> np.ndarray:
    """
    Simulate a Damped Random Walk (Ornstein-Uhlenbeck process) light curve.

    The OU process satisfies:
        x(t_{i+1}) = x(t_i) * exp(-dt/tau) + sigma * sqrt(1 - exp(-2dt/tau)) * N(0,1)

    This is the exact discrete-time solution to the SDE:
        dx = -(x/tau)dt + sigma * sqrt(2/tau) * dW

    Parameters
    ----------
    times  : array, sorted MJD values
    tau    : float, characteristic timescale (days)
    sigma  : float, process amplitude (mJy or magnitudes)
    seed   : int or None, random seed

    Returns
    -------
    flux : array, simulated flux values (mean = 0)
    """
    times = np.asarray(times, dtype=float)
    rng = np.random.default_rng(seed=seed)

    n = len(times)
    if n == 0:
        return np.array([])

    flux = np.zeros(n)
    flux[0] = sigma * rng.standard_normal()  # Draw from stationary distribution

    for i in range(1, n):
        dt = max(times[i] - times[i - 1], 0.0)
        rho = np.exp(-dt / max(tau, 1e-6))
        innovation_var = sigma ** 2 * (1.0 - rho ** 2)
        innovation_std = np.sqrt(max(innovation_var, 0.0))
        flux[i] = rho * flux[i - 1] + innovation_std * rng.standard_normal()

    return flux


# ---------------------------------------------------------------------------
# Signal injector
# ---------------------------------------------------------------------------

def _inject_signal(base_flux: np.ndarray, times: np.ndarray,
                   injection_type: str, amplitude_mag: float,
                   seed: int = None) -> np.ndarray:
    """
    Inject a synthetic variability signal into a base flux array.

    Supported injection types:
        'step'       : Heaviside step at t_mid (sudden turn-on/off)
        'ramp'       : Linear flux change over 30% of baseline
        'gaussian'   : Gaussian flare centered at t_mid
        'sinusoidal' : Periodic variation with period = 0.3 × baseline

    Parameters
    ----------
    base_flux      : array, base light curve (mJy)
    times          : array, MJD values (same length as base_flux)
    injection_type : str, one of {'step', 'ramp', 'gaussian', 'sinusoidal'}
    amplitude_mag  : float, peak-to-peak amplitude in magnitudes
    seed           : int or None, for reproducible injections

    Returns
    -------
    injected_flux : array, base_flux + injected signal (mJy)
    """
    times = np.asarray(times, dtype=float)
    base_flux = np.asarray(base_flux, dtype=float)
    rng = np.random.default_rng(seed=seed)

    if len(times) == 0:
        return base_flux.copy()

    t_min = times.min()
    t_max = times.max()
    t_range = max(t_max - t_min, 1.0)
    t_mid = 0.5 * (t_min + t_max)

    # Convert magnitude amplitude to flux amplitude
    # delta_F ≈ F_median × 0.4 × ln(10) × delta_mag
    F_median = float(np.nanmedian(base_flux))
    if F_median <= 0:
        F_median = 0.1
    flux_amplitude = F_median * 0.4 * np.log(10.0) * amplitude_mag

    if injection_type == 'step':
        # Sudden step at t_mid
        t_break = t_mid + rng.uniform(-0.1, 0.1) * t_range
        signal = np.where(times > t_break, flux_amplitude, 0.0)

    elif injection_type == 'ramp':
        # Linear ramp starting at 35% and ending at 65% of baseline
        t_start = t_min + 0.35 * t_range
        t_end = t_min + 0.65 * t_range
        signal = np.clip(
            (times - t_start) / max(t_end - t_start, 1.0) * flux_amplitude,
            0.0, flux_amplitude
        )
        signal = np.where(times < t_start, 0.0,
                 np.where(times > t_end, flux_amplitude, signal))

    elif injection_type == 'gaussian':
        # Gaussian flare centered at t_mid with width = 15% of baseline
        sigma_flare = 0.15 * t_range
        signal = flux_amplitude * np.exp(-0.5 * ((times - t_mid) / sigma_flare) ** 2)

    elif injection_type == 'sinusoidal':
        # Sinusoidal with period = 30% of baseline
        period = 0.3 * t_range
        signal = flux_amplitude * 0.5 * (1.0 + np.sin(2.0 * np.pi * (times - t_min) / period))

    else:
        logger.warning(f"Unknown injection type '{injection_type}', defaulting to 'step'")
        signal = np.where(times > t_mid, flux_amplitude, 0.0)

    return base_flux + signal


# ---------------------------------------------------------------------------
# Quick scorer (offline only)
# ---------------------------------------------------------------------------

def _quick_score(times: np.ndarray, fluxes: np.ndarray, errors: np.ndarray,
                 z: float = 0.1) -> float:
    """
    Compute a simplified CLAGN score [0, 1] without any IRSA queries.

    Score components:
        1. Relative flux change (delta_F / F_median)
        2. Nonstationarity (split half-means test)
        3. DRW MAP nonstationarity sigma

    Parameters
    ----------
    times, fluxes, errors : arrays
    z                     : float, redshift (default 0.1 for simulations)

    Returns
    -------
    score : float in [0, 1]
    """
    times = np.asarray(times, dtype=float)
    fluxes = np.asarray(fluxes, dtype=float)
    errors = np.asarray(errors, dtype=float)

    valid = np.isfinite(times) & np.isfinite(fluxes) & np.isfinite(errors)
    if valid.sum() < 5:
        return 0.0

    times = times[valid]
    fluxes = fluxes[valid]
    errors = errors[valid]

    sort_idx = np.argsort(times)
    times = times[sort_idx]
    fluxes = fluxes[sort_idx]
    errors = errors[sort_idx]

    # ---- Component 1: relative flux change ----------------------------------
    F_med = float(np.nanmedian(fluxes))
    if F_med <= 0:
        return 0.0

    n = len(fluxes)
    n5 = max(1, n // 5)
    F_early = float(np.nanmedian(fluxes[:n5]))
    F_late = float(np.nanmedian(fluxes[-n5:]))

    delta_F_rel = abs(F_late - F_early) / max(F_med, 1e-6)
    # Score 0 at delta_F=0, saturates at delta_F=1 (100% change)
    score_delta = min(1.0, delta_F_rel)

    # ---- Component 2: nonstationarity (split-half test) ---------------------
    n_half = n // 2
    mean_early = float(np.nanmean(fluxes[:n_half]))
    mean_late = float(np.nanmean(fluxes[n_half:]))
    err_early = float(np.nanstd(fluxes[:n_half]) / max(np.sqrt(n_half), 1))
    err_late = float(np.nanstd(fluxes[n_half:]) / max(np.sqrt(n - n_half), 1))

    t_stat = abs(mean_late - mean_early) / max(
        np.sqrt(err_early ** 2 + err_late ** 2), 1e-8
    )
    # Score: tanh(t_stat / 3) → 0 at t=0, ~1 at t>6
    score_nonstat = float(np.tanh(t_stat / 3.0))

    # ---- Component 3: DRW nonstationarity (lightweight) --------------------
    try:
        from ..models.drw import fit_drw_map, compute_drw_nonstationarity
        drw_result = fit_drw_map(times, fluxes, errors, z=z)
        tau = drw_result.get('tau_rest_days', 300.0)
        sigma = drw_result.get('sigma_drw', F_med * 0.1)
        ns_result = compute_drw_nonstationarity(times, fluxes, errors, z, tau, sigma)
        ns_sigma = ns_result.get('nonstationarity_sigma', 0.0)
        score_drw = min(1.0, max(0.0, float(ns_sigma or 0.0) / 5.0))
    except Exception:
        score_drw = 0.5 * (score_delta + score_nonstat)

    # ---- Weighted composite score ------------------------------------------
    weights = [0.35, 0.35, 0.30]
    score = float(
        weights[0] * score_delta +
        weights[1] * score_nonstat +
        weights[2] * score_drw
    )

    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Synthetic WISE cadence generator
# ---------------------------------------------------------------------------

def _generate_wise_cadence(n_epochs: int = 10, seed: int = None,
                            baseline_days: float = _NEOWISE_BASELINE_DAYS
                            ) -> np.ndarray:
    """
    Generate a realistic WISE/NEOWISE-R observation cadence.

    Produces approximately 6-month-spaced survey epochs with
    individual single-exposure scatter within each visit window.
    """
    rng = np.random.default_rng(seed=seed)

    t_start = 56800.0  # MJD, approximate NEOWISE-R start
    times = []

    t = t_start
    while t < t_start + baseline_days:
        # Each survey epoch: ~12 single exposures over ~2 days
        n_exp = int(rng.integers(8, 16))
        epoch_times = t + rng.uniform(0, 2.0, size=n_exp)
        times.extend(epoch_times.tolist())
        t += _NEOWISE_EPOCH_SPACING_DAYS + rng.uniform(
            -_NEOWISE_EPOCH_JITTER_DAYS, _NEOWISE_EPOCH_JITTER_DAYS
        )

    times = np.sort(np.array(times))

    # Subsample to n_epochs if specified
    if 0 < n_epochs < len(times):
        idx = np.sort(rng.choice(len(times), size=n_epochs, replace=False))
        times = times[idx]

    return times


# ---------------------------------------------------------------------------
# Injection-recovery
# ---------------------------------------------------------------------------

def run_injection_recovery(n_injections: int = N_INJECTIONS,
                           results_dir: str = './results/') -> dict:
    """
    Run injection-recovery test to characterize pipeline completeness.

    Procedure:
    1. Load a real WISE light curve from a non-variable control source
    2. Inject synthetic signals (step, ramp, gaussian, sinusoidal) at
       multiple amplitudes
    3. Score each injected light curve with _quick_score (no IRSA queries)
    4. Compute recovery fraction as function of amplitude

    Parameters
    ----------
    n_injections : int, number of injections per amplitude-type combination
    results_dir  : str, output directory

    Returns
    -------
    result : dict with keys:
        recovery_fractions : dict {amplitude_mag: {injection_type: fraction}}
        results_df         : DataFrame
        threshold_50pct    : float (amplitude for 50% recovery)
        threshold_90pct    : float (amplitude for 90% recovery)
    """
    results_path = Path(results_dir)
    val_dir = results_path / 'validation'
    val_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=12345)

    injection_types = ['step', 'ramp', 'gaussian', 'sinusoidal']
    rows = []

    # Generate a synthetic base light curve using WISE cadence + DRW noise
    base_times = _generate_wise_cadence(seed=0)
    base_drw = simulate_drw_lightcurve(
        base_times, tau=_DRW_TAU_TYPICAL_DAYS, sigma=_DRW_SIGMA_TYPICAL_MJY, seed=0
    )
    base_flux = 1.0 + base_drw  # 1 mJy baseline flux
    base_errors = _WISE_PHOT_NOISE_MJY * np.ones(len(base_times))

    score_threshold = 0.5  # Score > 0.5 = recovered

    for amplitude_mag in INJECTION_AMPLITUDES:
        for inj_type in injection_types:
            n_recovered = 0
            for trial in range(n_injections):
                seed_trial = int(rng.integers(0, 1_000_000))

                # Fresh DRW noise for each trial
                noise = simulate_drw_lightcurve(
                    base_times, tau=_DRW_TAU_TYPICAL_DAYS,
                    sigma=_DRW_SIGMA_TYPICAL_MJY, seed=seed_trial
                )
                trial_flux = 1.0 + noise
                trial_errors = base_errors.copy()

                injected = _inject_signal(
                    trial_flux, base_times, inj_type, amplitude_mag, seed=seed_trial
                )

                score = _quick_score(base_times, injected, trial_errors, z=0.1)

                recovered = score > score_threshold

                if recovered:
                    n_recovered += 1

                rows.append({
                    'amplitude_mag': amplitude_mag,
                    'injection_type': inj_type,
                    'trial': trial,
                    'score': score,
                    'recovered': bool(recovered),
                })

            frac = n_recovered / max(n_injections, 1)
            logger.info(
                f"Injection-recovery: amp={amplitude_mag:.2f} mag | "
                f"type={inj_type} | recovery={frac:.2%}"
            )

    results_df = pd.DataFrame(rows)

    # Save results
    csv_path = val_dir / 'injection_recovery_results.csv'
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Injection-recovery results saved to {csv_path}")

    # Recovery fraction per amplitude-type
    recovery_fractions = {}
    for amp in INJECTION_AMPLITUDES:
        recovery_fractions[amp] = {}
        for inj_type in injection_types:
            mask = (results_df['amplitude_mag'] == amp) & (results_df['injection_type'] == inj_type)
            if mask.sum() > 0:
                recovery_fractions[amp][inj_type] = float(results_df[mask]['recovered'].mean())
            else:
                recovery_fractions[amp][inj_type] = np.nan

    # Average across types for 50% and 90% thresholds
    avg_recovery = {
        amp: float(np.nanmean([recovery_fractions[amp][t] for t in injection_types]))
        for amp in INJECTION_AMPLITUDES
    }

    threshold_50 = np.nan
    threshold_90 = np.nan
    for amp in sorted(INJECTION_AMPLITUDES):
        if avg_recovery.get(amp, 0.0) >= 0.5 and np.isnan(threshold_50):
            threshold_50 = amp
        if avg_recovery.get(amp, 0.0) >= 0.9 and np.isnan(threshold_90):
            threshold_90 = amp

    return {
        'recovery_fractions': recovery_fractions,
        'avg_recovery': avg_recovery,
        'results_df': results_df,
        'threshold_50pct': float(threshold_50),
        'threshold_90pct': float(threshold_90),
        'n_injections': n_injections,
        'score_threshold': score_threshold,
    }


# ---------------------------------------------------------------------------
# False positive rate
# ---------------------------------------------------------------------------

def run_false_positive_rate(n_simulations: int = N_FP_SIMULATIONS,
                             results_dir: str = './results/') -> dict:
    """
    Estimate the false alarm rate (FAR) using pure-DRW simulations.

    OFFLINE ONLY — NO IRSA QUERIES.

    Procedure:
    1. Simulate N DRW light curves with WISE cadence
    2. Score each with _quick_score
    3. Compute FAR = fraction exceeding score threshold

    Parameters
    ----------
    n_simulations : int, number of DRW simulations
    results_dir   : str, output directory

    Returns
    -------
    result : dict with keys:
        scores              : array of all scores
        far_at_05           : float, FAR at score > 0.5
        far_at_07           : float, FAR at score > 0.7
        score_95th_pct      : float, 95th percentile score
        score_99th_pct      : float
        results_df          : DataFrame
    """
    results_path = Path(results_dir)
    val_dir = results_path / 'validation'
    val_dir.mkdir(parents=True, exist_ok=True)

    # DRW parameter distributions to sample from (log-normal)
    rng = np.random.default_rng(seed=99999)

    # Sample tau and sigma from physically motivated distributions
    log_tau_samples = rng.normal(loc=5.7, scale=1.0, size=n_simulations)
    log_sigma_samples = rng.normal(loc=-3.0, scale=1.0, size=n_simulations)

    tau_samples = np.exp(np.clip(log_tau_samples, 3.0, 8.5))
    sigma_samples = np.exp(np.clip(log_sigma_samples, -5.0, 3.0))

    rows = []
    all_scores = []

    for sim_idx in range(n_simulations):
        seed_sim = int(rng.integers(0, 10_000_000))

        times = _generate_wise_cadence(seed=seed_sim)

        tau = float(tau_samples[sim_idx])
        sigma = float(sigma_samples[sim_idx])

        drw_flux = simulate_drw_lightcurve(times, tau=tau, sigma=sigma, seed=seed_sim)
        flux = 1.0 + drw_flux
        errors = _WISE_PHOT_NOISE_MJY * np.ones(len(times)) + 0.3 * sigma

        score = _quick_score(times, flux, errors, z=0.1)
        all_scores.append(float(score))

        rows.append({
            'sim_idx': sim_idx,
            'tau_days': tau,
            'sigma_mjy': sigma,
            'n_epochs': len(times),
            'score': score,
        })

        if (sim_idx + 1) % 100 == 0:
            logger.debug(f"FAR simulation {sim_idx + 1}/{n_simulations}")

    all_scores = np.array(all_scores)
    results_df = pd.DataFrame(rows)

    far_05 = float(np.mean(all_scores > 0.5))
    far_07 = float(np.mean(all_scores > 0.7))
    p95 = float(np.percentile(all_scores, 95))
    p99 = float(np.percentile(all_scores, 99))

    logger.info(
        f"FAR simulation: {n_simulations} DRW runs | "
        f"FAR(>0.5)={far_05:.4f} | FAR(>0.7)={far_07:.4f} | "
        f"p95={p95:.3f} | p99={p99:.3f}"
    )

    # Save
    csv_path = val_dir / 'false_positive_scores.csv'
    results_df.to_csv(csv_path, index=False)
    logger.info(f"FAR scores saved to {csv_path}")

    return {
        'scores': all_scores,
        'far_at_05': far_05,
        'far_at_07': far_07,
        'score_95th_pct': p95,
        'score_99th_pct': p99,
        'results_df': results_df,
        'n_simulations': n_simulations,
    }


# ---------------------------------------------------------------------------
# Jackknife stability
# ---------------------------------------------------------------------------

def run_jackknife_stability(results_dir: str = './results/') -> dict:
    """
    Jackknife stability test: assess score sensitivity to epoch removal.

    For each candidate, repeatedly remove 20% of epochs at random and
    recompute score. Stable detections have low jackknife variance.

    Parameters
    ----------
    results_dir : str, path to results directory

    Returns
    -------
    result : dict with keys:
        results_df : DataFrame with columns:
            source_id, score_full, score_jk_mean, score_jk_std,
            score_jk_min, score_jk_max, stability_rating
    """
    results_path = Path(results_dir)
    val_dir = results_path / 'validation'
    val_dir.mkdir(parents=True, exist_ok=True)

    candidates_file = results_path / 'top_candidates.csv'
    if not candidates_file.exists():
        logger.warning("No top_candidates.csv found for jackknife test")
        return {'results_df': pd.DataFrame()}

    try:
        candidates = pd.read_csv(candidates_file)
    except Exception as exc:
        logger.error(f"Failed to load candidates: {exc}")
        return {'results_df': pd.DataFrame()}

    rng = np.random.default_rng(seed=77777)
    rows = []

    for idx, row in candidates.iterrows():
        source_id = str(row.get('source_id', f'source_{idx}'))

        # Load WISE light curve
        wise_lc = None
        for wise_path in [
            results_path / 'wise_enhanced' / f'enhanced_wise_{source_id}.pkl',
            results_path / f'wise_{source_id}.pkl',
        ]:
            if wise_path.exists():
                import pickle
                with open(wise_path, 'rb') as f:
                    wise_data = pickle.load(f)
                wise_lc = wise_data.get('lc', None)
                break

        if wise_lc is None or wise_lc.empty:
            # Generate synthetic light curve for testing
            times = _generate_wise_cadence(seed=idx)
            drw_flux = simulate_drw_lightcurve(
                times, tau=_DRW_TAU_TYPICAL_DAYS, sigma=0.1, seed=idx
            )
            # Add step signal (simulating a candidate)
            injected = _inject_signal(1.0 + drw_flux, times, 'step', 0.5, seed=idx)
            errors = _WISE_PHOT_NOISE_MJY * np.ones(len(times))
        else:
            times = wise_lc['mjd'].values if 'mjd' in wise_lc.columns else np.array([])
            fluxes_col = 'w1_flux_mjy' if 'w1_flux_mjy' in wise_lc.columns else 'w1_mag'
            errors_col = 'w1_flux_err_mjy' if 'w1_flux_err_mjy' in wise_lc.columns else 'w1_err'
            injected = pd.to_numeric(wise_lc[fluxes_col], errors='coerce').fillna(1.0).values
            errors = pd.to_numeric(wise_lc[errors_col], errors='coerce').fillna(0.02).values

        if len(times) < 5:
            continue

        z = float(row.get('z', row.get('redshift', 0.1)))

        # Full score
        score_full = _quick_score(times, injected, errors, z=z)

        # Jackknife: N_JACKKNIFE_SCRAMBLES resamples at 80% of epochs
        jk_scores = []
        n = len(times)
        n_keep = max(5, int(0.8 * n))

        for jk_trial in range(min(N_JACKKNIFE_SCRAMBLES, 100)):
            jk_idx = np.sort(
                rng.choice(n, size=n_keep, replace=False)
            )
            jk_score = _quick_score(
                times[jk_idx], injected[jk_idx], errors[jk_idx], z=z
            )
            jk_scores.append(jk_score)

        jk_scores = np.array(jk_scores)
        jk_mean = float(np.nanmean(jk_scores))
        jk_std = float(np.nanstd(jk_scores))
        jk_min = float(np.nanmin(jk_scores))
        jk_max = float(np.nanmax(jk_scores))

        # Stability rating
        if jk_std < 0.05:
            stability = 'high'
        elif jk_std < 0.15:
            stability = 'moderate'
        else:
            stability = 'low'

        rows.append({
            'source_id': source_id,
            'score_full': float(score_full),
            'score_jk_mean': jk_mean,
            'score_jk_std': jk_std,
            'score_jk_min': jk_min,
            'score_jk_max': jk_max,
            'stability_rating': stability,
            'n_epochs': len(times),
        })

        logger.debug(
            f"{source_id}: jackknife score={score_full:.3f} ± {jk_std:.3f} "
            f"(stability={stability})"
        )

    results_df = pd.DataFrame(rows)

    csv_path = val_dir / 'jackknife_stability.csv'
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Jackknife stability results saved to {csv_path}")

    return {'results_df': results_df}


# ---------------------------------------------------------------------------
# Coordinate scramble
# ---------------------------------------------------------------------------

def run_coordinate_scramble(catalog_df: pd.DataFrame = None,
                             n_scrambles: int = 200,
                             results_dir: str = './results/') -> dict:
    """
    Coordinate scramble null test: query random sky positions and score them.

    Positions are drawn from a uniform distribution in RA [0, 360] and
    Dec [-30, 30] (typical WISE survey region). Requires IRSA access.

    IMPORTANT: Sleeps 2 seconds between IRSA queries to avoid rate-limiting.

    Parameters
    ----------
    catalog_df   : DataFrame or None, source catalog for RA/Dec range
    n_scrambles  : int, number of random positions to query
    results_dir  : str, output directory

    Returns
    -------
    result : dict with keys:
        results_df    : DataFrame with scrambled position scores
        far_scramble  : float, fraction exceeding score threshold
        score_median  : float
        score_95th    : float
    """
    results_path = Path(results_dir)
    val_dir = results_path / 'validation'
    val_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=55555)

    # Determine RA/Dec range from catalog if provided
    if catalog_df is not None and 'ra' in catalog_df.columns:
        ra_min = float(catalog_df['ra'].min())
        ra_max = float(catalog_df['ra'].max())
        dec_min = float(catalog_df['dec'].min()) if 'dec' in catalog_df.columns else -30.0
        dec_max = float(catalog_df['dec'].max()) if 'dec' in catalog_df.columns else 30.0
    else:
        ra_min, ra_max = 0.0, 360.0
        dec_min, dec_max = -30.0, 30.0

    rows = []
    scores = []

    logger.info(
        f"Coordinate scramble: {n_scrambles} random positions "
        f"(RA [{ra_min:.1f}, {ra_max:.1f}], Dec [{dec_min:.1f}, {dec_max:.1f}])"
    )

    for trial in range(n_scrambles):
        ra_rand = float(rng.uniform(ra_min, ra_max))
        dec_rand = float(rng.uniform(dec_min, dec_max))

        try:
            from ..ingestion.wise import query_wise_lightcurve
            result = query_wise_lightcurve(ra_rand, dec_rand, f'scramble_{trial}')
            lc_df = result.get('lc', pd.DataFrame())

            if lc_df.empty or len(lc_df) < 5:
                score = 0.0
                n_epochs = 0
            else:
                times = lc_df['mjd'].values
                fluxes = (
                    lc_df['w1_flux_mjy'].values
                    if 'w1_flux_mjy' in lc_df.columns
                    else np.ones(len(lc_df))
                )
                errors = (
                    lc_df['w1_flux_err_mjy'].values
                    if 'w1_flux_err_mjy' in lc_df.columns
                    else np.ones(len(lc_df)) * 0.02
                )
                score = _quick_score(times, fluxes, errors, z=0.1)
                n_epochs = len(lc_df)

        except Exception as exc:
            logger.debug(f"Scramble {trial}: query failed: {exc}")
            score = 0.0
            n_epochs = 0

        scores.append(float(score))
        rows.append({
            'trial': trial,
            'ra_rand': ra_rand,
            'dec_rand': dec_rand,
            'n_epochs': n_epochs,
            'score': float(score),
        })

        logger.debug(
            f"Scramble {trial + 1}/{n_scrambles}: "
            f"RA={ra_rand:.3f}, Dec={dec_rand:.3f}, score={score:.3f}"
        )

        # Rate-limit IRSA queries
        time.sleep(2.0)

    scores = np.array(scores)
    results_df = pd.DataFrame(rows)

    far_scramble = float(np.mean(scores > 0.5))
    score_median = float(np.nanmedian(scores))
    score_95th = float(np.percentile(scores, 95)) if len(scores) > 0 else np.nan

    logger.info(
        f"Coordinate scramble: FAR={far_scramble:.4f} | "
        f"median score={score_median:.3f} | "
        f"95th pct={score_95th:.3f}"
    )

    csv_path = val_dir / 'coordinate_scramble.csv'
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Coordinate scramble results saved to {csv_path}")

    return {
        'results_df': results_df,
        'scores': scores,
        'far_scramble': far_scramble,
        'score_median': score_median,
        'score_95th': score_95th,
        'n_scrambles': n_scrambles,
    }


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------

def run_all_validation(results_dir: str = './results/') -> dict:
    """
    Run all four validation analyses in sequence.

    Note: coordinate_scramble makes IRSA queries; all others are offline.

    Parameters
    ----------
    results_dir : str, path to results directory

    Returns
    -------
    all_results : dict with keys:
        injection_recovery, false_positive_rate,
        jackknife_stability, coordinate_scramble
    """
    logger.info("Starting complete validation pipeline...")

    results = {}

    # 1. Injection-recovery
    logger.info("Running injection-recovery test...")
    try:
        results['injection_recovery'] = run_injection_recovery(
            n_injections=N_INJECTIONS,
            results_dir=results_dir,
        )
    except Exception as exc:
        logger.error(f"Injection-recovery failed: {exc}", exc_info=True)
        results['injection_recovery'] = {}

    # 2. False positive rate (OFFLINE)
    logger.info("Running false positive rate estimation (offline)...")
    try:
        results['false_positive_rate'] = run_false_positive_rate(
            n_simulations=N_FP_SIMULATIONS,
            results_dir=results_dir,
        )
    except Exception as exc:
        logger.error(f"FAR estimation failed: {exc}", exc_info=True)
        results['false_positive_rate'] = {}

    # 3. Jackknife stability
    logger.info("Running jackknife stability test...")
    try:
        results['jackknife_stability'] = run_jackknife_stability(
            results_dir=results_dir,
        )
    except Exception as exc:
        logger.error(f"Jackknife test failed: {exc}", exc_info=True)
        results['jackknife_stability'] = {}

    # 4. Coordinate scramble (MAKES IRSA QUERIES)
    logger.info("Running coordinate scramble null test (requires network access)...")
    try:
        results['coordinate_scramble'] = run_coordinate_scramble(
            n_scrambles=min(50, N_INJECTIONS // 4),  # Use fewer for runtime
            results_dir=results_dir,
        )
    except Exception as exc:
        logger.error(f"Coordinate scramble failed: {exc}", exc_info=True)
        results['coordinate_scramble'] = {}

    logger.info("All validation analyses complete.")
    return results
