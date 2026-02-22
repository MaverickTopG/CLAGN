"""
gaia_lightcurve.py — GAIA DR3 epoch G-band photometry and astrometric variability.

Provides optical time-domain support for WISE-selected CLAGN candidates.
Gaia G-band (330–1050 nm) epoch photometry directly tests whether optical
variability accompanies the IR changes seen in WISE.

References:
    Gaia Collaboration et al. 2022, A&A 674, A1 (Gaia DR3)
    Holl et al. 2023, A&A 674, A25 (Gaia variability)
    Lindegren et al. 2021, A&A 649, A4 (astrometric solution)
"""
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ADQL query builder
# ---------------------------------------------------------------------------

def _gaia_tap_query(adql: str, max_retries: int = 3) -> pd.DataFrame:
    """Execute a Gaia TAP+ query via astroquery with retry."""
    import time
    last_exc = None

    for attempt in range(max_retries):
        try:
            from astroquery.gaia import Gaia
            job = Gaia.launch_job(adql)
            table = job.get_results()
            return table.to_pandas()
        except Exception as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning(
                f"Gaia TAP attempt {attempt + 1}/{max_retries} failed: {exc}. "
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Gaia TAP query failed after {max_retries} attempts: {last_exc}"
    )


# ---------------------------------------------------------------------------
# Epoch photometry
# ---------------------------------------------------------------------------

def query_gaia_epoch_photometry(source_id: str, gaia_source_id: int) -> dict:
    """
    Query Gaia DR3 epoch G-band photometry for a single source.

    Uses the gaiadr3.epoch_photometry table which contains individual
    CCD transit measurements forming the G-band light curve.

    Parameters
    ----------
    source_id      : str, internal pipeline source identifier
    gaia_source_id : int, Gaia DR3 source_id (64-bit integer)

    Returns
    -------
    result : dict with keys:
        available      : bool
        n_epochs       : int
        times_tcb      : array, Barycentric Coordinate Time (TCB) days
        g_mag          : array, G-band magnitudes
        g_mag_err      : array, G-band magnitude uncertainties
        g_flux         : array, G-band flux (e- s^-1)
        g_flux_err     : array, G-band flux error
        median_g_mag   : float
        g_rms          : float, RMS variability
        delta_g_mag    : float, peak-to-peak amplitude
        rejection_reason : str or None
    """
    _empty = {
        'available': False,
        'n_epochs': 0,
        'times_tcb': np.array([]),
        'g_mag': np.array([]),
        'g_mag_err': np.array([]),
        'g_flux': np.array([]),
        'g_flux_err': np.array([]),
        'median_g_mag': np.nan,
        'g_rms': np.nan,
        'delta_g_mag': np.nan,
        'rejection_reason': 'Not attempted',
    }

    if gaia_source_id is None or not np.isfinite(float(gaia_source_id)):
        _empty['rejection_reason'] = 'No valid Gaia source_id'
        return _empty

    adql = (
        f"SELECT TOP 1000 "
        f"source_id, transit_id, time, mag, flux, flux_error, "
        f"rejected_by_photometry, rejected_by_variability "
        f"FROM gaiadr3.epoch_photometry "
        f"WHERE source_id = {int(gaia_source_id)}"
    )

    try:
        df = _gaia_tap_query(adql)
    except Exception as exc:
        _empty['rejection_reason'] = f'Query failed: {exc}'
        return _empty

    if df is None or df.empty:
        _empty['rejection_reason'] = 'No epoch photometry returned from Gaia'
        return _empty

    # Filter rejected transits
    for col in ['rejected_by_photometry', 'rejected_by_variability']:
        if col in df.columns:
            df = df[~df[col].astype(bool)]

    if df.empty:
        _empty['rejection_reason'] = 'All epochs rejected by Gaia pipeline'
        return _empty

    # Extract arrays
    times = pd.to_numeric(df.get('time', pd.Series(dtype=float)), errors='coerce').values
    flux = pd.to_numeric(df.get('flux', pd.Series(dtype=float)), errors='coerce').values
    flux_err = pd.to_numeric(df.get('flux_error', pd.Series(dtype=float)), errors='coerce').values
    mag = pd.to_numeric(df.get('mag', pd.Series(dtype=float)), errors='coerce').values

    valid = np.isfinite(times) & np.isfinite(flux) & np.isfinite(mag)
    times = times[valid]
    flux = flux[valid]
    flux_err = flux_err[valid] if len(flux_err) == len(valid) else np.ones_like(flux) * 0.01
    mag = mag[valid]

    # Sort by time
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    flux = flux[sort_idx]
    flux_err = flux_err[sort_idx]
    mag = mag[sort_idx]

    # Magnitude errors from flux SNR: σ_mag = 1.086 / SNR
    snr = np.where(flux_err > 0, flux / flux_err, 1.0)
    mag_err = np.where(snr > 0, 1.086 / snr, 0.1)

    n = len(times)
    if n < 3:
        _empty['rejection_reason'] = f'Only {n} valid Gaia epochs'
        return _empty

    median_g = float(np.nanmedian(mag))
    g_rms = float(np.nanstd(mag))
    delta_g = float(np.nanmax(mag) - np.nanmin(mag)) if n > 1 else np.nan

    logger.info(
        f"{source_id}: Gaia epoch photometry — {n} transits, "
        f"median G={median_g:.3f} mag, RMS={g_rms:.4f} mag"
    )

    return {
        'available': True,
        'n_epochs': n,
        'times_tcb': times,
        'g_mag': mag,
        'g_mag_err': mag_err,
        'g_flux': flux,
        'g_flux_err': flux_err,
        'median_g_mag': median_g,
        'g_rms': g_rms,
        'delta_g_mag': delta_g,
        'rejection_reason': None,
    }


# ---------------------------------------------------------------------------
# Optical/IR variability ratio
# ---------------------------------------------------------------------------

def compute_optical_ir_variability_ratio(gaia_lc: dict, wise_lc: dict) -> float:
    """
    Compute the ratio of optical (Gaia G) to infrared (WISE W1) variability amplitudes.

    R_var = delta_G_mag / delta_W1_mag

    For dust-echo CLAGN: R_var > 1 (optical leads, IR follows with lag).
    For dust-dominated variability: R_var < 1.
    For accretion disk changes: R_var ~ 1–3.

    Parameters
    ----------
    gaia_lc : dict, output of query_gaia_epoch_photometry
    wise_lc : dict, output of query_wise_lightcurve or query_all_wise_epochs

    Returns
    -------
    R_var : float, NaN if either amplitude is unavailable
    """
    g_rms = gaia_lc.get('g_rms', np.nan)

    # WISE W1 RMS from light curve DataFrame
    w_lc_df = wise_lc.get('lc', pd.DataFrame())
    if not w_lc_df.empty and 'w1_mag' in w_lc_df.columns:
        w1_mags = pd.to_numeric(w_lc_df['w1_mag'], errors='coerce').dropna().values
        w1_rms = float(np.nanstd(w1_mags)) if len(w1_mags) > 1 else np.nan
    else:
        w1_rms = np.nan

    if not (np.isfinite(g_rms) and np.isfinite(w1_rms) and w1_rms > 0):
        return np.nan

    return float(g_rms / w1_rms)


# ---------------------------------------------------------------------------
# Structure function
# ---------------------------------------------------------------------------

def compute_gaia_structure_function(gaia_times: np.ndarray,
                                    gaia_mags: np.ndarray,
                                    gaia_errors: np.ndarray) -> dict:
    """
    Compute the first-order structure function of the Gaia G-band light curve.

    SF(tau) = <[m(t+tau) - m(t)]^2 - sigma_m^2> ^(1/2)

    where sigma_m^2 is the mean photometric variance (noise floor correction).

    Parameters
    ----------
    gaia_times  : array, times in days (TCB)
    gaia_mags   : array, G-band magnitudes
    gaia_errors : array, photometric uncertainties

    Returns
    -------
    result : dict with keys:
        lag_days     : array
        sf_values    : array
        sf_errors    : array (bootstrap 1-sigma)
        sf_slope     : float (power-law index from log-log fit)
        sf_amplitude : float (SF at tau=100 days)
        n_lag_bins   : int
    """
    times = np.asarray(gaia_times, dtype=float)
    mags = np.asarray(gaia_mags, dtype=float)
    errors = np.asarray(gaia_errors, dtype=float)

    if len(times) < 5:
        return {
            'lag_days': np.array([]),
            'sf_values': np.array([]),
            'sf_errors': np.array([]),
            'sf_slope': np.nan,
            'sf_amplitude': np.nan,
            'n_lag_bins': 0,
        }

    n = len(times)
    lags = []
    diffs_sq = []
    noise_sq = []

    # Compute all pairwise structure function values
    for i in range(n):
        for j in range(i + 1, n):
            lag = abs(times[j] - times[i])
            diff_sq = (mags[j] - mags[i]) ** 2
            noise = errors[i] ** 2 + errors[j] ** 2
            lags.append(lag)
            diffs_sq.append(diff_sq)
            noise_sq.append(noise)

    lags = np.array(lags)
    diffs_sq = np.array(diffs_sq)
    noise_sq = np.array(noise_sq)

    # Bin in log-lag
    lag_min = max(lags.min(), 1.0)
    lag_max = lags.max()

    if lag_max <= lag_min:
        return {
            'lag_days': np.array([]),
            'sf_values': np.array([]),
            'sf_errors': np.array([]),
            'sf_slope': np.nan,
            'sf_amplitude': np.nan,
            'n_lag_bins': 0,
        }

    n_bins = 15
    edges = np.logspace(np.log10(lag_min), np.log10(lag_max), n_bins + 1)
    lag_centers = []
    sf_vals = []
    sf_errs = []

    for k in range(n_bins):
        mask = (lags >= edges[k]) & (lags < edges[k + 1])
        if mask.sum() < 3:
            continue
        sf_sq = np.maximum(diffs_sq[mask] - noise_sq[mask], 0.0)
        sf_mean = float(np.sqrt(np.nanmean(sf_sq)))
        # Bootstrap error
        n_boot = 100
        boot_vals = []
        idx_all = np.where(mask)[0]
        rng = np.random.default_rng(seed=42)
        for _ in range(n_boot):
            boot_idx = rng.choice(idx_all, size=len(idx_all), replace=True)
            boot_sq = np.maximum(diffs_sq[boot_idx] - noise_sq[boot_idx], 0.0)
            boot_vals.append(float(np.sqrt(np.nanmean(boot_sq))))
        sf_err = float(np.nanstd(boot_vals))

        lag_centers.append(float(0.5 * (edges[k] + edges[k + 1])))
        sf_vals.append(sf_mean)
        sf_errs.append(sf_err)

    lag_arr = np.array(lag_centers)
    sf_arr = np.array(sf_vals)
    sf_err_arr = np.array(sf_errs)

    # Power-law fit in log-log space
    sf_slope = np.nan
    sf_amp = np.nan
    if len(lag_arr) >= 3:
        try:
            good = sf_arr > 0
            if good.sum() >= 3:
                log_lag = np.log10(lag_arr[good])
                log_sf = np.log10(sf_arr[good])
                coeffs = np.polyfit(log_lag, log_sf, 1)
                sf_slope = float(coeffs[0])
                # Amplitude at tau=100 days
                sf_amp = float(10 ** (coeffs[1] + coeffs[0] * np.log10(100.0)))
        except Exception:
            pass

    return {
        'lag_days': lag_arr,
        'sf_values': sf_arr,
        'sf_errors': sf_err_arr,
        'sf_slope': sf_slope,
        'sf_amplitude': sf_amp,
        'n_lag_bins': len(lag_arr),
    }


# ---------------------------------------------------------------------------
# Astrometric variability
# ---------------------------------------------------------------------------

def analyze_gaia_astrometric_variability(gaia_data: dict) -> dict:
    """
    Assess astrometric variability from Gaia DR3 astrometric_excess_noise.

    The astrometric_excess_noise (epsilon) quantifies the scatter in the
    astrometric fit beyond formal measurement errors. For point sources,
    epsilon > 0 may indicate:
    - Photocentric variability (CLAGN candidate — accretion disk brightening)
    - Binarity or blending
    - Astrometric calibration residuals

    A significant astrometric_excess_noise_sig (>2) with epsilon > 0.5 mas
    is considered a positive indicator.

    Parameters
    ----------
    gaia_data : dict with keys from Gaia main catalog query:
        astrometric_excess_noise, astrometric_excess_noise_sig,
        ruwe, pmra, pmra_error, pmdec, pmdec_error

    Returns
    -------
    result : dict with keys:
        excess_noise_mas    : float
        excess_noise_sig    : float
        is_astrometrically_variable : bool
        ruwe                : float
        pm_sig              : float (max of pmra_sig, pmdec_sig in sigma units)
        is_likely_star      : bool (PM > 3 sigma AND ruwe < 1.4)
        astrometric_score   : float [0, 1]
    """
    aen = float(gaia_data.get('astrometric_excess_noise', 0.0) or 0.0)
    aen_sig = float(gaia_data.get('astrometric_excess_noise_sig', 0.0) or 0.0)
    ruwe = float(gaia_data.get('ruwe', 1.0) or 1.0)

    pmra = float(gaia_data.get('pmra', 0.0) or 0.0)
    pmra_e = float(gaia_data.get('pmra_error', 1.0) or 1.0)
    pmdec = float(gaia_data.get('pmdec', 0.0) or 0.0)
    pmdec_e = float(gaia_data.get('pmdec_error', 1.0) or 1.0)

    pm_sig_ra = abs(pmra) / max(pmra_e, 1e-6)
    pm_sig_dec = abs(pmdec) / max(pmdec_e, 1e-6)
    pm_sig = float(max(pm_sig_ra, pm_sig_dec))

    # FIX 9: Add parallax significance test (BEST star rejection test)
    # abs(parallax / parallax_error) > 3 → foreground star
    parallax = gaia_data.get('parallax', None)
    parallax_error = gaia_data.get('parallax_error', None)
    plx_sig = 0.0
    star_reasons = []

    if parallax is not None and parallax_error is not None:
        try:
            plx = float(parallax or 0.0)
            plx_e = float(parallax_error or 1.0)
            plx_sig = abs(plx) / max(plx_e, 1e-10)
            if plx_sig > 3.0:
                star_reasons.append(f'parallax_sig={plx_sig:.1f}>3 (nearby star)')
        except (TypeError, ValueError):
            plx_sig = 0.0

    # Test 1: proper motion significance
    if pm_sig > 3.0:
        star_reasons.append(f'pm_sig={pm_sig:.1f}>3')

    # Test 3: RUWE (non-point-source morphology)
    if ruwe > 1.4:
        star_reasons.append(f'ruwe={ruwe:.2f}>1.4')

    # Test 4: GAIA DSC quasar classification
    qso_prob = gaia_data.get('classprob_dsc_combmod_quasar', None)
    if qso_prob is not None:
        try:
            qso_prob_f = float(qso_prob or 0.0)
            if qso_prob_f < 0.1:
                star_reasons.append(f'gaia_qso_prob={qso_prob_f:.2f}<0.1')
        except (TypeError, ValueError):
            pass

    # Source is flagged as likely star if ANY criterion triggers
    is_likely_star = len(star_reasons) > 0

    is_astrometrically_variable = (aen > 0.5) and (aen_sig > 2.0)

    # Score: high if significant astrometric excess, NOT star
    astrometric_score = 0.0
    if is_astrometrically_variable and not is_likely_star:
        astrometric_score = min(1.0, aen_sig / 10.0)

    return {
        'excess_noise_mas': aen,
        'excess_noise_sig': aen_sig,
        'is_astrometrically_variable': bool(is_astrometrically_variable),
        'ruwe': ruwe,
        'pm_sig': pm_sig,
        'parallax_sig': float(plx_sig),
        'is_likely_star': bool(is_likely_star),
        'star_rejection_reasons': star_reasons,
        'astrometric_score': float(astrometric_score),
    }


# ---------------------------------------------------------------------------
# Color variability
# ---------------------------------------------------------------------------

def compute_gaia_color_variability(gaia_data: dict) -> dict:
    """
    Assess Gaia color variability using mean photometry and variability flag.

    Uses G, BP, RP magnitudes and the phot_variable_flag from Gaia DR3
    main catalog. The BP-RP color traces the spectral slope of the AGN
    SED from 330 to 1050 nm.

    Parameters
    ----------
    gaia_data : dict with keys:
        phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
        phot_variable_flag, bp_rp, classprob_dsc_combmod_quasar

    Returns
    -------
    result : dict with keys:
        g_mag           : float
        bp_rp           : float
        is_variable     : bool
        quasar_prob     : float
        gaia_qso_flag   : bool
        color_class     : str ('blue', 'red', 'intermediate')
    """
    g_mag = float(gaia_data.get('phot_g_mean_mag', np.nan) or np.nan)
    bp_rp = float(gaia_data.get('bp_rp', np.nan) or np.nan)
    if not np.isfinite(bp_rp):
        bp_g = gaia_data.get('phot_bp_mean_mag', np.nan)
        rp_g = gaia_data.get('phot_rp_mean_mag', np.nan)
        if np.isfinite(float(bp_g or np.nan)) and np.isfinite(float(rp_g or np.nan)):
            bp_rp = float(bp_g) - float(rp_g)

    var_flag = str(gaia_data.get('phot_variable_flag', '') or '')
    is_variable = 'VARIABLE' in var_flag.upper()

    quasar_prob = float(gaia_data.get('classprob_dsc_combmod_quasar', 0.0) or 0.0)
    gaia_qso_flag = quasar_prob > 0.5

    # BP-RP color classification for AGN
    if np.isfinite(bp_rp):
        if bp_rp < 0.5:
            color_class = 'blue'  # Type 1 AGN / quasar
        elif bp_rp > 1.5:
            color_class = 'red'   # Reddened / Type 2 AGN
        else:
            color_class = 'intermediate'
    else:
        color_class = 'unknown'

    return {
        'g_mag': float(g_mag),
        'bp_rp': float(bp_rp),
        'is_variable': bool(is_variable),
        'quasar_prob': float(quasar_prob),
        'gaia_qso_flag': bool(gaia_qso_flag),
        'color_class': color_class,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_gaia_photometry(results_dir: str = './results/') -> None:
    """
    Run Gaia epoch photometry analysis for all top candidates.

    Loads top_candidates.csv (requires columns: source_id, gaia_source_id).
    For each candidate:
    - Queries Gaia epoch photometry
    - Computes structure function and astrometric variability
    - Saves gaia_{source_id}.pkl and gaia_summary.csv

    Parameters
    ----------
    results_dir : str, path to results directory
    """
    import pickle
    import time

    results_path = Path(results_dir)
    candidates_file = results_path / 'top_candidates.csv'

    if not candidates_file.exists():
        logger.error(f"Candidates file not found: {candidates_file}")
        return

    try:
        candidates = pd.read_csv(candidates_file)
    except Exception as exc:
        logger.error(f"Failed to load candidates: {exc}")
        return

    gaia_dir = results_path / 'gaia_photometry'
    gaia_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for idx, row in candidates.iterrows():
        source_id = str(row.get('source_id', f'source_{idx}'))
        gaia_source_id = row.get('gaia_source_id', None)

        if gaia_source_id is None or (
            isinstance(gaia_source_id, float) and not np.isfinite(gaia_source_id)
        ):
            logger.info(f"{source_id}: No Gaia source_id, skipping")
            continue

        logger.info(f"Gaia photometry for {source_id} (gaia_id={gaia_source_id})")

        try:
            lc = query_gaia_epoch_photometry(source_id, int(gaia_source_id))

            result = {'lc': lc}

            if lc['available'] and len(lc['times_tcb']) >= 5:
                # Structure function
                sf = compute_gaia_structure_function(
                    lc['times_tcb'], lc['g_mag'], lc['g_mag_err']
                )
                result['structure_function'] = sf

            # Astrometric variability from catalog data
            astro = analyze_gaia_astrometric_variability(row.to_dict())
            result['astrometric'] = astro

            # Color variability
            color = compute_gaia_color_variability(row.to_dict())
            result['color'] = color

            # Save per-source pickle
            pkl_path = gaia_dir / f'gaia_{source_id}.pkl'
            with open(pkl_path, 'wb') as f:
                pickle.dump(result, f, protocol=4)

            summary_rows.append({
                'source_id': source_id,
                'gaia_source_id': gaia_source_id,
                'n_epochs': lc['n_epochs'],
                'median_g_mag': lc['median_g_mag'],
                'g_rms': lc['g_rms'],
                'delta_g_mag': lc['delta_g_mag'],
                'is_variable': color['is_variable'],
                'quasar_prob': color['quasar_prob'],
                'astrometric_score': astro['astrometric_score'],
                'is_likely_star': astro['is_likely_star'],
                'available': lc['available'],
            })

        except Exception as exc:
            logger.error(f"{source_id}: Gaia analysis failed: {exc}", exc_info=True)

        time.sleep(1.0)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = results_path / 'gaia_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Gaia photometry summary saved to {summary_path}")
    else:
        logger.warning("No Gaia results to save.")
