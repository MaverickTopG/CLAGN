"""
structure_function.py — SF(Δt) computation with bootstrapped uncertainties.

The Structure Function is a core diagnostic for AGN variability and CLAGN
detection. It measures variability amplitude as a function of time lag.

Normal AGN: SF ∝ Δt^β (β ~ 0.3–0.5) then plateau at Δt >> τ_DRW
CLAGN: SF shows excess power at long lags OR a step-change at the state
       transition epoch — both deviate from a stationary DRW.

SF is defined and computed in MAGNITUDE space.
"""
import logging
import numpy as np
from scipy.optimize import curve_fit

from ..config import (
    SF_N_LAG_BINS, SF_N_BOOTSTRAP, SF_MIN_PAIRS_PER_BIN,
)
from ..utils.photometry import flux_to_mag

logger = logging.getLogger(__name__)


def compute_structure_function(times, mags, mag_errors, n_lag_bins=SF_N_LAG_BINS):
    """
    Compute the observed Structure Function SF(Δt) for a light curve.

    SF(Δt) = <(m(t+Δt) - m(t))² > - <2σ²>

    The second term subtracts the contribution of measurement noise.
    Uses log-spaced lag bins from Δt_min = cadence to Δt_max = baseline.
    Bootstrap uncertainties: SF_N_BOOTSTRAP resamples per lag bin (one bin
    at a time to bound memory usage).

    Parameters
    ----------
    times      : array, MJD
    mags       : array, magnitudes (NOT fluxes — SF is in mag space)
    mag_errors : array, magnitude uncertainties
    n_lag_bins : int

    Returns
    -------
    result : dict with keys:
        lag_centers  (array), sf_values (array), sf_errors (array),
        sf_slope     (float), sf_plateau (float),
        valid_bins   (bool array)
    """
    times = np.asarray(times, dtype=float)
    mags  = np.asarray(mags,  dtype=float)
    mag_errors = np.asarray(mag_errors, dtype=float)

    # Remove NaN entries
    ok = np.isfinite(times) & np.isfinite(mags) & np.isfinite(mag_errors)
    times, mags, mag_errors = times[ok], mags[ok], mag_errors[ok]

    n = len(times)
    if n < 10:
        logger.warning(f"SF: Only {n} clean mag points — insufficient for SF")
        return _empty_sf_result()

    # ---- Build all unique pairs --------------------------------------------
    i_idx, j_idx = np.triu_indices(n, k=1)

    # Cap at 50,000 pairs to bound memory usage
    MAX_PAIRS = 50_000
    if len(i_idx) > MAX_PAIRS:
        rng = np.random.default_rng(seed=42)
        sel = rng.choice(len(i_idx), size=MAX_PAIRS, replace=False)
        i_idx = i_idx[sel]
        j_idx = j_idx[sel]

    dt_all    = np.abs(times[i_idx] - times[j_idx])          # days
    dm_sq_all = (mags[i_idx] - mags[j_idx])**2
    noise_sq  = mag_errors[i_idx]**2 + mag_errors[j_idx]**2   # 2σ² term

    # ---- Log-spaced lag bins ------------------------------------------------
    dt_min = np.percentile(dt_all[dt_all > 0], 5)   # 5th percentile of non-zero lags
    dt_max = dt_all.max()
    bin_edges = np.logspace(np.log10(max(dt_min, 1.0)),
                             np.log10(dt_max), n_lag_bins + 1)

    lag_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])   # geometric mean

    sf_values = np.full(n_lag_bins, np.nan)
    sf_errors = np.full(n_lag_bins, np.nan)
    n_pairs   = np.zeros(n_lag_bins, dtype=int)

    rng = np.random.default_rng(seed=42)

    for b in range(n_lag_bins):
        in_bin = (dt_all >= bin_edges[b]) & (dt_all < bin_edges[b + 1])
        dm_sq_bin  = dm_sq_all[in_bin]
        noise_bin  = noise_sq[in_bin]
        n_b = in_bin.sum()
        n_pairs[b] = n_b

        if n_b < SF_MIN_PAIRS_PER_BIN:
            continue

        # SF = mean(dm²) - mean(2σ²)
        sf_val = float(np.mean(dm_sq_bin) - np.mean(noise_bin))
        sf_val = max(sf_val, 0.0)   # SF must be non-negative
        sf_values[b] = sf_val

        # Bootstrap uncertainty: resample pairs (with replacement)
        boot_sf = np.empty(SF_N_BOOTSTRAP)
        for k in range(SF_N_BOOTSTRAP):
            idx_b = rng.integers(0, n_b, size=n_b)
            boot_sf[k] = max(
                float(np.mean(dm_sq_bin[idx_b]) - np.mean(noise_bin[idx_b])),
                0.0
            )
        sf_errors[b] = float(np.std(boot_sf))

    valid = np.isfinite(sf_values) & (sf_values > 0)

    # ---- Fit power-law slope to short-lag SF --------------------------------
    sf_slope = np.nan
    sf_plateau = np.nan

    if valid.sum() >= 4:
        try:
            sf_slope = _fit_sf_slope_short_lags(
                lag_centers[valid], sf_values[valid], sf_errors[valid]
            )
        except Exception as exc:
            logger.debug(f"SF slope fit failed: {exc}")

        # Plateau = median of SF at the longest 30% of valid lags
        n_long = max(1, valid.sum() * 3 // 10)
        long_idx = np.where(valid)[0][-n_long:]
        sf_plateau = float(np.nanmedian(sf_values[long_idx]))

    return {
        'lag_centers': lag_centers,
        'sf_values': sf_values,
        'sf_errors': sf_errors,
        'sf_slope': sf_slope,
        'sf_plateau': sf_plateau,
        'valid_bins': valid,
        'n_pairs': n_pairs,
    }


def _fit_sf_slope_short_lags(lags, sf_vals, sf_errs):
    """Fit SF ∝ Δt^β to the short-lag portion (lowest 50% of valid lags)."""
    n_fit = max(3, len(lags) // 2)
    lags_fit = lags[:n_fit]
    sf_fit = sf_vals[:n_fit]
    sf_err_fit = sf_errs[:n_fit]

    # Fit in log-log space: log(SF) = β × log(Δt) + const
    log_lag = np.log(lags_fit)
    log_sf  = np.log(np.maximum(sf_fit, 1e-10))
    log_err = np.where(sf_fit > 0, sf_err_fit / sf_fit, 1.0)

    # Simple weighted linear fit in log-log space
    weights = 1.0 / np.maximum(log_err, 0.01)**2
    A = np.vstack([log_lag, np.ones(len(log_lag))]).T
    result = np.linalg.lstsq(A * weights[:, None], log_sf * weights, rcond=None)
    beta = float(result[0][0])

    return beta


def detect_sf_break(lag_centers, sf_values, sf_errors, tau_drw):
    """
    Detect a slope break in the structure function — the key CLAGN signature.

    Normal stochastic AGN: SF ∝ Δt^β (β ~ 0.3–0.5), plateau at Δt >> τ_DRW.
    CLAGN: SF shows excess power at long lags compared to DRW prediction,
           or a step change corresponding to the epoch of the state change.

    Method:
    1. Fit single power law vs broken power law to SF(Δt)
    2. Compare BIC (ΔBIC > 6 = strong evidence for break)
    3. Compute SF_obs(Δt_max) / SF_DRW(Δt_max) — ratio >> 1 indicates CLAGN

    Parameters
    ----------
    lag_centers : array
    sf_values   : array (may contain NaN for empty bins)
    sf_errors   : array
    tau_drw     : float, DRW rest-frame timescale (days)

    Returns
    -------
    result : dict with keys:
        sf_excess_ratio, break_detected, break_lag_days,
        sf_break_significance, delta_bic_sf
    """
    valid = np.isfinite(sf_values) & np.isfinite(sf_errors) & (sf_errors > 0) & (sf_values > 0)

    if valid.sum() < 4:
        return _empty_sf_break_result()

    lags = lag_centers[valid]
    sfs  = sf_values[valid]
    errs = sf_errors[valid]

    # ---- Model A: single power law SF = A × Δt^β ---------------------------
    def single_pl(x, A, beta):
        return A * x**beta

    try:
        pA, _ = curve_fit(single_pl, lags, sfs, p0=[np.median(sfs), 0.4],
                           sigma=errs, absolute_sigma=True,
                           maxfev=5000, bounds=([0, -2], [np.inf, 5]))
        residA = (sfs - single_pl(lags, *pA)) / errs
        chi2_A = float(np.sum(residA**2))
        k_A = 2   # 2 params
    except Exception:
        chi2_A = np.sum((sfs / np.mean(sfs))**2) * len(sfs)
        k_A = 2

    # ---- Model B: broken power law ------------------------------------------
    # SF = A*x^β1 for x < x_break, B*x^β2 for x >= x_break
    def broken_pl(x, A, beta1, x_break, beta2):
        x_break = max(x_break, lags.min())
        result = np.where(x < x_break, A * x**beta1, A * x_break**(beta1-beta2) * x**beta2)
        return np.maximum(result, 1e-20)

    try:
        x_break_init = np.median(lags)
        pB, _ = curve_fit(
            broken_pl, lags, sfs,
            p0=[np.median(sfs), 0.4, x_break_init, 0.1],
            sigma=errs, absolute_sigma=True,
            maxfev=10000,
            bounds=([0, -2, lags.min(), -2], [np.inf, 5, lags.max(), 5]),
        )
        residB = (sfs - broken_pl(lags, *pB)) / errs
        chi2_B = float(np.sum(residB**2))
        k_B = 4
        break_lag = float(pB[2])
        break_fit_ok = True
    except Exception:
        chi2_B = chi2_A  # No improvement
        k_B = 4
        break_lag = float(np.median(lags))
        break_fit_ok = False

    n = len(sfs)
    bic_A = k_A * np.log(n) + chi2_A
    bic_B = k_B * np.log(n) + chi2_B
    delta_bic = float(bic_A - bic_B)   # positive = broken PL preferred

    break_detected = break_fit_ok and (delta_bic > 6.0)

    # ---- SF excess at longest lag vs DRW prediction -----------------------
    # DRW theoretical SF: σ²(1 - exp(-Δt/τ))
    # Note: we don't have σ_DRW here in mag units, so we normalize differently.
    # Use the plateau of the short-lag SF as the expected DRW asymptote.
    tau_drw_safe = max(float(tau_drw), 1.0)
    sf_plateau_drw = float(np.nanmedian(sfs[:max(2, len(sfs)//3)]))   # short-lag mean as reference
    if sf_plateau_drw <= 0:
        sf_plateau_drw = float(np.nanmedian(sfs))

    # DRW SF at the longest lag
    dt_max = float(lags[-1])
    sf_drw_at_max = sf_plateau_drw * (1.0 - np.exp(-dt_max / tau_drw_safe))
    sf_obs_at_max = float(sfs[-1])

    sf_excess_ratio = (sf_obs_at_max / sf_drw_at_max
                       if sf_drw_at_max > 0 else 1.0)

    # Break significance: ΔBIC normalized to sigma-like value
    sf_break_sig = float(np.sqrt(max(delta_bic, 0.0)))

    return {
        'sf_excess_ratio': float(sf_excess_ratio),
        'break_detected': break_detected,
        'break_lag_days': break_lag if break_detected else np.nan,
        'sf_break_significance': sf_break_sig,
        'delta_bic_sf': delta_bic,
    }


def compute_structure_function_from_flux(times, fluxes, flux_errors,
                                          band='W1', n_lag_bins=SF_N_LAG_BINS):
    """
    Convenience wrapper: converts flux→mag then calls compute_structure_function.

    Parameters
    ----------
    times, fluxes, flux_errors : arrays
    band : str ('W1' or 'W2')

    Returns
    -------
    Same as compute_structure_function
    """
    mags, mag_errs = flux_to_mag(fluxes, flux_errors, band)
    ok = np.isfinite(mags)
    return compute_structure_function(
        times[ok], mags[ok], mag_errs[ok], n_lag_bins=n_lag_bins
    )


def _empty_sf_result():
    n = SF_N_LAG_BINS
    return {
        'lag_centers': np.full(n, np.nan),
        'sf_values':   np.full(n, np.nan),
        'sf_errors':   np.full(n, np.nan),
        'sf_slope':    np.nan,
        'sf_plateau':  np.nan,
        'valid_bins':  np.zeros(n, dtype=bool),
        'n_pairs':     np.zeros(n, dtype=int),
    }


def _empty_sf_break_result():
    return {
        'sf_excess_ratio': 1.0,
        'break_detected': False,
        'break_lag_days': np.nan,
        'sf_break_significance': 0.0,
        'delta_bic_sf': 0.0,
    }
