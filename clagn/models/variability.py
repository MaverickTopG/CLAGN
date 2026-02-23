"""
variability.py — Correct variability amplitude computation for CLAGN.

FIX 1: delta_mag must use seasonal median fluxes, NOT raw single epochs.

Published papers use seasonal median flux comparison (Sheng+2017, Graham+2020,
Hon+2020). The pipeline MUST use this method to produce physically meaningful
amplitude estimates.

SIGN CONVENTION (from config.DELTA_MAG_SIGN_CONVENTION):
    delta_mag > 0  → source BRIGHTENED (flux increased, turn-ON)
    delta_mag < 0  → source FADED     (flux decreased, turn-OFF)
    delta_mag = -2.5 * log10(F_early / F_late) = +2.5 * log10(F_late / F_early)
    flux_ratio = F_late / F_early  (> 1 = brightened)

References:
    Sheng et al. 2017, ApJL 846, L7
    Graham et al. 2020, MNRAS 491, 4925
    Wright et al. 2010, AJ 140, 1868
"""
import logging

import numpy as np

from ..config import WISE_SEASON_ANCHOR_MJD, DRW_STRICT_SIGN_CHECK

logger = logging.getLogger(__name__)


def compute_delta_mag_correct(times, w1_flux_mjy, w1_flux_err_mjy, z=0.0):
    """
    Compute delta_mag the way published papers actually do it.

    Step 1: Bin light curve into 182.625-day WISE seasons (half-year bins chosen
            to match the ~6-month WISE survey cadence per sky visit; Wright+2010).
    Step 2: Compute SEASONAL MEDIAN flux per season (robust against outliers).
            MAD-based uncertainty: robust_sigma = 1.4826 * MAD;
            stat_err = robust_sigma / sqrt(n_season).
    Step 3: Calibration floor 2.8% per epoch (Wright+2010).
    Step 4: Require >= 4 valid seasons.
    Step 5: Compare INVERSE-VARIANCE weighted mean of first 2 vs last 2 seasons.
    Step 6: delta_mag = +2.5 * log10(F_late / F_early) — full error propagation.

    SIGN CONVENTION (critical — consistent throughout codebase):
    delta_mag > 0 = source BRIGHTENED (flux INCREASED = turn-ON)
    delta_mag < 0 = source FADED      (flux DECREASED = turn-OFF)

    DO NOT use: max(mags) - min(mags)
    DO NOT use: single epoch comparison
    DO NOT use: magnitude subtraction instead of flux ratio

    Parameters
    ----------
    times : array-like
        Observation times in MJD (observer frame)
    w1_flux_mjy : array-like
        W1 flux density in mJy (already converted from Vega mag)
    w1_flux_err_mjy : array-like
        W1 flux uncertainty in mJy (positive, finite)
    z : float
        Source redshift (reserved for future rest-frame season binning)

    Returns
    -------
    dict or None
        None if insufficient data. Otherwise a dict with:
        delta_mag, delta_mag_err, flux_ratio,
        F_early_mjy, F_early_err_mjy, F_late_mjy, F_late_err_mjy,
        n_seasons_used, seasons_early, seasons_late,
        season_anchor_mjd, binning_frame, estimator
    """
    times = np.asarray(times, dtype=float)
    w1_flux_mjy = np.asarray(w1_flux_mjy, dtype=float)
    w1_flux_err_mjy = np.asarray(w1_flux_err_mjy, dtype=float)

    # Clean: require positive, finite flux and errors
    valid = (
        np.isfinite(times) &
        np.isfinite(w1_flux_mjy) & (w1_flux_mjy > 0) &
        np.isfinite(w1_flux_err_mjy) & (w1_flux_err_mjy > 0)
    )
    if valid.sum() < 4:
        return None

    t = times[valid]
    f = w1_flux_mjy[valid]

    # Sort by time
    sort_idx = np.argsort(t)
    t = t[sort_idx]
    f = f[sort_idx]

    # 182.625-day bins: half-year WISE seasons anchored globally (FLAW A2 fix)
    season_id = np.floor((t - WISE_SEASON_ANCHOR_MJD) / 182.625).astype(int)

    # Seasonal MEDIAN flux + MAD-based uncertainty + 2.8% calibration floor (Wright+2010)
    season_flux = {}
    season_flux_err = {}
    for s in np.unique(season_id):
        mask = season_id == s
        n_s = mask.sum()
        if n_s < 3:
            continue   # Skip seasons with fewer than 3 epochs
        f_s = f[mask]
        med = float(np.median(f_s))
        mad = float(np.median(np.abs(f_s - med)))
        robust_sigma = 1.4826 * mad
        stat_err = robust_sigma / np.sqrt(n_s)
        # Wright+2010: 2.8% calibration floor per season measurement
        cal_floor_flux = 0.028 * abs(med)
        total_err = float(np.sqrt(stat_err ** 2 + cal_floor_flux ** 2))
        season_flux[s] = med
        season_flux_err[s] = max(total_err, 1e-10)

    if len(season_flux) < 4:
        logger.debug(
            "compute_delta_mag_correct: only %d seasons with >= 3 epochs — returning None",
            len(season_flux)
        )
        return None

    seasons = sorted(season_flux.keys())
    seasons_early = seasons[:2]
    seasons_late  = seasons[-2:]

    # Inverse-variance weighted mean for early and late baselines
    def _ivw_mean(season_list):
        fluxes_s = np.array([season_flux[s] for s in season_list])
        errs_s   = np.array([season_flux_err[s] for s in season_list])
        weights  = 1.0 / np.maximum(errs_s ** 2, 1e-30)
        w_sum    = weights.sum()
        mean_f   = float(np.sum(weights * fluxes_s) / w_sum)
        mean_err = float(1.0 / np.sqrt(w_sum))
        return mean_f, mean_err

    F_early, F_early_err = _ivw_mean(seasons_early)
    F_late,  F_late_err  = _ivw_mean(seasons_late)

    if F_early <= 0 or F_late <= 0:
        return None

    # Flux ratio: F_late / F_early  (> 1 = brightened, < 1 = faded)
    flux_ratio = F_late / F_early

    # delta_mag = +2.5 * log10(F_late / F_early) — positive = brightened
    delta_mag = float(2.5 * np.log10(flux_ratio))

    # Full error propagation: quadrature sum of fractional errors
    frac_err_early = F_early_err / max(abs(F_early), 1e-30)
    frac_err_late  = F_late_err  / max(abs(F_late),  1e-30)
    delta_mag_err = float((2.5 / np.log(10)) * np.sqrt(frac_err_early ** 2 + frac_err_late ** 2))

    # Sign check: log warning on inconsistency (raise only if DRW_STRICT_SIGN_CHECK)
    if flux_ratio > 0 and np.isfinite(delta_mag):
        sign_dm = np.sign(delta_mag)
        sign_fr = np.sign(np.log(flux_ratio))
        if sign_dm != 0 and sign_fr != 0 and sign_dm != sign_fr:
            msg = (
                f"SIGN INCONSISTENCY: delta_mag={delta_mag:.4f} but "
                f"log(flux_ratio)={np.log(flux_ratio):.4f}. "
                f"F_early={F_early:.4f}, F_late={F_late:.4f}."
            )
            if DRW_STRICT_SIGN_CHECK:
                raise ValueError(msg)
            logger.warning(msg + " Forcing consistent sign.")
            delta_mag = float(2.5 * np.log10(flux_ratio))

    logger.debug(
        "delta_mag_correct: %d seasons, F_early=%.4f mJy, F_late=%.4f mJy, "
        "delta_mag=%.4f, flux_ratio=%.4f",
        len(seasons), F_early, F_late, delta_mag, flux_ratio
    )

    return {
        'delta_mag':         float(delta_mag),
        'delta_mag_err':     float(delta_mag_err),
        'flux_ratio':        float(flux_ratio),
        'F_early_mjy':       float(F_early),
        'F_early_err_mjy':   float(F_early_err),
        'F_late_mjy':        float(F_late),
        'F_late_err_mjy':    float(F_late_err),
        'n_seasons_used':    len(seasons),
        'seasons_early':     seasons_early,
        'seasons_late':      seasons_late,
        'season_anchor_mjd': float(WISE_SEASON_ANCHOR_MJD),
        'binning_frame':     'observer_frame',
        'estimator':         'seasonal_median',
    }


def compute_delta_mag_w2(times, w2_flux_mjy, w2_flux_err_mjy, z=0.0):
    """
    Same as compute_delta_mag_correct but for W2 band.

    Returns
    -------
    dict or None  (same schema as compute_delta_mag_correct)
    """
    return compute_delta_mag_correct(times, w2_flux_mjy, w2_flux_err_mjy, z=z)
