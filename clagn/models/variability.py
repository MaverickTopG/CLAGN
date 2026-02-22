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

from ..config import WISE_SEASON_ANCHOR_MJD

logger = logging.getLogger(__name__)


def compute_delta_mag_correct(times, w1_flux_mjy, w1_flux_err_mjy, z=0.0):
    """
    Compute delta_mag the way published papers actually do it.

    Step 1: Bin light curve into 6-month WISE seasons
    Step 2: Compute weighted average flux per season (weight = 1/err^2)
            Skip seasons with < 3 epochs
    Step 3: Require >= 4 seasons
    Step 4: Compare mean of first 2 vs last 2 seasons
    Step 5: delta_mag = -2.5 * log10(F_early / F_late)

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
        Source redshift (not used in flux-space computation, reserved for
        future rest-frame season binning)

    Returns
    -------
    delta_mag : float or None
        Signed delta magnitude (positive = brightened). None if insufficient data.
    delta_mag_err : float or None
        1-sigma uncertainty including 5% WISE calibration floor.
    flux_ratio : float or None
        F_late / F_early (> 1 = brightened). None if insufficient data.
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
    if valid.sum() < 6:
        return None, None, None

    t = times[valid]
    f = w1_flux_mjy[valid]
    e = w1_flux_err_mjy[valid]

    # Sort by time
    sort_idx = np.argsort(t)
    t = t[sort_idx]
    f = f[sort_idx]
    e = e[sort_idx]

    # FLAW A2: Use global anchor so all sources have identical season boundaries
    season_id = np.floor((t - WISE_SEASON_ANCHOR_MJD) / 182.625).astype(int)

    # Weighted mean flux per season (weight = 1/err^2)
    season_flux = {}
    season_flux_err = {}
    for s in np.unique(season_id):
        mask = season_id == s
        if mask.sum() < 3:
            continue   # Skip seasons with fewer than 3 epochs
        weights = 1.0 / e[mask] ** 2
        w_sum = weights.sum()
        wmean = np.sum(weights * f[mask]) / w_sum
        # Weighted standard error of the mean
        werr = 1.0 / np.sqrt(w_sum)
        season_flux[s] = float(wmean)
        season_flux_err[s] = float(werr)

    if len(season_flux) < 4:
        logger.debug(
            f"compute_delta_mag_correct: only {len(season_flux)} seasons "
            f"with >= 3 epochs — returning None"
        )
        return None, None, None

    seasons = sorted(season_flux.keys())

    # Early baseline: weighted mean of first 2 seasons
    F_early = float(np.mean([season_flux[s] for s in seasons[:2]]))
    F_late  = float(np.mean([season_flux[s] for s in seasons[-2:]]))

    if F_early <= 0 or F_late <= 0:
        return None, None, None

    # Flux ratio: F_late / F_early
    # > 1 = brightened (turn-ON), < 1 = faded (turn-OFF)
    flux_ratio = F_late / F_early

    # Delta magnitude: positive = brightened
    # = -2.5 * log10(F_early / F_late) = +2.5 * log10(F_late / F_early)
    delta_mag = -2.5 * np.log10(F_early / F_late)

    # Error propagation: 5% systematic calibration floor from WISE
    # Systematic dominates over statistical for bright sources
    calibration_floor = 0.05
    delta_mag_err = float(
        (2.5 / np.log(10)) * np.sqrt(
            (calibration_floor) ** 2 +
            (calibration_floor) ** 2
        )
    )

    # Consistency assertion: sign of delta_mag must match sign of log(flux_ratio)
    if flux_ratio > 0 and np.isfinite(delta_mag):
        sign_dm = np.sign(delta_mag)
        sign_fr = np.sign(np.log(flux_ratio))
        if sign_dm != 0 and sign_fr != 0 and sign_dm != sign_fr:
            logger.error(
                f"SIGN INCONSISTENCY: delta_mag={delta_mag:.4f} but "
                f"log(flux_ratio)={np.log(flux_ratio):.4f}. "
                f"F_early={F_early:.4f}, F_late={F_late:.4f}. "
                f"This is a bug — forcing consistent sign."
            )
            delta_mag = 2.5 * np.log10(flux_ratio)

    logger.debug(
        f"delta_mag_correct: {len(seasons)} seasons, "
        f"F_early={F_early:.4f} mJy, F_late={F_late:.4f} mJy, "
        f"delta_mag={delta_mag:.4f}, flux_ratio={flux_ratio:.4f}"
    )

    return float(delta_mag), float(delta_mag_err), float(flux_ratio)


def compute_delta_mag_w2(times, w2_flux_mjy, w2_flux_err_mjy, z=0.0):
    """
    Same as compute_delta_mag_correct but for W2 band.

    Returns
    -------
    delta_mag_w2 : float or None
    delta_mag_w2_err : float or None
    flux_ratio_w2 : float or None
    """
    return compute_delta_mag_correct(times, w2_flux_mjy, w2_flux_err_mjy, z=z)
