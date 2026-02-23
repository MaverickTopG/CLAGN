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

from ..config import WISE_SEASON_ANCHOR_MJD, DRW_STRICT_SIGN_CHECK, SEASON_GAP_DAYS

logger = logging.getLogger(__name__)


def cluster_epochs_into_seasons(times, gap_days=None):
    """
    Cluster observation epochs into seasons using gap-based grouping.

    More robust than fixed 182.625-day calendar bins because it adapts to
    the actual WISE survey cadence (which varies by sky position) and
    naturally handles the WISE hibernation gap.

    Parameters
    ----------
    times : array-like
        Observation times in MJD (need not be sorted).
    gap_days : float or None
        Minimum temporal gap (days) that defines a season boundary.
        Defaults to SEASON_GAP_DAYS from config (120 days).

    Returns
    -------
    season_ids : ndarray of int
        Season label for each input epoch (0-indexed, ascending).
        All epochs receive a non-negative label.
    """
    if gap_days is None:
        gap_days = SEASON_GAP_DAYS

    times = np.asarray(times, dtype=float)
    n = len(times)
    if n == 0:
        return np.array([], dtype=int)

    # Sort internally; unsort at the end to preserve input order
    sort_idx = np.argsort(times)
    t_sorted = times[sort_idx]

    season_ids_sorted = np.zeros(n, dtype=int)
    current_season = 0
    for i in range(1, n):
        if t_sorted[i] - t_sorted[i - 1] > gap_days:
            current_season += 1
        season_ids_sorted[i] = current_season

    # Map back to original ordering
    season_ids = np.empty(n, dtype=int)
    season_ids[sort_idx] = season_ids_sorted
    return season_ids


def _delta_mag_with_err(F_early, F_early_err, F_late, F_late_err):
    """
    Compute delta_mag and propagated uncertainty from two flux measurements.

    delta_mag = +2.5 * log10(F_late / F_early)   [positive = brightened]
    delta_mag_err via quadrature of fractional flux errors.

    Parameters
    ----------
    F_early, F_late : float, flux values in mJy (must be > 0)
    F_early_err, F_late_err : float, flux uncertainties in mJy

    Returns
    -------
    delta_mag : float
    delta_mag_err : float
    flux_ratio : float  (F_late / F_early)
    """
    if F_early <= 0 or F_late <= 0:
        return np.nan, np.nan, np.nan
    flux_ratio = F_late / F_early
    if flux_ratio <= 0:
        return np.nan, np.nan, np.nan
    delta_mag = float(2.5 * np.log10(flux_ratio))
    frac_err_early = F_early_err / max(abs(F_early), 1e-30)
    frac_err_late  = F_late_err  / max(abs(F_late),  1e-30)
    delta_mag_err = float((2.5 / np.log(10)) * np.sqrt(frac_err_early ** 2 + frac_err_late ** 2))
    return delta_mag, delta_mag_err, float(flux_ratio)


def compute_amplitude_metrics(times, w1_flux_mjy, w1_flux_err_mjy, z=0.0):
    """
    Compute multiple amplitude estimates and return the highest-significance one.

    R2 FIX 2: Catches CLAGN that transition in the middle of the baseline
    (not just early→late), which the standard first-vs-last-2-seasons method misses.

    Three methods are tried:
      1. Standard: first-2 vs last-2 seasonal medians (IVW combined)
      2. Min-max:  minimum vs maximum seasonal median (full dynamic range)
      3. Max-contrast: all non-adjacent season pairs, pick max |δm|/σ

    The best method is the one with the highest detection significance
    (|delta_mag| / delta_mag_err).

    Parameters
    ----------
    times : array-like, MJD
    w1_flux_mjy : array-like, flux density in mJy
    w1_flux_err_mjy : array-like, flux uncertainty in mJy
    z : float, redshift (reserved for future rest-frame binning)

    Returns
    -------
    dict or None
        None if insufficient data (< 4 valid seasons). Otherwise:
        delta_mag, delta_mag_err, flux_ratio,
        F_early_mjy, F_late_mjy,
        amplitude_method ('standard' | 'min_max' | 'max_contrast'),
        n_seasons_used, significance
    """
    times = np.asarray(times, dtype=float)
    w1_flux_mjy = np.asarray(w1_flux_mjy, dtype=float)
    w1_flux_err_mjy = np.asarray(w1_flux_err_mjy, dtype=float)

    valid = (
        np.isfinite(times) &
        np.isfinite(w1_flux_mjy) & (w1_flux_mjy > 0) &
        np.isfinite(w1_flux_err_mjy) & (w1_flux_err_mjy > 0)
    )
    if valid.sum() < 4:
        return None

    t = times[valid]
    f = w1_flux_mjy[valid]
    e = w1_flux_err_mjy[valid]

    sort_idx = np.argsort(t)
    t = t[sort_idx]
    f = f[sort_idx]
    e = e[sort_idx]

    # Gap-based season clustering
    season_id = cluster_epochs_into_seasons(t)

    season_flux = {}
    season_flux_err = {}
    for s in np.unique(season_id[season_id >= 0]):
        mask = season_id == s
        n_s = mask.sum()
        if n_s < 3:
            continue
        f_s = f[mask]
        med = float(np.median(f_s))
        mad = float(np.median(np.abs(f_s - med)))
        robust_sigma = 1.4826 * mad
        stat_err = robust_sigma / np.sqrt(n_s)
        cal_floor_flux = 0.028 * abs(med)
        total_err = float(np.sqrt(stat_err ** 2 + cal_floor_flux ** 2))
        season_flux[s] = med
        season_flux_err[s] = max(total_err, 1e-10)

    if len(season_flux) < 4:
        return None

    seasons = sorted(season_flux.keys())
    n_seasons = len(seasons)
    flux_vals = np.array([season_flux[s] for s in seasons])
    flux_errs = np.array([season_flux_err[s] for s in seasons])

    def _ivw(season_list):
        fs = np.array([season_flux[s] for s in season_list])
        es = np.array([season_flux_err[s] for s in season_list])
        w  = 1.0 / np.maximum(es ** 2, 1e-30)
        ws = w.sum()
        return float(np.sum(w * fs) / ws), float(1.0 / np.sqrt(ws))

    # Method 1: Standard first-2 vs last-2
    Fe_std, Fe_err_std = _ivw(seasons[:2])
    Fl_std, Fl_err_std = _ivw(seasons[-2:])
    dm_std, dme_std, fr_std = _delta_mag_with_err(Fe_std, Fe_err_std, Fl_std, Fl_err_std)

    # Method 2: Min-vs-max seasonal median
    min_idx = int(np.argmin(flux_vals))
    max_idx = int(np.argmax(flux_vals))
    F_min, e_min = flux_vals[min_idx], flux_errs[min_idx]
    F_max, e_max = flux_vals[max_idx], flux_errs[max_idx]
    if max_idx > min_idx:
        dm_mm, dme_mm, fr_mm = _delta_mag_with_err(F_min, e_min, F_max, e_max)
        Fe_mm, Fl_mm = F_min, F_max
    else:
        dm_mm, dme_mm, fr_mm = _delta_mag_with_err(F_max, e_max, F_min, e_min)
        Fe_mm, Fl_mm = F_max, F_min

    # Method 3: Max-contrast non-adjacent pair
    best_sig3, best_dm3, best_dme3 = 0.0, np.nan, np.nan
    best_fr3, best_Fe3, best_Fl3 = np.nan, np.nan, np.nan
    for i in range(n_seasons):
        for j in range(i + 1, n_seasons):
            if j == i + 1 and n_seasons > 3:
                continue   # skip adjacent seasons for short-baseline pairs
            dm_ij, dme_ij, fr_ij = _delta_mag_with_err(
                flux_vals[i], flux_errs[i], flux_vals[j], flux_errs[j])
            if not (np.isfinite(dm_ij) and np.isfinite(dme_ij) and dme_ij > 0):
                continue
            sig = abs(dm_ij) / dme_ij
            if sig > best_sig3:
                best_sig3, best_dm3, best_dme3 = sig, dm_ij, dme_ij
                best_fr3, best_Fe3, best_Fl3 = fr_ij, flux_vals[i], flux_vals[j]

    # Collect valid candidates and pick highest significance
    candidates = []
    if np.isfinite(dm_std) and np.isfinite(dme_std) and dme_std > 0:
        candidates.append(('standard', dm_std, dme_std, fr_std,
                           Fe_std, Fl_std, abs(dm_std) / dme_std))
    if np.isfinite(dm_mm) and np.isfinite(dme_mm) and dme_mm > 0:
        candidates.append(('min_max', dm_mm, dme_mm, fr_mm,
                           Fe_mm, Fl_mm, abs(dm_mm) / dme_mm))
    if np.isfinite(best_dm3) and np.isfinite(best_dme3) and best_dme3 > 0:
        candidates.append(('max_contrast', best_dm3, best_dme3, best_fr3,
                           best_Fe3, best_Fl3, best_sig3))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[6], reverse=True)
    method, delta_mag, delta_mag_err, flux_ratio, F_e, F_l, significance = candidates[0]

    logger.debug(
        "compute_amplitude_metrics: best=%s, delta_mag=%.4f±%.4f, sig=%.2f",
        method, delta_mag, delta_mag_err, significance
    )
    return {
        'delta_mag':        float(delta_mag),
        'delta_mag_err':    float(delta_mag_err),
        'flux_ratio':       float(flux_ratio),
        'F_early_mjy':      float(F_e),
        'F_late_mjy':       float(F_l),
        'amplitude_method': method,
        'n_seasons_used':   n_seasons,
        'significance':     float(significance),
    }


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

    # R2 FIX 1: Gap-based season clustering (replaces fixed 182.625-day calendar bins)
    season_id = cluster_epochs_into_seasons(t)

    # Seasonal MEDIAN flux + MAD-based uncertainty + 2.8% calibration floor (Wright+2010)
    season_flux = {}
    season_flux_err = {}
    season_stat_err = {}  # R2 FIX 4: track stat-only error separately from cal floor
    for s in np.unique(season_id[season_id >= 0]):
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
        season_stat_err[s] = max(float(stat_err), 1e-10)  # stat-only (no cal floor)

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
    def _ivw_mean(season_list, err_dict=None):
        if err_dict is None:
            err_dict = season_flux_err
        fluxes_s = np.array([season_flux[s] for s in season_list])
        errs_s   = np.array([err_dict[s] for s in season_list])
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

    # Full error propagation: quadrature sum of fractional errors (stat + cal floor)
    frac_err_early = F_early_err / max(abs(F_early), 1e-30)
    frac_err_late  = F_late_err  / max(abs(F_late),  1e-30)
    delta_mag_err = float((2.5 / np.log(10)) * np.sqrt(frac_err_early ** 2 + frac_err_late ** 2))

    # R2 FIX 4: Stat-only error (calibration floor removed) for systematic decomposition
    _, stat_err_early = _ivw_mean(seasons_early, err_dict=season_stat_err)
    _, stat_err_late  = _ivw_mean(seasons_late,  err_dict=season_stat_err)
    frac_stat_early = stat_err_early / max(abs(F_early), 1e-30)
    frac_stat_late  = stat_err_late  / max(abs(F_late),  1e-30)
    delta_mag_err_stat_only = float(
        (2.5 / np.log(10)) * np.sqrt(frac_stat_early ** 2 + frac_stat_late ** 2)
    )

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
        'delta_mag':              float(delta_mag),
        'delta_mag_err':          float(delta_mag_err),
        'delta_mag_err_stat_only': float(delta_mag_err_stat_only),  # R2 FIX 4
        'flux_ratio':             float(flux_ratio),
        'F_early_mjy':            float(F_early),
        'F_early_err_mjy':        float(F_early_err),
        'F_late_mjy':             float(F_late),
        'F_late_err_mjy':         float(F_late_err),
        'n_seasons_used':         len(seasons),
        'seasons_early':          seasons_early,
        'seasons_late':           seasons_late,
        'season_anchor_mjd':      float(WISE_SEASON_ANCHOR_MJD),
        'binning_frame':          'observer_frame',
        'estimator':              'gap_clustered_seasonal_median',  # R2 FIX 1
    }


def compute_delta_mag_w2(times, w2_flux_mjy, w2_flux_err_mjy, z=0.0):
    """
    Same as compute_delta_mag_correct but for W2 band.

    Returns
    -------
    dict or None  (same schema as compute_delta_mag_correct)
    """
    return compute_delta_mag_correct(times, w2_flux_mjy, w2_flux_err_mjy, z=z)
