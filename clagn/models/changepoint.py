"""
changepoint.py — Bayesian structural break detection and monotonic trend test.

Model comparison:
- M0 (null): single stationary mean μ, variance σ²
- M1 (CLAGN): two segments with means μ1, μ2; break at time t_break

Mann-Kendall test via scipy.stats.kendalltau (uses .statistic attribute).
Theil-Sen slope via scipy.stats.theilslopes.
"""
import logging
import numpy as np
from scipy.stats import kendalltau, theilslopes

from ..config import (
    CP_MIN_SEGMENT_FRACTION, CP_STRONG_BIC_THRESHOLD, CP_VERY_STRONG_BIC,
)

logger = logging.getLogger(__name__)


def _bic_single_segment(fluxes):
    """
    BIC for M0: single Gaussian with estimated mean and variance.

    BIC = k × ln(N) - 2 × log-likelihood
    k = 2 params (μ, σ²)
    """
    n = len(fluxes)
    if n < 2:
        return 0.0

    mu_hat = np.mean(fluxes)
    var_hat = np.var(fluxes, ddof=1)
    if var_hat <= 0:
        var_hat = 1e-12

    ll = -0.5 * n * np.log(2.0 * np.pi * var_hat) \
         - 0.5 * np.sum((fluxes - mu_hat)**2) / var_hat
    k = 2
    return float(k * np.log(n) - 2.0 * ll)


def _bic_two_segments(fluxes_1, fluxes_2):
    """
    BIC for M1: two independent Gaussians.

    Total k = 4 params (μ1, σ1², μ2, σ2²) + 1 for the break location.
    We add an extra ln(N) penalty for the break time parameter.
    """
    n = len(fluxes_1) + len(fluxes_2)
    bic1 = _bic_single_segment(fluxes_1)
    bic2 = _bic_single_segment(fluxes_2)
    # Extra parameter for t_break: add ln(N) penalty
    return float(bic1 + bic2 + np.log(n))


def bayesian_changepoint_detection(times, fluxes, flux_errors):
    """
    Rigorous Bayesian detection of a structural break in the light curve mean.

    Scans all candidate break times in (0.15*N, 0.85*N) index range.
    Computes ΔBIC = BIC(M0) - BIC(M1) for each candidate.
    The break time maximizing ΔBIC is the best-fit state transition.

    ΔBIC > 6:  strong evidence (Kass & Raftery 1995)
    ΔBIC > 10: very strong evidence

    Break duration estimate: Gaussian fit to ΔBIC profile (used by false
    positive rejection to discriminate CLAGN from supernovae).

    Parameters
    ----------
    times       : array, MJD
    fluxes      : array, mJy
    flux_errors : array, mJy

    Returns
    -------
    result : dict with keys:
        best_break_mjd, delta_bic, break_significance,
        pre_break_mean, post_break_mean, mean_ratio,
        break_duration_days (FWHM of ΔBIC profile)
    """
    times = np.asarray(times, dtype=float)
    fluxes = np.asarray(fluxes, dtype=float)

    # Sort by time
    idx_sort = np.argsort(times)
    times  = times[idx_sort]
    fluxes = fluxes[idx_sort]

    n = len(times)
    if n < 10:
        return _empty_cp_result()

    # Null model BIC
    bic_null = _bic_single_segment(fluxes)

    # Candidate break indices (not too close to edges)
    i_min = max(2, int(n * CP_MIN_SEGMENT_FRACTION))
    i_max = min(n - 2, int(n * (1.0 - CP_MIN_SEGMENT_FRACTION)))

    if i_min >= i_max:
        return _empty_cp_result()

    delta_bics = np.full(n, np.nan)

    for i in range(i_min, i_max + 1):
        seg1 = fluxes[:i]
        seg2 = fluxes[i:]
        if len(seg1) < 2 or len(seg2) < 2:
            continue
        bic_alt = _bic_two_segments(seg1, seg2)
        delta_bics[i] = bic_null - bic_alt

    # Find best break
    valid_breaks = np.where(np.isfinite(delta_bics))[0]
    if len(valid_breaks) == 0:
        return _empty_cp_result()

    best_i = int(valid_breaks[np.argmax(delta_bics[valid_breaks])])
    best_delta_bic = float(delta_bics[best_i])
    best_break_mjd = float(times[best_i])

    pre_mean  = float(np.mean(fluxes[:best_i]))
    post_mean = float(np.mean(fluxes[best_i:]))
    mean_ratio = (post_mean / pre_mean
                  if pre_mean > 0 else
                  (pre_mean / post_mean if post_mean > 0 else 1.0))
    mean_ratio = abs(mean_ratio)

    # ---- Break significance: posterior probability estimate ----------------
    # P(break) ≈ 1 - exp(-ΔBIC/2) for log-odds approximation
    break_sig_approx = float(1.0 - np.exp(-best_delta_bic / 2.0))

    # ---- Break duration: FWHM of ΔBIC profile (proxy for transition time) --
    break_duration = _estimate_break_duration(
        times[valid_breaks], delta_bics[valid_breaks],
        best_i, best_break_mjd
    )

    logger.debug(
        f"Changepoint: ΔBIC={best_delta_bic:.1f} at MJD={best_break_mjd:.1f} | "
        f"pre={pre_mean:.4f} mJy → post={post_mean:.4f} mJy "
        f"(ratio={mean_ratio:.2f})"
    )

    return {
        'best_break_mjd': best_break_mjd,
        'delta_bic': best_delta_bic,
        'break_significance': break_sig_approx,
        'pre_break_mean': pre_mean,
        'post_break_mean': post_mean,
        'mean_ratio': mean_ratio,
        'break_duration_days': break_duration,
    }


def _estimate_break_duration(times, delta_bics, best_i, best_mjd):
    """
    Estimate break duration from the FWHM of the ΔBIC profile.

    A sharp, narrow peak → short transition (SN-like).
    A broad peak → long transition (genuine CS-AGN).

    Returns FWHM in days.
    """
    if len(delta_bics) < 5:
        return np.nan

    db_max = float(np.nanmax(delta_bics))
    if db_max <= 0:
        return np.nan

    half_max = db_max / 2.0
    above_half = delta_bics >= half_max

    if above_half.sum() < 2:
        return 365.0   # Default: assume ~1 year if can't compute

    t_above = times[above_half]
    fwhm = float(t_above.max() - t_above.min())
    return max(fwhm, 1.0)   # At least 1 day


def test_monotonic_trend(times, fluxes, flux_errors):
    """
    Test for a sustained monotonic flux trend — another CLAGN signature.

    Normal AGN: symmetric, mean-reverting variability (DRW)
    CLAGN turn-off: sustained decline over years
    CLAGN turn-on: sustained rise over years

    Uses Mann-Kendall trend test (non-parametric, robust to outliers) via
    scipy.stats.kendalltau and Theil-Sen slope estimator.

    NOTE: kendalltau returns SignificanceResult with .statistic and .pvalue
    in scipy >= 1.7. Access as result.statistic (NOT result.correlation).

    Parameters
    ----------
    times, fluxes, flux_errors : arrays

    Returns
    -------
    result : dict with keys:
        mk_tau, mk_pvalue, trend_slope (mJy/year), trend_dir
    """
    times  = np.asarray(times,  dtype=float)
    fluxes = np.asarray(fluxes, dtype=float)

    if len(times) < 5:
        return {
            'mk_tau': np.nan, 'mk_pvalue': np.nan,
            'trend_slope': np.nan, 'trend_dir': 'none',
        }

    # Sort by time
    idx_sort = np.argsort(times)
    t_sorted = times[idx_sort]
    f_sorted = fluxes[idx_sort]

    # Mann-Kendall: Kendall's tau between the integer rank index and flux values
    # This is equivalent to the classic Mann-Kendall S statistic.
    x_ranks = np.arange(len(f_sorted), dtype=float)
    mk_result = kendalltau(x_ranks, f_sorted)
    mk_tau    = float(mk_result.statistic)   # .statistic in scipy >= 1.7
    mk_pvalue = float(mk_result.pvalue)

    # Theil-Sen slope in mJy/day → convert to mJy/year
    ts = theilslopes(f_sorted, t_sorted)
    trend_slope_per_year = float(ts.slope * 365.25)

    if mk_pvalue < 0.01:
        trend_dir = 'rising' if mk_tau > 0 else 'falling'
    else:
        trend_dir = 'none'

    return {
        'mk_tau': mk_tau,
        'mk_pvalue': mk_pvalue,
        'trend_slope': trend_slope_per_year,
        'trend_dir': trend_dir,
    }


def _empty_cp_result():
    return {
        'best_break_mjd': np.nan,
        'delta_bic': 0.0,
        'break_significance': 0.0,
        'pre_break_mean': np.nan,
        'post_break_mean': np.nan,
        'mean_ratio': 1.0,
        'break_duration_days': np.nan,
    }
