"""
clagn_score.py — 7-component composite CLAGN scoring engine.

Each component is normalized to [0, 1] before weighting.
Final score is a weighted sum divided by MAX_SCORE (= 14.0).

NaN component values contribute 0.0 — they don't crash the score.

Scientific basis: Ricci & Trakhtenbrot 2022 (arXiv:2211.05132)

FIX 1: delta_mag uses seasonal median flux method (compute_delta_mag_correct)
FIX 6: Host galaxy contamination correction
FIX 8: Sign convention — positive delta_mag = brightened (turn-ON)
FIX 10: Composite score v3 avoids double-counting correlated components
FIX 13: W1/W2 coherence cross-check
"""
import logging
import numpy as np

from ..config import (
    SCORE_WEIGHTS, MAX_SCORE,
    FP_SN_BREAK_DURATION_DAYS, FP_GALACTIC_LAT_THRESHOLD, FP_LOW_QUASAR_PROB,
    FP_SN_SCORE_PENALTY, FP_STELLAR_SCORE_PENALTY,
    FP_HOST_SCORE_PENALTY, FP_ARTIFACT_SCORE_PENALTY,
    FP_SN_MORPH_PENALTY, FP_SN_XMATCH_PENALTY,
    FP_BLAZAR_SCORE_PENALTY, FP_DUST_SCORE_PENALTY,
    TRANSIENT_XMATCH_RADIUS_ARCSEC, TRANSIENT_XMATCH_MAX_DT_DAYS,
    SIGMA_EXCESS_NORM_FLUX_MJY, SIGMA_EXCESS_LUM_SLOPE,
    MIN_MAG_CHANGE_W1,
    WISE_SEASON_ANCHOR_MJD,
    SCORE_GROUP_WEIGHTS_V3,
)
from ..utils.crossmatch import galactic_latitude
from ..models.variability import (
    compute_delta_mag_correct, compute_delta_mag_w2,
    compute_amplitude_metrics, cluster_epochs_into_seasons,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Component normalization helpers
# ---------------------------------------------------------------------------

def _safe(val, default=0.0):
    """Return default if val is NaN or None, else float(val)."""
    if val is None:
        return default
    try:
        v = float(val)
        return default if not np.isfinite(v) else v
    except (TypeError, ValueError):
        return default


def _norm_drw_nonstat(sigma):
    """DRW nonstationarity: min(σ / 10, 1)."""
    return float(np.clip(_safe(sigma) / 10.0, 0.0, 1.0))


def _norm_frac_flux_change(frac_change):
    """
    Normalized fractional flux change: input is |flux_ratio - 1| (not magnitude delta).
    A factor-2 change → flux_ratio=2 → frac_change=1.0. Normalized by dividing by 2.
    """
    return float(np.clip(_safe(frac_change) / 2.0, 0.0, 1.0))


def _norm_changepoint_bic(delta_bic):
    """Bayesian changepoint: min(ΔBIC / 20, 1)."""
    return float(np.clip(_safe(delta_bic) / 20.0, 0.0, 1.0))


def _norm_sf_break(sf_excess_ratio):
    """SF excess at long lags: min((ratio-1) / 4, 1)."""
    return float(np.clip((_safe(sf_excess_ratio) - 1.0) / 4.0, 0.0, 1.0))


def _norm_color_evolution(delta_color):
    """W1-W2 color change: min(|Δcolor| / 0.3, 1)."""
    return float(np.clip(abs(_safe(delta_color)) / 0.3, 0.0, 1.0))


def _norm_sigma_excess(sigma_excess):
    """DRW sigma excess vs expectation: min((excess-1) / 3, 1)."""
    return float(np.clip((_safe(sigma_excess) - 1.0) / 3.0, 0.0, 1.0))


def _norm_gaia_var(variable_flag):
    """GAIA optical variability: 1.0 if VARIABLE, else 0.0."""
    return 1.0 if str(variable_flag).strip().upper() == 'VARIABLE' else 0.0


# ---------------------------------------------------------------------------
# DRW sigma excess vs luminosity expectation (Vanden Berk+2004)
# ---------------------------------------------------------------------------

def _compute_flux_variability_excess(sigma_drw_obs, w1_flux_mean_mjy):
    """
    Compute DRW flux variability amplitude excess vs the luminosity-variability
    anti-correlation (Vanden Berk+2004). sigma_expected ∝ L^(-0.5).

    flux_variability_excess = sigma_drw_obs / sigma_expected

    Parameters
    ----------
    sigma_drw_obs : float, observed DRW amplitude (mJy)
    w1_flux_mean_mjy : float, mean W1 flux (mJy)

    Returns
    -------
    flux_variability_excess : float (> 1 = more variable than expected)
    """
    if not (np.isfinite(sigma_drw_obs) and sigma_drw_obs > 0 and
            np.isfinite(w1_flux_mean_mjy) and w1_flux_mean_mjy > 0):
        return 1.0

    sigma_expected = (SIGMA_EXCESS_NORM_FLUX_MJY *
                      (w1_flux_mean_mjy / SIGMA_EXCESS_NORM_FLUX_MJY) **
                      SIGMA_EXCESS_LUM_SLOPE)

    if sigma_expected <= 0:
        return 1.0

    return float(sigma_drw_obs / sigma_expected)


# ---------------------------------------------------------------------------
# FIX 6: Host galaxy contamination estimation and correction
# ---------------------------------------------------------------------------

def estimate_host_contamination_fraction(z):
    """
    Estimate fractional contribution of host galaxy to total WISE W1 flux.

    Empirical calibration from Assef+2013 AGN SED decomposition:
    f_host ~ 0.5 * exp(-z / 0.15) for typical Seyfert luminosities

    At z < 0.05: host likely dominates. At z > 0.3: host contribution negligible.

    Parameters
    ----------
    z : float, source redshift

    Returns
    -------
    f_host : float [0, 0.95]
    f_agn : float [0.05, 1.0]
    contamination_flag : str
    """
    f_host = 0.5 * np.exp(-float(z) / 0.15)
    f_host = float(np.clip(f_host, 0.0, 0.95))
    f_agn = 1.0 - f_host

    if f_host > 0.5:
        flag = 'severe_contamination'
    elif f_host > 0.3:
        flag = 'moderate_contamination'
    elif f_host > 0.1:
        flag = 'low_contamination'
    else:
        flag = 'negligible'

    return f_host, f_agn, flag


def correct_delta_mag_for_host(delta_mag_observed, f_agn):
    """
    Correct observed delta_mag for host galaxy dilution.

    If host contributes fraction (1 - f_agn) of total flux at constant level,
    the observed variability is diluted relative to the true AGN variability.

    Correction:
        flux_ratio_total = 10^(delta_mag_observed / 2.5)
        flux_ratio_agn = 1 + (flux_ratio_total - 1) / f_agn

    Parameters
    ----------
    delta_mag_observed : float, observed (diluted) delta_mag
    f_agn : float, AGN fraction [0, 1]

    Returns
    -------
    delta_mag_agn : float, host-corrected AGN-only delta_mag
    """
    if not (np.isfinite(delta_mag_observed) and 0 < f_agn <= 1):
        return delta_mag_observed

    # score-H6: Don't correct when AGN fraction < 20% — correction unreliable
    if f_agn < 0.20:
        return delta_mag_observed

    flux_ratio_total = 10.0 ** (delta_mag_observed / 2.5)
    flux_ratio_agn = 1.0 + (flux_ratio_total - 1.0) / f_agn
    flux_ratio_agn = float(np.clip(flux_ratio_agn, 0.01, 100.0))

    if flux_ratio_agn <= 0:
        return delta_mag_observed

    return float(2.5 * np.log10(flux_ratio_agn))


# ---------------------------------------------------------------------------
# FIX 1 + FIX 8: Delta-magnitude computation using seasonal median flux method
# ---------------------------------------------------------------------------

def _compute_delta_mag_seasonal(wise_result, z=0.1):
    """
    Compute delta_mag using seasonal median flux method (published papers standard).

    FIX 1: Use compute_delta_mag_correct() instead of raw epoch comparison.
    FIX 8: Return SIGNED delta_mag (positive = brightened).

    Also returns:
        delta_mag_w1 : float, signed (positive = brightened, turn-ON)
        delta_mag_w2 : float, signed
        flux_ratio   : float, F_late / F_early (> 1 = brightened)
        delta_mag_method : str, always 'seasonal_median_flux'
        frac_change  : float, |flux_ratio - 1| for backward compat
    """
    lc = wise_result.get('lc')
    if lc is None or len(lc) < 6:
        return 0.0, 0.0, 0.0, 1.0, 'seasonal_median_flux'

    mjd = lc['mjd'].values if 'mjd' in lc.columns else None
    w1_flux = lc['w1_flux_mjy'].values if 'w1_flux_mjy' in lc.columns else None
    w1_err = lc['w1_flux_err_mjy'].values if 'w1_flux_err_mjy' in lc.columns else None
    w2_flux = lc['w2_flux_mjy'].values if 'w2_flux_mjy' in lc.columns else None
    w2_err = lc['w2_flux_err_mjy'].values if 'w2_flux_err_mjy' in lc.columns else None

    if mjd is None or w1_flux is None:
        return 0.0, 0.0, 0.0, 1.0, 'seasonal_median_flux'

    # Default errors if missing
    if w1_err is None:
        w1_err = np.where(np.isfinite(w1_flux), w1_flux * 0.05, np.nan)

    # FIX 1: Use seasonal median flux method (dict return)
    dm_result_w1 = compute_delta_mag_correct(mjd, w1_flux, w1_err, z=z)
    if dm_result_w1 is not None:
        delta_mag_w1 = dm_result_w1['delta_mag']
        flux_ratio_w1 = dm_result_w1['flux_ratio']
    else:
        delta_mag_w1 = None
        flux_ratio_w1 = None

    # W2
    delta_mag_w2 = 0.0
    if w2_flux is not None:
        if w2_err is None:
            w2_err = np.where(np.isfinite(w2_flux), w2_flux * 0.05, np.nan)
        dm_result_w2 = compute_delta_mag_w2(mjd, w2_flux, w2_err, z=z)
        if dm_result_w2 is not None:
            delta_mag_w2 = float(dm_result_w2['delta_mag'])

    if delta_mag_w1 is None:
        delta_mag_w1 = 0.0
        flux_ratio_w1 = 1.0

    # Fractional change for normalization (absolute value, backward compat)
    flux_ratio_w1 = flux_ratio_w1 if flux_ratio_w1 is not None else 1.0
    frac_change = abs(flux_ratio_w1 - 1.0)

    # FIX 8: NO abs() — keep the sign
    return (float(delta_mag_w1), float(delta_mag_w2),
            float(frac_change), float(flux_ratio_w1), 'seasonal_median_flux')


# ---------------------------------------------------------------------------
# FIX 13: W1/W2 coherence check
# ---------------------------------------------------------------------------

def check_w1_w2_coherence(times, w1_flux, w2_flux, w1_err=None, w2_err=None):
    """
    Test whether W1 and W2 variability is coherent (physically expected).

    Real AGN variability: W1 and W2 are highly correlated (r > 0.6).
    CLAGN state change: W1 and W2 change in the SAME direction.
    Artifacts: W1 and W2 may be uncorrelated or anti-correlated.

    R2 FIX 3: Uses gap-based cluster_epochs_into_seasons() for season
    assignment instead of fixed 182.625-day calendar bins. Seasonal medians
    replace weighted means for robustness against outliers.

    Parameters
    ----------
    times : array, MJD
    w1_flux : array, W1 flux in mJy
    w2_flux : array, W2 flux in mJy
    w1_err : array or None
    w2_err : array or None

    Returns
    -------
    dict with keys:
        coherence_score : float [0, 1]
        pearson_r       : float
        same_direction  : bool or None
        coherence_flag  : str ('coherent'|'marginal'|'incoherent'|'insufficient_data')
    """
    _insufficient = {
        'coherence_score': 0.0, 'pearson_r': np.nan,
        'same_direction': None, 'coherence_flag': 'insufficient_data',
    }

    times = np.asarray(times, dtype=float)
    w1_flux = np.asarray(w1_flux, dtype=float)
    w2_flux = np.asarray(w2_flux, dtype=float)

    valid = (np.isfinite(times) & np.isfinite(w1_flux) & (w1_flux > 0) &
             np.isfinite(w2_flux) & (w2_flux > 0))

    if valid.sum() < 6:
        return _insufficient

    t = times[valid]
    f1 = w1_flux[valid]
    f2 = w2_flux[valid]

    if w1_err is not None:
        e1 = np.asarray(w1_err, dtype=float)[valid]
    else:
        e1 = f1 * 0.05

    if w2_err is not None:
        e2 = np.asarray(w2_err, dtype=float)[valid]
    else:
        e2 = f2 * 0.05

    # R2 FIX 3: Gap-based season clustering (replaces fixed 182.625-day bins)
    season_id = cluster_epochs_into_seasons(t)

    w1_season = {}
    w2_season = {}
    w1_season_err = {}
    w2_season_err = {}

    for s in np.unique(season_id[season_id >= 0]):
        mask = season_id == s
        if mask.sum() < 3:
            continue
        # R2 FIX 3: Use seasonal MEDIANS (robust) instead of weighted means
        w1_med = float(np.median(f1[mask]))
        w2_med = float(np.median(f2[mask]))
        w1_mad = float(np.median(np.abs(f1[mask] - w1_med)))
        w2_mad = float(np.median(np.abs(f2[mask] - w2_med)))
        n_s = mask.sum()
        w1_season[s] = w1_med
        w2_season[s] = w2_med
        w1_season_err[s] = max(1.4826 * w1_mad / np.sqrt(n_s), 1e-10)
        w2_season_err[s] = max(1.4826 * w2_mad / np.sqrt(n_s), 1e-10)

    common_seasons = sorted(set(w1_season) & set(w2_season))
    if len(common_seasons) < 4:
        return _insufficient

    w1_vals = np.array([w1_season[s] for s in common_seasons])
    w2_vals = np.array([w2_season[s] for s in common_seasons])
    w1_errs = np.array([w1_season_err[s] for s in common_seasons])
    w2_errs = np.array([w2_season_err[s] for s in common_seasons])

    # Uncertainty-weighted Pearson r
    combined_weights = 1.0 / np.maximum(w1_errs ** 2 + w2_errs ** 2, 1e-30)
    combined_weights /= combined_weights.sum()

    w1_wmean = np.average(w1_vals, weights=combined_weights)
    w2_wmean = np.average(w2_vals, weights=combined_weights)

    cov  = np.sum(combined_weights * (w1_vals - w1_wmean) * (w2_vals - w2_wmean))
    std1 = np.sqrt(np.sum(combined_weights * (w1_vals - w1_wmean) ** 2))
    std2 = np.sqrt(np.sum(combined_weights * (w2_vals - w2_wmean) ** 2))

    pearson_r = float(cov / (std1 * std2 + 1e-10))
    if not np.isfinite(pearson_r):
        pearson_r = 0.0

    # Same-direction change (indeterminate when changes are too flat)
    from ..config import COHERENCE_FLATNESS_THRESH
    seasons_sorted = sorted(common_seasons)
    if len(seasons_sorted) >= 4:
        f1_early = np.mean([w1_season[s] for s in seasons_sorted[:2]])
        f1_late  = np.mean([w1_season[s] for s in seasons_sorted[-2:]])
        f2_early = np.mean([w2_season[s] for s in seasons_sorted[:2]])
        f2_late  = np.mean([w2_season[s] for s in seasons_sorted[-2:]])
        delta_w1 = f1_late - f1_early
        delta_w2 = f2_late - f2_early
        thresh1 = COHERENCE_FLATNESS_THRESH * (abs(f1_early) + abs(f1_late)) / 2.0
        thresh2 = COHERENCE_FLATNESS_THRESH * (abs(f2_early) + abs(f2_late)) / 2.0
        if abs(delta_w1) < thresh1 or abs(delta_w2) < thresh2:
            same_direction = None
        else:
            same_direction = bool(np.sign(delta_w1) == np.sign(delta_w2))
    else:
        d1 = w1_vals[-1] - w1_vals[0]
        d2 = w2_vals[-1] - w2_vals[0]
        same_direction = bool(np.sign(d1) == np.sign(d2))

    if same_direction is None:
        direction_factor = 0.75
    elif same_direction:
        direction_factor = 1.0
    else:
        direction_factor = 0.5

    coherence_score = float(np.clip(max(0.0, pearson_r) * direction_factor, 0.0, 1.0))

    if pearson_r > 0.6 and same_direction is True:
        coherence_flag = 'coherent'
    elif pearson_r > 0.6 and same_direction is None:
        coherence_flag = 'marginal'
    elif pearson_r > 0.3:
        coherence_flag = 'marginal'
    else:
        coherence_flag = 'incoherent'

    return {
        'coherence_score': coherence_score,
        'pearson_r':       pearson_r,
        'same_direction':  same_direction,
        'coherence_flag':  coherence_flag,
    }


# ---------------------------------------------------------------------------
# FIX 10: Composite score v3 — eliminates double-counting
# ---------------------------------------------------------------------------

def compute_composite_score_v3(components):
    """
    Composite score v3: avoids double-counting correlated evidence.

    Component groups (take MAX within each group, not sum):

    Group A: Amplitude evidence (how much did it change?)
        - score_delta_mag  (from seasonal median flux)
        - score_flux_ratio (same quantity, different normalization)
        → A_score = max(score_delta_mag, score_flux_ratio)

    Group B: Temporal structure evidence (when did it change?)
        - score_changepoint_bic
        - score_drw_nonstat
        - score_broken_drw_bic
        → B_score = max of the three

    Group C: Color evidence (did the spectrum change?)
        - score_color_evolution  [independent]

    Group D: Statistical model evidence (is it unusual?)
        - score_sf_excess
        - score_drw_sigma_excess
        → D_score = max of the two

    Group E: Astrometric evidence (is it a real AGN?)
        - score_gaia_variable

    Weights: w_A=3, w_B=3, w_C=2, w_D=1.5, w_E=0.5

    Parameters
    ----------
    components : dict with component score keys

    Returns
    -------
    composite : float [0, 1]
    group_scores : dict of A-E group scores
    """
    A_score = max(
        _safe(components.get('delta_mag', 0)),
        _safe(components.get('flux_ratio', 0))
    )
    B_score = max(
        _safe(components.get('changepoint_bic', 0)),
        _safe(components.get('drw_nonstat', 0)),
        _safe(components.get('broken_drw_bic', 0))
    )
    C_score = _safe(components.get('color_evolution', 0))
    D_score = max(
        _safe(components.get('sf_excess', 0)),
        _safe(components.get('drw_sigma_excess', 0))
    )
    E_score = _safe(components.get('gaia_variable', 0))

    # score-M12: Use weights from config (SCORE_GROUP_WEIGHTS_V3)
    w = SCORE_GROUP_WEIGHTS_V3
    w_A = w['amplitude']
    w_B = w['temporal']
    w_C = w['color']
    w_D = w['statistical']
    w_E = w['astrometric']
    total_w = sum(w.values())

    composite = (w_A * A_score + w_B * B_score + w_C * C_score +
                 w_D * D_score + w_E * E_score) / total_w

    return float(np.clip(composite, 0.0, 1.0)), {
        'A_amplitude': float(A_score),
        'B_temporal': float(B_score),
        'C_color': float(C_score),
        'D_statistical': float(D_score),
        'E_astrometric': float(E_score),
    }


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def compute_composite_clagn_score(source_record, wise_result, drw_results,
                                    sf_results, cp_results, gaia_data):
    """
    7-component composite CLAGN scoring engine.

    Each component is normalized to [0, 1] before weighting.
    Final score uses compute_composite_score_v3 to avoid double-counting.

    FIX 1: delta_mag computed via seasonal median flux (not raw epochs)
    FIX 8: Signed delta_mag (positive = brightened, NO abs() wrapper)
    FIX 10: v3 scoring avoids double-counting correlated components

    Parameters
    ----------
    source_record : dict (ra, dec, redshift, source_id)
    wise_result   : dict from ingestion.wise.query_wise_lightcurve
    drw_results   : dict from models.drw.fit_drw_map
    sf_results    : dict from models.structure_function.detect_sf_break
    cp_results    : dict from models.changepoint.bayesian_changepoint_detection
    gaia_data     : dict from ingestion.gaia.query_gaia_dr3

    Returns
    -------
    result : dict with all component scores + composite + label
    """
    z = float(source_record.get('redshift', source_record.get('z', 0.1)))

    # Extract light curve early so it's available to all components below
    lc = wise_result.get('lc')

    # ---- Component 1: DRW Nonstationarity -----------------------------------
    nonstat_sigma = _safe(drw_results.get('nonstationarity_sigma'))
    score_drw_nonstat = _norm_drw_nonstat(nonstat_sigma)

    # ---- Component 2: Delta Magnitude W1 — FIX 1 (seasonal median flux) ----
    (delta_mag_w1, delta_mag_w2, frac_change,
     flux_ratio_w1, delta_mag_method) = _compute_delta_mag_seasonal(
        wise_result, z=z
    )

    # R2 FIX 2: Try compute_amplitude_metrics for max-contrast amplitude
    amplitude_metrics_result = None
    amplitude_method = delta_mag_method
    if lc is not None and len(lc) >= 6:
        mjd_am = lc['mjd'].values if 'mjd' in lc.columns else None
        w1_am  = lc['w1_flux_mjy'].values if 'w1_flux_mjy' in lc.columns else None
        w1e_am = lc['w1_flux_err_mjy'].values if 'w1_flux_err_mjy' in lc.columns else None
        if mjd_am is not None and w1_am is not None and w1e_am is not None:
            try:
                amplitude_metrics_result = compute_amplitude_metrics(
                    mjd_am, w1_am, w1e_am, z=z)
            except Exception:
                pass

    if amplitude_metrics_result is not None:
        delta_mag_best = amplitude_metrics_result['delta_mag']
        amplitude_method = amplitude_metrics_result['amplitude_method']
        frac_change_best = abs(10.0 ** (delta_mag_best / 2.5) - 1.0)
        score_delta_mag = _norm_frac_flux_change(frac_change_best)
        frac_change = frac_change_best
    else:
        # FIX 8: NO abs() — use signed delta_mag for scoring (|frac_change| for norm)
        score_delta_mag = _norm_frac_flux_change(frac_change)

    # Score for flux ratio (same amplitude evidence, different normalization)
    flux_ratio_score = float(np.clip(abs(flux_ratio_w1 - 1.0) / 1.0, 0.0, 1.0))

    # ---- Component 3: Bayesian Changepoint BIC ------------------------------
    delta_bic = _safe(cp_results.get('delta_bic'), default=0.0)
    score_changepoint = _norm_changepoint_bic(delta_bic)

    # ---- Component 4: Structure Function Break ------------------------------
    sf_excess_ratio = _safe(sf_results.get('sf_excess_ratio'), default=1.0)
    score_sf_break = _norm_sf_break(sf_excess_ratio)

    # ---- Component 5: W1-W2 Color Evolution ---------------------------------
    w1w2_early = _safe(wise_result.get('w1_minus_w2_early'), default=np.nan)
    w1w2_late  = _safe(wise_result.get('w1_minus_w2_late'),  default=np.nan)
    if np.isfinite(w1w2_early) and np.isfinite(w1w2_late):
        delta_color = w1w2_late - w1w2_early
    else:
        delta_color = 0.0
    score_color = _norm_color_evolution(delta_color)

    # ---- Component 6: DRW Sigma Excess vs Luminosity ------------------------
    sigma_drw_obs = _safe(drw_results.get('sigma_drw'), default=0.0)
    # lc already extracted above (R2 FIX 2)
    w1_flux_mean = (float(np.nanmean(lc['w1_flux_mjy'].values))
                    if lc is not None and 'w1_flux_mjy' in lc.columns
                    else 1.0)
    sigma_excess = _compute_flux_variability_excess(sigma_drw_obs, w1_flux_mean)
    score_sigma_excess = _norm_sigma_excess(sigma_excess)

    # ---- Component 7: GAIA Optical Variability ------------------------------
    gaia_var_flag = gaia_data.get('phot_variable_flag', 'NOT_AVAILABLE')
    score_gaia = _norm_gaia_var(gaia_var_flag)

    # ---- FIX 6: Host galaxy contamination -----------------------------------
    # FLAW B4: Labels as estimate (not source-specific SED decomposition)
    f_host, f_agn, contamination_flag = estimate_host_contamination_fraction(z)
    delta_mag_host_corrected_estimate = correct_delta_mag_for_host(delta_mag_w1, f_agn)
    host_fraction_prior_estimate = f_host

    # ---- FIX 13: W1/W2 coherence check -------------------------------------
    w1_w2_coherence_score = 0.5   # default (neutral)
    w1_w2_pearson_r = np.nan
    w1_w2_same_direction = None
    w1_w2_coherence_flag = 'insufficient_data'
    if lc is not None and len(lc) >= 6:
        mjd_arr = lc['mjd'].values if 'mjd' in lc.columns else None
        w1_arr = lc['w1_flux_mjy'].values if 'w1_flux_mjy' in lc.columns else None
        w2_arr = lc['w2_flux_mjy'].values if 'w2_flux_mjy' in lc.columns else None
        w1e_arr = lc['w1_flux_err_mjy'].values if 'w1_flux_err_mjy' in lc.columns else None
        w2e_arr = lc['w2_flux_err_mjy'].values if 'w2_flux_err_mjy' in lc.columns else None

        if mjd_arr is not None and w1_arr is not None and w2_arr is not None:
            _coh = check_w1_w2_coherence(
                mjd_arr, w1_arr, w2_arr, w1e_arr, w2e_arr
            )
            w1_w2_coherence_score = _coh['coherence_score']
            w1_w2_pearson_r       = _coh['pearson_r']
            w1_w2_same_direction  = _coh['same_direction']
            w1_w2_coherence_flag  = _coh['coherence_flag']

    # ---- FIX 10: Composite score v3 — no double-counting -------------------
    components_v3 = {
        'delta_mag':        score_delta_mag,
        'flux_ratio':       flux_ratio_score,
        'changepoint_bic':  score_changepoint,
        'drw_nonstat':      score_drw_nonstat,
        'broken_drw_bic':   0.0,   # Not computed in v1 path
        'color_evolution':  score_color,
        'sf_excess':        score_sf_break,
        'drw_sigma_excess': score_sigma_excess,
        'gaia_variable':    score_gaia,
    }
    composite, group_scores = compute_composite_score_v3(components_v3)

    # ---- Physical interpretation (R2 FIX 10: dict return with candidate language)
    label_dict = classify_clagn_type(
        {
            'score_drw_nonstat': score_drw_nonstat,
            'score_delta_mag': score_delta_mag,
            'score_changepoint': score_changepoint,
            'delta_mag_w1': delta_mag_w1,
        },
        drw_results, cp_results, wise_result
    )
    label = label_dict['clagn_type']

    return {
        'score_drw_nonstat':    score_drw_nonstat,
        'score_delta_mag':      score_delta_mag,
        'score_changepoint':    score_changepoint,
        'score_sf_break':       score_sf_break,
        'score_color':          score_color,
        'score_sigma_excess':   score_sigma_excess,
        'score_gaia':           score_gaia,
        'composite':            float(composite),
        'label':                label,
        # R2 FIX 10: Full classification dict
        'clagn_type':                          label_dict['clagn_type'],
        'classification_basis':                label_dict['classification_basis'],
        'spectroscopic_confirmation_required': label_dict['spectroscopic_confirmation_required'],
        'classification_confidence':           label_dict['classification_confidence'],
        # FIX 1, 8: Signed delta_mag from seasonal median flux method
        'delta_mag_w1':                          delta_mag_w1,
        'delta_mag_w2':                          delta_mag_w2,
        'delta_mag_method':                      delta_mag_method,
        # R2 FIX 2: Best-amplitude method used for scoring
        'amplitude_method':                      amplitude_method,
        'amplitude_metrics':                     amplitude_metrics_result,
        # FLAW B4: renamed to clearly label as estimate
        'delta_mag_host_corrected_estimate':     delta_mag_host_corrected_estimate,
        'host_fraction_prior_estimate':          host_fraction_prior_estimate,
        'host_model':                            'redshift_prior_heuristic_Assef2013',
        'host_correction_note':                  'Rough prior; not source-specific SED decomposition',
        'host_contamination_flag':               contamination_flag,
        'delta_color':                 delta_color,
        'sigma_excess':                sigma_excess,
        'frac_change':                 frac_change,
        'w1_flux_ratio':               flux_ratio_w1,
        # FIX 13: W1/W2 coherence
        'w1_w2_coherence_score':       float(w1_w2_coherence_score),
        'w1_w2_pearson_r':             float(w1_w2_pearson_r) if np.isfinite(w1_w2_pearson_r) else np.nan,
        'w1_w2_same_direction':        bool(w1_w2_same_direction) if w1_w2_same_direction is not None else None,
        'w1_w2_coherence_flag':        w1_w2_coherence_flag,
        # FIX 10: Group scores breakdown
        'group_A_amplitude':           group_scores['A_amplitude'],
        'group_B_temporal':            group_scores['B_temporal'],
        'group_C_color':               group_scores['C_color'],
        'group_D_statistical':         group_scores['D_statistical'],
        'group_E_astrometric':         group_scores['E_astrometric'],
    }


def _compute_flux_ratio(wise_result):
    """Compute max/min ratio of rolling 1-year median W1 fluxes."""
    lc = wise_result.get('lc')
    if lc is None or len(lc) < 6:
        return 1.0

    from ..utils.photometry import rolling_median_flux
    mjd = lc['mjd'].values
    w1  = lc['w1_flux_mjy'].values if 'w1_flux_mjy' in lc.columns else None
    if w1 is None:
        return 1.0

    rolling = rolling_median_flux(mjd, w1, window_years=1.0)
    valid = rolling[np.isfinite(rolling)]
    if len(valid) < 2 or valid.min() <= 0:
        return 1.0

    return float(valid.max() / valid.min())


# ---------------------------------------------------------------------------
# CLAGN type classification
# ---------------------------------------------------------------------------

def classify_clagn_type(scores, drw_results, cp_results, wise_result):
    """
    Assign a physical interpretation label based on the Ricci & Trakhtenbrot
    (2022) CLAGN taxonomy.

    R2 FIX 10: Returns a dict (not a plain str) using photometric candidate
    language. Spectroscopic confirmation is always required.

    Classification decision tree:
    1. Rapid Transition photometric candidate
    2. CS-AGN Turn-Off photometric candidate
    3. CS-AGN Turn-On photometric candidate
    4. CO-AGN photometric candidate (cloud eclipse)
    5. Ambiguous photometric CLAGN candidate

    Parameters
    ----------
    scores : dict with delta_mag_w1 and component scores
    drw_results, cp_results, wise_result : pipeline result dicts

    Returns
    -------
    dict with keys:
        clagn_type : str  (photometric candidate label)
        classification_basis : str  (which evidence drove the label)
        spectroscopic_confirmation_required : bool  (always True)
        classification_confidence : str  ('high'|'moderate'|'low')
    """
    delta_mag     = _safe(scores.get('delta_mag_w1', 0.0))
    abs_delta_mag = abs(delta_mag)
    pre_mean      = _safe(cp_results.get('pre_break_mean', 0.0))
    post_mean     = _safe(cp_results.get('post_break_mean', 0.0))
    break_dur     = _safe(cp_results.get('break_duration_days', 365.0), default=365.0)
    tau           = _safe(drw_results.get('tau_rest_days', 300.0), default=300.0)

    w1w2_early = _safe(wise_result.get('w1_minus_w2_early'), default=np.nan)
    w1w2_late  = _safe(wise_result.get('w1_minus_w2_late'),  default=np.nan)
    color_change = np.nan
    if np.isfinite(w1w2_early) and np.isfinite(w1w2_late):
        color_change = w1w2_late - w1w2_early

    def _make(clagn_type, basis, confidence):
        return {
            'clagn_type':                          clagn_type,
            'classification_basis':                basis,
            'spectroscopic_confirmation_required': True,
            'classification_confidence':           confidence,
        }

    # Rapid Transition: short break duration AND large amplitude (either direction)
    if abs_delta_mag > 0.75 and np.isfinite(break_dur) and break_dur < 2 * 365.25:
        return _make(
            'rapid_transition_photometric_candidate',
            'large_amplitude_short_timescale_photometric',
            'moderate'
        )

    # Turn-Off with confirmed color change
    if (delta_mag < -0.5 and pre_mean > 0 and post_mean < pre_mean and
            np.isfinite(color_change) and color_change < -0.05):
        return _make(
            'cs_agn_turn_off_photometric_candidate',
            'fading_flux_and_bluer_color_photometric',
            'high'
        )

    # Turn-On with confirmed color change
    if (delta_mag > 0.5 and post_mean > pre_mean and
            np.isfinite(color_change) and color_change > 0.05):
        return _make(
            'cs_agn_turn_on_photometric_candidate',
            'rising_flux_and_redder_color_photometric',
            'high'
        )

    # Strong turn-off without confirmed color change
    if delta_mag < -0.5 and pre_mean > 0 and post_mean < pre_mean:
        return _make(
            'cs_agn_turn_off_photometric_candidate',
            'fading_flux_photometric_no_color_confirmation',
            'moderate'
        )

    # Strong turn-on without confirmed color change
    if delta_mag > 0.5 and post_mean > pre_mean:
        return _make(
            'cs_agn_turn_on_photometric_candidate',
            'rising_flux_photometric_no_color_confirmation',
            'moderate'
        )

    # CO-AGN: short DRW timescale, moderate amplitude
    if tau < 100.0 and 0.1 < abs_delta_mag < 0.4:
        return _make(
            'co_agn_photometric_candidate',
            'short_tau_drw_moderate_amplitude_photometric',
            'low'
        )

    # Marginal / ambiguous
    if 0.3 <= abs_delta_mag <= 0.5:
        return _make(
            'ambiguous_photometric_clagn_candidate',
            'marginal_amplitude_photometric',
            'low'
        )

    return _make(
        'ambiguous_photometric_clagn_candidate',
        'insufficient_discriminating_evidence',
        'low'
    )


# ---------------------------------------------------------------------------
# False positive rejection
# ---------------------------------------------------------------------------

def apply_false_positive_rejection(candidates_df, lc_dict, wise_dict_map):
    """
    Apply false positive rejection logic before finalizing top 6.

    Checks:
    1. Supernovae contamination (short break duration + declining flux)
    2. Stellar variability leakage (low quasar prob + Galactic plane)
    3. WISE single-epoch artifacts (single-epoch-driven variability)
    4. Host galaxy contamination (marginal CLAGN at z < 0.05)

    Applies multiplicative score penalties (not hard rejections) with a
    contamination_flag explaining the reason.

    Parameters
    ----------
    candidates_df : pd.DataFrame with composite_score column
    lc_dict       : dict {source_id: lc_DataFrame}
    wise_dict_map : dict {source_id: wise_result_dict}

    Returns
    -------
    candidates_df : updated with contamination_flag column and adjusted scores
    """
    import pandas as pd

    candidates_df = candidates_df.copy()
    candidates_df['contamination_flag'] = 'none'

    for idx in candidates_df.index:
        row = candidates_df.loc[idx]
        flags = []
        penalty = 1.0

        source_id = str(row.get('source_id', ''))
        wise_result = wise_dict_map.get(source_id, {})

        # ---- Check 1: Supernovae contamination ----------------------------
        break_dur = _safe(row.get('changepoint_break_duration_days'), default=np.nan)
        post_mean = _safe(row.get('changepoint_post_break_mean'), default=np.nan)
        pre_mean  = _safe(row.get('changepoint_pre_break_mean'),  default=np.nan)

        if (np.isfinite(break_dur) and
                break_dur < FP_SN_BREAK_DURATION_DAYS and
                np.isfinite(post_mean) and np.isfinite(pre_mean) and
                post_mean < pre_mean):
            flags.append(f'possible_SN_contamination(break_dur={break_dur:.0f}d)')
            penalty *= FP_SN_SCORE_PENALTY

        # ---- Check 2: Stellar variability leakage -------------------------
        ra  = _safe(row.get('ra'),  default=np.nan)
        dec = _safe(row.get('dec'), default=np.nan)
        quasar_prob = _safe(row.get('gaia_quasar_prob'), default=np.nan)

        if (np.isfinite(ra) and np.isfinite(dec) and
                np.isfinite(quasar_prob) and
                quasar_prob < FP_LOW_QUASAR_PROB):
            try:
                gal_lat = abs(galactic_latitude(ra, dec))
                if gal_lat < FP_GALACTIC_LAT_THRESHOLD:
                    flags.append(
                        f'stellar_contamination_risk(|b|={gal_lat:.1f}deg,'
                        f'quasar_prob={quasar_prob:.2f})'
                    )
                    penalty *= FP_STELLAR_SCORE_PENALTY
            except Exception:
                pass

        # ---- Check 3: WISE single-epoch artifact ---------------------------
        lc = lc_dict.get(source_id)
        if lc is not None and len(lc) >= 10 and 'w1_flux_mjy' in lc.columns:
            artifact_flag = _check_wise_artifact(lc)
            if artifact_flag:
                flags.append('possible_wise_artifact(single_epoch_driven)')
                penalty *= FP_ARTIFACT_SCORE_PENALTY

        # ---- Check 4: Host galaxy contamination ---------------------------
        # R2 FIX 9: Flag host contamination risk but do NOT apply a score penalty.
        # Applying a penalty here double-penalises low-z sources (they already score
        # lower due to diluted delta_mag). The flag is preserved for transparency.
        host_risk = bool(row.get('host_contamination_risk', False))
        delta_mag = _safe(
            row.get('delta_mag_w1',
            row.get('w1_delta_mag',
            row.get('delta_mag', 0.0))),
            default=0.0
        )
        if host_risk and delta_mag < 0.5:
            flags.append(
                'host_contamination_risk(z<0.05,marginal_delta_mag,annotation_only)'
            )
            # NOTE: No penalty *= FP_HOST_SCORE_PENALTY (R2 FIX 9)

        # ---- Apply penalty --------------------------------------------------
        if penalty < 1.0:
            old_score = _safe(row.get('composite_score'), default=0.0)
            candidates_df.loc[idx, 'composite_score'] = old_score * penalty
            candidates_df.loc[idx, 'contamination_flag'] = '; '.join(flags)
            logger.info(
                f"{source_id}: FP penalty {penalty:.2f}× → "
                f"score {old_score:.3f} → {old_score*penalty:.3f} "
                f"[{'; '.join(flags)}]"
            )

    return candidates_df


def apply_contaminant_filters(candidates_df, lc_dict, wise_dict_map, allow_network=True):
    """
    Apply contaminant war-plan filters (SN morphology, transient catalogs, dust, blazars).

    Returns updated DataFrame with contamination_flag and adjusted composite_score.
    """
    from ..models.variability import cluster_epochs_into_seasons
    try:
        from validation.sn_rejection import check_sn_morphology, transient_catalog_xmatch, color_reversal_timescale_flag
    except Exception:
        check_sn_morphology = None
        transient_catalog_xmatch = None
        color_reversal_timescale_flag = None
    try:
        from validation.dust_obscuration import color_flux_correlation
    except Exception:
        color_flux_correlation = None
    try:
        from validation.blazar_rejection import is_blazar
    except Exception:
        is_blazar = None

    import pandas as pd
    df = apply_false_positive_rejection(candidates_df, lc_dict, wise_dict_map)

    for idx in df.index:
        row = df.loc[idx]
        source_id = str(row.get('source_id', ''))
        lc = lc_dict.get(source_id)
        flags = []
        penalty = 1.0

        # ---- SN morphology + transient catalogs ---------------------------
        if lc is not None and len(lc) >= 6 and 'w1_flux_mjy' in lc.columns:
            times = lc['mjd'].values
            flux = lc['w1_flux_mjy'].values
            season_ids = cluster_epochs_into_seasons(times)
            season_medians = {}
            for s in np.unique(season_ids):
                season_medians[int(s)] = float(np.nanmedian(flux[season_ids == s]))

            if check_sn_morphology:
                sn = check_sn_morphology(times, flux, season_medians)
                if sn.get('sn_like_morphology', False):
                    flags.append('sn_like_morphology')
                    penalty *= FP_SN_MORPH_PENALTY

            # Transient catalog crossmatch (uses changepoint_mjd if present)
            if transient_catalog_xmatch:
                transition_mjd = row.get('changepoint_mjd', row.get('changepoint_mjd', np.nan))
                ra = _safe(row.get('ra'), default=np.nan)
                dec = _safe(row.get('dec'), default=np.nan)
                if np.isfinite(ra) and np.isfinite(dec) and np.isfinite(transition_mjd):
                    matches = transient_catalog_xmatch(
                        ra, dec, transition_mjd,
                        max_sep_arcsec=TRANSIENT_XMATCH_RADIUS_ARCSEC,
                        max_dt_days=TRANSIENT_XMATCH_MAX_DT_DAYS
                    )
                    if matches:
                        flags.append('transient_catalog_match')
                        penalty *= FP_SN_XMATCH_PENALTY

            # ---- Dust obscuration ----------------------------------------
            if color_flux_correlation and 'w2_flux_mjy' in lc.columns:
                w2_flux = lc['w2_flux_mjy'].values
                w2_season_medians = {}
                for s in np.unique(season_ids):
                    w2_season_medians[int(s)] = float(np.nanmedian(w2_flux[season_ids == s]))
                dust = color_flux_correlation(season_medians, w2_season_medians)
                if dust.get('dust_obscuration_flag', False):
                    flags.append('dust_obscuration_signature')
                    penalty *= FP_DUST_SCORE_PENALTY

                if color_reversal_timescale_flag:
                    if color_reversal_timescale_flag(season_medians, w2_season_medians):
                        flags.append('sn_color_reversal')
                        penalty *= FP_SN_MORPH_PENALTY

        # ---- Blazar rejection (radio crossmatch) -------------------------
        ra = _safe(row.get('ra'), default=np.nan)
        dec = _safe(row.get('dec'), default=np.nan)
        if is_blazar and np.isfinite(ra) and np.isfinite(dec):
            bz = is_blazar(source_id, ra, dec, allow_network=allow_network)
            if bz.get('blazar_flag', False):
                flags.append('blazar_radio_match')
                penalty *= FP_BLAZAR_SCORE_PENALTY

        if penalty < 1.0:
            old_score = _safe(row.get('composite_score'), default=_safe(row.get('composite_score_v2'), default=0.0))
            df.loc[idx, 'composite_score'] = old_score * penalty
            existing_flag = str(df.loc[idx, 'contamination_flag']) if 'contamination_flag' in df.columns else ''
            existing_flag = '' if existing_flag in ('none', 'nan') else existing_flag
            if existing_flag and flags:
                df.loc[idx, 'contamination_flag'] = existing_flag + '; ' + '; '.join(flags)
            elif flags:
                df.loc[idx, 'contamination_flag'] = '; '.join(flags)

    return df


def _check_wise_artifact(lc):
    """
    Check if variability is driven by a single extreme WISE epoch.

    score-H8: Uses seasonal delta_mag method. Finds most deviant epoch by
    MAD z-score. If removing that epoch causes delta_mag to drop by > 40%,
    the variability is artifact-driven.

    Returns True if artifact suspected.
    """
    if 'mjd' not in lc.columns or 'w1_flux_mjy' not in lc.columns:
        return False

    mjd    = lc['mjd'].values
    fluxes = lc['w1_flux_mjy'].values
    errors = (lc['w1_flux_err_mjy'].values
              if 'w1_flux_err_mjy' in lc.columns
              else np.where(np.isfinite(fluxes) & (fluxes > 0), fluxes * 0.05, np.nan))

    valid = np.isfinite(fluxes) & (fluxes > 0) & np.isfinite(errors) & (errors > 0)
    if valid.sum() < 10:
        return False

    f = fluxes[valid]
    t = mjd[valid]
    e = errors[valid]

    # Find most deviant epoch by MAD z-score
    med = np.nanmedian(f)
    mad = np.nanmedian(np.abs(f - med))
    if mad <= 0:
        return False

    z_scores = np.abs(f - med) / (1.4826 * mad + 1e-30)
    worst_idx = int(np.argmax(z_scores))
    if z_scores[worst_idx] < 3.0:
        return False   # No extreme outlier — not artifact-driven

    # Compute delta_mag on full dataset
    dm_full = compute_delta_mag_correct(t, f, e)
    if dm_full is None:
        return False
    delta_full = abs(dm_full['delta_mag'])

    # Compute delta_mag without the worst epoch
    keep = np.ones(len(f), dtype=bool)
    keep[worst_idx] = False
    dm_clean = compute_delta_mag_correct(t[keep], f[keep], e[keep])
    if dm_clean is None:
        # If we can't compute without that epoch, it's definitely artifact-driven
        return True
    delta_clean = abs(dm_clean['delta_mag'])

    # If single epoch drove > 40% of amplitude, flag as artifact
    if delta_full > 0 and (delta_clean < delta_full * 0.6):
        return True
    return False


# ---------------------------------------------------------------------------
# Expansion 6: Score v2 — 10-component scorer with MCMC/broken DRW inputs
# The existing compute_composite_clagn_score() is PRESERVED unchanged above.
# ---------------------------------------------------------------------------

from ..config import SCORE_WEIGHTS_V2, MAX_SCORE_V2


def _norm_broken_drw_bic(delta_bic_broken):
    """Broken DRW BIC improvement: min(ΔBIC/20, 1)."""
    return float(np.clip(_safe(delta_bic_broken) / 20.0, 0.0, 1.0))


def _norm_nonstationary_gp(log_bayes_factor):
    """Non-stationary GP log Bayes factor: min(lnBF/5, 1)."""
    return float(np.clip(_safe(log_bayes_factor) / 5.0, 0.0, 1.0))


def _norm_flux_bimodality(w1_fluxes):
    """Bimodality coefficient of W1 flux distribution.

    BC = (skewness^2 + 1) / (kurtosis + 3*(n-1)^2/((n-2)*(n-3)))
    BC > 0.555 → bimodal → score = min((BC - 0.555) / 0.3, 1.0)
    """
    if w1_fluxes is None:
        return 0.0
    try:
        from scipy.stats import skew, kurtosis
        f = np.asarray(w1_fluxes, dtype=float)
        f = f[np.isfinite(f)]
        n = len(f)
        if n < 8:
            return 0.0
        sk = skew(f)
        ku = kurtosis(f, fisher=True)  # excess kurtosis
        # Bimodality coefficient
        bc_denom = ku + 3.0 * (n - 1.0) ** 2 / max((n - 2.0) * (n - 3.0), 1.0)
        if bc_denom == 0:
            return 0.0
        bc = (sk ** 2 + 1.0) / bc_denom
        if bc > 0.555:
            return float(np.clip((bc - 0.555) / 0.3, 0.0, 1.0))
        return 0.0
    except Exception:
        return 0.0


def _norm_gaia_var_v2(gaia_lc, gaia_photometry):
    """Enhanced GAIA variability score using G-band RMS and flag.

    Returns 0.0–1.0 based on:
    - phot_variable_flag = VARIABLE: 0.5 base
    - G-band RMS > 0.05 mag: up to +0.5 additional
    """
    score = 0.0
    if gaia_photometry and isinstance(gaia_photometry, dict):
        if gaia_photometry.get('is_variable', False):
            score = 0.5
    if gaia_lc and isinstance(gaia_lc, dict):
        g_rms = gaia_lc.get('g_rms', np.nan)
        if g_rms is not None and np.isfinite(g_rms):
            score += float(np.clip(g_rms / 0.1, 0.0, 0.5))
    return float(np.clip(score, 0.0, 1.0))


def _bootstrap_score_uncertainty(source_data, n_bootstrap=100):
    """Block bootstrap uncertainty estimate for composite score.

    R2 FIX 7: Resamples SEASONS (blocks) instead of individual epochs.
    This preserves within-season temporal structure, which is critical for
    the seasonal-median delta_mag computation. Individual-epoch resampling
    destroys the seasonal grouping and produces overconfident (narrow)
    uncertainty estimates.

    Parameters
    ----------
    source_data : dict with keys 'times', 'w1_flux_mjy', 'w1_flux_err_mjy', 'z'
    n_bootstrap : int

    Returns
    -------
    dict with: score_err, score_err_lo, score_err_hi, n_bootstrap,
               bootstrap_component, bootstrap_method
    """
    t = np.asarray(source_data.get('times', []), dtype=float)
    f = np.asarray(source_data.get('w1_flux_mjy', []), dtype=float)
    e = np.asarray(source_data.get('w1_flux_err_mjy', []), dtype=float)
    z = float(source_data.get('z', 0.1))

    _empty = {'score_err': np.nan, 'score_err_lo': np.nan, 'score_err_hi': np.nan,
              'n_bootstrap': 0, 'bootstrap_component': 'delta_mag',
              'bootstrap_method': 'block_by_season'}

    if len(t) < 10:
        return _empty

    try:
        # Assign season labels via gap-based clustering
        season_ids = cluster_epochs_into_seasons(t)
        unique_seasons = np.unique(season_ids[season_ids >= 0])
        if len(unique_seasons) < 4:
            return _empty

        # Group epoch indices by season
        season_groups = {s: np.where(season_ids == s)[0] for s in unique_seasons}

        rng = np.random.default_rng(42)
        bootstrap_scores = []

        for _ in range(n_bootstrap):
            # Resample seasons WITH replacement (block bootstrap)
            sampled_seasons = rng.choice(unique_seasons,
                                         size=len(unique_seasons), replace=True)
            idx_list = []
            for s in sampled_seasons:
                idx_list.extend(season_groups[s].tolist())
            idx_arr = np.array(idx_list, dtype=int)

            t_b = t[idx_arr]
            f_b = f[idx_arr]
            e_b = e[idx_arr]

            # Sort by time for seasonal-median computation
            sort_order = np.argsort(t_b)
            t_b = t_b[sort_order]
            f_b = f_b[sort_order]
            e_b = e_b[sort_order]

            dm_result = compute_delta_mag_correct(t_b, f_b, e_b, z=z)
            if dm_result is not None:
                frac_change = abs(dm_result['flux_ratio'] - 1.0)
                c_delta = _norm_frac_flux_change(frac_change)
            else:
                c_delta = 0.0
            bootstrap_scores.append(float(c_delta))

        scores_arr = np.array(bootstrap_scores)
        score_std  = float(np.std(scores_arr))
        pct = np.percentile(scores_arr, [16, 84])
        return {
            'score_err':           score_std,
            'score_err_lo':        float(pct[0]),
            'score_err_hi':        float(pct[1]),
            'n_bootstrap':         n_bootstrap,
            'bootstrap_component': 'delta_mag',
            'bootstrap_method':    'block_by_season',
        }
    except Exception:
        return _empty


def compute_composite_score_v2(source, lc_data=None, drw_map=None, drw_mcmc_results=None,
                                broken_drw=None, nonstat_gp=None, sf_results=None,
                                cp_results=None, variability_stats=None,
                                gaia_lc=None, gaia_photometry=None, physics_results=None):
    """10-component CLAGN scorer v2.

    Incorporates broken DRW, non-stationary GP, flux bimodality, and
    enhanced GAIA variability. Backfills from v1 results when advanced
    inputs are None.

    Parameters
    ----------
    source : dict or pd.Series — source catalog info (needs 'z' or 'redshift')
    lc_data : dict or None — light curve data with 'lc_df' or 'combined_lc'
    drw_map : dict or None — MAP DRW results (fit_drw_map output)
    drw_mcmc_results : dict or None — MCMC DRW results (fit_drw_full_mcmc output)
    broken_drw : dict or None — broken DRW fit (fit_broken_drw output)
    nonstat_gp : dict or None — non-stationary GP fit (fit_nonstationary_gp output)
    sf_results : dict or None — structure function results
    cp_results : dict or None — changepoint detection results
    variability_stats : dict or None — variability statistics (delta_mag etc.)
    gaia_lc : dict or None — GAIA epoch photometry results
    gaia_photometry : dict or None — GAIA photometry/color info
    physics_results : dict or None — physical parameter results

    Returns
    -------
    dict with:
        composite_score_v2 : float [0,1]
        composite_score_v2_raw : float [0, MAX_SCORE_V2]
        composite_score_err : float (bootstrap uncertainty)
        components : dict of 10 component scores
        weights : dict (SCORE_WEIGHTS_V2)
        max_score : float (MAX_SCORE_V2)
    """
    z = float(source.get('z', source.get('redshift', 0.1))) if source else 0.1

    # Extract light curve arrays if available
    times = fluxes = errors = None
    if lc_data is not None:
        df = None
        if isinstance(lc_data, dict):
            df = lc_data.get('lc_df') or lc_data.get('combined_lc')
        elif hasattr(lc_data, 'columns'):
            df = lc_data

        # score-C3: fallback to 'epochs_df' key in source dict
        if df is None and isinstance(source, dict) and 'epochs_df' in source:
            df = source['epochs_df']

        if df is not None and len(df) > 5:
            for col_t in ['mjd']:
                if col_t in df.columns:
                    times = df[col_t].values
                    break
            # score-C3: try canonical column names first
            for col_f in ['w1_flux_mjy', 'w1flux', 'w1flux_ep', 'w1_flux']:
                if col_f in df.columns:
                    fluxes = df[col_f].values
                    break
            for col_e in ['w1_flux_err_mjy', 'w1flux_err', 'w1sigflux_ep', 'w1sigflux', 'w1_flux_err']:
                if col_e in df.columns:
                    errors = df[col_e].values
                    break

    # --- Component 1: DRW nonstationarity ---
    nonstat_sigma = 0.0
    if drw_map and 'nonstationarity_sigma' in drw_map:
        nonstat_sigma = drw_map.get('nonstationarity_sigma', 0.0)
    elif variability_stats and 'nonstationarity_sigma' in variability_stats:
        nonstat_sigma = variability_stats.get('nonstationarity_sigma', 0.0)
    c1_drw_nonstat = _norm_drw_nonstat(nonstat_sigma)

    # --- Component 2: Broken DRW BIC ---
    delta_bic_broken = np.nan
    if broken_drw and broken_drw.get('success', False):
        delta_bic_broken = broken_drw.get('delta_bic_broken', np.nan)
    c2_broken_drw = _norm_broken_drw_bic(delta_bic_broken)

    # --- Component 3: Non-stationary GP ---
    ns_log_bf = np.nan
    if nonstat_gp and nonstat_gp.get('success', False):
        ns_log_bf = nonstat_gp.get('log_bayes_factor', np.nan)
    c3_ns_gp = _norm_nonstationary_gp(ns_log_bf)

    # --- Component 4: Delta mag W1 (R2 FIX 8: explicit mag→fraction conversion) ---
    delta_mag_frac = 0.0
    if variability_stats:
        # Read as magnitude (Pogson scale), then convert to fractional flux change
        delta_mag_mag = float(_safe(
            variability_stats.get(
                'delta_mag_w1_seasonal_pogson',
                variability_stats.get('delta_mag_w1',
                variability_stats.get('delta_mag', 0.0))
            ), default=0.0
        ))
        flux_ratio_v2 = 10.0 ** (delta_mag_mag / 2.5)
        delta_mag_frac = abs(flux_ratio_v2 - 1.0)
    elif fluxes is not None and len(fluxes) > 10:
        # Fallback: raw flux median comparison → true fractional change
        n = len(fluxes)
        mid = n // 2
        f_early = np.median(fluxes[:mid])
        f_late = np.median(fluxes[mid:])
        if f_early > 0:
            flux_ratio_v2 = f_late / f_early
            delta_mag_frac = abs(flux_ratio_v2 - 1.0)
    c4_delta_mag = _norm_frac_flux_change(delta_mag_frac)

    # --- Component 5: Changepoint BIC ---
    cp_delta_bic = np.nan
    if cp_results:
        cp_delta_bic = _safe(cp_results.get('delta_bic', cp_results.get('bic_delta', np.nan)))
    c5_cp_bic = _norm_changepoint_bic(cp_delta_bic)

    # --- Component 6: SF excess (from sf_results) ---
    sf_ratio = np.nan
    if sf_results and isinstance(sf_results, dict):
        sf_ratio = _safe(sf_results.get('sf_ratio', sf_results.get('break_ratio', np.nan)))
    c6_sf = _norm_sf_break(sf_ratio)

    # --- Component 7: Color evolution W1-W2 ---
    color_change = np.nan
    if variability_stats:
        color_change = _safe(variability_stats.get('w1w2_color_change',
                             variability_stats.get('delta_color', np.nan)))
    c7_color = _norm_color_evolution(color_change)

    # --- Component 8: Flux bimodality ---
    w1_fluxes_arr = None
    if fluxes is not None:
        w1_fluxes_arr = fluxes
    elif lc_data is not None and isinstance(lc_data, dict):
        df = lc_data.get('lc_df') or lc_data.get('combined_lc')
        if df is not None:
            for col in ['w1flux', 'w1flux_ep']:
                if col in df.columns:
                    w1_fluxes_arr = df[col].values
                    break
    c8_bimodal = _norm_flux_bimodality(w1_fluxes_arr)

    # --- Component 9: GAIA variability (enhanced) ---
    c9_gaia = _norm_gaia_var_v2(gaia_lc, gaia_photometry)

    # --- Component 10: DRW sigma excess ---
    sigma_excess = np.nan
    if drw_map:
        sigma_excess = _safe(drw_map.get('sigma_excess', np.nan))
    c10_sigma_excess = _norm_sigma_excess(sigma_excess)

    # --- Weighted sum ---
    w = SCORE_WEIGHTS_V2
    components = {
        'drw_nonstationarity': c1_drw_nonstat,
        'broken_drw_bic': c2_broken_drw,
        'nonstationary_gp': c3_ns_gp,
        'delta_mag_w1': c4_delta_mag,
        'changepoint_bic': c5_cp_bic,
        'sf_excess': c6_sf,
        'color_evolution': c7_color,
        'flux_bimodality': c8_bimodal,
        'gaia_variability': c9_gaia,
        'drw_sigma_excess': c10_sigma_excess,
    }

    raw_score = sum(w[k] * components[k] for k in w)
    # score-M13: Verify MAX_SCORE_V2 matches sum of weights (guards against drift)
    max_score_actual = sum(SCORE_WEIGHTS_V2.values())
    if abs(max_score_actual - MAX_SCORE_V2) > 0.01:
        raise ValueError(
            f"MAX_SCORE_V2={MAX_SCORE_V2} != sum(SCORE_WEIGHTS_V2)={max_score_actual}. "
            f"Update MAX_SCORE_V2 in config.py to match SCORE_WEIGHTS_V2."
        )
    composite_score_v2 = raw_score / max_score_actual

    # Bootstrap uncertainty estimate (score-C1: dict signature)
    composite_score_err = np.nan
    if times is not None and fluxes is not None and errors is not None:
        valid = np.isfinite(times) & np.isfinite(fluxes) & np.isfinite(errors) & (errors > 0)
        if valid.sum() >= 10:
            composite_score_err = _bootstrap_score_uncertainty(
                {'times': times[valid], 'w1_flux_mjy': fluxes[valid],
                 'w1_flux_err_mjy': errors[valid], 'z': z},
                n_bootstrap=100
            ).get('score_err', np.nan)

    return {
        'composite_score_v2': float(np.clip(composite_score_v2, 0.0, 1.0)),
        'composite_score_v2_raw': float(raw_score),
        'composite_score_err': float(composite_score_err) if np.isfinite(composite_score_err) else np.nan,
        'components': components,
        'weights': dict(w),
        'max_score': MAX_SCORE_V2,
        'nonstat_sigma': float(nonstat_sigma),
        'delta_bic_broken': float(delta_bic_broken) if np.isfinite(delta_bic_broken) else np.nan,
        'ns_log_bf': float(ns_log_bf) if np.isfinite(ns_log_bf) else np.nan,
    }


def rescore_all_v2(results_dir='./results/'):
    """Load all pickled results for top candidates, run v2 scoring, write enhanced CSV.

    Parameters
    ----------
    results_dir : str

    Returns
    -------
    pd.DataFrame with v2 scores appended
    """
    import os
    import pickle
    import glob as glob_mod
    import pandas as pd

    csv_path = os.path.join(results_dir, 'top_candidates.csv')
    if not os.path.exists(csv_path):
        logger.error("top_candidates.csv not found in %s", results_dir)
        return None

    candidates = pd.read_csv(csv_path)
    logger.info("Rescoring %d sources with v2 scorer", len(candidates))

    v2_rows = []
    for _, row in candidates.iterrows():
        source_id = str(row.get('source_id', row.get('name', 'unknown')))

        # Load all available result pickles for this source
        lc_data = drw_map = drw_mcmc = broken_drw = nonstat_gp = None
        sf_results = cp_results = variability_stats = None
        gaia_lc = gaia_photometry = physics_results = None

        for pkl_f in glob_mod.glob(os.path.join(results_dir, f'*{source_id}*.pkl')):
            try:
                with open(pkl_f, 'rb') as f:
                    tmp = pickle.load(f)
                if not isinstance(tmp, dict):
                    continue

                if 'lc_df' in tmp or 'combined_lc' in tmp:
                    lc_data = tmp
                if 'tau_rest_days' in tmp and drw_map is None:
                    drw_map = tmp
                if 'tau_med' in tmp or 'chain' in tmp:
                    drw_mcmc = tmp
                if 'delta_bic_broken' in tmp:
                    broken_drw = tmp
                if 'log_bayes_factor' in tmp and 't_break_ns' in tmp:
                    nonstat_gp = tmp
                if 'sf_ratio' in tmp or 'break_ratio' in tmp:
                    sf_results = tmp
                if 'delta_bic' in tmp or 'change_time_mjd' in tmp:
                    cp_results = tmp
                if 'delta_mag_frac' in tmp or 'w1w2_color_change' in tmp:
                    variability_stats = tmp
                if 'epoch_photometry' in tmp:
                    gaia_lc = tmp.get('epoch_photometry')
                    gaia_photometry = tmp.get('color_variability')
                if 'L_bol' in tmp or 'M_BH_solar' in tmp:
                    physics_results = tmp
            except Exception:
                continue

        # Also try mcmc-specific pickle
        mcmc_pkl = os.path.join(results_dir, f'drw_mcmc_{source_id}.pkl')
        if os.path.exists(mcmc_pkl):
            try:
                with open(mcmc_pkl, 'rb') as f:
                    mcmc_data = pickle.load(f)
                if isinstance(mcmc_data, dict):
                    drw_mcmc = mcmc_data.get('mcmc')
                    if broken_drw is None:
                        broken_drw = mcmc_data.get('broken_drw')
                    if nonstat_gp is None:
                        nonstat_gp = mcmc_data.get('nonstationary_gp')
            except Exception:
                pass

        # Variability stats from candidate row as fallback
        if variability_stats is None:
            variability_stats = {
                'delta_mag_frac': row.get('delta_mag_frac', row.get('delta_mag_w1', np.nan)),
                'w1w2_color_change': row.get('w1w2_color_change', row.get('delta_color', np.nan)),
                'nonstationarity_sigma': row.get('nonstationarity_sigma', np.nan),
            }

        score_result = compute_composite_score_v2(
            source=row,
            lc_data=lc_data,
            drw_map=drw_map,
            drw_mcmc_results=drw_mcmc,
            broken_drw=broken_drw,
            nonstat_gp=nonstat_gp,
            sf_results=sf_results,
            cp_results=cp_results,
            variability_stats=variability_stats,
            gaia_lc=gaia_lc,
            gaia_photometry=gaia_photometry,
            physics_results=physics_results,
        )

        v2_row = {'source_id': source_id, **dict(row)}
        v2_row['composite_score_v2'] = score_result['composite_score_v2']
        v2_row['composite_score_v2_raw'] = score_result['composite_score_v2_raw']
        v2_row['composite_score_err'] = score_result['composite_score_err']
        for comp_name, comp_val in score_result['components'].items():
            v2_row[f'v2_{comp_name}'] = comp_val
        v2_rows.append(v2_row)
        logger.info("%s: v2_score=%.3f (err=%.3f)", source_id,
                    score_result['composite_score_v2'],
                    score_result['composite_score_err'] if np.isfinite(score_result['composite_score_err']) else -1)

    if not v2_rows:
        return None

    v2_df = pd.DataFrame(v2_rows)
    out_path = os.path.join(results_dir, 'top_candidates_v2.csv')
    v2_df.to_csv(out_path, index=False)
    logger.info("Saved v2 scores to %s", out_path)

    return v2_df
