"""
clagn_score.py — 7-component composite CLAGN scoring engine.

Each component is normalized to [0, 1] before weighting.
Final score is a weighted sum divided by MAX_SCORE (= 14.0).

NaN component values contribute 0.0 — they don't crash the score.

Scientific basis: Ricci & Trakhtenbrot 2022 (arXiv:2211.05132)
"""
import logging
import numpy as np

from ..config import (
    SCORE_WEIGHTS, MAX_SCORE,
    FP_SN_BREAK_DURATION_DAYS, FP_GALACTIC_LAT_THRESHOLD, FP_LOW_QUASAR_PROB,
    FP_SN_SCORE_PENALTY, FP_STELLAR_SCORE_PENALTY,
    FP_HOST_SCORE_PENALTY, FP_ARTIFACT_SCORE_PENALTY,
    SIGMA_EXCESS_NORM_FLUX_MJY, SIGMA_EXCESS_LUM_SLOPE,
    MIN_MAG_CHANGE_W1,
)
from ..utils.crossmatch import galactic_latitude

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


def _norm_delta_mag(frac_change):
    """
    Fractional flux change: |flux_early - flux_late| / flux_mean.
    A factor-2 change → frac_change ~ 1.0. Normalized by dividing by 2.
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

def _compute_sigma_excess(sigma_drw_obs, w1_flux_mean_mjy):
    """
    Compute DRW amplitude excess vs the luminosity-variability anti-correlation.

    AGN variability anti-correlates with luminosity (Vanden Berk+2004).
    sigma_expected ∝ L^(-0.5)

    sigma_excess = sigma_drw_obs / sigma_expected

    Parameters
    ----------
    sigma_drw_obs : float, observed DRW amplitude (mJy)
    w1_flux_mean_mjy : float, mean W1 flux (mJy)

    Returns
    -------
    sigma_excess : float (> 1 = more variable than expected)
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
# Delta-magnitude computation (W1)
# ---------------------------------------------------------------------------

def _compute_delta_mag_frac(wise_result):
    """
    Compute the fractional flux change between early and late epochs.

    Uses rolling 1-year medians to suppress short-term flares.
    Computes: |flux_late - flux_early| / flux_mean

    Also returns delta_mag_w1 and delta_mag_w2 in magnitudes.
    """
    lc = wise_result.get('lc')
    if lc is None or len(lc) < 6:
        return 0.0, 0.0, 0.0

    mjd = lc['mjd'].values
    w1_flux = lc['w1_flux_mjy'].values if 'w1_flux_mjy' in lc.columns else None
    w1_mag  = lc['w1_mag'].values if 'w1_mag' in lc.columns else None
    w2_mag  = lc['w2_mag'].values if 'w2_mag' in lc.columns else None

    if w1_flux is None or not np.any(np.isfinite(w1_flux)):
        return 0.0, 0.0, 0.0

    # Split into early and late 33% of timeline
    n = len(mjd)
    n_third = max(3, n // 3)
    idx_sort = np.argsort(mjd)

    early_idx = idx_sort[:n_third]
    late_idx  = idx_sort[-n_third:]

    flux_early = float(np.nanmedian(w1_flux[early_idx]))
    flux_late  = float(np.nanmedian(w1_flux[late_idx]))
    flux_mean  = float(np.nanmean(w1_flux[np.isfinite(w1_flux)]))

    frac_change = (abs(flux_late - flux_early) / flux_mean
                   if flux_mean > 0 else 0.0)

    # Delta magnitudes
    if w1_mag is not None:
        delta_mag_w1 = float(abs(np.nanmedian(w1_mag[early_idx]) -
                                  np.nanmedian(w1_mag[late_idx])))
    else:
        delta_mag_w1 = 0.0

    if w2_mag is not None:
        delta_mag_w2 = float(abs(np.nanmedian(w2_mag[early_idx]) -
                                  np.nanmedian(w2_mag[late_idx])))
    else:
        delta_mag_w2 = 0.0

    return frac_change, delta_mag_w1, delta_mag_w2


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def compute_composite_clagn_score(source_record, wise_result, drw_results,
                                    sf_results, cp_results, gaia_data):
    """
    7-component composite CLAGN scoring engine.

    Each component is normalized to [0, 1] before weighting.
    Final score = weighted sum / MAX_SCORE ∈ [0, 1].

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
    result : dict with all 7 component scores + composite + label
    """
    # ---- Component 1: DRW Nonstationarity (weight 3.0) ---------------------
    nonstat_sigma = _safe(drw_results.get('nonstationarity_sigma'))
    score_drw_nonstat = _norm_drw_nonstat(nonstat_sigma)

    # ---- Component 2: Delta Magnitude W1 (weight 2.5) ----------------------
    frac_change, delta_mag_w1, delta_mag_w2 = _compute_delta_mag_frac(wise_result)
    score_delta_mag = _norm_delta_mag(frac_change)

    # ---- Component 3: Bayesian Changepoint BIC (weight 2.5) ----------------
    delta_bic = _safe(cp_results.get('delta_bic'), default=0.0)
    score_changepoint = _norm_changepoint_bic(delta_bic)

    # ---- Component 4: Structure Function Break (weight 2.0) ----------------
    sf_excess_ratio = _safe(sf_results.get('sf_excess_ratio'), default=1.0)
    score_sf_break = _norm_sf_break(sf_excess_ratio)

    # ---- Component 5: W1-W2 Color Evolution (weight 1.5) -------------------
    w1w2_early = _safe(wise_result.get('w1_minus_w2_early'), default=np.nan)
    w1w2_late  = _safe(wise_result.get('w1_minus_w2_late'),  default=np.nan)
    if np.isfinite(w1w2_early) and np.isfinite(w1w2_late):
        delta_color = w1w2_late - w1w2_early
    else:
        delta_color = 0.0
    score_color = _norm_color_evolution(delta_color)

    # ---- Component 6: DRW Sigma Excess vs Luminosity (weight 1.5) ----------
    sigma_drw_obs  = _safe(drw_results.get('sigma_drw'), default=0.0)
    lc = wise_result.get('lc')
    w1_flux_mean = (float(np.nanmean(lc['w1_flux_mjy'].values))
                    if lc is not None and 'w1_flux_mjy' in lc.columns
                    else 1.0)
    sigma_excess = _compute_sigma_excess(sigma_drw_obs, w1_flux_mean)
    score_sigma_excess = _norm_sigma_excess(sigma_excess)

    # ---- Component 7: GAIA Optical Variability (weight 1.0) ----------------
    gaia_var_flag = gaia_data.get('phot_variable_flag', 'NOT_AVAILABLE')
    gaia_var_score = gaia_data.get('gaia_variability_score', 0.0)
    score_gaia = _norm_gaia_var(gaia_var_flag)

    # ---- Weighted sum -------------------------------------------------------
    component_scores = [
        score_drw_nonstat,
        score_delta_mag,
        score_changepoint,
        score_sf_break,
        score_color,
        score_sigma_excess,
        score_gaia,
    ]
    weights = list(SCORE_WEIGHTS.values())

    composite = sum(w * s for w, s in zip(weights, component_scores)) / MAX_SCORE

    # ---- Physical interpretation --------------------------------------------
    label = classify_clagn_type(
        {
            'score_drw_nonstat': score_drw_nonstat,
            'score_delta_mag': score_delta_mag,
            'score_changepoint': score_changepoint,
            'delta_mag_w1': delta_mag_w1,
        },
        drw_results, cp_results, wise_result
    )

    # ---- W1 flux ratio (max/min rolling 1-year medians) --------------------
    w1_flux_ratio = _compute_flux_ratio(wise_result)

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
        'delta_mag_w1':         delta_mag_w1,
        'delta_mag_w2':         delta_mag_w2,
        'delta_color':          delta_color,
        'sigma_excess':         sigma_excess,
        'frac_change':          frac_change,
        'w1_flux_ratio':        w1_flux_ratio,
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

    Classification decision tree:
    1. Rapid Transition (possible TDE-in-AGN or magnetic disk event)
    2. CS-AGN Turn-Off (fading accretion)
    3. CS-AGN Turn-On (rising accretion)
    4. CO-AGN candidate (cloud eclipse)
    5. Unknown/Ambiguous (spectroscopic followup required)

    Parameters
    ----------
    scores : dict with delta_mag_w1 and component scores
    drw_results, cp_results, wise_result : pipeline result dicts

    Returns
    -------
    label : str
    """
    delta_mag  = _safe(scores.get('delta_mag_w1', 0.0))
    pre_mean   = _safe(cp_results.get('pre_break_mean', 0.0))
    post_mean  = _safe(cp_results.get('post_break_mean', 0.0))
    break_dur  = _safe(cp_results.get('break_duration_days', 365.0), default=365.0)
    tau        = _safe(drw_results.get('tau_rest_days', 300.0), default=300.0)

    w1w2_early = _safe(wise_result.get('w1_minus_w2_early'), default=np.nan)
    w1w2_late  = _safe(wise_result.get('w1_minus_w2_late'),  default=np.nan)
    color_change = np.nan
    if np.isfinite(w1w2_early) and np.isfinite(w1w2_late):
        color_change = w1w2_late - w1w2_early

    # Rapid Transition: short break duration AND large flux change
    if delta_mag > 0.75 and np.isfinite(break_dur) and break_dur < 2 * 365.25:
        return (
            "Rapid transition — possible TDE-in-AGN or magnetic disk event "
            "(Ricci+2022 §3.6.2)"
        )

    # Turn-Off: flux declining AND color becoming bluer
    if (delta_mag > 0.5 and pre_mean > 0 and post_mean < pre_mean and
            np.isfinite(color_change) and color_change < -0.05):
        return (
            "CS-AGN Turn-Off (fading accretion, BLR likely disappearing)"
        )

    # Turn-On: flux rising AND color becoming redder
    if (delta_mag > 0.5 and post_mean > pre_mean and
            np.isfinite(color_change) and color_change > 0.05):
        return (
            "CS-AGN Turn-On (rising accretion, BLR likely emerging)"
        )

    # General strong turn-off without confirmed color change
    if delta_mag > 0.5 and pre_mean > 0 and post_mean < pre_mean:
        return "CS-AGN Turn-Off candidate (spectroscopic followup required)"

    # General strong turn-on without confirmed color change
    if delta_mag > 0.5 and post_mean > pre_mean:
        return "CS-AGN Turn-On candidate (spectroscopic followup required)"

    # CO-AGN: short DRW timescale, moderate amplitude, no sustained trend
    if tau < 100.0 and 0.1 < delta_mag < 0.4:
        return (
            "CO-AGN candidate — possible BLR cloud eclipse "
            "(short τ_DRW ~ BLR cloud crossing timescale)"
        )

    # Marginal candidates
    if 0.3 <= delta_mag <= 0.5:
        return "Marginal CLAGN candidate — type ambiguous, spectroscopic followup required"

    return "CLAGN candidate — type ambiguous, spectroscopic followup required"


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
        host_risk = bool(row.get('host_contamination_risk', False))
        delta_mag  = _safe(row.get('w1_delta_mag'), default=0.0)
        if host_risk and delta_mag < 0.5:
            flags.append(
                'host_contamination_risk(z<0.05,marginal_delta_mag)'
            )
            penalty *= FP_HOST_SCORE_PENALTY

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


def _check_wise_artifact(lc):
    """
    Check if variability is driven by a single extreme WISE epoch.

    Method: Find the single most deviant epoch (largest |flux - median|).
    Remove it and recompute delta_flux. If delta_flux drops by > 50%,
    the variability is artifact-driven.

    Returns True if artifact suspected.
    """
    fluxes = lc['w1_flux_mjy'].values
    valid = np.isfinite(fluxes)
    if valid.sum() < 10:
        return False

    f = fluxes[valid]
    median_flux = np.nanmedian(f)
    deviations  = np.abs(f - median_flux)
    delta_flux_full = float(f.max() - f.min())

    if delta_flux_full <= 0:
        return False

    # Remove the single most deviant epoch
    worst_idx = np.argmax(deviations)
    f_reduced = np.delete(f, worst_idx)
    delta_flux_reduced = float(f_reduced.max() - f_reduced.min())

    reduction_fraction = 1.0 - (delta_flux_reduced / delta_flux_full)
    return reduction_fraction > 0.50
