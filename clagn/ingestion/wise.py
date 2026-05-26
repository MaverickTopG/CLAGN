"""
wise.py — WISE AllWISE MEPT + NEOWISE-R multi-epoch queries and cleaning.

Query strategy:
1. AllWISE Multi-Epoch Photometry Table (MEPT) — 2010-2011 (pre-hibernation)
2. NEOWISE-R Single Exposure Source Table — 2013-present
Together these give 10+ year baselines. The 2011-2013 gap is handled explicitly.

IMPORTANT column differences between AllWISE MEPT and NEOWISE-R:
  AllWISE MEPT (allwise_p3as_mep):
    - Photometry columns: w1mpro_ep, w1sigmpro_ep (with _ep suffix)
    - NO ph_qual column; NO w1snr/w2snr → derive SNR = flux/sigflux
    - moon_masked is a 4-char string (position [0] = W1)
    - qi_fact: quality factor for the frame

  NEOWISE-R (neowiser_p1bs_psd):
    - Photometry columns: w1mpro, w1sigmpro (no suffix)
    - ph_qual is 2-char string: position [0]=W1, [1]=W2
    - w1snr, w2snr explicit columns
    - moon_masked is a 4-char string (position [0] = W1)
"""
import logging
import time
import warnings
import numpy as np
import pandas as pd

from ..config import (
    WISE_ALLWISE_SEARCH_RADIUS_ARCSEC, WISE_NEOWISE_SEARCH_RADIUS_ARCSEC,
    WISE_SEARCH_RADIUS_ARCSEC,
    WISE_HIBERNATION_MJD_START, WISE_HIBERNATION_MJD_END,
    IRSA_ALLWISE_TABLE, IRSA_NEOWISE_TABLE,
    ALLWISE_MEPT_COLUMNS, NEOWISE_COLUMNS,
    MIN_BASELINE_YEARS, MIN_EPOCHS, MIN_EPOCHS_PRE_GAP, MIN_EPOCHS_POST_GAP,
    SIGMA_CLIP_SIGMA, SIGMA_CLIP_ITERS,
)
from ..utils.photometry import (
    mag_to_flux_mjy, compute_epoch_weights,
    sigma_clip_lightcurve, check_host_contamination,
)

logger = logging.getLogger(__name__)

# Minimum SNR threshold per single exposure.
# WISE single-exposure depth: W1 5σ ~ 17.5 mag. For faint/variable AGN, individual
# exposures often have SNR 2-5. We accept SNR > 3 and rely on the DRW model to
# handle heteroscedastic errors. High-quality coadds use SNR > 5, but per-epoch
# photometry for variability studies commonly uses SNR > 2-3 (Chen+2018, NEOWISE-var).
MIN_SNR = 3.0
MAX_NB  = 2

# cc_flags positions that indicate severe artifact contamination (W1 band = position 0).
# 'D' = diffraction spike, 'P' = persistence, 'O' = optical ghost.
# 'h'/'H' = halo of nearby bright star — photometry is often still usable.
# We reject only the most severe artifacts for the W1 band.
_BAD_CC_FLAGS_W1 = {'D', 'P', 'O', 'd', 'p', 'o'}


# ---------------------------------------------------------------------------
# Quality filter functions
# ---------------------------------------------------------------------------

def _apply_quality_filters_allwise(df):
    """
    Apply WISE quality filters to an AllWISE MEPT DataFrame.

    AllWISE MEPT specifics:
    - NO ph_qual column: instead filter on SNR derived from flux/sigflux
    - moon_masked is 4-char string; position [0] = W1 (use != '1' for unmasked)
    - qi_fact: frame quality factor (= 1 for good)
    - cc_flags: 4-char string; '0000' = clean
    - nb <= 2: not blended with many sources
    """
    mask = pd.Series(True, index=df.index)

    # qi_fact == 1 (good frame)
    if 'qi_fact' in df.columns:
        qi = pd.to_numeric(df['qi_fact'], errors='coerce').fillna(0)
        mask &= qi == 1

    # saa_sep > 5
    if 'saa_sep' in df.columns:
        saa = pd.to_numeric(df['saa_sep'], errors='coerce').fillna(0)
        mask &= saa > 5

    # moon_masked: string (2-char or 4-char). Position [0] = W1. '0' = not masked.
    if 'moon_masked' in df.columns:
        mm = df['moon_masked'].fillna('1').astype(str).str.strip()
        # Pad to at least 1 character; check W1 position
        mask &= mm.str[0] == '0'

    # cc_flags: reject W1 severe artifact contamination (position [0]).
    # 'D'/'d' = diffraction spike, 'P'/'p' = persistence, 'O'/'o' = optical ghost.
    # 'h'/'H' = halo of bright star — photometry often still valid; we accept it.
    # '0' = clean.
    if 'cc_flags' in df.columns:
        cc = df['cc_flags'].fillna('X').astype(str).str.strip()
        cc_w1 = cc.str[0]
        mask &= ~cc_w1.isin(_BAD_CC_FLAGS_W1)

    # SNR > MIN_SNR: derived as flux / sigflux (AllWISE MEPT has no w1snr column).
    # Require only W1 SNR (primary science band); W2 is advisory.
    if 'w1flux_ep' in df.columns and 'w1sigflux_ep' in df.columns:
        w1f = pd.to_numeric(df['w1flux_ep'], errors='coerce').fillna(0)
        w1sf = pd.to_numeric(df['w1sigflux_ep'], errors='coerce').fillna(np.inf)
        with np.errstate(divide='ignore', invalid='ignore'):
            w1snr = np.where(w1sf > 0, w1f / w1sf, 0.0)
        mask &= pd.Series(w1snr, index=df.index) > MIN_SNR

    # nb <= 2
    if 'nb' in df.columns:
        nb = pd.to_numeric(df['nb'], errors='coerce').fillna(99)
        mask &= nb <= MAX_NB

    return mask


def _apply_quality_filters_neowise(df):
    """
    Apply WISE quality filters to a NEOWISE-R DataFrame.

    NEOWISE-R specifics:
    - ph_qual is 2-char string: [0]=W1, [1]=W2. 'A' or 'B' = good.
    - moon_masked is 4-char string: [0]=W1. '0' = not masked.
    - w1snr, w2snr are explicit columns
    """
    mask = pd.Series(True, index=df.index)

    # qi_fact == 1
    if 'qi_fact' in df.columns:
        qi = pd.to_numeric(df['qi_fact'], errors='coerce').fillna(0)
        mask &= qi == 1

    # saa_sep > 5
    if 'saa_sep' in df.columns:
        saa = pd.to_numeric(df['saa_sep'], errors='coerce').fillna(0)
        mask &= saa > 5

    # moon_masked: string (2-char or 4-char). Position [0] = W1. '0' = not masked.
    if 'moon_masked' in df.columns:
        mm = df['moon_masked'].fillna('1').astype(str).str.strip()
        mask &= mm.str[0] == '0'

    # cc_flags: reject W1 severe artifact contamination (position [0]).
    # 'h'/'H' = halo — accept; 'D'/'P'/'O' = severe artifacts — reject.
    if 'cc_flags' in df.columns:
        cc = df['cc_flags'].fillna('X').astype(str).str.strip()
        cc_w1 = cc.str[0]
        mask &= ~cc_w1.isin(_BAD_CC_FLAGS_W1)

    # ph_qual: 2-char string in NEOWISE-R. [0]=W1, [1]=W2.
    # Accept A (>7σ), B (5-7σ), C (3-5σ) for W1. Exclude U (<3σ = non-detection).
    # W2 quality is advisory only — do not filter on it.
    if 'ph_qual' in df.columns:
        pq = df['ph_qual'].fillna('UU').astype(str).str.strip().str.ljust(2, 'U')
        mask &= pq.str[0].isin(['A', 'B', 'C'])

    # w1snr > MIN_SNR (primary science band). W2 SNR is advisory only.
    if 'w1snr' in df.columns:
        mask &= pd.to_numeric(df['w1snr'], errors='coerce').fillna(0) > MIN_SNR

    # nb <= 2
    if 'nb' in df.columns:
        nb = pd.to_numeric(df['nb'], errors='coerce').fillna(99)
        mask &= nb <= MAX_NB

    return mask


# ---------------------------------------------------------------------------
# IRSA TAP query with retry
# ---------------------------------------------------------------------------

def _irsa_query_with_retry(adql, max_retries=4):
    """Execute an IRSA TAP query with exponential backoff retry logic."""
    from astroquery.ipac.irsa import Irsa

    last_exc = None
    for attempt in range(max_retries):
        try:
            result = Irsa.query_tap(adql)
            return result.to_table().to_pandas()
        except Exception as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning(
                f"IRSA query attempt {attempt+1}/{max_retries} failed: {exc}. "
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)

    raise RuntimeError(f"IRSA query failed after {max_retries} attempts: {last_exc}")


def _query_allwise_mept(ra, dec, radius_arcsec=WISE_ALLWISE_SEARCH_RADIUS_ARCSEC):
    """Query AllWISE Multi-Epoch Photometry Table for a single source."""
    radius_deg = radius_arcsec / 3600.0
    adql = (
        f"SELECT {ALLWISE_MEPT_COLUMNS} "
        f"FROM {IRSA_ALLWISE_TABLE} "
        f"WHERE CONTAINS("
        f"POINT('ICRS', ra, dec),"
        f"CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius_deg:.8f})"
        f") = 1"
    )
    logger.debug(f"AllWISE MEPT query: {adql}")
    return _irsa_query_with_retry(adql)


def _query_neowise_r(ra, dec, radius_arcsec=WISE_NEOWISE_SEARCH_RADIUS_ARCSEC):
    """Query NEOWISE-R Single Exposure Source Table for a single source."""
    radius_deg = radius_arcsec / 3600.0
    adql = (
        f"SELECT {NEOWISE_COLUMNS} "
        f"FROM {IRSA_NEOWISE_TABLE} "
        f"WHERE CONTAINS("
        f"POINT('ICRS', ra, dec),"
        f"CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius_deg:.8f})"
        f") = 1"
    )
    logger.debug(f"NEOWISE-R query: {adql}")
    return _irsa_query_with_retry(adql)


# ---------------------------------------------------------------------------
# WISE gap handling
# ---------------------------------------------------------------------------

def flag_wise_gap(mjd_array):
    """
    Flag epochs that fall within the WISE hibernation period
    (2011-02-17 to 2012-08-08, MJD 55593–56141).

    Real data should have NO epochs in this window.
    """
    mjd_array = np.asarray(mjd_array)
    gap_mask = (
        (mjd_array > WISE_HIBERNATION_MJD_START) &
        (mjd_array < WISE_HIBERNATION_MJD_END)
    )
    if gap_mask.any():
        warnings.warn(
            f"Found {gap_mask.sum()} epochs in WISE hibernation window — removing."
        )
    return mjd_array[~gap_mask], ~gap_mask


# ---------------------------------------------------------------------------
# Baseline validation
# ---------------------------------------------------------------------------

def validate_baseline(mjd_array, source_id):
    """
    Compute and validate the temporal baseline of the light curve.
    Returns dict with baseline_years, passes_baseline, rejection_reason, epoch counts.
    """
    mjd_array = np.asarray(mjd_array)
    n_total = len(mjd_array)

    if n_total == 0:
        return {
            'baseline_years': 0.0,
            'passes_baseline': False,
            'rejection_reason': 'No usable epochs after quality filtering',
            'n_epochs_pre_gap': 0,
            'n_epochs_post_gap': 0,
            'n_epochs_total': 0,
        }

    baseline_years = (mjd_array.max() - mjd_array.min()) / 365.25
    n_pre_gap  = int(np.sum(mjd_array <= WISE_HIBERNATION_MJD_START))
    n_post_gap = int(np.sum(mjd_array >= WISE_HIBERNATION_MJD_END))

    # NEOWISE-R spans 2013–present (~11 yr). For sources lacking AllWISE MEPT
    # coverage (saturation, extended morphology, or catalog gaps), we allow the
    # source to proceed if NEOWISE-R alone provides ≥10 yr baseline and ≥20 epochs.
    # The pre-gap constraint is relaxed in this case with a logged warning.
    neowise_only_baseline = (
        n_pre_gap == 0 and
        n_post_gap >= MIN_EPOCHS and
        baseline_years >= MIN_BASELINE_YEARS
    )

    if n_total < MIN_EPOCHS:
        reason = f"Only {n_total} epochs (need {MIN_EPOCHS})"
    elif baseline_years < MIN_BASELINE_YEARS:
        reason = f"Baseline {baseline_years:.1f} yr < {MIN_BASELINE_YEARS} yr"
    elif n_pre_gap < MIN_EPOCHS_PRE_GAP and not neowise_only_baseline:
        reason = (
            f"Only {n_pre_gap} pre-gap epochs (need {MIN_EPOCHS_PRE_GAP}); "
            f"AllWISE data missing or insufficient"
        )
    elif n_post_gap < MIN_EPOCHS_POST_GAP:
        reason = (
            f"Only {n_post_gap} post-gap epochs (need {MIN_EPOCHS_POST_GAP}); "
            f"NEOWISE-R data missing or insufficient"
        )
    else:
        reason = None

    return {
        'baseline_years': float(baseline_years),
        'passes_baseline': reason is None,
        'rejection_reason': reason,
        'n_epochs_pre_gap': n_pre_gap,
        'n_epochs_post_gap': n_post_gap,
        'n_epochs_total': n_total,
    }


# ---------------------------------------------------------------------------
# Main query and cleaning function
# ---------------------------------------------------------------------------

def query_wise_lightcurve(ra, dec, source_id, z=0.0):
    """
    Query both AllWISE-MEPT and NEOWISE-R for a single source.
    Returns merged, cleaned, calibrated light curve as DataFrame.

    Returned DataFrame columns:
        mjd, w1_mag, w1_err, w2_mag, w2_err,
        w1_flux_mjy, w1_flux_err_mjy, w2_flux_mjy, w2_flux_err_mjy,
        w1snr, w2snr, w1_minus_w2, epoch_weight, dataset_flag

    Parameters
    ----------
    ra, dec    : float, ICRS degrees
    source_id  : str
    z          : float, source redshift

    Returns
    -------
    result : dict
    """
    _empty = {
        'lc': pd.DataFrame(),
        'passes_baseline': False,
        'baseline_years': 0.0,
        'n_epochs_total': 0,
        'n_epochs_pre_gap': 0,
        'n_epochs_post_gap': 0,
        'w1_minus_w2_early': np.nan,
        'w1_minus_w2_late': np.nan,
        'w1_minus_w2_allwise': np.nan,
        'host_contamination_risk': check_host_contamination(z),
    }

    # ---- Query AllWISE MEPT -------------------------------------------------
    try:
        aw_raw = _query_allwise_mept(ra, dec)
        logger.debug(f"{source_id}: AllWISE MEPT returned {len(aw_raw)} raw rows")
    except Exception as exc:
        logger.error(f"{source_id}: AllWISE MEPT query failed: {exc}")
        aw_raw = pd.DataFrame()

    # ---- Query NEOWISE-R ----------------------------------------------------
    try:
        neo_raw = _query_neowise_r(ra, dec)
        logger.debug(f"{source_id}: NEOWISE-R returned {len(neo_raw)} raw rows")
    except Exception as exc:
        logger.error(f"{source_id}: NEOWISE-R query failed: {exc}")
        neo_raw = pd.DataFrame()

    # ---- Clean and normalize AllWISE MEPT -----------------------------------
    frames = []

    if len(aw_raw) > 0:
        aw_mask = _apply_quality_filters_allwise(aw_raw)
        aw_clean = aw_raw[aw_mask].copy()

        # Rename _ep columns to standard names
        aw_clean = aw_clean.rename(columns={
            'w1mpro_ep':    'w1_mag',
            'w1sigmpro_ep': 'w1_err',
            'w2mpro_ep':    'w2_mag',
            'w2sigmpro_ep': 'w2_err',
        })

        # Compute SNR from flux / sigflux (no native SNR column in AllWISE MEPT)
        if 'w1flux_ep' in aw_clean.columns and 'w1sigflux_ep' in aw_clean.columns:
            w1f  = pd.to_numeric(aw_clean['w1flux_ep'], errors='coerce').fillna(0)
            w1sf = pd.to_numeric(aw_clean['w1sigflux_ep'], errors='coerce').replace(0, np.nan)
            aw_clean['w1snr'] = (w1f / w1sf).fillna(0)
        else:
            aw_clean['w1snr'] = 5.0

        if 'w2flux_ep' in aw_clean.columns and 'w2sigflux_ep' in aw_clean.columns:
            w2f  = pd.to_numeric(aw_clean['w2flux_ep'], errors='coerce').fillna(0)
            w2sf = pd.to_numeric(aw_clean['w2sigflux_ep'], errors='coerce').replace(0, np.nan)
            aw_clean['w2snr'] = (w2f / w2sf).fillna(0)
        else:
            aw_clean['w2snr'] = 5.0

        aw_clean['dataset_flag'] = 'allwise'
        logger.debug(f"{source_id}: AllWISE after QA: {len(aw_clean)} epochs")

        common_aw = ['mjd', 'w1_mag', 'w1_err', 'w2_mag', 'w2_err',
                     'w1snr', 'w2snr', 'saa_sep', 'dataset_flag']
        available = [c for c in common_aw if c in aw_clean.columns]
        frames.append(aw_clean[available].copy())

    # ---- Clean and normalize NEOWISE-R --------------------------------------
    if len(neo_raw) > 0:
        neo_mask = _apply_quality_filters_neowise(neo_raw)
        neo_clean = neo_raw[neo_mask].copy()

        neo_clean = neo_clean.rename(columns={
            'w1mpro':    'w1_mag',
            'w1sigmpro': 'w1_err',
            'w2mpro':    'w2_mag',
            'w2sigmpro': 'w2_err',
        })
        neo_clean['dataset_flag'] = 'neowise'
        logger.debug(f"{source_id}: NEOWISE-R after QA: {len(neo_clean)} epochs")

        common_neo = ['mjd', 'w1_mag', 'w1_err', 'w2_mag', 'w2_err',
                      'w1snr', 'w2snr', 'saa_sep', 'dataset_flag']
        available = [c for c in common_neo if c in neo_clean.columns]
        frames.append(neo_clean[available].copy())

    if not frames:
        return {**_empty,
                'rejection_reason': 'No data returned from IRSA queries or failed all QA filters'}

    # ---- Merge and sort by MJD ----------------------------------------------
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values('mjd').reset_index(drop=True)

    # Ensure numeric types
    for col in ['mjd', 'w1_mag', 'w1_err', 'w2_mag', 'w2_err']:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors='coerce')

    # Drop rows with NaN in essential columns
    combined = combined.dropna(subset=['mjd', 'w1_mag', 'w1_err']).copy()

    # ---- Remove hibernation gap epochs (sanity check) -----------------------
    _, keep_mask = flag_wise_gap(combined['mjd'].values)
    combined = combined[keep_mask].reset_index(drop=True)

    # ---- AllWISE-only color for AGN pre-selection ---------------------------
    aw_sub = combined[combined['dataset_flag'] == 'allwise'].copy()
    if len(aw_sub) >= 3:
        w1_aw = pd.to_numeric(aw_sub['w1_mag'], errors='coerce')
        w2_aw = pd.to_numeric(aw_sub['w2_mag'], errors='coerce')
        w1_minus_w2_allwise = float(np.nanmedian(w1_aw.values - w2_aw.values))
    else:
        w1_minus_w2_allwise = np.nan

    # ---- Convert magnitudes → flux (mJy) ------------------------------------
    w1_mag_arr = combined['w1_mag'].values
    w1_err_arr = combined['w1_err'].values
    w2_mag_arr = combined['w2_mag'].values if 'w2_mag' in combined.columns else w1_mag_arr
    w2_err_arr = combined['w2_err'].values if 'w2_err' in combined.columns else w1_err_arr

    w1_flux, w1_flux_err = mag_to_flux_mjy(w1_mag_arr, w1_err_arr, 'W1')
    w2_flux, w2_flux_err = mag_to_flux_mjy(w2_mag_arr, w2_err_arr, 'W2')

    combined['w1_flux_mjy']     = w1_flux
    combined['w1_flux_err_mjy'] = w1_flux_err
    combined['w2_flux_mjy']     = w2_flux
    combined['w2_flux_err_mjy'] = w2_flux_err

    # ---- W1-W2 color --------------------------------------------------------
    combined['w1_minus_w2'] = combined['w1_mag'] - combined['w2_mag']

    # ---- Epoch weights ------------------------------------------------------
    snr_w1   = combined['w1snr'].fillna(5).values if 'w1snr' in combined.columns else np.full(len(combined), 5.0)
    snr_w2   = combined['w2snr'].fillna(5).values if 'w2snr' in combined.columns else np.full(len(combined), 5.0)
    moon_sep = combined['saa_sep'].fillna(45).values if 'saa_sep' in combined.columns else np.full(len(combined), 45.0)
    combined['epoch_weight'] = compute_epoch_weights(snr_w1, snr_w2, moon_sep)

    # ---- Remove NaN fluxes before sigma-clipping ----------------------------
    valid_flux = np.isfinite(combined['w1_flux_mjy'].values)
    combined = combined[valid_flux].reset_index(drop=True)

    if len(combined) < 5:
        return {**_empty,
                'rejection_reason': f'Only {len(combined)} valid W1 flux measurements',
                'w1_minus_w2_allwise': w1_minus_w2_allwise,
                'n_epochs_total': len(combined)}

    # ---- 4σ sigma-clipping on W1 flux (3 iters) ----------------------------
    _, _, _, keep = sigma_clip_lightcurve(
        combined['mjd'].values,
        combined['w1_flux_mjy'].values,
        combined['w1_flux_err_mjy'].values,
        sigma=SIGMA_CLIP_SIGMA,
        maxiters=SIGMA_CLIP_ITERS,
    )
    combined = combined[keep].reset_index(drop=True)

    # ---- Baseline validation ------------------------------------------------
    baseline_info = validate_baseline(combined['mjd'].values, source_id)

    # ---- Color: early vs late 20% -------------------------------------------
    mjd_vals   = combined['mjd'].values
    color_vals = combined['w1_minus_w2'].values
    n = len(mjd_vals)
    idx_sort = np.argsort(mjd_vals)
    n_fifth   = max(3, n // 5)

    early_idx = idx_sort[:n_fifth]
    late_idx  = idx_sort[-n_fifth:]
    w1w2_early = float(np.nanmedian(color_vals[early_idx]))
    w1w2_late  = float(np.nanmedian(color_vals[late_idx]))

    logger.info(
        f"{source_id}: {len(combined)} clean epochs | "
        f"baseline={baseline_info['baseline_years']:.1f} yr | "
        f"pre-gap={baseline_info['n_epochs_pre_gap']} | "
        f"post-gap={baseline_info['n_epochs_post_gap']}"
    )

    return {
        'lc': combined,
        'passes_baseline': baseline_info['passes_baseline'],
        'baseline_years': baseline_info['baseline_years'],
        'n_epochs_total': baseline_info['n_epochs_total'],
        'n_epochs_pre_gap': baseline_info['n_epochs_pre_gap'],
        'n_epochs_post_gap': baseline_info['n_epochs_post_gap'],
        'w1_minus_w2_early': w1w2_early,
        'w1_minus_w2_late': w1w2_late,
        'w1_minus_w2_allwise': w1_minus_w2_allwise,
        'host_contamination_risk': check_host_contamination(z),
        'rejection_reason': baseline_info['rejection_reason'],
    }
