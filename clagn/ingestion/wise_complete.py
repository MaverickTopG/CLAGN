"""
wise_complete.py — Full multi-dataset WISE ingestion covering all 5 survey phases.

Survey phases:
    AllSky    : 2010-01-14 – 2010-08-06 (4-band, fully cryogenic)
    3Band     : 2010-08-06 – 2010-09-29 (W1/W2/W3, partial cryogen)
    PostCryo  : 2010-09-29 – 2011-02-01 (W1/W2 only, no cryogen)
    [HIBERNATION 2011-02-17 – 2013-12-13, MJD 55593-56987]  (wise-H6 corrected)
    NEOWISE-R : 2013-12-13 – present   (W1/W2 only)
    AllWISE   : legacy catalog (handled by .wise module)

References:
    Wright et al. 2010, AJ 140, 1868 (WISE mission)
    Mainzer et al. 2014, ApJ 792, 30 (NEOWISE reactivation)
"""
import logging
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (
    WISE_ALLSKY_TABLE,
    WISE_3BAND_TABLE,
    WISE_POSTCRYO_TABLE,
    WISE_FULL_BASELINE_START_MJD,
    WISE_HIBERNATION_MJD_START,
    WISE_HIBERNATION_MJD_END,
    WISE_HIBERNATION_START_MJD,
    WISE_HIBERNATION_END_MJD,
    WISE_ZERO_POINTS,
    WISE_NEOWISE_SEARCH_RADIUS_ARCSEC,
    WISE_SEASON_ANCHOR_MJD,
    WISE_TABLE_REQUIRED_COLS,
    SIGMA_CLIP_SIGMA,
    SIGMA_CLIP_ITERS,
    flag_systematic_epochs,
    check_wise_saturation,
)
from ..models.variability import cluster_epochs_into_seasons
from .wise import query_wise_lightcurve

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_EARLY_COLUMNS = (
    "ra, dec, mjd, w1mpro, w1sigmpro, w2mpro, w2sigmpro, "
    "w1flux, w1sigflux, w2flux, w2sigflux, "
    "cc_flags, qi_fact, saa_sep, moon_masked, nb"
)

_BAD_CC_W1 = {'D', 'P', 'O', 'd', 'p', 'o'}
_MIN_SNR = 3.0
_MAX_NB = 2
_DEDUP_WINDOW_DAYS = 0.01   # 15 minutes in days

# W3 band search uses wider radius (larger PSF)
_W3_SEARCH_RADIUS_ARCSEC = 12.0


# ---------------------------------------------------------------------------
# Internal IRSA query helpers
# ---------------------------------------------------------------------------

def _irsa_tap_query(adql: str, max_retries: int = 3) -> pd.DataFrame:
    """Execute an IRSA TAP ADQL query with exponential backoff retry."""
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
                f"IRSA TAP attempt {attempt + 1}/{max_retries} failed: {exc}. "
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)

    raise RuntimeError(
        f"IRSA TAP query failed after {max_retries} attempts: {last_exc}"
    )


def irsa_query_with_retry(catalog, coords, radius, columns=None,
                           max_retries=5, base_sleep=1.5):
    """
    Wrapper around Irsa.query_region() with exponential backoff retry,
    rate-limit handling, and graceful None return on persistent failure.

    NOTE: This function uses the cone-search API (Irsa.query_region).
    The primary pipeline uses TAP queries via _irsa_tap_query().
    Retained as fallback if TAP endpoint is unavailable.

    Parameters
    ----------
    catalog : str
        IRSA catalog name e.g. 'neowiser_p1bs_psd'
    coords : SkyCoord
        Target coordinates
    radius : astropy Quantity
        Search radius e.g. 6*u.arcsec
    columns : str or None
        Comma-separated column list, or None for all columns
    max_retries : int
        Maximum attempts before giving up (default 5)
    base_sleep : float
        Base sleep time in seconds; doubles on each retry (default 1.0)

    Returns
    -------
    astropy Table or None
        Query result, or None if all retries exhausted or source not found
    """
    from astroquery.ipac.irsa import Irsa
    try:
        from astroquery.exceptions import TimeoutError as AQTimeoutError
    except ImportError:
        AQTimeoutError = Exception  # fallback

    kwargs = dict(catalog=catalog, spatial='Cone',
                  radius=radius, coordinates=coords)
    if columns:
        kwargs['columns'] = columns

    for attempt in range(max_retries):
        try:
            sleep_time = base_sleep * (2 ** attempt) if attempt > 0 else 0.5
            time.sleep(sleep_time)

            result = Irsa.query_region(**kwargs)
            return result

        except AQTimeoutError:
            logger.warning(f"IRSA timeout on attempt {attempt + 1}/{max_retries}")
        except Exception as e:
            err_str = str(e).lower()
            if 'no sources' in err_str or 'empty' in err_str or 'no rows' in err_str:
                return None   # Source genuinely not in catalog — not an error
            if 'rate' in err_str or '429' in err_str or 'too many' in err_str:
                logger.warning(f"IRSA rate limited on attempt {attempt + 1}/{max_retries}")
            else:
                logger.warning(f"IRSA query error attempt {attempt + 1}/{max_retries}: {e}")

    logger.error(f"IRSA query failed after {max_retries} retries: catalog={catalog}")
    return None


def _query_wise_table_safe(catalog_name: str, ra: float, dec: float,
                            radius_arcsec: float, primary_columns: str,
                            fallback_columns: str = None):
    """
    Schema-safe early WISE table query (Fix #11).

    Tries full column list first; on column-not-found failure retries with
    minimal fallback columns. Returns (result_df, schema_used) where
    schema_used is one of 'full_schema', 'fallback_schema', 'failed'.

    Parameters
    ----------
    catalog_name : str
        IRSA TAP table name
    ra, dec : float
        ICRS degrees
    radius_arcsec : float
        Cone search radius in arcseconds
    primary_columns : str
        Comma-separated primary column list
    fallback_columns : str or None
        Minimal fallback column list; if None uses 'ra,dec,mjd,w1mpro,w1sigmpro'

    Returns
    -------
    df : pd.DataFrame
    schema_used : str
    """
    if fallback_columns is None:
        fallback_columns = "ra, dec, mjd, w1mpro, w1sigmpro, w2mpro, w2sigmpro"

    adql_full = _build_cone_adql(catalog_name, primary_columns, ra, dec, radius_arcsec)
    try:
        df = _irsa_tap_query(adql_full)
        return df, 'full_schema'
    except Exception as exc:
        err_msg = str(exc).lower()
        if 'column' in err_msg or 'field' in err_msg or 'unknown' in err_msg:
            logger.warning(
                f"Schema mismatch for {catalog_name}, retrying with fallback columns: {exc}"
            )
            try:
                adql_fallback = _build_cone_adql(
                    catalog_name, fallback_columns, ra, dec, radius_arcsec
                )
                df = _irsa_tap_query(adql_fallback)
                return df, 'fallback_schema'
            except Exception as exc2:
                logger.error(f"Fallback query also failed for {catalog_name}: {exc2}")
                return pd.DataFrame(), 'failed'
        raise


def _build_cone_adql(table: str, columns: str, ra: float, dec: float,
                     radius_arcsec: float) -> str:
    """Build a standard cone-search ADQL query string."""
    radius_deg = radius_arcsec / 3600.0
    return (
        f"SELECT {columns} "
        f"FROM {table} "
        f"WHERE CONTAINS("
        f"POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius_deg:.8f})"
        f") = 1"
    )


# ---------------------------------------------------------------------------
# Quality filtering
# ---------------------------------------------------------------------------

def _quality_filter_early(df: pd.DataFrame, dataset: str) -> pd.Series:
    """
    Apply quality filters to AllSky / 3Band / PostCryo DataFrames.

    These tables share a common schema with w1mpro (no _ep suffix) and
    have a ph_qual-like column for some versions; we filter on SNR when
    ph_qual is absent.
    """
    mask = pd.Series(True, index=df.index)

    # qi_fact == 1 (good scan frame)
    if 'qi_fact' in df.columns:
        qi = pd.to_numeric(df['qi_fact'], errors='coerce').fillna(0)
        mask &= qi == 1

    # saa_sep > 5 degrees (not in South Atlantic Anomaly)
    if 'saa_sep' in df.columns:
        saa = pd.to_numeric(df['saa_sep'], errors='coerce').fillna(0)
        mask &= saa > 5

    # moon_masked: position [0] == '0' means W1 is not moon-masked
    if 'moon_masked' in df.columns:
        mm = df['moon_masked'].fillna('1').astype(str).str.strip()
        mask &= mm.str[0] == '0'

    # cc_flags: reject severe W1 artifacts
    if 'cc_flags' in df.columns:
        cc = df['cc_flags'].fillna('X').astype(str).str.strip()
        mask &= ~cc.str[0].isin(_BAD_CC_W1)

    # ph_qual if present (A or B in W1)
    if 'ph_qual' in df.columns:
        pq = df['ph_qual'].fillna('UU').astype(str).str.strip().str.ljust(2, 'U')
        mask &= pq.str[0].isin(['A', 'B'])

    # Derive SNR from flux / sigflux when ph_qual is absent
    if 'ph_qual' not in df.columns:
        if 'w1flux' in df.columns and 'w1sigflux' in df.columns:
            w1f = pd.to_numeric(df['w1flux'], errors='coerce').fillna(0)
            w1sf = pd.to_numeric(df['w1sigflux'], errors='coerce').replace(0, np.nan)
            with np.errstate(divide='ignore', invalid='ignore'):
                snr = (w1f / w1sf).fillna(0)
            mask &= snr > _MIN_SNR

    # nb <= 2 (not blended with many neighbours)
    if 'nb' in df.columns:
        nb = pd.to_numeric(df['nb'], errors='coerce').fillna(99)
        mask &= nb <= _MAX_NB

    return mask


# ---------------------------------------------------------------------------
# wise-C4: Schema-safe moon_masked parser + table schema validator
# ---------------------------------------------------------------------------

def _parse_moon_masked(series):
    """
    Parse WISE moon_masked column safely across table schema variants.

    Known formats across WISE releases:
    - Integer 0/1: 0=not masked, 1=masked (applies to all bands)
    - String '0000': per-band, character per band W1W2W3W4
    - String '0': single character, applies to all bands
    - NaN/None: treat as masked (conservative)

    Returns dict of bool arrays: w1_moon, w2_moon, parse_format
    """
    n = len(series)
    w1_moon = np.zeros(n, dtype=bool)
    w2_moon = np.zeros(n, dtype=bool)

    # Fill NaN conservatively (unknown = assume masked)
    filled = series.fillna('1')

    try:
        # Try numeric first
        numeric = pd.to_numeric(filled, errors='coerce')
        if numeric.notna().all():
            # Pure integer format — applies to all bands
            masked = numeric.astype(int) != 0
            return {'w1_moon': masked.values, 'w2_moon': masked.values,
                    'parse_format': 'integer'}
    except Exception:
        pass

    # String format
    s = filled.astype(str).str.strip()

    if s.str.len().max() >= 2:
        # Per-band string encoding
        w1_moon = (s.str[0] != '0').values
        w2_moon = (s.str[1] != '0').values
        fmt = 'per_band_string'
    else:
        # Single character — applies to all bands
        masked = (s.str[0] != '0').values
        w1_moon = masked
        w2_moon = masked
        fmt = 'single_char_string'

    return {'w1_moon': w1_moon, 'w2_moon': w2_moon, 'parse_format': fmt}


def _validate_table_schema(df, table_name, required_cols):
    """
    Hard-fail if required columns are missing — never silently proceed.

    Parameters
    ----------
    df : DataFrame
    table_name : str
    required_cols : list of str

    Returns
    -------
    True if all required columns are present.

    Raises
    ------
    ValueError if any required columns are missing.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Table '{table_name}' missing required columns: {missing}. "
            f"Available: {list(df.columns)}. "
            f"Check IRSA table schema — it may have changed."
        )
    return True


# ---------------------------------------------------------------------------
# wise-C3: Per-band quality flag parsing
# ---------------------------------------------------------------------------

def parse_wise_quality_flags(df):
    """
    Parse WISE cc_flags, qi_fact, moon_masked into per-band boolean masks.

    cc_flags encoding (WISE Explanatory Supplement, Cutri+2012):
        Character 0: W1 artifact contamination
        Character 1: W2 artifact contamination
        Character 2: W3 artifact contamination
        Character 3: W4 artifact contamination
        '0' = no artifact, other = contaminated

    Previously only W1 (char[0]) was checked — W2 artifacts were silently
    passed through, producing fake W1-W2 color evolution (wise-C3).

    Returns df with added columns:
        w1_quality_ok : bool — W1 epoch passes quality cuts
        w2_quality_ok : bool — W2 epoch passes quality cuts
        quality_parse_method : str — which parsing path was used
    """
    n = len(df)
    w1_ok = np.ones(n, dtype=bool)
    w2_ok = np.ones(n, dtype=bool)
    parse_method = 'default'

    if 'cc_flags' in df.columns:
        cc = df['cc_flags'].fillna('0000').astype(str).str.strip().str.ljust(4, '0')
        # W1: character 0, W2: character 1 (separate per-band — wise-C3 fix)
        w1_ok &= (cc.str[0] == '0').values
        w2_ok &= (cc.str[1] == '0').values
        parse_method = 'cc_flags_per_band'

    if 'qi_fact' in df.columns:
        qi = pd.to_numeric(df['qi_fact'], errors='coerce').fillna(0)
        w1_ok &= (qi > 0).values
        w2_ok &= (qi > 0).values

    if 'moon_masked' in df.columns:
        # Schema-safe moon_masked parsing (wise-C4)
        mm = _parse_moon_masked(df['moon_masked'])
        w1_ok &= ~mm['w1_moon']
        w2_ok &= ~mm['w2_moon']

    df = df.copy()
    df['w1_quality_ok']        = w1_ok
    df['w2_quality_ok']        = w2_ok
    df['quality_parse_method'] = parse_method

    return df


# ---------------------------------------------------------------------------
# wise-C1: Within-season sigma clipping (replaces global sigma clip)
# ---------------------------------------------------------------------------

def sigma_clip_within_seasons(times_mjd, flux_mjy, flux_err_mjy,
                               season_id, sigma_thresh=4.0):
    """
    Sigma clip WITHIN each season, not globally.

    Why within-season (wise-C1):
    - Each season has a roughly stable flux level
    - Outliers within a season are likely instrumental artifacts
    - Cross-season flux differences ARE the CLAGN signal — never clip those
    - Global sigma clip destroys the bimodal flux distribution of a CLAGN turn-on

    Why sigma_thresh=4.0 (conservative):
    - CLAGN transitions can produce ~2-3 sigma excursions at season level
    - 4-sigma within a season is almost certainly an artifact
    - Never go below 3.5 for CLAGN science

    Parameters
    ----------
    times_mjd : array of MJD
    flux_mjy : array of flux densities (mJy)
    flux_err_mjy : array of flux uncertainties (mJy)
    season_id : 1D int array from cluster_epochs_into_seasons()
    sigma_thresh : float, clip threshold (default 4.0)

    Returns
    -------
    keep : bool array — True = keep, False = clip
    clip_reason : str array — reason for each clipped point
    """
    keep = np.ones(len(flux_mjy), dtype=bool)
    clip_reason = np.full(len(flux_mjy), '', dtype=object)

    for s in np.unique(season_id):
        if s < 0:
            continue
        mask = season_id == s
        if mask.sum() < 4:
            continue

        f_s = flux_mjy[mask]

        # Robust season median
        med_s = float(np.median(f_s))

        # MAD-based robust scatter
        mad_s = float(np.median(np.abs(f_s - med_s)))
        rob_sigma = 1.4826 * mad_s

        if rob_sigma < 1e-10:
            continue

        # Clip points beyond sigma_thresh * rob_sigma from season median
        deviant = np.abs(f_s - med_s) > sigma_thresh * rob_sigma

        # Apply back to global keep array
        season_indices = np.where(mask)[0]
        for i, d in zip(season_indices, deviant):
            if d:
                keep[i] = False
                clip_reason[i] = f'within_season_s{s}_>{sigma_thresh}sigma'

    return keep, clip_reason


# ---------------------------------------------------------------------------
# wise-C2: Visit-level weighted mean aggregation (replaces _deduplicate)
# ---------------------------------------------------------------------------

def aggregate_to_visits(df, max_visit_gap_days=0.5,
                         time_col='mjd',
                         w1_flux_col='w1_flux_mjy',
                         w1_err_col='w1_flux_err_mjy',
                         w2_flux_col='w2_flux_mjy',
                         w2_err_col='w2_flux_err_mjy'):
    """
    Aggregate individual WISE exposures into per-visit inverse-variance weighted means.

    A "visit" is a group of consecutive exposures with gaps < max_visit_gap_days.
    0.5 days captures within-night WISE scanning without merging distinct visits.

    Replaces _deduplicate() which kept only the best-SNR single point — statistically
    wrong because it discards information and biases flux toward low-scatter exposures.

    For each visit:
    - Compute inverse-variance weighted mean flux (W1 and W2)
    - Propagate formal uncertainty: sigma_visit = 1/sqrt(sum(1/sigma_i^2))
    - Store n_exp (number of exposures) and intra-visit scatter

    Returns
    -------
    DataFrame with one row per visit.
    """
    if df.empty:
        return df

    df_sorted = df.sort_values(time_col).copy()
    t = df_sorted[time_col].values

    # Assign visit IDs by gap
    visit_id = np.zeros(len(t), dtype=int)
    vid = 0
    for i in range(1, len(t)):
        if (t[i] - t[i - 1]) > max_visit_gap_days:
            vid += 1
        visit_id[i] = vid
    df_sorted['visit_id'] = visit_id

    rows = []
    for vid, grp in df_sorted.groupby('visit_id'):
        row = {'mjd': float(grp[time_col].median()),
               'n_exp': len(grp)}

        # W1 weighted mean
        if w1_flux_col in grp.columns:
            f1 = pd.to_numeric(grp[w1_flux_col], errors='coerce').values
            e1 = pd.to_numeric(grp[w1_err_col],  errors='coerce').values
            valid1 = np.isfinite(f1) & np.isfinite(e1) & (e1 > 0) & (f1 > 0)
            if valid1.sum() > 0:
                w1 = 1.0 / e1[valid1] ** 2
                row['w1_flux_mjy']           = float(np.sum(w1 * f1[valid1]) / np.sum(w1))
                row['w1_flux_err_mjy']       = float(np.sqrt(1.0 / np.sum(w1)))
                row['n_exp_w1']              = int(valid1.sum())
                row['w1_intravisit_scatter'] = float(np.std(f1[valid1])) if valid1.sum() > 1 else 0.0

        # W2 weighted mean
        if w2_flux_col in grp.columns:
            f2 = pd.to_numeric(grp[w2_flux_col], errors='coerce').values
            e2 = pd.to_numeric(grp[w2_err_col],  errors='coerce').values
            valid2 = np.isfinite(f2) & np.isfinite(e2) & (e2 > 0) & (f2 > 0)
            if valid2.sum() > 0:
                w2 = 1.0 / e2[valid2] ** 2
                row['w2_flux_mjy']           = float(np.sum(w2 * f2[valid2]) / np.sum(w2))
                row['w2_flux_err_mjy']       = float(np.sqrt(1.0 / np.sum(w2)))
                row['n_exp_w2']              = int(valid2.sum())
                row['w2_intravisit_scatter'] = float(np.std(f2[valid2])) if valid2.sum() > 1 else 0.0

        # Pass through other columns from first exposure
        for col in ['cc_flags', 'qi_fact', 'moon_masked', 'saa_sep', 'qual_frame',
                    'dataset', 'in_gap', 'w1_mag', 'w1_err', 'w2_mag', 'w2_err',
                    'w1_minus_w2', 'w1_quality_ok', 'w2_quality_ok']:
            if col in grp.columns:
                row[col] = grp[col].iloc[0]

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# wise-M8: W3 color computation with paired masks
# ---------------------------------------------------------------------------

def compute_w1_w3_color(df):
    """
    Compute W1-W3 color using only epochs where BOTH bands are valid.

    Never drop NaNs from one band without dropping from the other —
    mismatched lengths produce nan W1-W3 even when valid pairs exist (wise-M8).

    Returns
    -------
    w1w3_median : float or np.nan
    n_paired : int
    """
    if 'w1mpro' not in df.columns or 'w3mpro' not in df.columns:
        return np.nan, 0

    w1 = pd.to_numeric(df['w1mpro'], errors='coerce').values
    w3 = pd.to_numeric(df['w3mpro'], errors='coerce').values

    # PAIRED mask — both must be finite (wise-M8 fix)
    paired = np.isfinite(w1) & np.isfinite(w3)

    if paired.sum() < 3:
        return np.nan, int(paired.sum())

    w1w3 = float(np.median(w1[paired] - w3[paired]))
    return w1w3, int(paired.sum())


# ---------------------------------------------------------------------------
# Per-dataset IRSA queries
# ---------------------------------------------------------------------------

def _query_allsky(ra: float, dec: float,
                  radius_arcsec: float = WISE_NEOWISE_SEARCH_RADIUS_ARCSEC
                  ) -> pd.DataFrame:
    """Query WISE AllSky single-exposure table via safe schema helper (wise-H5)."""
    required = WISE_TABLE_REQUIRED_COLS.get(WISE_ALLSKY_TABLE, [])
    df, schema = _query_wise_table_safe(
        WISE_ALLSKY_TABLE, ra, dec, radius_arcsec, _EARLY_COLUMNS
    )
    logger.debug(f"AllSky query: schema={schema}, rows={len(df)}")
    if not df.empty and required:
        _validate_table_schema(df, WISE_ALLSKY_TABLE, required)
    return df


def _query_3band(ra: float, dec: float,
                 radius_arcsec: float = WISE_NEOWISE_SEARCH_RADIUS_ARCSEC
                 ) -> pd.DataFrame:
    """Query WISE 3-Band Cryo single-exposure table via safe schema helper (wise-H5)."""
    required = WISE_TABLE_REQUIRED_COLS.get(WISE_3BAND_TABLE, [])
    df, schema = _query_wise_table_safe(
        WISE_3BAND_TABLE, ra, dec, radius_arcsec, _EARLY_COLUMNS
    )
    logger.debug(f"3Band query: schema={schema}, rows={len(df)}")
    if not df.empty and required:
        _validate_table_schema(df, WISE_3BAND_TABLE, required)
    return df


def _query_postcryo(ra: float, dec: float,
                    radius_arcsec: float = WISE_NEOWISE_SEARCH_RADIUS_ARCSEC
                    ) -> pd.DataFrame:
    """Query WISE Post-Cryo (2-band) single-exposure table via safe schema helper (wise-H5)."""
    required = WISE_TABLE_REQUIRED_COLS.get(WISE_POSTCRYO_TABLE, [])
    df, schema = _query_wise_table_safe(
        WISE_POSTCRYO_TABLE, ra, dec, radius_arcsec, _EARLY_COLUMNS
    )
    logger.debug(f"PostCryo query: schema={schema}, rows={len(df)}")
    if not df.empty and required:
        _validate_table_schema(df, WISE_POSTCRYO_TABLE, required)
    return df


# ---------------------------------------------------------------------------
# Magnitude → flux conversion
# ---------------------------------------------------------------------------

def _mag_to_flux_mjy(mag: np.ndarray, mag_err: np.ndarray,
                     band: str) -> tuple:
    """Convert Vega magnitude to flux density in mJy."""
    zp = WISE_ZERO_POINTS.get(band, 309.540)  # Default to W1 if unknown
    flux_jy = zp * 10.0 ** (-0.4 * np.asarray(mag, dtype=float))
    flux_mjy = flux_jy * 1e3
    # Propagate magnitude error: δF/F = 0.4 * ln(10) * δm
    flux_err_mjy = flux_mjy * 0.4 * np.log(10.0) * np.abs(np.asarray(mag_err, dtype=float))
    return flux_mjy, flux_err_mjy


# ---------------------------------------------------------------------------
# Unified schema builder
# ---------------------------------------------------------------------------

def _normalise_early_frame(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """
    Convert a raw early-phase WISE DataFrame to unified schema.

    Output columns:
        mjd, w1_mag, w1_err, w2_mag, w2_err,
        w1_flux_mjy, w1_flux_err_mjy, w2_flux_mjy, w2_flux_err_mjy,
        w1_minus_w2, dataset, in_gap
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Apply quality mask
    mask = _quality_filter_early(df, dataset)
    df = df[mask].copy()

    if df.empty:
        return pd.DataFrame()

    # Rename magnitude columns (early tables use w1mpro without _ep)
    rename_map = {
        'w1mpro': 'w1_mag', 'w1sigmpro': 'w1_err',
        'w2mpro': 'w2_mag', 'w2sigmpro': 'w2_err',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Ensure numeric MJD
    df['mjd'] = pd.to_numeric(df['mjd'], errors='coerce')
    df = df.dropna(subset=['mjd'])

    # Dataset label
    df['dataset'] = dataset

    # Hibernation gap flag
    mjd_vals = df['mjd'].values
    df['in_gap'] = (
        (mjd_vals > WISE_HIBERNATION_MJD_START) &
        (mjd_vals < WISE_HIBERNATION_MJD_END)
    )

    # Flux conversion — Fix #13: no synthetic error injection
    for band in ['W1', 'W2']:
        mag_col = 'w1_mag' if band == 'W1' else 'w2_mag'
        err_col = 'w1_err' if band == 'W1' else 'w2_err'
        flux_col  = f'w{band[1]}_flux_mjy'
        ferr_col  = f'w{band[1]}_flux_err_mjy'
        raw_flux_col = 'w1flux' if band == 'W1' else 'w2flux'
        raw_ferr_col = 'w1sigflux' if band == 'W1' else 'w2sigflux'

        if mag_col in df.columns and err_col in df.columns:
            mag_arr = pd.to_numeric(df[mag_col], errors='coerce').values

            # Fix #13: derive mag error from flux/sigflux when available;
            # fall back to reported mag error; never inject synthetic 0.1
            if raw_flux_col in df.columns and raw_ferr_col in df.columns:
                f_raw  = pd.to_numeric(df[raw_flux_col], errors='coerce').values
                df_raw = pd.to_numeric(df[raw_ferr_col], errors='coerce').values
                err_from_flux = np.where(
                    (f_raw > 0) & np.isfinite(df_raw),
                    1.08574 * np.abs(df_raw) / f_raw,
                    np.nan
                )
                err_arr = pd.Series(err_from_flux, index=df.index)
            else:
                err_arr = pd.to_numeric(df[err_col], errors='coerce')
            # Preserve provenance flag
            df[f'{err_col}_source'] = np.where(
                pd.to_numeric(err_arr, errors='coerce').isna(), 'missing', 'real'
            )
            err_arr = err_arr.values

            valid = np.isfinite(mag_arr)
            flux = np.full(len(df), np.nan)
            ferr = np.full(len(df), np.nan)
            if valid.any():
                # Use NaN errors where not available (no synthetic fill)
                err_for_valid = np.where(np.isfinite(err_arr[valid]), err_arr[valid], np.nan)
                flux[valid], ferr[valid] = _mag_to_flux_mjy(
                    mag_arr[valid], err_for_valid, band
                )
            df[flux_col] = flux
            df[ferr_col] = ferr
        else:
            df[flux_col] = np.nan
            df[ferr_col] = np.nan

    # W1-W2 color
    if 'w1_mag' in df.columns and 'w2_mag' in df.columns:
        df['w1_minus_w2'] = (
            pd.to_numeric(df['w1_mag'], errors='coerce') -
            pd.to_numeric(df['w2_mag'], errors='coerce')
        )
    else:
        df['w1_minus_w2'] = np.nan

    # Select and return unified columns
    keep = [
        'mjd', 'w1_mag', 'w1_err', 'w2_mag', 'w2_err',
        'w1_flux_mjy', 'w1_flux_err_mjy', 'w2_flux_mjy', 'w2_flux_err_mjy',
        'w1_minus_w2', 'dataset', 'in_gap',
    ]
    present = [c for c in keep if c in df.columns]
    return df[present].copy()


def _normalise_neowise_frame(lc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adapt the output of query_wise_lightcurve (existing wise.py) to the
    unified schema used by this module.
    """
    if lc_df is None or lc_df.empty:
        return pd.DataFrame()

    df = lc_df.copy()

    # Rename dataset_flag → dataset
    if 'dataset_flag' in df.columns:
        df = df.rename(columns={'dataset_flag': 'dataset'})
    else:
        df['dataset'] = 'neowise'

    # in_gap flag
    mjd_vals = df['mjd'].values
    df['in_gap'] = (
        (mjd_vals > WISE_HIBERNATION_MJD_START) &
        (mjd_vals < WISE_HIBERNATION_MJD_END)
    )

    # Ensure w1_minus_w2
    if 'w1_minus_w2' not in df.columns:
        if 'w1_mag' in df.columns and 'w2_mag' in df.columns:
            df['w1_minus_w2'] = (
                pd.to_numeric(df['w1_mag'], errors='coerce') -
                pd.to_numeric(df['w2_mag'], errors='coerce')
            )
        else:
            df['w1_minus_w2'] = np.nan

    keep = [
        'mjd', 'w1_mag', 'w1_err', 'w2_mag', 'w2_err',
        'w1_flux_mjy', 'w1_flux_err_mjy', 'w2_flux_mjy', 'w2_flux_err_mjy',
        'w1_minus_w2', 'dataset', 'in_gap',
    ]
    present = [c for c in keep if c in df.columns]
    return df[present].copy()


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _deduplicate(df: pd.DataFrame, window: float = _DEDUP_WINDOW_DAYS) -> pd.DataFrame:
    """
    Remove duplicate exposures within `window` days.

    Within each time cluster, keep the observation with the smallest
    w1_err (highest SNR). Sorts by mjd before processing.
    """
    if df.empty:
        return df

    df = df.sort_values('mjd').reset_index(drop=True)
    mjd = df['mjd'].values
    err = df['w1_err'].fillna(np.inf).values

    keep_indices = []
    i = 0
    while i < len(mjd):
        # Collect cluster
        cluster = [i]
        j = i + 1
        # Fix #14: adjacency-based clustering (each point within window of previous)
        while j < len(mjd) and (mjd[j] - mjd[j - 1]) < window:
            cluster.append(j)
            j += 1
        # Keep best SNR (smallest err)
        best = cluster[int(np.argmin(err[cluster]))]
        keep_indices.append(best)
        i = j

    return df.iloc[keep_indices].reset_index(drop=True)


# ---------------------------------------------------------------------------
# FIX 7: Sigma clipping in flux space only
# ---------------------------------------------------------------------------

def sigma_clip_in_flux_space(times, fluxes, flux_errors, sigma=4.0, maxiters=5):
    """
    SUPERSEDED by sigma_clip_within_seasons() (FIX wise-C1).
    Global flux-space sigma clip retained for reference only — DO NOT USE.
    sigma_clip_within_seasons() clips within seasons, preserving cross-season
    flux differences (the CLAGN signal). This global clip is scientifically
    backwards for CLAGN detection — it can delete the bimodal flux distribution.

    Sigma clip light curves in FLUX SPACE only.

    NEVER sigma clip in magnitude space.

    Why: magnitude errors are asymmetric (log scale).
    A magnitude outlier of +0.5 mag corresponds to flux 58% below median.
    A magnitude outlier of -0.5 mag corresponds to flux 58% ABOVE median.
    These are equal in magnitude space but the flux excursions are equal.

    In flux space, the distribution is approximately Gaussian (central limit
    theorem) and sigma clipping is statistically valid.

    Parameters
    ----------
    times : array (MJD, already in mJy)
    fluxes : array (flux density in mJy)
    flux_errors : array (flux uncertainties in mJy)
    sigma : float, clipping threshold in standard deviations (default 4.0)
    maxiters : int, maximum iterations (default 5)

    Returns
    -------
    mask : boolean array, True = keep
    n_clipped : int, number of clipped epochs
    """
    from astropy.stats import sigma_clip as astropy_sigma_clip

    fluxes = np.asarray(fluxes, dtype=float)

    # CRITICAL: clip on flux values, not magnitudes
    with np.errstate(invalid='ignore'):
        clipped = astropy_sigma_clip(fluxes, sigma=sigma, maxiters=maxiters,
                                     masked=True, copy=True)

    mask = ~np.ma.getmaskarray(clipped)
    n_clipped = int(np.ma.getmaskarray(clipped).sum())

    return mask, n_clipped


# ---------------------------------------------------------------------------
# Seasonal structure function
# ---------------------------------------------------------------------------

def compute_wise_seasonal_structure(times: np.ndarray, fluxes: np.ndarray,
                                    errors: np.ndarray) -> dict:
    """
    Bin the WISE light curve into ~6-month seasons and compute per-season statistics.

    Parameters
    ----------
    times  : array of MJD values
    fluxes : array of W1 flux densities in mJy
    errors : array of flux uncertainties

    Returns
    -------
    dict with key 'seasons' (list of per-season dicts), each containing:
        season_center_mjd, season_index, season_label, n_epochs,
        median_flux, mad_flux, weighted_mean_flux, mean_flux_mjy,
        variability_rms_flux_mjy, wise_season_anchor_mjd
    """
    times = np.asarray(times, dtype=float)
    fluxes = np.asarray(fluxes, dtype=float)
    errors = np.asarray(errors, dtype=float)

    if len(times) < 2:
        return {'seasons': []}

    # Use global anchor so all sources have identical season boundaries
    bin_width = 182.625  # 6 months in days
    first_bin = int(np.floor((times.min() - WISE_SEASON_ANCHOR_MJD) / bin_width))
    last_bin  = int(np.floor((times.max() - WISE_SEASON_ANCHOR_MJD) / bin_width)) + 1
    edges = WISE_SEASON_ANCHOR_MJD + np.arange(first_bin, last_bin + 1) * bin_width

    rows = []

    for i in range(len(edges) - 1):
        t_lo, t_hi = edges[i], edges[i + 1]
        mask = (times >= t_lo) & (times < t_hi)
        n = int(mask.sum())
        if n < 1:
            continue

        f = fluxes[mask]
        e = errors[mask]

        med_f = float(np.nanmedian(f))
        mad_f = float(np.nanmedian(np.abs(f - med_f)))

        # Fix #9: inverse-variance weighted mean and CENTERED variability RMS
        w = 1.0 / np.where(e > 0, e ** 2, 1e-6)
        w_sum = np.sum(w)
        wmean = float(np.sum(w * f) / w_sum) if w_sum > 0 else med_f
        mu_f  = wmean
        rms_f = float(np.sqrt(np.average((f - mu_f) ** 2, weights=w)))

        # Fix #15: season label with signed index, zero-padded
        season_index = int(first_bin + i)
        season_label = f"S{season_index:+04d}"   # e.g. S+003, S-002

        rows.append({
            'season_center_mjd': float(0.5 * (t_lo + t_hi)),
            'season_index': season_index,
            'season_label': season_label,
            'n_epochs': n,
            'median_flux': med_f,
            'mad_flux': mad_f,
            'weighted_mean_flux': wmean,
            'mean_flux_mjy': mu_f,
            'variability_rms_flux_mjy': rms_f,
            'wise_season_anchor_mjd': WISE_SEASON_ANCHOR_MJD,
        })

    return {'seasons': rows}


# ---------------------------------------------------------------------------
# Epoch weights
# ---------------------------------------------------------------------------

def compute_wise_epoch_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Compute per-epoch inverse-variance weights for DRW fitting.

    Fix #3: proper unit-aware logic; never mixes magnitude and flux units.
    Three priority cases:
        1. w1_flux_err_mjy present → use directly (flux units)
        2. w1_err + w1_flux_mjy present → convert mag error to flux error
        3. Neither → uniform weights

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    weights : array, same length as df (un-normalized inverse-variance)
    """
    if df.empty:
        return np.array([])

    FLUX_ERROR_FLOOR_MJY = 0.005  # mJy systematic noise floor

    # Case 1: flux errors in mJy — use directly
    if 'w1_flux_err_mjy' in df.columns:
        err = pd.to_numeric(df['w1_flux_err_mjy'], errors='coerce').values
    # Case 2: magnitude error — convert to flux error: dF = F * 0.92103 * |dmag|
    elif {'w1_err', 'w1_flux_mjy'}.issubset(df.columns):
        f  = pd.to_numeric(df['w1_flux_mjy'], errors='coerce').values
        dm = pd.to_numeric(df['w1_err'],      errors='coerce').values
        err = f * 0.92103 * np.abs(dm)
    # Case 3: no error — uniform weights
    else:
        return np.ones(len(df))

    err = np.where(np.isfinite(err) & (err > 0), err, np.nan)
    err_floored = np.sqrt(err ** 2 + FLUX_ERROR_FLOOR_MJY ** 2)
    weights = np.where(np.isfinite(err_floored), 1.0 / err_floored ** 2, 0.0)
    return weights


# ---------------------------------------------------------------------------
# W3 variability (AllSky only)
# ---------------------------------------------------------------------------

def extract_w3_variability(ra: float, dec: float, source_id: str) -> dict:
    """
    Extract W3 (12 micron) variability from the AllSky survey.

    W3 probes the warm dust component. Its variability relative to W1
    constrains the dust covering factor evolution.

    Parameters
    ----------
    ra, dec    : float, ICRS degrees
    source_id  : str, source identifier

    Returns
    -------
    result : dict with keys:
        available, n_epochs, median_w3_mag, rms_w3_mag, var_amplitude_w3,
        w1_w3_color_median, rejection_reason
    """
    _empty = {
        'available': False,
        'n_epochs': 0,
        'median_w3_mag': np.nan,
        'rms_w3_mag': np.nan,
        'var_amplitude_w3': np.nan,
        'w1_w3_color_median': np.nan,
        'rejection_reason': 'Not attempted',
    }

    w3_columns = (
        "ra, dec, mjd, w1mpro, w1sigmpro, w3mpro, w3sigmpro, "
        "cc_flags, qi_fact, saa_sep, moon_masked, nb"
    )

    try:
        adql = _build_cone_adql(
            WISE_ALLSKY_TABLE, w3_columns, ra, dec, _W3_SEARCH_RADIUS_ARCSEC
        )
        df = _irsa_tap_query(adql)
    except Exception as exc:
        _empty['rejection_reason'] = f'Query failed: {exc}'
        return _empty

    if df.empty:
        _empty['rejection_reason'] = 'No AllSky rows returned'
        return _empty

    # Quality filter on W3: require w3sigmpro < 0.5
    if 'w3mpro' in df.columns and 'w3sigmpro' in df.columns:
        w3m = pd.to_numeric(df['w3mpro'], errors='coerce')
        w3e = pd.to_numeric(df['w3sigmpro'], errors='coerce')
        mask = np.isfinite(w3m) & np.isfinite(w3e) & (w3e < 0.5) & (w3e > 0)
    else:
        _empty['rejection_reason'] = 'W3 columns missing'
        return _empty

    df = df[mask].copy()
    if len(df) < 3:
        _empty['rejection_reason'] = f'Only {len(df)} good W3 epochs'
        _empty['available'] = False
        return _empty

    w3_mags = pd.to_numeric(df['w3mpro'], errors='coerce').dropna().values
    w1_mags = pd.to_numeric(df['w1mpro'], errors='coerce').values

    median_w3 = float(np.nanmedian(w3_mags))
    rms_w3 = float(np.nanstd(w3_mags))
    amp_w3 = float(np.nanmax(w3_mags) - np.nanmin(w3_mags)) if len(w3_mags) > 1 else np.nan
    w1w3 = float(np.nanmedian(w1_mags - w3_mags)) if len(w1_mags) == len(w3_mags) else np.nan

    logger.info(
        f"{source_id}: W3 variability — {len(df)} epochs, "
        f"median={median_w3:.3f} mag, RMS={rms_w3:.4f} mag"
    )

    return {
        'available': True,
        'n_epochs': len(df),
        'median_w3_mag': median_w3,
        'rms_w3_mag': rms_w3,
        'var_amplitude_w3': amp_w3,
        'w1_w3_color_median': w1w3,
        'rejection_reason': None,
    }


# ---------------------------------------------------------------------------
# Main per-source function
# ---------------------------------------------------------------------------

def query_all_wise_epochs(ra: float, dec: float, source_id: str) -> dict:
    """
    Query all five WISE survey phases for a single source and return a
    unified, quality-filtered, deduplicated light curve.

    Parameters
    ----------
    ra, dec    : float, ICRS degrees
    source_id  : str, source identifier

    Returns
    -------
    result : dict with keys:
        lc            : DataFrame (unified schema, sorted by MJD)
        baseline_years: float
        n_epochs_total: int
        n_epochs_allsky, n_epochs_3band, n_epochs_postcryo, n_epochs_neowise
        mjd_first, mjd_last
        w1_minus_w2_median: float
        in_gap_count : int
        seasonal_structure : DataFrame
        epoch_weights : array
        rejection_reason : str or None
    """
    _empty = {
        'lc': pd.DataFrame(),
        'baseline_years': 0.0,
        'n_epochs_total': 0,
        'n_epochs_allsky': 0,
        'n_epochs_3band': 0,
        'n_epochs_postcryo': 0,
        'n_epochs_neowise': 0,
        'mjd_first': np.nan,
        'mjd_last': np.nan,
        'w1_minus_w2_median': np.nan,
        'in_gap_count': 0,
        'seasonal_structure': pd.DataFrame(),
        'epoch_weights': np.array([]),
        'rejection_reason': 'Initialization',
    }

    frames = []

    # ---- AllSky ----------------------------------------------------------------
    try:
        raw_allsky = _query_allsky(ra, dec)
        norm_allsky = _normalise_early_frame(raw_allsky, 'allsky')
        if not norm_allsky.empty:
            frames.append(norm_allsky)
            logger.debug(
                f"{source_id}: AllSky — {len(norm_allsky)} good epochs"
            )
    except Exception as exc:
        logger.warning(f"{source_id}: AllSky query failed: {exc}")

    # ---- 3-Band ----------------------------------------------------------------
    try:
        raw_3band = _query_3band(ra, dec)
        norm_3band = _normalise_early_frame(raw_3band, '3band')
        if not norm_3band.empty:
            frames.append(norm_3band)
            logger.debug(
                f"{source_id}: 3Band — {len(norm_3band)} good epochs"
            )
    except Exception as exc:
        logger.warning(f"{source_id}: 3Band query failed: {exc}")

    # ---- PostCryo --------------------------------------------------------------
    try:
        raw_postcryo = _query_postcryo(ra, dec)
        norm_postcryo = _normalise_early_frame(raw_postcryo, 'postcryo')
        if not norm_postcryo.empty:
            frames.append(norm_postcryo)
            logger.debug(
                f"{source_id}: PostCryo — {len(norm_postcryo)} good epochs"
            )
    except Exception as exc:
        logger.warning(f"{source_id}: PostCryo query failed: {exc}")

    # ---- AllWISE + NEOWISE-R (via existing wise.py) ----------------------------
    try:
        wise_result = query_wise_lightcurve(ra, dec, source_id)
        lc_neo = wise_result.get('lc', pd.DataFrame())
        norm_neo = _normalise_neowise_frame(lc_neo)
        if not norm_neo.empty:
            frames.append(norm_neo)
            logger.debug(
                f"{source_id}: AllWISE+NEOWISE — {len(norm_neo)} good epochs"
            )
    except Exception as exc:
        logger.warning(f"{source_id}: AllWISE/NEOWISE query failed: {exc}")

    # ---- Merge -----------------------------------------------------------------
    if not frames:
        _empty['rejection_reason'] = 'No data from any WISE survey phase'
        return _empty

    combined = pd.concat(frames, ignore_index=True, sort=False)

    # Ensure numeric MJD and sort
    combined['mjd'] = pd.to_numeric(combined['mjd'], errors='coerce')
    combined = combined.dropna(subset=['mjd']).sort_values('mjd').reset_index(drop=True)

    # Drop rows inside hibernation gap
    in_gap_mask = combined['in_gap'].astype(bool) if 'in_gap' in combined.columns \
        else pd.Series(False, index=combined.index)
    in_gap_count = int(in_gap_mask.sum())
    combined = combined[~in_gap_mask].reset_index(drop=True)

    # ---- Stage 0: Within-season sigma clip (wise-C1) -------------------------
    # REPLACED: global iterative MAD sigma clip (scientifically backwards for CLAGN).
    # Global clip on bimodal CLAGN flux distribution can delete the bright or faint state.
    # Within-season clip preserves cross-season flux differences (the CLAGN signal).
    n_before_clip = len(combined)
    if 'w1_flux_mjy' in combined.columns:
        times_mjd = pd.to_numeric(combined['mjd'], errors='coerce').values
        flux_arr  = pd.to_numeric(combined['w1_flux_mjy'], errors='coerce').values
        ferr_col  = combined.get('w1_flux_err_mjy',
                                  pd.Series(np.ones(len(combined)) * 0.01, index=combined.index))
        ferr_arr  = pd.to_numeric(ferr_col, errors='coerce').values

        # Assign season IDs first (1D ndarray — cluster_epochs_into_seasons returns ndarray)
        season_id = cluster_epochs_into_seasons(times_mjd)

        # Within-season clip only — never clip across seasons
        keep_clip, clip_reasons = sigma_clip_within_seasons(
            times_mjd, flux_arr, ferr_arr,
            season_id=season_id,
            sigma_thresh=SIGMA_CLIP_SIGMA   # from config, should be 4.0
        )
        combined = combined[keep_clip].reset_index(drop=True)
        season_id = season_id[keep_clip]    # keep season_id aligned
    else:
        season_id = cluster_epochs_into_seasons(
            pd.to_numeric(combined['mjd'], errors='coerce').values
        )
    n_sigma_clipped = n_before_clip - len(combined)
    logger.info(
        f"{source_id}: Within-season clip removed {n_sigma_clipped} epochs "
        f"(sigma_thresh={SIGMA_CLIP_SIGMA})"
    )

    # ---- Visit aggregation (wise-C2) ------------------------------------------
    # REPLACED: _deduplicate() which kept only the best-SNR single exposure.
    # aggregate_to_visits() computes the inverse-variance weighted mean per visit,
    # preserving all information and correctly propagating uncertainties.
    len_before_agg = len(combined)
    combined = aggregate_to_visits(combined, max_visit_gap_days=0.5)
    logger.info(
        f"{source_id}: Aggregated {len_before_agg} exposures into "
        f"{len(combined)} visits"
    )

    if combined.empty:
        _empty['rejection_reason'] = 'No epochs remain after quality filtering'
        return _empty

    # ---- wise-M9: Assign gap-based season IDs to combined DataFrame -----------
    # cluster_epochs_into_seasons is called at ingestion, not just at analysis time.
    # This ensures season_id is available for all downstream operations.
    combined['season_id'] = cluster_epochs_into_seasons(
        combined['mjd'].values
    )

    # ---- Per-dataset counts ---------------------------------------------------
    n_by_dataset = combined['dataset'].value_counts().to_dict() if 'dataset' in combined.columns else {}

    # ---- Stage 1: Baseline metrics (Fix #1) ----------------------------------
    mjd_vals = combined['mjd'].values
    mjd_first = float(np.nanmin(mjd_vals))
    mjd_last  = float(np.nanmax(mjd_vals))
    # Fix #1: actual span between first and last observed epoch
    baseline_years_actual = (mjd_last - mjd_first) / 365.25
    # Additional: time since mission start (for context)
    baseline_years_since_mission_start = (mjd_last - WISE_FULL_BASELINE_START_MJD) / 365.25
    baseline_years = baseline_years_actual  # canonical value

    # ---- Stage 2: Seasonal structure -----------------------------------------
    if 'w1_flux_mjy' in combined.columns:
        flux_arr = pd.to_numeric(combined['w1_flux_mjy'], errors='coerce').values
        ferr_arr = pd.to_numeric(
            combined.get('w1_flux_err_mjy', pd.Series(np.ones(len(combined)) * 0.01)),
            errors='coerce'
        ).values
        seasonal = compute_wise_seasonal_structure(mjd_vals, flux_arr, ferr_arr)
    else:
        seasonal = {'seasons': []}

    # ---- Stage 3: Epoch weights ----------------------------------------------
    epoch_weights = compute_wise_epoch_weights(combined)

    # ---- Stage 4: Color ------------------------------------------------------
    color_vals = pd.to_numeric(
        combined.get('w1_minus_w2', pd.Series(np.nan, index=combined.index)),
        errors='coerce'
    ).values
    w1w2_median = float(np.nanmedian(color_vals)) if np.any(np.isfinite(color_vals)) else np.nan

    # ---- Stage 5: Flag W2 systematic epochs (MJD 57000-57071) ---------------
    w2_systematic_flag = np.zeros(len(combined), dtype=bool)
    if 'mjd' in combined.columns:
        mjd_arr = combined['mjd'].values
        w2_systematic_flag = flag_systematic_epochs(mjd_arr, 'W2')
        combined = combined.copy()
        combined['w2_systematic_flag'] = w2_systematic_flag
        n_w2_flagged = int(w2_systematic_flag.sum())
        if n_w2_flagged > 0:
            logger.info(
                f"{source_id}: Flagged {n_w2_flagged} W2 epochs in MJD 57000-57071 "
                f"systematic window (NEOWISE docs)"
            )

    # ---- Stage 6: Per-epoch saturation check (Fix #5) -----------------------
    w1_mags_arr = pd.to_numeric(
        combined.get('w1_mag', pd.Series(np.nan, index=combined.index)),
        errors='coerce'
    ).values
    w2_mags_arr = pd.to_numeric(
        combined.get('w2_mag', pd.Series(np.nan, index=combined.index)),
        errors='coerce'
    ).values
    sat_result = check_wise_saturation_complete(w1_mags_arr, w2_mags_arr, mjd_vals)
    saturation_flag = sat_result['saturation_flag']
    if saturation_flag == 'saturated':
        logger.warning(
            f"{source_id}: WISE saturation detected — "
            f"W1={sat_result['n_w1_saturated_epochs']} epochs, "
            f"W2={sat_result['n_w2_saturated_epochs']} epochs"
        )

    # Legacy scalar medians (for backward compat with check_wise_saturation)
    w1_mag_median = sat_result['w1_median_mag']
    w2_mag_median = sat_result['w2_median_mag']
    w1_reliable, w2_reliable, _ = check_wise_saturation(w1_mag_median, w2_mag_median)

    logger.info(
        f"{source_id}: Complete WISE — {len(combined)} epochs | "
        f"baseline={baseline_years:.1f} yr | "
        f"datasets={n_by_dataset}"
    )

    return {
        'lc': combined,
        # Fix #1: actual baseline (first-to-last epoch)
        'baseline_years': float(baseline_years_actual),
        'baseline_years_since_mission_start': float(baseline_years_since_mission_start),
        'n_epochs_total': len(combined),
        'n_epochs_allsky': n_by_dataset.get('allsky', 0),
        'n_epochs_3band': n_by_dataset.get('3band', 0),
        'n_epochs_postcryo': n_by_dataset.get('postcryo', 0),
        'n_epochs_neowise': (
            n_by_dataset.get('neowise', 0) +
            n_by_dataset.get('allwise', 0)
        ),
        'mjd_first': mjd_first,
        'mjd_last': mjd_last,
        'w1_minus_w2_median': w1w2_median,
        'in_gap_count': in_gap_count,
        'seasonal_structure': seasonal,
        'epoch_weights': epoch_weights,
        'w2_systematic_flag_count': int(w2_systematic_flag.sum()),
        'w1_mag_median': float(w1_mag_median) if np.isfinite(w1_mag_median) else np.nan,
        'w2_median_mag': float(w2_mag_median) if np.isfinite(w2_mag_median) else np.nan,
        'w1_photometry_reliable': bool(w1_reliable),
        'w2_photometry_reliable': bool(w2_reliable),
        'saturation_flag': saturation_flag,
        'n_w1_saturated_epochs': sat_result['n_w1_saturated_epochs'],
        'n_w2_saturated_epochs': sat_result['n_w2_saturated_epochs'],
        'n_sigma_clipped': n_sigma_clipped,
        'rejection_reason': None,
    }


# ---------------------------------------------------------------------------
# FLAW A7: Symmetric W1/W2 quality masks
# ---------------------------------------------------------------------------

def compute_wise_quality_masks(table):
    """
    Compute SEPARATE quality masks for W1 and W2.
    Do NOT use a W1-only mask for both bands.

    Fix #4: schema-safe — uses helper getters that return safe defaults for
    missing columns and returns missing_cols for transparency.

    Returns
    -------
    mask_w1 : pd.Series[bool] — epochs where W1 is reliable
    mask_w2 : pd.Series[bool] — epochs where W2 is reliable
    mask_joint : pd.Series[bool] — epochs where BOTH are reliable
    missing_cols : list[str] — column names absent from table
    """
    n = len(table)
    idx = table.index if hasattr(table, 'index') else range(n)
    missing_cols = []

    def _col(col, fill=0.0):
        if col not in table.columns:
            missing_cols.append(col)
            return pd.Series(fill, index=idx, dtype=float)
        return pd.to_numeric(table[col], errors='coerce').fillna(fill)

    def _str_col(col):
        if col not in table.columns:
            missing_cols.append(col)
            return pd.Series(['X'] * n, index=idx)
        return table[col].fillna('X').astype(str).str.strip()

    cc = _str_col('cc_flags')
    w1cc_ok = cc.str[:1].isin(['0', 'H', 'h'])
    w2cc_ok = cc.apply(lambda c: c[1] in ('0', 'H', 'h') if len(c) > 1 else False)

    mask_w1 = (
        (_col('w1snr',    fill=0.0) >= 5.0) &
        (_col('w1rchi2',  fill=99.0) < 5.0) &
        (_col('w1sat',    fill=1.0) == 0)   &
        (_col('w1sigmpro', fill=0.0) > 0)   &
        w1cc_ok
    )

    mask_w2 = (
        (_col('w2snr',    fill=0.0) >= 5.0) &
        (_col('w2rchi2',  fill=99.0) < 5.0) &
        (_col('w2sat',    fill=1.0) == 0)   &
        (_col('w2sigmpro', fill=0.0) > 0)   &
        w2cc_ok
    )

    mask_joint = mask_w1 & mask_w2
    return mask_w1, mask_w2, mask_joint, list(set(missing_cols))


# ---------------------------------------------------------------------------
# FLAW B1: Per-season flux and uncertainty with calibration floor
# ---------------------------------------------------------------------------

def compute_seasonal_flux_and_uncertainty(times, fluxes, errors, season_id):
    """
    Per season: compute weighted mean AND uncertainty of weighted mean.
    Combine with calibration floor in quadrature.

    Returns per-season dict with mean_flux, total_err, n_epochs,
    intra_season_scatter, formal_err.
    """
    WISE_CALIBRATION_FLOOR_FRACTION = 0.028  # 2.8% from WISE docs (W1 RMS)

    results = {}
    for s in np.unique(season_id):
        mask = season_id == s
        n = int(mask.sum())

        f_s = fluxes[mask]
        e_s = errors[mask]

        # Fix #10: handle singleton seasons conservatively
        if n == 1:
            mean_flux  = float(f_s[0])
            formal_err = float(e_s[0])
            cal_floor  = WISE_CALIBRATION_FLOOR_FRACTION * abs(mean_flux)
            total_err  = float(np.sqrt(formal_err ** 2 + cal_floor ** 2))
            results[int(s)] = {
                'mean_flux':              mean_flux,
                'total_err':              total_err,
                'n_epochs':               1,
                'intra_season_scatter':   np.nan,
                'formal_err':             formal_err,
                'singleton':              True,
            }
            continue

        w_s = 1.0 / np.maximum(e_s ** 2, 1e-30)

        # Weighted mean
        mean_flux = np.average(f_s, weights=w_s)

        # Formal uncertainty of weighted mean
        formal_err = 1.0 / np.sqrt(w_s.sum())

        # Intra-season scatter (may exceed formal error due to real variability)
        scatter = np.sqrt(np.average((f_s - mean_flux) ** 2, weights=w_s))

        # Calibration floor
        cal_floor = WISE_CALIBRATION_FLOOR_FRACTION * abs(mean_flux)

        # Total uncertainty: max of formal, scatter/sqrt(N), floor
        total_err = np.sqrt(
            max(formal_err, scatter / np.sqrt(mask.sum())) ** 2 +
            cal_floor ** 2
        )

        results[int(s)] = {
            'mean_flux':            float(mean_flux),
            'total_err':            float(total_err),
            'n_epochs':             n,
            'intra_season_scatter': float(scatter),
            'formal_err':           float(formal_err),
            'singleton':            False,
        }

    return results


# ---------------------------------------------------------------------------
# FLAW B2: Per-epoch saturation check (not just global median)
# ---------------------------------------------------------------------------

def check_wise_saturation_complete(w1_mags, w2_mags, times):
    """
    Check saturation on per-epoch basis, not just median.

    Fix #5: Returns per-epoch boolean arrays aligned to input arrays.
    A source entering saturation during a bright state is the most
    dangerous case — exactly the epochs most relevant to CLAGN detection.

    Parameters
    ----------
    w1_mags, w2_mags : array-like, W1/W2 Vega magnitudes (NaN allowed)
    times : array-like, MJD values aligned to w1_mags/w2_mags

    Returns
    -------
    dict with per-epoch masks and scalar summary statistics
    """
    W1_SAT_LIMIT   = 8.0    # mag, Vega
    W2_SAT_LIMIT   = 6.7
    W1_FAINT_LIMIT = 14.5
    W2_FAINT_LIMIT = 13.7

    w1_mags = np.asarray(w1_mags, dtype=float)
    w2_mags = np.asarray(w2_mags, dtype=float)
    times   = np.asarray(times,   dtype=float)
    n = len(w1_mags)

    # Per-epoch masks (aligned to input)
    w1_fin = np.isfinite(w1_mags)
    w2_fin = np.isfinite(w2_mags)

    w1_saturated_epoch_mask  = w1_fin & (w1_mags < W1_SAT_LIMIT)
    w2_saturated_epoch_mask  = w2_fin & (w2_mags < W2_SAT_LIMIT)
    w1_too_faint_mask        = w1_fin & (w1_mags > W1_FAINT_LIMIT)
    w2_too_faint_mask        = w2_fin & (w2_mags > W2_FAINT_LIMIT)
    w1_reliable_epoch_mask   = w1_fin & ~w1_saturated_epoch_mask & ~w1_too_faint_mask
    w2_reliable_epoch_mask   = w2_fin & ~w2_saturated_epoch_mask & ~w2_too_faint_mask

    # Saturated MJDs
    w1_saturated_mjds = times[w1_saturated_epoch_mask].tolist()
    w2_saturated_mjds = times[w2_saturated_epoch_mask].tolist()

    # W2 median on reliable epochs only
    w2_reliable_mags = w2_mags[w2_reliable_epoch_mask]
    w2_median_mag = float(np.median(w2_reliable_mags)) if len(w2_reliable_mags) > 0 else np.nan

    any_sat = bool(w1_saturated_epoch_mask.any() or w2_saturated_epoch_mask.any())

    return {
        'w1_saturated_epoch_mask':  w1_saturated_epoch_mask,
        'w2_saturated_epoch_mask':  w2_saturated_epoch_mask,
        'w1_reliable_epoch_mask':   w1_reliable_epoch_mask,
        'w2_reliable_epoch_mask':   w2_reliable_epoch_mask,
        'w1_saturated_mjds':        w1_saturated_mjds,
        'w2_saturated_mjds':        w2_saturated_mjds,
        'w2_median_mag':            w2_median_mag,
        'n_w1_saturated_epochs':    int(w1_saturated_epoch_mask.sum()),
        'n_w2_saturated_epochs':    int(w2_saturated_epoch_mask.sum()),
        'w1_any_saturated':         bool(w1_saturated_epoch_mask.any()),
        'w2_any_saturated':         bool(w2_saturated_epoch_mask.any()),
        'w1_min_mag':               float(w1_mags[w1_fin].min()) if w1_fin.any() else np.nan,
        'w2_min_mag':               float(w2_mags[w2_fin].min()) if w2_fin.any() else np.nan,
        'w1_median_mag':            float(np.median(w1_mags[w1_fin])) if w1_fin.any() else np.nan,
        'saturation_flag':          'saturated' if any_sat else 'clear',
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_complete_wise_ingestion(results_dir: str = './results/') -> None:
    """
    Run complete WISE ingestion for all sources in top_candidates.csv.

    For each source:
    - Calls query_all_wise_epochs to fetch all survey phases
    - Saves enhanced_wise_{source_id}.pkl per source
    - Accumulates and saves enhanced_wise_all.csv

    Parameters
    ----------
    results_dir : str, path to results directory containing top_candidates.csv
    """
    results_path = Path(results_dir)
    candidates_file = results_path / 'top_candidates.csv'

    if not candidates_file.exists():
        logger.error(f"Candidates file not found: {candidates_file}")
        return

    try:
        candidates = pd.read_csv(candidates_file)
    except Exception as exc:
        logger.error(f"Failed to load candidates file: {exc}")
        return

    logger.info(
        f"Starting complete WISE ingestion for {len(candidates)} candidates"
    )

    # Output directories
    pkl_dir = results_path / 'wise_enhanced'
    pkl_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for idx, row in candidates.iterrows():
        source_id = str(row.get('source_id', f'source_{idx}'))
        ra = float(row.get('ra', row.get('RA', np.nan)))
        dec = float(row.get('dec', row.get('DEC', np.nan)))

        if not (np.isfinite(ra) and np.isfinite(dec)):
            logger.warning(f"{source_id}: Invalid coordinates (ra={ra}, dec={dec}), skipping")
            continue

        logger.info(f"Processing {source_id} ({idx + 1}/{len(candidates)})")

        try:
            result = query_all_wise_epochs(ra, dec, source_id)

            # Per-source pickle
            pkl_path = pkl_dir / f'enhanced_wise_{source_id}.pkl'
            with open(pkl_path, 'wb') as f:
                pickle.dump(result, f, protocol=4)
            logger.debug(f"{source_id}: Saved to {pkl_path}")

            # Summary row — Fix #23: include all diagnostic fields
            summary_rows.append({
                'source_id':                    source_id,
                'ra':                           ra,
                'dec':                          dec,
                'baseline_years':               result['baseline_years'],
                'n_epochs_total':               result['n_epochs_total'],
                'n_epochs_allsky':              result['n_epochs_allsky'],
                'n_epochs_3band':               result['n_epochs_3band'],
                'n_epochs_postcryo':            result['n_epochs_postcryo'],
                'n_epochs_neowise':             result['n_epochs_neowise'],
                'mjd_first':                    result['mjd_first'],
                'mjd_last':                     result['mjd_last'],
                'w1_minus_w2_median':           result['w1_minus_w2_median'],
                'rejection_reason':             result.get('rejection_reason'),
                # Fix #23: additional diagnostic fields
                'baseline_years_actual':        result.get('baseline_years', np.nan),
                'baseline_years_since_mission': result.get('baseline_years_since_mission_start', np.nan),
                'w2_median_mag':                result.get('w2_median_mag', np.nan),
                'saturation_flag':              result.get('saturation_flag', 'unknown'),
                'n_w1_saturated_epochs':        result.get('n_w1_saturated_epochs', 0),
                'n_w2_saturated_epochs':        result.get('n_w2_saturated_epochs', 0),
                'n_sigma_clipped':              result.get('n_sigma_clipped', 0),
                'schema_used':                  result.get('schema_used', 'unknown'),
                'missing_quality_cols':         str(result.get('missing_quality_cols', [])),
            })

        except Exception as exc:
            logger.error(f"{source_id}: Ingestion failed: {exc}", exc_info=True)
            summary_rows.append({
                'source_id': source_id,
                'ra': ra,
                'dec': dec,
                'rejection_reason': str(exc),
            })

        # Polite IRSA delay
        time.sleep(0.5)

    # Save summary CSV
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = results_path / 'enhanced_wise_all.csv'
        summary_df.to_csv(summary_path, index=False)
        logger.info(
            f"Complete WISE ingestion finished. "
            f"Summary saved to {summary_path}"
        )
    else:
        logger.warning("No results to save.")
