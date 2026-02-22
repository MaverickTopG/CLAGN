"""
wise_complete.py — Full multi-dataset WISE ingestion covering all 5 survey phases.

Survey phases:
    AllSky    : 2010-01-14 – 2010-08-06 (4-band, fully cryogenic)
    3Band     : 2010-08-06 – 2010-09-29 (W1/W2/W3, partial cryogen)
    PostCryo  : 2010-09-29 – 2011-02-01 (W1/W2 only, no cryogen)
    [HIBERNATION 2011-02-17 – 2012-08-08, MJD 55593-56141]
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
    WISE_ALLSKY_MJD_END,
    WISE_3BAND_MJD_END,
    WISE_POSTCRYO_MJD_END,
    WISE_FULL_BASELINE_START_MJD,
    WISE_HIBERNATION_MJD_START,
    WISE_HIBERNATION_MJD_END,
    WISE_ZERO_POINTS,
    WISE_NEOWISE_SEARCH_RADIUS_ARCSEC,
    WISE_SEASON_ANCHOR_MJD,
    SIGMA_CLIP_SIGMA,
    SIGMA_CLIP_ITERS,
    flag_systematic_epochs,
    check_wise_saturation,
)
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
                           max_retries=5, base_sleep=1.0):
    """
    Wrapper around Irsa.query_region() with exponential backoff retry,
    rate-limit handling, and graceful None return on persistent failure.

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
# Per-dataset IRSA queries
# ---------------------------------------------------------------------------

def _query_allsky(ra: float, dec: float,
                  radius_arcsec: float = WISE_NEOWISE_SEARCH_RADIUS_ARCSEC
                  ) -> pd.DataFrame:
    """Query WISE AllSky single-exposure table."""
    adql = _build_cone_adql(WISE_ALLSKY_TABLE, _EARLY_COLUMNS, ra, dec, radius_arcsec)
    logger.debug(f"AllSky query: {adql[:120]}...")
    return _irsa_tap_query(adql)


def _query_3band(ra: float, dec: float,
                 radius_arcsec: float = WISE_NEOWISE_SEARCH_RADIUS_ARCSEC
                 ) -> pd.DataFrame:
    """Query WISE 3-Band Cryo single-exposure table."""
    adql = _build_cone_adql(WISE_3BAND_TABLE, _EARLY_COLUMNS, ra, dec, radius_arcsec)
    logger.debug(f"3Band query: {adql[:120]}...")
    return _irsa_tap_query(adql)


def _query_postcryo(ra: float, dec: float,
                    radius_arcsec: float = WISE_NEOWISE_SEARCH_RADIUS_ARCSEC
                    ) -> pd.DataFrame:
    """Query WISE Post-Cryo (2-band) single-exposure table."""
    adql = _build_cone_adql(WISE_POSTCRYO_TABLE, _EARLY_COLUMNS, ra, dec, radius_arcsec)
    logger.debug(f"PostCryo query: {adql[:120]}...")
    return _irsa_tap_query(adql)


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

    # Flux conversion
    for band in ['W1', 'W2']:
        mag_col = 'w1_mag' if band == 'W1' else 'w2_mag'
        err_col = 'w1_err' if band == 'W1' else 'w2_err'
        flux_col = f'w{band[1]}_flux_mjy'
        ferr_col = f'w{band[1]}_flux_err_mjy'

        if mag_col in df.columns and err_col in df.columns:
            mag_arr = pd.to_numeric(df[mag_col], errors='coerce').values
            err_arr = pd.to_numeric(df[err_col], errors='coerce').fillna(0.1).values
            valid = np.isfinite(mag_arr)
            flux = np.full(len(df), np.nan)
            ferr = np.full(len(df), np.nan)
            if valid.any():
                flux[valid], ferr[valid] = _mag_to_flux_mjy(
                    mag_arr[valid], err_arr[valid], band
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
        while j < len(mjd) and (mjd[j] - mjd[cluster[0]]) < window:
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
                                    errors: np.ndarray) -> pd.DataFrame:
    """
    Bin the WISE light curve into ~6-month seasons and compute per-season statistics.

    Parameters
    ----------
    times  : array of MJD values
    fluxes : array of W1 flux densities in mJy
    errors : array of flux uncertainties

    Returns
    -------
    DataFrame with columns:
        season_center_mjd, n_epochs, median_flux, mad_flux, weighted_mean_flux,
        flux_rms, season_label
    """
    times = np.asarray(times, dtype=float)
    fluxes = np.asarray(fluxes, dtype=float)
    errors = np.asarray(errors, dtype=float)

    if len(times) < 2:
        return pd.DataFrame()

    # FLAW A2: Use global anchor so all sources have identical season boundaries
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
        rms_f = float(np.sqrt(np.nanmean(f ** 2)))

        # Inverse-variance weighted mean
        w = 1.0 / np.where(e > 0, e ** 2, 1e-6)
        wmean = float(np.sum(w * f) / np.sum(w)) if np.sum(w) > 0 else med_f

        rows.append({
            'season_center_mjd': float(0.5 * (t_lo + t_hi)),
            'n_epochs': n,
            'median_flux': med_f,
            'mad_flux': mad_f,
            'weighted_mean_flux': wmean,
            'flux_rms': rms_f,
            'season_label': f"S{first_bin + i + 1:02d}",
            'wise_season_anchor_mjd': WISE_SEASON_ANCHOR_MJD,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Epoch weights
# ---------------------------------------------------------------------------

def compute_wise_epoch_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Compute per-epoch weights for DRW fitting.

    Weight = 1 / (w1_err^2 + floor^2)
    where floor = 0.01 mJy (systematic noise floor).
    Weights are normalized to have a maximum of 1.

    Parameters
    ----------
    df : DataFrame with columns w1_err and/or w1_flux_err_mjy

    Returns
    -------
    weights : array, same length as df
    """
    if df.empty:
        return np.array([])

    # Prefer flux errors in mJy; fall back to magnitude errors
    if 'w1_flux_err_mjy' in df.columns:
        err = pd.to_numeric(df['w1_flux_err_mjy'], errors='coerce').fillna(0.1).values
    elif 'w1_err' in df.columns:
        err = pd.to_numeric(df['w1_err'], errors='coerce').fillna(0.1).values
    else:
        return np.ones(len(df))

    floor = 0.01  # mJy systematic floor
    var = err ** 2 + floor ** 2
    weights = 1.0 / np.where(var > 0, var, 1e-6)

    # Normalize
    w_max = weights.max()
    if w_max > 0:
        weights /= w_max

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

    # ---- Sigma clipping on W1 flux -------------------------------------------
    if 'w1_flux_mjy' in combined.columns:
        flux = pd.to_numeric(combined['w1_flux_mjy'], errors='coerce').values
        ferr = pd.to_numeric(
            combined.get('w1_flux_err_mjy', pd.Series(np.ones(len(combined)) * 0.01)),
            errors='coerce'
        ).values

        valid = np.isfinite(flux) & np.isfinite(ferr) & (ferr > 0)
        keep = valid.copy()

        for _ in range(SIGMA_CLIP_ITERS):
            if keep.sum() < 5:
                break
            med = np.nanmedian(flux[keep])
            mad = np.nanmedian(np.abs(flux[keep] - med))
            sigma_mad = 1.4826 * mad
            if sigma_mad <= 0:
                break
            keep &= np.abs(flux - med) < SIGMA_CLIP_SIGMA * sigma_mad

        combined = combined[keep].reset_index(drop=True)

    # ---- Deduplicate -----------------------------------------------------------
    combined = _deduplicate(combined)

    if combined.empty:
        _empty['rejection_reason'] = 'No epochs remain after quality filtering'
        return _empty

    # ---- Per-dataset counts ---------------------------------------------------
    n_by_dataset = combined['dataset'].value_counts().to_dict() if 'dataset' in combined.columns else {}

    # ---- Baseline metrics ----------------------------------------------------
    mjd_vals = combined['mjd'].values
    mjd_first = float(mjd_vals.min())
    mjd_last = float(mjd_vals.max())
    baseline_years = (mjd_last - WISE_FULL_BASELINE_START_MJD) / 365.25

    # ---- Seasonal structure ---------------------------------------------------
    if 'w1_flux_mjy' in combined.columns:
        flux_arr = pd.to_numeric(combined['w1_flux_mjy'], errors='coerce').values
        ferr_arr = pd.to_numeric(
            combined.get('w1_flux_err_mjy', pd.Series(np.ones(len(combined)) * 0.01)),
            errors='coerce'
        ).values
        seasonal = compute_wise_seasonal_structure(mjd_vals, flux_arr, ferr_arr)
    else:
        seasonal = pd.DataFrame()

    # ---- Epoch weights -------------------------------------------------------
    epoch_weights = compute_wise_epoch_weights(combined)

    # ---- Color ---------------------------------------------------------------
    color_vals = pd.to_numeric(
        combined.get('w1_minus_w2', pd.Series(np.nan, index=combined.index)),
        errors='coerce'
    ).values
    w1w2_median = float(np.nanmedian(color_vals)) if np.any(np.isfinite(color_vals)) else np.nan

    # ---- FIX 4: Flag W2 systematic epochs (MJD 57000-57071) ----------------
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

    # ---- FIX 14: Saturation check -------------------------------------------
    w1_mag_median = np.nan
    w2_mag_median = np.nan
    if 'w1_mag' in combined.columns:
        w1_mags = pd.to_numeric(combined['w1_mag'], errors='coerce').values
        w1_mag_median = float(np.nanmedian(w1_mags[np.isfinite(w1_mags)])) if np.any(np.isfinite(w1_mags)) else np.nan
    if 'w2_mag' in combined.columns:
        w2_mags = pd.to_numeric(combined['w2_mag'], errors='coerce').values
        w2_mag_median = float(np.nanmedian(w2_mags[np.isfinite(w2_mags)])) if np.any(np.isfinite(w2_mags)) else np.nan

    w1_reliable, w2_reliable, saturation_flag = check_wise_saturation(
        w1_mag_median, w2_mag_median
    )
    if saturation_flag:
        logger.warning(f"{source_id}: WISE photometry reliability warning: {saturation_flag}")

    logger.info(
        f"{source_id}: Complete WISE — {len(combined)} epochs | "
        f"baseline={baseline_years:.1f} yr | "
        f"datasets={n_by_dataset}"
    )

    return {
        'lc': combined,
        'baseline_years': float(baseline_years),
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
        'w1_mag_median': float(w1_mag_median),
        'w2_mag_median': float(w2_mag_median),
        'w1_photometry_reliable': bool(w1_reliable),
        'w2_photometry_reliable': bool(w2_reliable),
        'saturation_flag': saturation_flag,
        'rejection_reason': None,
    }


# ---------------------------------------------------------------------------
# FLAW A7: Symmetric W1/W2 quality masks
# ---------------------------------------------------------------------------

def compute_wise_quality_masks(table):
    """
    Compute SEPARATE quality masks for W1 and W2.
    Do NOT use a W1-only mask for both bands.

    Returns three masks:
    - mask_w1: epochs where W1 is reliable
    - mask_w2: epochs where W2 is reliable
    - mask_joint: epochs where BOTH are reliable (use for color/coherence)
    """
    # W1 quality criteria
    mask_w1 = (
        (pd.to_numeric(table['w1snr'],    errors='coerce').fillna(0) >= 5.0) &
        (pd.to_numeric(table['w1rchi2'],  errors='coerce').fillna(99) < 5.0) &
        (pd.to_numeric(table['w1sat'],    errors='coerce').fillna(1) == 0)   &
        (pd.to_numeric(table['w1sigmpro'], errors='coerce').fillna(0) > 0)   &
        np.array([str(c)[0] in ('0', 'H', 'h')
                  for c in table['cc_flags']])  # W1 cc_flag char 0
    )

    # W2 quality criteria — SAME stringency as W1
    mask_w2 = (
        (pd.to_numeric(table['w2snr'],    errors='coerce').fillna(0) >= 5.0) &
        (pd.to_numeric(table['w2rchi2'],  errors='coerce').fillna(99) < 5.0) &
        (pd.to_numeric(table['w2sat'],    errors='coerce').fillna(1) == 0)   &
        (pd.to_numeric(table['w2sigmpro'], errors='coerce').fillna(0) > 0)   &
        np.array([str(c)[1] in ('0', 'H', 'h') if len(str(c)) > 1 else False
                  for c in table['cc_flags']])  # W2 cc_flag char 1
    )

    mask_joint = mask_w1 & mask_w2

    return mask_w1, mask_w2, mask_joint


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
        if mask.sum() < 2:
            continue

        f_s = fluxes[mask]
        e_s = errors[mask]
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
            'mean_flux': float(mean_flux),
            'total_err': float(total_err),
            'n_epochs': int(mask.sum()),
            'intra_season_scatter': float(scatter),
            'formal_err': float(formal_err),
        }

    return results


# ---------------------------------------------------------------------------
# FLAW B2: Per-epoch saturation check (not just global median)
# ---------------------------------------------------------------------------

def check_wise_saturation_complete(w1_mags, w2_mags, times):
    """
    Check saturation on per-epoch basis, not just median.

    A source entering saturation during a bright state is the most
    dangerous case — exactly the epochs most relevant to CLAGN detection.
    """
    W1_SAT_LIMIT = 8.0   # mag, Vega (from WISE docs)
    W2_SAT_LIMIT = 6.7

    w1_mags = np.asarray(w1_mags, dtype=float)
    w2_mags = np.asarray(w2_mags, dtype=float)

    w1_finite = w1_mags[np.isfinite(w1_mags)]
    w2_finite = w2_mags[np.isfinite(w2_mags)]

    w1_sat_mask = w1_finite < W1_SAT_LIMIT
    w2_sat_mask = w2_finite < W2_SAT_LIMIT

    return {
        'w1_any_saturated': bool(w1_sat_mask.any()) if len(w1_sat_mask) > 0 else False,
        'w2_any_saturated': bool(w2_sat_mask.any()) if len(w2_sat_mask) > 0 else False,
        'n_w1_saturated_epochs': int(w1_sat_mask.sum()),
        'n_w2_saturated_epochs': int(w2_sat_mask.sum()),
        'w1_min_mag': float(w1_finite.min()) if len(w1_finite) > 0 else np.nan,
        'w2_min_mag': float(w2_finite.min()) if len(w2_finite) > 0 else np.nan,
        'w1_median_mag': float(np.median(w1_finite)) if len(w1_finite) > 0 else np.nan,
        'saturation_flag': (
            'saturated' if (
                (w1_sat_mask.any() if len(w1_sat_mask) > 0 else False) or
                (w2_sat_mask.any() if len(w2_sat_mask) > 0 else False)
            ) else 'clear'
        ),
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

            # Summary row
            summary_rows.append({
                'source_id': source_id,
                'ra': ra,
                'dec': dec,
                'baseline_years': result['baseline_years'],
                'n_epochs_total': result['n_epochs_total'],
                'n_epochs_allsky': result['n_epochs_allsky'],
                'n_epochs_3band': result['n_epochs_3band'],
                'n_epochs_postcryo': result['n_epochs_postcryo'],
                'n_epochs_neowise': result['n_epochs_neowise'],
                'mjd_first': result['mjd_first'],
                'mjd_last': result['mjd_last'],
                'w1_minus_w2_median': result['w1_minus_w2_median'],
                'rejection_reason': result.get('rejection_reason'),
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
