"""
catalog.py — Load input catalog, validate redshifts, apply AGN pre-selection.

Supports CSV and FITS formats. Every removal is logged.
"""
import logging
import numpy as np
import pandas as pd

from ..config import (
    MIN_REDSHIFT, MAX_REDSHIFT, WISE_AGN_COLOR_CUT,
)

logger = logging.getLogger(__name__)

# Canonical column name aliases: maps any variant to the standard name
_COLUMN_ALIASES = {
    'RA': 'ra', 'Ra': 'ra', 'RA_DEG': 'ra', 'raj2000': 'ra',
    'DEC': 'dec', 'Dec': 'dec', 'DE': 'dec', 'DEC_DEG': 'dec', 'dej2000': 'dec',
    'Z': 'redshift', 'z': 'redshift', 'REDSHIFT': 'redshift', 'z_spec': 'redshift',
    'zspec': 'redshift', 'ZSPEC': 'redshift', 'Z_SPEC': 'redshift',
    'NAME': 'source_id', 'name': 'source_id', 'ID': 'source_id', 'id': 'source_id',
    'SDSS_NAME': 'source_id', 'OBJID': 'source_id', 'objid': 'source_id',
    'Source_Name': 'source_id',
}


def load_and_validate_catalog(filepath):
    """
    Load source catalog, apply all pre-flight quality filters.
    Supports CSV and FITS formats.
    Returns clean DataFrame with standardized column names.

    Required columns (or aliases): ra, dec, source_id, redshift

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    df : pd.DataFrame with columns: source_id, ra, dec, redshift
         plus any additional columns from the input file.
    """
    filepath = str(filepath)

    # ---- Load ----------------------------------------------------------------
    if filepath.lower().endswith(('.fits', '.fit', '.fts')):
        from astropy.table import Table
        tbl = Table.read(filepath)
        df = tbl.to_pandas()
        # Decode bytes columns (FITS stores strings as bytes in pandas)
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].apply(
                lambda x: x.decode('utf-8').strip() if isinstance(x, bytes) else x
            )
    else:
        df = pd.read_csv(filepath)

    logger.info(f"Loaded {len(df)} sources from {filepath}")

    # ---- Standardize column names --------------------------------------------
    df = df.rename(columns={k: v for k, v in _COLUMN_ALIASES.items()
                             if k in df.columns})

    # Check required columns
    required = {'ra', 'dec', 'redshift'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Catalog missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    # Generate source_id if absent
    if 'source_id' not in df.columns:
        logger.warning("No source_id column found — generating from coordinates.")
        df['source_id'] = [
            f"J{row.ra:.4f}{row.dec:+.4f}"
            for row in df[['ra', 'dec']].itertuples()
        ]

    # Ensure numeric types
    for col in ('ra', 'dec', 'redshift'):
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['source_id'] = df['source_id'].astype(str)

    n_start = len(df)

    # ---- Step 1: Remove NaN redshifts ---------------------------------------
    mask_nan_z = df['redshift'].isna()
    n_removed = mask_nan_z.sum()
    df = df[~mask_nan_z].copy()
    if n_removed:
        logger.info(f"Removed {n_removed} sources with NaN redshift")

    # ---- Step 2: Remove non-positive and near-zero redshifts ----------------
    mask_low_z = df['redshift'] <= MIN_REDSHIFT
    n_removed = mask_low_z.sum()
    df = df[~mask_low_z].copy()
    if n_removed:
        logger.info(
            f"Removed {n_removed} sources with z <= {MIN_REDSHIFT} "
            f"(Galactic foreground)"
        )

    # ---- Step 3: Remove unphysically high redshifts -------------------------
    mask_high_z = df['redshift'] >= MAX_REDSHIFT
    n_removed = mask_high_z.sum()
    df = df[~mask_high_z].copy()
    if n_removed:
        logger.info(
            f"Removed {n_removed} sources with z >= {MAX_REDSHIFT} "
            f"(WISE W1/W2 probes UV at high-z, unreliable)"
        )

    # ---- Step 4: Require valid sky coordinates ------------------------------
    mask_bad_coord = ~(
        df['ra'].between(0, 360, inclusive='both') &
        df['dec'].between(-90, 90, inclusive='both')
    )
    n_removed = mask_bad_coord.sum()
    df = df[~mask_bad_coord].copy()
    if n_removed:
        logger.info(f"Removed {n_removed} sources with invalid sky coordinates")

    logger.info(
        f"Final catalog: {len(df)} sources after validation "
        f"(started with {n_start})"
    )

    return df.reset_index(drop=True)


def apply_wise_agn_color_selection(df, w1_minus_w2_col='w1_minus_w2_allwise'):
    """
    Apply W1-W2 > 0.8 mag (Vega) AGN color cut (Stern et al. 2012).

    This selects AGN with >95% reliability and removes:
    - Normal galaxies (W1-W2 ~ 0)
    - Stars (W1-W2 < 0.5)
    - Starburst galaxies

    Sources without AllWISE colors are retained with a warning flag.

    NOTE: This function is called AFTER the WISE query in main.py, using the
    median AllWISE color from the light curve. It does NOT query IRSA itself.
    It returns flags rather than dropping rows — the caller decides what to do.

    Parameters
    ----------
    df : pd.DataFrame with a column named by `w1_minus_w2_col`

    Returns
    -------
    df : same DataFrame with added columns:
         'passes_agn_color' : bool
         'agn_color_unknown': bool (True if color was not available)
    """
    df = df.copy()

    if w1_minus_w2_col not in df.columns:
        logger.warning(
            f"Column '{w1_minus_w2_col}' not found — "
            f"cannot apply AGN color selection. Flagging all as unknown."
        )
        df['passes_agn_color'] = True
        df['agn_color_unknown'] = True
        return df

    color = pd.to_numeric(df[w1_minus_w2_col], errors='coerce')
    missing = color.isna()

    df['passes_agn_color'] = (color > WISE_AGN_COLOR_CUT).fillna(False)
    df['agn_color_unknown'] = missing

    # Retain sources with unknown colors but with a warning
    df.loc[missing, 'passes_agn_color'] = True  # Benefit of the doubt

    n_pass = df['passes_agn_color'].sum()
    n_total = len(df)
    n_unknown = missing.sum()
    logger.info(
        f"AGN color selection: {n_pass}/{n_total} pass W1-W2 > {WISE_AGN_COLOR_CUT} "
        f"({n_unknown} with unknown color retained)"
    )

    return df
