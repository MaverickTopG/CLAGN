"""
catalog.py — Load input catalog, validate redshifts, apply AGN pre-selection.

Supports CSV and FITS formats. Every removal is logged.
Also provides setup_irsa_login() and build_large_agn_catalog() for large-scale runs.
"""
import logging
import os
import numpy as np
import pandas as pd

from ..config import (
    MIN_REDSHIFT, MAX_REDSHIFT, WISE_AGN_COLOR_CUT,
    mag_to_flux_mjy, flux_mjy_to_mag,  # Re-exported for pipeline-wide import
)

# Re-export for verification test:
# from clagn.ingestion.catalog import mag_to_flux_mjy
__all__ = ['load_and_validate_catalog', 'apply_wise_agn_color_selection',
           'mag_to_flux_mjy', 'flux_mjy_to_mag',
           'setup_irsa_login', 'build_large_agn_catalog']

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


# ---------------------------------------------------------------------------
# IRSA authentication
# ---------------------------------------------------------------------------

def setup_irsa_login():
    """
    IRSA authentication via ~/.netrc (astroquery >= 0.4.7 compatible).

    The .login() method was removed in newer astroquery versions.
    Authentication now happens automatically if ~/.netrc contains:

        machine irsa.ipac.caltech.edu
        login your_email
        password your_password

    This function checks whether authentication is likely working
    and prints the rate limit status. It never crashes the pipeline.
    """
    import netrc as netrc_lib
    from astroquery.ipac.irsa import Irsa  # noqa: F401 — ensures module loaded

    netrc_path = os.path.expanduser('~/.netrc')
    authenticated = False

    if os.path.exists(netrc_path):
        try:
            nrc = netrc_lib.netrc(netrc_path)
            if 'irsa.ipac.caltech.edu' in nrc.hosts:
                authenticated = True
                print("IRSA: ~/.netrc found with IRSA credentials")
                print("IRSA: Authenticated access — ~10,000 requests/hour")
            else:
                print("IRSA: ~/.netrc exists but no IRSA entry found")
        except Exception as e:
            print(f"IRSA: ~/.netrc parse error: {e}")

    if not authenticated:
        print("IRSA: No credentials found — using anonymous access")
        print("IRSA: Rate limit ~1,000 requests/hour (slower but works)")
        print("IRSA: To authenticate, add to ~/.netrc:")
        print("      machine irsa.ipac.caltech.edu")
        print("      login your_email@example.com")
        print("      password your_password")
        print("      Then: chmod 600 ~/.netrc")

    return authenticated


# ---------------------------------------------------------------------------
# Large-scale AGN catalog builder
# ---------------------------------------------------------------------------

def build_large_agn_catalog(target_size=500000, output='large_agn_catalog.fits',
                              seed=None):
    """
    Assemble a catalog of >= target_size AGN from multiple real, publicly
    available astronomical databases and save as FITS.

    Sources queried (in order of preference):
      1. SDSS DR16 Quasar Catalog (Lyke+2020) — spectroscopic quasars
      2. Milliquas v8 (Flesch 2023) — confirmed + probable quasars (VizieR)
      3. WISE AGN R90 (Assef+2018) — IRSA TAP
      4. GAIA DR3 QSO candidates — astroquery.gaia

    Each source is queried independently, results are positionally
    cross-matched (2 arcsec) to deduplicate, and the merged catalog is
    written as a FITS binary table.

    Parameters
    ----------
    target_size : int
        Minimum number of unique AGN to assemble. Default 500,000.
    output : str
        Path for the output FITS file.
    seed : int or None
        Random seed for reproducibility of any sampling steps.

    Returns
    -------
    df : pd.DataFrame  (also saved to `output`)
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    if seed is not None:
        np.random.seed(seed)

    frames = []

    # Per-source query limit: request generously so we have room to deduplicate
    # and still reach target_size.  For small targets (mini run), cap sensibly.
    per_source_limit = min(max(target_size * 3 // 4, 5000), 800000)

    # ------------------------------------------------------------------
    # SOURCE 1: SDSS DR16 Quasar Catalog via SkyServer SQL API
    # ------------------------------------------------------------------
    print("Querying SDSS DR16Q via SkyServer...")
    try:
        import urllib.request
        import urllib.parse
        import json

        limit_sdss = min(per_source_limit, 750000)
        sql = (
            f"SELECT TOP {limit_sdss} "
            f"ra, dec, z AS redshift, SDSS_NAME AS source_id "
            f"FROM SpecObj "
            f"WHERE class='QSO' AND zWarning=0 "
            f"AND z>{MIN_REDSHIFT} AND z<{MAX_REDSHIFT}"
        )
        url = (
            "http://skyserver.sdss.org/dr16/SkyServerWS/SearchTools/SqlSearch"
            f"?cmd={urllib.parse.quote(sql)}&format=json"
        )
        req = urllib.request.urlopen(url, timeout=60)
        data = json.loads(req.read().decode())
        rows = data[0].get("Rows", []) if isinstance(data, list) else []
        if rows:
            sdss_df = pd.DataFrame(rows)
            sdss_df.columns = [c.lower() for c in sdss_df.columns]
            if 'source_id' not in sdss_df.columns:
                sdss_df['source_id'] = [
                    f"SDSS_J{r:.4f}{d:+.4f}"
                    for r, d in zip(sdss_df['ra'], sdss_df['dec'])
                ]
            sdss_df['redshift_source'] = 'SDSS_DR16Q_spec'
            sdss_df = sdss_df[['source_id', 'ra', 'dec', 'redshift',
                                'redshift_source']].copy()
            frames.append(sdss_df)
            print(f"  SDSS DR16Q: {len(sdss_df):,} sources")
        else:
            print("  SDSS DR16Q: no rows returned")
    except Exception as e:
        print(f"  SDSS DR16Q query failed: {e}")

    # ------------------------------------------------------------------
    # SOURCE 2: Milliquas v8 via VizieR
    # ------------------------------------------------------------------
    print("Querying Milliquas v8 via VizieR...")
    try:
        from astroquery.vizier import Vizier
        viz = Vizier(columns=['RAJ2000', 'DEJ2000', 'z', 'Name', 'Type'],
                     row_limit=min(per_source_limit, 900000))
        result = viz.get_catalogs('VII/290')
        if not result:
            result = viz.get_catalogs('VII/284')
        if result:
            tbl = result[0].to_pandas()
            ra_col  = next((c for c in tbl.columns
                            if c.lower() in ('raj2000', 'ra', '_raj2000')), None)
            dec_col = next((c for c in tbl.columns
                            if c.lower() in ('dej2000', 'dec', '_dej2000')), None)
            z_col   = next((c for c in tbl.columns
                            if c.lower() in ('z', 'zsp', 'redshift')), None)
            type_col = next((c for c in tbl.columns
                             if 'type' in c.lower()), None)
            name_col = next((c for c in tbl.columns
                             if c.lower() in ('name', 'id', '_name')), None)
            if not (ra_col and dec_col and z_col):
                raise ValueError(f"Unexpected Milliquas columns: {list(tbl.columns)}")
            mq = tbl.rename(columns={ra_col: 'ra', dec_col: 'dec', z_col: 'redshift'})
            if name_col:
                mq = mq.rename(columns={name_col: 'source_id'})
            else:
                mq['source_id'] = [
                    f"MQ_J{r:.4f}{d:+.4f}"
                    for r, d in zip(mq['ra'], mq['dec'])
                ]
            if type_col:
                mq = mq[mq[type_col].isin(['Q', 'A', 'K', 'q', 'a', 'k'])]
            for col in ('ra', 'dec', 'redshift'):
                mq[col] = pd.to_numeric(mq[col], errors='coerce')
            mq = mq[mq['redshift'] > 0]
            mq['redshift_source'] = 'Milliquas_v8_spec'
            mq = mq[['source_id', 'ra', 'dec', 'redshift',
                      'redshift_source']].dropna()
            frames.append(mq)
            print(f"  Milliquas v8: {len(mq):,} sources")
        else:
            print("  Milliquas v8: no catalog returned from VizieR")
    except Exception as e:
        print(f"  Milliquas v8 query failed: {e}")

    # ------------------------------------------------------------------
    # SOURCE 3: AllWISE AGN candidates (W1-W2 > 0.8) via IRSA TAP
    # ------------------------------------------------------------------
    print("Querying AllWISE AGN candidates via IRSA TAP...")
    try:
        from .wise_complete import _irsa_tap_query
        limit_wise = min(per_source_limit, 4500000)
        adql = (
            f"SELECT TOP {limit_wise} ra, dec, w1mpro, w2mpro "
            f"FROM allwise_p3as_psd "
            f"WHERE (w1mpro - w2mpro) > 0.8 "
            f"AND w1mpro < 17.0"
        )
        tbl = _irsa_tap_query(adql)
        if len(tbl):
            tbl['source_id'] = [
                f"WISE_J{r:.4f}{d:+.4f}"
                for r, d in zip(tbl['ra'], tbl['dec'])
            ]
            tbl['redshift'] = np.nan
            tbl['redshift_source'] = 'WISE_photometric'
            wise_df = tbl[['source_id', 'ra', 'dec', 'redshift',
                            'redshift_source']].copy()
            frames.append(wise_df)
            print(f"  AllWISE AGN: {len(wise_df):,} sources "
                  f"(photometric; redshift from cross-match)")
        else:
            print("  AllWISE AGN: no results")
    except Exception as e:
        print(f"  AllWISE AGN query failed: {e}")

    # ------------------------------------------------------------------
    # SOURCE 4: GAIA DR3 QSO candidates
    # ------------------------------------------------------------------
    print("Querying GAIA DR3 QSO candidates...")
    try:
        from astroquery.gaia import Gaia
        limit_gaia = min(per_source_limit, 6600000)
        adql = (
            f"SELECT TOP {limit_gaia} "
            f"source_id, ra, dec, redshift_qso "
            f"FROM gaiadr3.qso_candidates "
            f"WHERE classprob_dsc_combmod_quasar > 0.8 "
            f"AND redshift_qso > {MIN_REDSHIFT} "
            f"AND redshift_qso < {MAX_REDSHIFT}"
        )
        job = Gaia.launch_job(adql)
        tbl = job.get_results().to_pandas()
        if len(tbl):
            tbl = tbl.rename(columns={'redshift_qso': 'redshift'})
            tbl['source_id'] = 'GAIA_' + tbl['source_id'].astype(str)
            tbl['redshift_source'] = 'GAIA_DR3_photo'
            gaia_df = tbl[['source_id', 'ra', 'dec', 'redshift',
                            'redshift_source']].copy()
            frames.append(gaia_df)
            print(f"  GAIA DR3 QSO: {len(gaia_df):,} sources")
        else:
            print("  GAIA DR3: no results")
    except Exception as e:
        print(f"  GAIA DR3 query failed: {e}")

    # ------------------------------------------------------------------
    # Merge and deduplicate
    # ------------------------------------------------------------------
    if not frames:
        raise RuntimeError(
            "All catalog queries failed. Check network access and try again."
        )

    print(f"\nMerging {len(frames)} source list(s)...")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    n_total_raw = len(combined)
    print(f"Total sources before dedup: {n_total_raw:,}")

    for col in ('ra', 'dec', 'redshift'):
        combined[col] = pd.to_numeric(combined[col], errors='coerce')
    combined = combined.dropna(subset=['ra', 'dec'])

    # Positional deduplication at 2 arcsec
    print("Deduplicating by position (2 arcsec) — this may take a minute...")
    coords = SkyCoord(
        ra=combined['ra'].values * u.deg,
        dec=combined['dec'].values * u.deg
    )
    keep_idx = _dedup_by_position(coords, radius_arcsec=2.0)
    combined = combined.iloc[keep_idx].copy()
    n_after_dedup = len(combined)
    print(f"After dedup: {n_after_dedup:,}")

    # Prefer spectroscopic redshifts over photometric (already sorted by
    # source order: SDSS spec first → Milliquas spec → WISE phot → GAIA phot)
    n_before_z = len(combined)
    combined = combined.dropna(subset=['redshift'])
    combined = combined[
        combined['redshift'].between(MIN_REDSHIFT, MAX_REDSHIFT,
                                     inclusive='neither')
    ]
    n_after_z = len(combined)
    print(f"After redshift filter: {n_after_z:,} "
          f"(removed {n_before_z - n_after_z:,} without valid z)")

    # Galactic plane cut: |b| < 10 deg
    sky = SkyCoord(
        ra=combined['ra'].values * u.deg,
        dec=combined['dec'].values * u.deg,
        frame='icrs'
    ).galactic
    mask_gal = np.abs(sky.b.deg) > 10.0
    combined = combined[mask_gal].copy()
    n_final = len(combined)
    print(f"After Galactic plane cut (|b|>10): {n_final:,}")
    print(f"FINAL CATALOG SIZE: {n_final:,}")

    if n_final < target_size:
        print(
            f"WARNING: Catalog has {n_final:,} sources but target was "
            f"{target_size:,}. Some data sources may have been unavailable."
        )

    combined = combined.reset_index(drop=True)

    # Save as FITS
    print(f"\nSaving to {output}...")
    from astropy.table import Table
    tbl_out = Table.from_pandas(combined)
    tbl_out.write(output, overwrite=True, format='fits')
    print(f"Saved {n_final:,} sources → {output}")

    return combined


def _dedup_by_position(coords, radius_arcsec=2.0):
    """
    Return indices of unique sources, removing positional duplicates within
    radius_arcsec.  Lower-index entries take precedence (they come from
    higher-priority catalogs: SDSS spec > Milliquas spec > WISE > GAIA).

    Uses chunked matching to stay within memory limits for large catalogs.
    """
    from astropy.coordinates import match_coordinates_sky
    import astropy.units as u

    n = len(coords)
    if n == 0:
        return np.array([], dtype=int)

    keep = np.ones(n, dtype=bool)
    radius = radius_arcsec * u.arcsec
    chunk = 50_000

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sub = coords[start:end]
        idx, sep, _ = match_coordinates_sky(sub, coords)
        for local_i, (global_match, separation) in enumerate(zip(idx, sep)):
            global_i = start + local_i
            if separation < radius and global_match < global_i:
                keep[global_i] = False

    return np.where(keep)[0]
