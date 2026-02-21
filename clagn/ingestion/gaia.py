"""
gaia.py — GAIA DR3 queries, proper motion filter, astrometric QA.

Uses astroquery.gaia with ADQL for efficiency.
Public DR3 queries do NOT require login.
"""
import logging
import warnings
import numpy as np

from ..config import (
    GAIA_SEARCH_RADIUS_ARCSEC, GAIA_MAIN_TABLE, GAIA_COLUMNS,
    MAX_PROPER_MOTION_SIG, MAX_RUWE,
)

logger = logging.getLogger(__name__)

# Suppress the ESA authentication warning at import time
warnings.filterwarnings('ignore', message='.*passwords of all user accounts.*')


def _null_gaia_result():
    """Return a fully populated GAIA result dict with null values."""
    return {
        'gaia_found': False,
        'gaia_source_id': None,
        'ruwe': np.nan,
        'pm_sig': np.nan,
        'gaia_pm_unknown': True,
        'passes_pm_filter': True,    # NaN PM → unknown, do not reject
        'passes_ruwe': True,
        'astrometric_excess_noise_sig': np.nan,
        'phot_g_mean_mag': np.nan,
        'phot_variable_flag': 'NOT_AVAILABLE',
        'gaia_variability_score': 0.0,
        'classprob_quasar': np.nan,
        'classprob_galaxy': np.nan,
    }


def query_gaia_dr3(ra, dec, source_id):
    """
    Query GAIA DR3 for astrometric and photometric properties.

    Uses astroquery.gaia with ADQL.
    Returns the brightest source (by G-band flux / error) within the search
    radius, which is the most reliable counterpart.

    Parameters
    ----------
    ra, dec    : float, ICRS degrees
    source_id  : str (for logging)

    Returns
    -------
    result : dict with all fields needed for filtering and output.
    """
    try:
        from astroquery.gaia import Gaia
    except ImportError:
        logger.error("astroquery.gaia not available")
        return _null_gaia_result()

    radius_deg = GAIA_SEARCH_RADIUS_ARCSEC / 3600.0

    query = (
        f"SELECT TOP 5 {GAIA_COLUMNS} "
        f"FROM {GAIA_MAIN_TABLE} "
        f"WHERE CONTAINS("
        f"  POINT('ICRS', ra, dec),"
        f"  CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius_deg:.8f})"
        f") = 1 "
        f"ORDER BY phot_g_mean_flux_over_error DESC"
    )

    try:
        # Suppress noisy ESA auth warnings during query
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            job = Gaia.launch_job(query)
            tbl = job.get_results()
    except Exception as exc:
        logger.warning(f"{source_id}: GAIA DR3 query failed: {exc}")
        return _null_gaia_result()

    if tbl is None or len(tbl) == 0:
        logger.debug(f"{source_id}: No GAIA DR3 counterpart found within "
                     f"{GAIA_SEARCH_RADIUS_ARCSEC}\"")
        return _null_gaia_result()

    # Take the best match (already sorted by SNR DESC)
    row = tbl[0]

    def _safe_float(col):
        val = row[col]
        if hasattr(val, 'unmasked'):
            val = val.unmasked
        try:
            v = float(val)
            return v if np.isfinite(v) else np.nan
        except (TypeError, ValueError):
            return np.nan

    def _safe_str(col):
        val = row[col]
        if hasattr(val, 'unmasked'):
            val = val.unmasked
        if val is None:
            return 'NOT_AVAILABLE'
        s = str(val).strip()
        return s if s not in ('', '--', 'None', 'nan') else 'NOT_AVAILABLE'

    pmra = _safe_float('pmra')
    pmra_err = _safe_float('pmra_error')
    pmdec = _safe_float('pmdec')
    pmdec_err = _safe_float('pmdec_error')
    ruwe = _safe_float('ruwe')
    excess_noise_sig = _safe_float('astrometric_excess_noise_sig')
    phot_var_flag = _safe_str('phot_variable_flag')
    classprob_quasar = _safe_float('classprob_dsc_combmod_quasar')
    classprob_galaxy = _safe_float('classprob_dsc_combmod_galaxy')
    g_mag = _safe_float('phot_g_mean_mag')
    gaia_id = row['source_id']

    # ---- Proper motion significance -----------------------------------------
    pm_sig, gaia_pm_unknown = compute_pm_significance(
        pmra, pmra_err, pmdec, pmdec_err
    )

    # ---- Point source quality -----------------------------------------------
    passes_ruwe, ruwe_quality_flag = check_point_source_quality(
        ruwe, excess_noise_sig
    )

    # ---- PM filter: reject stars (pm_sig >= 3.0) ----------------------------
    if gaia_pm_unknown:
        passes_pm = True
    elif not np.isfinite(pm_sig):
        passes_pm = True
    else:
        passes_pm = pm_sig < MAX_PROPER_MOTION_SIG

    # ---- GAIA variability score --------------------------------------------
    gaia_var_score = 1.0 if phot_var_flag == 'VARIABLE' else 0.0

    # ---- Quasar probability advisory check ----------------------------------
    check_gaia_quasar_classification(classprob_quasar, source_id)

    logger.debug(
        f"{source_id}: GAIA match | RUWE={ruwe:.2f} | "
        f"pm_sig={pm_sig:.1f}{'(unknown)' if gaia_pm_unknown else ''} | "
        f"variable={phot_var_flag}"
    )

    return {
        'gaia_found': True,
        'gaia_source_id': int(gaia_id) if gaia_id is not None else None,
        'ruwe': ruwe,
        'pm_sig': pm_sig,
        'gaia_pm_unknown': gaia_pm_unknown,
        'passes_pm_filter': passes_pm,
        'passes_ruwe': passes_ruwe,
        'astrometric_excess_noise_sig': excess_noise_sig,
        'phot_g_mean_mag': g_mag,
        'phot_variable_flag': phot_var_flag,
        'gaia_variability_score': gaia_var_score,
        'classprob_quasar': classprob_quasar,
        'classprob_galaxy': classprob_galaxy,
    }


def compute_pm_significance(pmra, pmra_err, pmdec, pmdec_err):
    """
    Compute total proper motion significance in sigma.

    A genuine extragalactic AGN has zero proper motion.
    Stars at any distance have measurable proper motion.

    pm_total = sqrt(pmra^2 + pmdec^2)
    pm_err   = sqrt((pmra*pmra_err)^2 + (pmdec*pmdec_err)^2) / pm_total
    pm_sig   = pm_total / pm_err

    NaN proper motions: GAIA may not measure PM for faint/crowded sources.
    NaN PM means GAIA couldn't measure it, NOT that it's zero.
    Treat as unknown — do not auto-reject, flag as gaia_pm_unknown=True.

    Parameters
    ----------
    pmra, pmra_err, pmdec, pmdec_err : float (mas/yr)

    Returns
    -------
    pm_sig : float (sigma) — NaN if unknown
    gaia_pm_unknown : bool
    """
    # Check for NaN inputs
    if any(not np.isfinite(v) for v in [pmra, pmra_err, pmdec, pmdec_err]):
        return np.nan, True

    pm_total = np.sqrt(pmra**2 + pmdec**2)

    # Guard against near-zero total PM (definitely not a star)
    if pm_total < 1e-10:
        return 0.0, False

    # Error propagation on the total PM
    pm_err = np.sqrt((pmra * pmra_err)**2 + (pmdec * pmdec_err)**2) / pm_total

    if pm_err <= 0 or not np.isfinite(pm_err):
        return np.nan, True

    pm_sig = pm_total / pm_err
    return float(pm_sig), False


def check_point_source_quality(ruwe, astrometric_excess_noise_sig):
    """
    Confirm the source is a point source (AGN) not an extended galaxy or blend.

    RUWE (Renormalised Unit Weight Error):
    - RUWE ~ 1.0: single point source, clean astrometry → ACCEPT
    - RUWE > 1.4: extended, blended, or binary → REJECT

    astrometric_excess_noise_sig:
    - < 2: consistent with point source → ACCEPT
    - >= 2: excess astrometric noise, possible extended source → FLAG

    Parameters
    ----------
    ruwe                       : float
    astrometric_excess_noise_sig : float

    Returns
    -------
    passes_quality : bool
    quality_flag   : str
    """
    if not np.isfinite(ruwe):
        return True, 'ruwe_unknown'

    if ruwe > MAX_RUWE:
        return False, f'ruwe={ruwe:.2f}_exceeds_{MAX_RUWE}'

    if np.isfinite(astrometric_excess_noise_sig) and astrometric_excess_noise_sig >= 2.0:
        # Flag but don't reject — extended sources can still host AGN
        return True, f'excess_astrometric_noise_sig={astrometric_excess_noise_sig:.1f}'

    return True, 'clean_point_source'


def check_gaia_quasar_classification(classprob_quasar, source_id=''):
    """
    GAIA DSC quasar probability advisory check.

    classprob_quasar > 0.5: strong independent AGN confirmation.
    classprob_quasar < 0.1: potentially misclassified — advisory warning only.
    This is advisory only — do not reject based on this alone.
    """
    if not np.isfinite(classprob_quasar):
        return  # No classification available
    if classprob_quasar < 0.1:
        logger.debug(
            f"{source_id}: Low GAIA quasar probability = {classprob_quasar:.3f} "
            f"(possible misclassification — advisory warning only)"
        )
