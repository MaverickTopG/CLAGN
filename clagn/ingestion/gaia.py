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
    GAIA_COLUMNS_WITH_PARALLAX,
    MAX_PROPER_MOTION_SIG, MAX_RUWE,
)

logger = logging.getLogger(__name__)

# Suppress the ESA authentication warning at import time
warnings.filterwarnings('ignore', message='.*passwords of all user accounts.*')


def _fmt(x, spec=".2f"):
    """NaN-safe log formatting helper (Fix #21). Returns 'nan' for non-finite or bad types."""
    try:
        return format(float(x), spec) if np.isfinite(float(x)) else "nan"
    except (TypeError, ValueError):
        return "nan"


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
        # parallax fields
        'parallax': np.nan,
        'parallax_error': np.nan,
        'gaia_parallax_sig': np.nan,
        # Fix #8: diagnostic fields
        'gaia_is_foreground_star': None,
        'gaia_rejection_reasons': None,
        'gaia_quality_flag': None,
        'gaia_quasar_class': None,
        'gaia_ang_sep_arcsec': np.nan,
        'gaia_match_method': None,
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

    # Fix #6: Order by angular separation (nearest first), not by SNR
    query = (
        f"SELECT TOP 5 {GAIA_COLUMNS_WITH_PARALLAX}, "
        f"DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {ra:.6f}, {dec:.6f})) AS ang_sep_deg "
        f"FROM {GAIA_MAIN_TABLE} "
        f"WHERE CONTAINS("
        f"  POINT('ICRS', ra, dec),"
        f"  CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius_deg:.8f})"
        f") = 1 "
        f"ORDER BY ang_sep_deg ASC"
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

    # Fix #6: Take nearest match (sorted by ang_sep_deg ASC)
    row = tbl[0]

    # Reject if nearest source exceeds match threshold
    MAX_MATCH_ARCSEC = 1.0
    ang_sep_arcsec = float(row['ang_sep_deg']) * 3600.0
    if ang_sep_arcsec > MAX_MATCH_ARCSEC:
        logger.debug(
            f"{source_id}: Nearest GAIA source at {ang_sep_arcsec:.2f}\" "
            f"exceeds {MAX_MATCH_ARCSEC}\" threshold — no match"
        )
        return _null_gaia_result()

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
    # FLAW A6: Extract parallax
    parallax = _safe_float('parallax')
    parallax_error = _safe_float('parallax_error')
    plx_sig = (abs(parallax) / parallax_error
               if (np.isfinite(parallax) and np.isfinite(parallax_error) and parallax_error > 0)
               else np.nan)

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
    quasar_class = check_gaia_quasar_classification(classprob_quasar, source_id)

    # ---- Foreground star check (Fix #8) -------------------------------------
    is_star, star_reasons = is_foreground_star({
        'pm_sig': pm_sig,
        'gaia_parallax_sig': float(plx_sig) if np.isfinite(plx_sig) else np.nan,
        'ruwe': ruwe,
    })

    logger.debug(
        f"{source_id}: GAIA match | RUWE={_fmt(ruwe)} | "
        f"pm_sig={_fmt(pm_sig, '.1f')}{'(unknown)' if gaia_pm_unknown else ''} | "
        f"variable={phot_var_flag} | ang_sep={_fmt(ang_sep_arcsec)}\""
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
        # parallax for foreground star rejection
        'parallax': parallax,
        'parallax_error': parallax_error,
        'gaia_parallax_sig': float(plx_sig) if np.isfinite(plx_sig) else np.nan,
        # Fix #8: diagnostic fields
        'gaia_is_foreground_star': is_star,
        'gaia_rejection_reasons': ', '.join(star_reasons) if star_reasons else None,
        'gaia_quality_flag': ruwe_quality_flag,
        'gaia_quasar_class': quasar_class,
        'gaia_ang_sep_arcsec': float(ang_sep_arcsec),
        'gaia_match_method': 'nearest_within_1arcsec',
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


def is_foreground_star(gaia_row):
    """
    FLAW A6: Reject foreground stars using ALL three GAIA criteria.

    Reject if ANY of these three criteria trigger:
    1. Proper motion significance > 3σ
    2. Parallax significance > 3σ (direct distance measurement — strongest test)
    3. RUWE > 1.4

    Parameters
    ----------
    gaia_row : dict (output of query_gaia_dr3)

    Returns
    -------
    is_star : bool
    reasons : list of str (empty if not a star)
    """
    reasons = []

    # 1. Proper motion significance
    try:
        pm_sig = gaia_row.get('pm_sig', np.nan)
        if np.isfinite(pm_sig) and pm_sig > MAX_PROPER_MOTION_SIG:
            reasons.append(f'pm_sig={pm_sig:.1f}')
    except (KeyError, TypeError):
        pass

    # 2. Parallax (strongest test — direct distance measurement)
    try:
        plx_sig = gaia_row.get('gaia_parallax_sig', np.nan)
        if np.isfinite(plx_sig) and plx_sig > 3.0:
            reasons.append(f'parallax_sig={plx_sig:.1f}')
    except (KeyError, TypeError):
        pass

    # 3. RUWE
    try:
        ruwe = gaia_row.get('ruwe', np.nan)
        if np.isfinite(ruwe) and ruwe > MAX_RUWE:
            reasons.append(f'ruwe={ruwe:.2f}')
    except (KeyError, TypeError):
        pass

    return len(reasons) > 0, reasons


def check_gaia_quasar_classification(classprob_quasar, source_id=''):
    """
    GAIA DSC quasar probability advisory check.

    classprob_quasar > 0.5: strong independent AGN confirmation.
    classprob_quasar < 0.1: potentially misclassified — advisory warning only.
    This is advisory only — do not reject based on this alone.

    Returns
    -------
    str: one of 'strong_quasar_support', 'neutral', 'low_quasar_prob', 'unknown'
    """
    if not np.isfinite(classprob_quasar):
        return 'unknown'
    if classprob_quasar > 0.5:
        return 'strong_quasar_support'
    if classprob_quasar < 0.1:
        logger.debug(
            f"{source_id}: Low GAIA quasar probability = {classprob_quasar:.3f} "
            f"(possible misclassification — advisory warning only)"
        )
        return 'low_quasar_prob'
    return 'neutral'
