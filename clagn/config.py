"""
config.py — All scientific constants and configuration for the CLAGN pipeline.

Every number used in the pipeline must be defined here.
Never hardcode values in other modules.

References:
    Jarrett et al. 2011 (WISE zero points)
    Wright et al. 2010, AJ 140, 1868 (WISE mission, zero points)
    Stern et al. 2012, ApJ 753 30 (WISE AGN color cut)
    Ricci & Trakhtenbrot 2022, arXiv:2211.05132
    Kelly et al. 2009, ApJ 698 895 (DRW model)
    Kozlowski et al. 2017, arXiv:1611.08248 (DRW reliability)
"""

import numpy as np

# ---------------------------------------------------------------------------
# WISE Vega zero points (Wright et al. 2010, Table 1 — AUTHORITATIVE VALUES)
# These MUST match Wright+2010 exactly. Any other value is wrong.
# ---------------------------------------------------------------------------
WISE_ZERO_POINTS = {
    'W1': 309.540,   # Jy (Wright+2010 Table 1)
    'W2': 171.787,   # Jy (Wright+2010 Table 1)
    'W3': 31.674,    # Jy
    'W4': 8.363      # Jy
}

# Alias for use in new functions
WISE_VEGA_ZERO_POINTS = WISE_ZERO_POINTS

# Scalar aliases used by validate_config() and external callers (config-C1)
WISE_W1_ZERO_POINT_JY = WISE_ZERO_POINTS['W1']   # = 309.540 Jy (Wright+2010 Table 1)
WISE_W2_ZERO_POINT_JY = WISE_ZERO_POINTS['W2']   # = 171.787 Jy (Wright+2010 Table 1)

# ---------------------------------------------------------------------------
# WISE global season anchor (FLAW A2)
# All sources use this same anchor so season boundaries are identical.
# MJD 55200 = 2010-01-14, approximate start of WISE all-sky survey.
# Season 0: MJD 55200–55382, Season 1: MJD 55383–55565, etc.
# ---------------------------------------------------------------------------
WISE_SEASON_ANCHOR_MJD = 55200.0

# ---------------------------------------------------------------------------
# FIX 8: Sign convention documentation (enforced everywhere)
# ---------------------------------------------------------------------------
DELTA_MAG_SIGN_CONVENTION = """
SIGN CONVENTION — ENFORCED EVERYWHERE IN THIS CODEBASE
=======================================================

delta_mag is defined as:
    delta_mag = mag_early - mag_late
             = -2.5 * log10(F_early / F_late)
             = +2.5 * log10(F_late / F_early)

POSITIVE delta_mag = source BRIGHTENED (flux INCREASED)
    = CS-AGN Turn-ON event
    = accretion rate increased
    = W1 flux went UP

NEGATIVE delta_mag = source FADED (flux DECREASED)
    = CS-AGN Turn-OFF event
    = accretion rate decreased
    = W1 flux went DOWN

flux_ratio is defined as:
    flux_ratio = F_late / F_early

flux_ratio > 1 = source brightened (= positive delta_mag)
flux_ratio < 1 = source faded     (= negative delta_mag)

Relationship:
    delta_mag = +2.5 * log10(flux_ratio)
    flux_ratio = 10^(delta_mag / 2.5)

For CLAGN detection threshold: |delta_mag| >= 0.3
(both turn-on and turn-off events qualify)

DO NOT use abs() on delta_mag in scoring.
DO NOT use max(mags) - min(mags).
DO NOT use single-epoch comparison.
USE: seasonal median flux method (compute_delta_mag_correct in variability.py)
"""


# ---------------------------------------------------------------------------
# FIX 2: Magnitude <-> flux conversion functions (Wright+2010 zero points)
# ---------------------------------------------------------------------------

def mag_to_flux_mjy(mag, mag_err, band):
    """
    Convert WISE Vega magnitude to flux density in mJy.

    Parameters
    ----------
    mag : float or array
        WISE Vega magnitude (w1mpro, w2mpro)
    mag_err : float or array
        Magnitude uncertainty (w1sigmpro, w2sigmpro)
    band : str
        'W1' or 'W2'

    Returns
    -------
    flux_mjy : float or array
        Flux density in mJy (millijansky)
    flux_err_mjy : float or array
        Flux uncertainty in mJy

    Notes
    -----
    WISE zero points from Wright+2010, Table 1.
    F0(W1) = 309.540 Jy, F0(W2) = 171.787 Jy.

    Conversion: F = F0 * 10^(-0.4 * mag)
    In mJy:     F_mJy = F0_Jy * 1000 * 10^(-0.4 * mag)
    Error:      dF = F * (0.4 * ln10) * dmag = F * 0.92103 * dmag
    """
    # FLAW C3: Use atleast_1d for robust scalar handling
    scalar_input = np.isscalar(mag)
    mag     = np.atleast_1d(np.asarray(mag,     dtype=float))
    mag_err = np.atleast_1d(np.asarray(mag_err, dtype=float))
    F0_mJy = WISE_VEGA_ZERO_POINTS[band] * 1000.0  # Jy -> mJy

    valid = (mag > 0) & (mag < 30) & np.isfinite(mag)
    flux_mjy = np.where(valid, F0_mJy * 10.0 ** (-0.4 * mag), np.nan)

    flux_err_mjy = np.where(
        valid & (mag_err > 0),
        flux_mjy * 0.92103 * mag_err,
        np.nan
    )

    # Return scalar if scalar was given
    if scalar_input:
        return float(flux_mjy[0]), float(flux_err_mjy[0])
    return flux_mjy, flux_err_mjy


def flux_mjy_to_mag(flux_mjy, flux_err_mjy, band):
    """
    Convert flux density in mJy back to WISE Vega magnitude.

    mag = -2.5 * log10(flux_mJy / F0_mJy)
    dmag = (2.5 / ln10) * dflux / flux = 1.08574 * dflux / flux
    """
    # FLAW C3: Use atleast_1d for robust scalar handling
    scalar_input = np.isscalar(flux_mjy)
    flux_mjy     = np.atleast_1d(np.asarray(flux_mjy,     dtype=float))
    flux_err_mjy = np.atleast_1d(np.asarray(flux_err_mjy, dtype=float))
    F0_mJy = WISE_VEGA_ZERO_POINTS[band] * 1000.0
    valid = (flux_mjy > 0) & np.isfinite(flux_mjy)
    mag = np.where(valid, -2.5 * np.log10(flux_mjy / F0_mJy), np.nan)
    mag_err = np.where(
        valid & (flux_err_mjy > 0),
        1.08574 * flux_err_mjy / flux_mjy,
        np.nan
    )
    if scalar_input:
        return float(mag[0]), float(mag_err[0])
    return mag, mag_err


# ---------------------------------------------------------------------------
# FIX wise-H6 / wise-M10: WISE systematic windows
# ---------------------------------------------------------------------------
# WISE_KNOWN_SYSTEMATICS has been cleared (config-C1 / wise-M10):
#   - Removed bogus W1 "hibernation gap" at MJD 55400-56200 (wrong dates; no data exists
#     there anyway — WISE was hibernating at MJD 55593-56987).
#   - Removed uncited W2 entry at MJD 57000-57071 (no NEOWISE release-note citation found).
#   An uncited systematic "correction" is worse than no correction (wise-M10).
WISE_KNOWN_SYSTEMATICS = []   # No validated WISE-cryo-era systematics to flag

# NEOWISE-R systematics: only include entries with explicit NEOWISE release notes citations.
# Currently empty — add entries when a verified citation is found.
WISE_NEOWISE_KNOWN_SYSTEMATICS = [
    # {
    #     'mjd_start':   ...,
    #     'mjd_end':     ...,
    #     'band':        'W1' or 'W2',
    #     'description': '...',
    #     'citation':    'NEOWISE YYYY Data Release Notes, Section X.X',  # REQUIRED
    #     'action':      'inflate_uncertainty',
    #     'sigma_inflation_factor': 2.0,
    # }
]


def flag_systematic_epochs(times, band):
    """
    Return boolean array: True = epoch affected by known WISE systematic.

    These epochs should be DOWN-WEIGHTED in DRW fitting, not removed.
    They are flagged in the output CSV for transparency.

    Parameters
    ----------
    times : array of MJD values
    band : str ('W1' or 'W2')

    Returns
    -------
    flagged : boolean array, True = affected epoch
    """
    times = np.asarray(times, dtype=float)
    flagged = np.zeros(len(times), dtype=bool)
    for sys in WISE_KNOWN_SYSTEMATICS:
        if sys['band'] == band and sys['action'] == 'flag_epochs':
            flagged |= (times >= sys['mjd_start']) & (times <= sys['mjd_end'])
    return flagged


# ---------------------------------------------------------------------------
# FIX 14: WISE saturation / reliability limits
# ---------------------------------------------------------------------------
WISE_RELIABLE_PHOTOMETRY_LIMITS = {
    'W1': {'bright': 8.0,  'faint': 14.5},   # mag Vega
    'W2': {'bright': 6.7,  'faint': 13.7},
}


def check_wise_saturation(w1_mag_median, w2_mag_median):
    """
    Check whether median WISE magnitudes fall in the reliable range.

    WISE W1 saturates at ~8.0 mag; profile-fit photometry unreliable
    outside [8.0, 14.5] (W1) and [6.7, 13.7] (W2).

    Parameters
    ----------
    w1_mag_median : float, median W1 magnitude
    w2_mag_median : float, median W2 magnitude

    Returns
    -------
    w1_reliable : bool
    w2_reliable : bool
    saturation_flag : str or None
    """
    lim = WISE_RELIABLE_PHOTOMETRY_LIMITS

    w1_ok = (lim['W1']['bright'] < w1_mag_median < lim['W1']['faint']
             if np.isfinite(w1_mag_median) else False)
    w2_ok = (lim['W2']['bright'] < w2_mag_median < lim['W2']['faint']
             if np.isfinite(w2_mag_median) else False)

    flags = []
    if np.isfinite(w1_mag_median) and not w1_ok:
        if w1_mag_median <= lim['W1']['bright']:
            flags.append(f'W1={w1_mag_median:.1f} saturated (limit: 8.0 mag)')
        else:
            flags.append(f'W1={w1_mag_median:.1f} too faint (limit: 14.5 mag)')
    if np.isfinite(w2_mag_median) and not w2_ok:
        if w2_mag_median <= lim['W2']['bright']:
            flags.append(f'W2={w2_mag_median:.1f} saturated (limit: 6.7 mag)')
        else:
            flags.append(f'W2={w2_mag_median:.1f} too faint (limit: 13.7 mag)')

    saturation_flag = '; '.join(flags) if flags else None
    return bool(w1_ok), bool(w2_ok), saturation_flag


# ---------------------------------------------------------------------------
# Quality thresholds
# NOTE: these are pipeline engineering choices, not universal physics laws.
# Sensitivity to these thresholds should be tested before publication.
# ---------------------------------------------------------------------------
MIN_BASELINE_YEARS     = 7.0     # config-H2: was 10.0; NEOWISE-R started Dec 2013 (MJD 56987);
                                  # as of 2026, max baseline is ~12 yr. 7 yr >> 10*tau for tau~300d.
                                  # SENSITIVITY TEST REQUIRED: re-run with 5.0 and 10.0.
MIN_EPOCHS             = 20      # Minimum WISE single-exposure epochs
MIN_EPOCHS_PRE_GAP     = 5       # Minimum epochs from AllWISE (pre-hibernation)
MIN_EPOCHS_POST_GAP    = 5       # Minimum epochs from NEOWISE-R (post-hibernation)
MIN_REDSHIFT           = 0.002   # z < 0.002 = likely Galactic (not hard physics — see config-H2)
                                  # SENSITIVITY TEST REQUIRED: check sources in 0.002-0.01.
MAX_REDSHIFT           = 5.0     # Beyond this WISE W1/W2 probe rest-frame UV, not IR torus
MAX_PROPER_MOTION_SIG  = 3.0     # Sigma threshold for star rejection via GAIA PM
MAX_RUWE               = 1.4     # GAIA RUWE soft-flag threshold (gaia-H3: NOT a hard cut alone)
MAX_PARALLAX_SIG       = 3.0     # Sigma threshold for parallax-based star rejection (gaia-C1)
RUWE_HARD_REJECT       = 2.5     # Reject on RUWE alone only above this (gaia-H3)
RUWE_SOFT_FLAG         = 1.4     # Flag for review; hard-reject only with corroboration (gaia-H3)

# Sensitivity analysis parameters (config-H2)
SENSITIVITY_THRESHOLDS_TO_TEST = {
    'MIN_BASELINE_YEARS':    [5.0, 7.0, 10.0],
    'MIN_REDSHIFT':          [0.002, 0.005, 0.01],
    'SIGMA_CLIP_SIGMA':      [3.5, 4.0, 5.0],
    'MIN_DELTA_MAG':         [0.2, 0.3, 0.4],
    'MIN_EPOCHS_PER_SEASON': [2, 3, 5],
    # Run sensitivity_test.py before paper submission
}

# ---------------------------------------------------------------------------
# AGN WISE color selection (Stern et al. 2012, ApJ 753 30)
# W1-W2 > 0.8 (Vega) selects AGN with >95% reliability
# ---------------------------------------------------------------------------
WISE_AGN_COLOR_CUT     = 0.8     # W1-W2 Vega magnitudes

# ---------------------------------------------------------------------------
# DRW model priors (log-space, physically motivated for AGN)
# ---------------------------------------------------------------------------
DRW_LOG_TAU_MIN        = 1.0     # ln(days), tau > e^1 ~ 3 days
DRW_LOG_TAU_MAX        = 8.5     # ln(days), tau < e^8.5 ~ 5000 days
DRW_LOG_SIGMA_MIN      = -5.0    # ln(flux amplitude)
DRW_LOG_SIGMA_MAX      = 5.0

# DRW subsampling: O(N^3) — subsample if N > this threshold
DRW_MAX_EPOCHS_FOR_FULL_FIT = 300

# ---------------------------------------------------------------------------
# Scoring weights (tuned to maximize separation from normal AGN variability)
# ---------------------------------------------------------------------------
SCORE_WEIGHTS = {
    'drw_nonstationarity': 3.0,   # Most important: deviation from stationary DRW
    'delta_mag_w1':        2.5,   # Raw flux change amplitude
    'changepoint_bic':     2.5,   # Bayesian evidence for structural break
    'sf_break':            2.0,   # Structure function slope flattening at long lags
    'color_evolution':     1.5,   # W1-W2 color change (dust temperature/accretion)
    'drw_sigma_excess':    1.5,   # Variability amplitude vs luminosity expectation
    'gaia_variability':    1.0,   # Supporting optical evidence from GAIA
}

MAX_SCORE = sum(SCORE_WEIGHTS.values())  # = 14.0

# ---------------------------------------------------------------------------
# Cone search radii
# ---------------------------------------------------------------------------
# AllWISE MEPT uses 6" because the AllWISE catalog position can differ from
# the input catalog by up to ~5" (WISE PSF FWHM = 6").
WISE_ALLWISE_SEARCH_RADIUS_ARCSEC = 8.0    # 8" to capture nearby Seyferts where WISE centroid offset from nucleus
WISE_NEOWISE_SEARCH_RADIUS_ARCSEC = 6.0   # 6" to capture high-dec sources with larger position scatter
WISE_SEARCH_RADIUS_ARCSEC  = 6.0   # Legacy / NEOWISE-R default
GAIA_SEARCH_RADIUS_ARCSEC  = 2.0    # gaia-H2: query radius (generous — catch potential matches)
GAIA_ACCEPT_RADIUS_ARCSEC  = 1.0    # gaia-H2: acceptance radius (tighter — reduce contamination)
# Two-stage justification: query 2.0" to find nearest match, accept only within 1.0" to
# limit contamination. This justification must appear in the paper Methods section.

# ---------------------------------------------------------------------------
# WISE gap: satellite was hibernating (do not interpolate across this)
# WISE mission timeline (from NASA/IPAC WISE mission descriptions):
#   Cryogenic:    MJD 55200 – 55457  (Jan 2010 – Aug 2010)
#   3-Band Cryo:  MJD 55457 – 55468  (Aug 2010)
#   Post-Cryo:    MJD 55468 – 55593  (Aug 2010 – Dec 2010)
#   Hibernation:  MJD 55593 – 56987  (Feb 2011 – Dec 2013)  ← CORRECT (wise-H6)
#   NEOWISE-R:    MJD 56987 – present (Dec 2013 – )
# ---------------------------------------------------------------------------
# Canonical names (wise-H6)
WISE_HIBERNATION_START_MJD = 55593   # 2011-02-17
WISE_HIBERNATION_END_MJD   = 56987   # 2013-12-13 (NEOWISE-R start — actual end of hibernation)

# Backward-compatible aliases (keep so existing imports don't break)
WISE_HIBERNATION_MJD_START = WISE_HIBERNATION_START_MJD
WISE_HIBERNATION_MJD_END   = WISE_HIBERNATION_END_MJD    # FIXED: was 56141 (wrong)

# ---------------------------------------------------------------------------
# Sigma clipping for outlier rejection before DRW fit
# ---------------------------------------------------------------------------
SIGMA_CLIP_SIGMA  = 4.0
SIGMA_CLIP_ITERS  = 3

# ---------------------------------------------------------------------------
# Minimum flux change to be a serious CLAGN candidate (Ricci+2022: factor >= 2)
# ---------------------------------------------------------------------------
MIN_FLUX_RATIO_CHANGE = 2.0   # max_flux / min_flux (rolling 1-year medians)
MIN_MAG_CHANGE_W1     = 0.3   # mag — conservative lower bound; strong CLAGN show > 0.75 mag

# ---------------------------------------------------------------------------
# IRSA TAP table names
# ---------------------------------------------------------------------------
IRSA_ALLWISE_TABLE = "allwise_p3as_mep"
IRSA_NEOWISE_TABLE = "neowiser_p1bs_psd"

# Columns to request from AllWISE MEPT.
# IMPORTANT: AllWISE MEPT uses _ep suffix for photometry columns.
# There is NO ph_qual or w1snr/w2snr — SNR is derived as flux/sigflux.
# moon_masked is a 4-char string (one char per band: W1, W2, W3, W4).
ALLWISE_MEPT_COLUMNS = (
    "ra, dec, mjd, w1mpro_ep, w1sigmpro_ep, w2mpro_ep, w2sigmpro_ep, "
    "w1flux_ep, w1sigflux_ep, w2flux_ep, w2sigflux_ep, "
    "cc_flags, qi_fact, saa_sep, moon_masked, nb"
)

# Columns to request from NEOWISE-R.
# ph_qual is a 2-char string: [0]=W1, [1]=W2.
# moon_masked is a 4-char string: [0]=W1.
NEOWISE_COLUMNS = (
    "ra, dec, mjd, w1mpro, w1sigmpro, w2mpro, w2sigmpro, "
    "w1snr, w2snr, w1flux, w1sigflux, w2flux, w2sigflux, "
    "cc_flags, ph_qual, qi_fact, saa_sep, moon_masked, nb"
)

# ---------------------------------------------------------------------------
# GAIA DR3 configuration
# ---------------------------------------------------------------------------
GAIA_MAIN_TABLE = "gaiadr3.gaia_source"
GAIA_COLUMNS = (
    "source_id, ra, dec, pmra, pmra_error, pmdec, pmdec_error, "
    "ruwe, astrometric_excess_noise, astrometric_excess_noise_sig, "
    "phot_g_mean_mag, phot_g_mean_flux_over_error, "
    "phot_variable_flag, non_single_star, "
    "classprob_dsc_combmod_quasar, classprob_dsc_combmod_galaxy"
)

# Fix #7 + #20: Canonical GAIA column list including parallax.
# Used in query_gaia_dr3(); DISTANCE() is appended inline in the query.
GAIA_COLUMNS_WITH_PARALLAX = (
    "source_id, ra, dec, "
    "parallax, parallax_error, "
    "pmra, pmra_error, pmdec, pmdec_error, "
    "ruwe, phot_g_mean_mag, phot_g_mean_flux_over_error, "
    "astrometric_excess_noise, astrometric_excess_noise_sig, "
    "phot_variable_flag, "
    "classprob_dsc_combmod_quasar, classprob_dsc_combmod_galaxy"
)

# ---------------------------------------------------------------------------
# Host galaxy contamination threshold
# ---------------------------------------------------------------------------
HOST_CONTAMINATION_Z_THRESHOLD = 0.05  # WISE PSF (6 arcsec) blends AGN + host at low-z

# ---------------------------------------------------------------------------
# Structure function
# ---------------------------------------------------------------------------
SF_N_LAG_BINS       = 20
SF_N_BOOTSTRAP      = 1000
SF_MIN_PAIRS_PER_BIN = 5

# ---------------------------------------------------------------------------
# Changepoint detection
# ---------------------------------------------------------------------------
CP_MIN_SEGMENT_FRACTION = 0.15   # Don't place break in first or last 15% of time series
CP_STRONG_BIC_THRESHOLD  = 6.0   # ΔBIC > 6: strong evidence (Kass & Raftery 1995)
CP_VERY_STRONG_BIC       = 10.0  # ΔBIC > 10: very strong evidence

# ---------------------------------------------------------------------------
# False positive rejection thresholds
# ---------------------------------------------------------------------------
FP_SN_BREAK_DURATION_DAYS = 100.0  # Break durations shorter than this may be supernovae
FP_GALACTIC_LAT_THRESHOLD = 20.0   # |b| < 20 deg + low quasar prob → stellar penalty
FP_SN_SCORE_PENALTY       = 0.70   # Multiplicative penalty for SN-like events
FP_STELLAR_SCORE_PENALTY  = 0.50   # Multiplicative penalty for possible stellar contaminants
FP_HOST_SCORE_PENALTY     = 0.50   # Multiplicative penalty for host-contaminated marginal sources
FP_ARTIFACT_SCORE_PENALTY = 0.30   # Multiplicative penalty for single-epoch-driven variability

# Quasar probability threshold below which sources in the Galactic plane are suspect
FP_LOW_QUASAR_PROB = 0.10

# ---------------------------------------------------------------------------
# Checkpointing and parallelism
# ---------------------------------------------------------------------------
CHECKPOINT_DIR_DEFAULT    = ".clagn_checkpoints"
PARALLEL_WORKERS_DEFAULT  = 4

# ---------------------------------------------------------------------------
# Luminosity-variability relation (Vanden Berk+2004)
# Used for DRW sigma excess scoring component
# ---------------------------------------------------------------------------
SIGMA_EXCESS_NORM_FLUX_MJY = 1.0    # mJy reference flux at z=0.1, W1=14 mag
SIGMA_EXCESS_LUM_SLOPE     = -0.5   # sigma ∝ L^(-0.5)

# ---------------------------------------------------------------------------
# Output format
# ---------------------------------------------------------------------------
OUTPUT_CSV_FLOAT_FORMAT = '%.6f'

# ---------------------------------------------------------------------------
# config-H3: Runtime schema verification for WISE tables
# ---------------------------------------------------------------------------
# Minimal columns required from each IRSA table — hard-fail if missing.
WISE_TABLE_REQUIRED_COLS = {
    'allwise_p3as_mep': [
        'ra', 'dec', 'w1mpro', 'w1sigmpro', 'w2mpro', 'w2sigmpro',
        'cc_flags', 'qi_fact', 'mjd',
    ],
    'neowiser_p1bs_psd': [
        'ra', 'dec', 'w1mpro', 'w1sigmpro', 'w2mpro', 'w2sigmpro',
        'cc_flags', 'qi_fact', 'mjd',
    ],
    'wise_allsky_4band_p1bs_psd': [
        'ra', 'dec', 'w1mpro', 'w1sigmpro', 'w2mpro', 'w2sigmpro',
        'cc_flags', 'qi_fact', 'mjd',
    ],
    'wise_3band_p1bs_psd': [
        'ra', 'dec', 'w1mpro', 'w1sigmpro', 'w2mpro', 'w2sigmpro',
        'cc_flags', 'qi_fact', 'mjd',
    ],
    'wise_postcryo': [
        'ra', 'dec', 'w1mpro', 'w1sigmpro', 'w2mpro', 'w2sigmpro',
        'cc_flags', 'qi_fact', 'mjd',
    ],
}

# Optional columns — warn if missing but don't hard-fail.
WISE_TABLE_OPTIONAL_COLS = {
    'allwise_p3as_mep':          ['moon_masked', 'saa_sep', 'qual_frame', 'w3mpro', 'w4mpro'],
    'neowiser_p1bs_psd':         ['moon_masked', 'saa_sep', 'qual_frame'],
    'wise_allsky_4band_p1bs_psd': ['moon_masked', 'saa_sep', 'nb', 'w3mpro', 'w4mpro'],
    'wise_3band_p1bs_psd':       ['moon_masked', 'saa_sep', 'nb', 'w3mpro'],
    'wise_postcryo':             ['moon_masked', 'saa_sep', 'nb'],
}

# ---------------------------------------------------------------------------
# WISE complete dataset tables (Expansion 1)
# ---------------------------------------------------------------------------
WISE_ALLSKY_TABLE     = "wise_allsky_4band_p1bs_psd"
WISE_3BAND_TABLE      = "wise_3band_p1bs_psd"
WISE_POSTCRYO_TABLE   = "wise_postcryo"
WISE_ALLSKY_MJD_END   = 55415.0    # 2010-08-06
WISE_3BAND_MJD_END    = 55468.0    # 2010-09-29
WISE_POSTCRYO_MJD_END = 55593.0    # 2011-02-01
WISE_FULL_BASELINE_START_MJD = 55210.0  # 2010-01-14

# ---------------------------------------------------------------------------
# Expansion 3: MCMC
# ---------------------------------------------------------------------------
DRW_MCMC_N_WALKERS    = 64
DRW_MCMC_N_STEPS      = 3000
DRW_MCMC_N_BURN       = 1000
DRW_GR_THRESHOLD      = 1.1       # Gelman-Rubin convergence

# ---------------------------------------------------------------------------
# Expansion 4: Physical parameters
# ---------------------------------------------------------------------------
KAPPA_IR_BOLOMETRIC   = 8.0       # IR bolometric correction (Richards+2006)
KAPPA_5100_BOLOMETRIC = 10.3      # 5100A bolometric correction
KAPPA_X_BOLOMETRIC    = 20.0      # X-ray bolometric correction (Lusso+2012)
DRW_MASS_SCALING_A    = 2.4       # log(tau/days) = A + B*log(L44) + C*log(M8) (Kelly+2009)
DRW_MASS_SCALING_B    = 0.17
DRW_MASS_SCALING_C    = 0.038

# ---------------------------------------------------------------------------
# Expansion 5: Validation
# ---------------------------------------------------------------------------
N_INJECTIONS          = 200
N_FP_SIMULATIONS      = 1000
INJECTION_AMPLITUDES  = [0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]  # delta_mag
N_JACKKNIFE_SCRAMBLES = 200

# ---------------------------------------------------------------------------
# Expansion 6: Score v2 weights (10 components)
# ---------------------------------------------------------------------------
SCORE_WEIGHTS_V2 = {
    'drw_nonstationarity':  3.0,
    'broken_drw_bic':       3.0,   # NEW
    'nonstationary_gp':     2.5,   # NEW
    'delta_mag_w1':         2.5,
    'changepoint_bic':      2.5,
    'sf_excess':            2.0,
    'color_evolution':      1.5,
    'flux_bimodality':      1.5,   # NEW
    'gaia_variability':     1.5,   # upgraded weight
    'drw_sigma_excess':     1.0,
}
MAX_SCORE_V2 = sum(SCORE_WEIGHTS_V2.values())  # Fix #24: computed, not hardcoded

# ---------------------------------------------------------------------------
# Expansion 7: Score v3 group weights (score-M12)
# ---------------------------------------------------------------------------
SCORE_GROUP_WEIGHTS_V3 = {
    'amplitude':   3.0,
    'temporal':    3.0,
    'color':       2.0,
    'statistical': 1.5,
    'astrometric': 0.5,
}

# DRW sign-check strictness: if True, raise on inconsistency; if False, log warning
DRW_STRICT_SIGN_CHECK = False


# ---------------------------------------------------------------------------
# Fix #18: Delta-mag / flux-ratio helpers (sign convention enforced)
# ---------------------------------------------------------------------------

def delta_mag_from_flux_ratio(flux_ratio):
    """Return delta_mag = +2.5 * log10(flux_ratio). flux_ratio > 0 required."""
    if flux_ratio <= 0:
        raise ValueError(f"flux_ratio must be positive, got {flux_ratio}")
    return 2.5 * np.log10(flux_ratio)


def flux_ratio_from_delta_mag(delta_mag):
    """Return flux_ratio = 10^(delta_mag / 2.5)."""
    return 10.0 ** (delta_mag / 2.5)


def delta_mag_from_fluxes(F_early, F_late):
    """Return delta_mag = +2.5 * log10(F_late / F_early). Both fluxes > 0 required."""
    if F_early <= 0 or F_late <= 0:
        raise ValueError(
            f"Both fluxes must be positive, got F_early={F_early}, F_late={F_late}"
        )
    return delta_mag_from_flux_ratio(F_late / F_early)


# ---------------------------------------------------------------------------
# R2 FIX 12: Threshold provenance grouping
# ---------------------------------------------------------------------------

# --- Literature-derived thresholds ---
# Ricci & Trakhtenbrot 2022 (arXiv:2211.05132): min amplitude for CLAGN
CLAGN_MIN_DELTA_MAG = 0.3          # mag; |delta_mag| >= 0.3 for CLAGN candidacy
# Stern et al. 2012 (ApJ 753 30): AGN wedge W1-W2 color cut (Vega)
CLAGN_W1_W2_AGN_MIN = 0.8          # W1-W2 Vega mag
# Kozlowski et al. 2017 (arXiv:1611.08248): DRW reliability baseline criterion
DRW_MIN_BASELINE_X_TAU = 10.0      # baseline_rest >= 10 × tau_rest for reliable fit

# --- Engineering/empirical thresholds ---
# Calibrated to WISE survey cadence: ~6-month visits, hibernate gap ~ 1.3 yr
SEASON_GAP_DAYS = 120.0            # min inter-observation gap (days) to split seasons
MIN_EPOCHS_PER_SEASON = 3          # min epochs per season for median to be robust
MIN_SEASONS_REQUIRED = 4           # min seasons for delta_mag computation
# Artifact rejection (score-H8): single-epoch driven threshold
ARTIFACT_AMPLITUDE_DROP = 0.40     # fraction: if removing worst epoch reduces |dm| by this, flag artifact
# W1/W2 coherence: direction is indeterminate when fractional change < this
COHERENCE_FLATNESS_THRESH = 0.02   # fractional flux change threshold for "flat" season

# ---------------------------------------------------------------------------
# Fix #22: Source ID type convention helpers
# ---------------------------------------------------------------------------

def normalize_source_id(sid):
    """Always return string. Use for all internal dict keys and CSV values."""
    return str(sid).strip()


def normalize_gaia_id(gaia_sid):
    """Always return int. Use when passing to GAIA queries."""
    return int(gaia_sid)


# ---------------------------------------------------------------------------
# config-C1: Programmatic validation of all critical constants
# ---------------------------------------------------------------------------

def validate_config():
    """
    Validate critical pipeline constants at startup.

    Called at the top of main.py to hard-fail on misconfiguration.
    Replaces 'must match exactly' comments with actual enforcement.

    Raises
    ------
    ValueError if any critical constant is wrong.

    Returns
    -------
    True if all checks pass.
    """
    errors = []

    # WISE zero points (Wright+2010, Table 1) — authoritative values
    W1_ZP_EXPECTED = 309.540   # Jy
    W2_ZP_EXPECTED = 171.787   # Jy
    if abs(WISE_W1_ZERO_POINT_JY - W1_ZP_EXPECTED) > 0.001:
        errors.append(
            f"WISE_W1_ZERO_POINT_JY={WISE_W1_ZERO_POINT_JY} "
            f"!= Wright+2010 value {W1_ZP_EXPECTED}"
        )
    if abs(WISE_W2_ZERO_POINT_JY - W2_ZP_EXPECTED) > 0.001:
        errors.append(
            f"WISE_W2_ZERO_POINT_JY={WISE_W2_ZERO_POINT_JY} "
            f"!= Wright+2010 value {W2_ZP_EXPECTED}"
        )

    # MAX_SCORE_V2 must equal sum of SCORE_WEIGHTS_V2
    max_score_computed = sum(SCORE_WEIGHTS_V2.values())
    if abs(max_score_computed - MAX_SCORE_V2) > 0.01:
        errors.append(
            f"MAX_SCORE_V2={MAX_SCORE_V2} != sum(SCORE_WEIGHTS_V2)={max_score_computed:.3f}"
        )

    # Season anchor must be before first WISE data
    if WISE_SEASON_ANCHOR_MJD > 55200.0:
        errors.append(
            f"WISE_SEASON_ANCHOR_MJD={WISE_SEASON_ANCHOR_MJD} > 55200 "
            f"(WISE first light). Anchor must precede all data."
        )

    # Hibernation dates must be logically consistent
    if WISE_HIBERNATION_START_MJD >= WISE_HIBERNATION_END_MJD:
        errors.append(
            f"WISE_HIBERNATION_START_MJD={WISE_HIBERNATION_START_MJD} >= "
            f"WISE_HIBERNATION_END_MJD={WISE_HIBERNATION_END_MJD} — invalid range"
        )

    if errors:
        raise ValueError(
            "Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    return True
