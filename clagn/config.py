"""
config.py — All scientific constants and configuration for the CLAGN pipeline.

Every number used in the pipeline must be defined here.
Never hardcode values in other modules.

References:
    Jarrett et al. 2011 (WISE zero points)
    Stern et al. 2012, ApJ 753 30 (WISE AGN color cut)
    Ricci & Trakhtenbrot 2022, arXiv:2211.05132
    Kelly et al. 2009, ApJ 698 895 (DRW model)
"""

# ---------------------------------------------------------------------------
# WISE Vega zero points (Jarrett et al. 2011)
# ---------------------------------------------------------------------------
WISE_ZERO_POINTS = {
    'W1': 309.540,   # Jy
    'W2': 171.787,   # Jy
    'W3': 31.674,    # Jy
    'W4': 8.363      # Jy
}

# ---------------------------------------------------------------------------
# Quality thresholds
# ---------------------------------------------------------------------------
MIN_BASELINE_YEARS     = 10.0    # Minimum light curve baseline (Ricci+2022: CS transitions need decades)
MIN_EPOCHS             = 20      # Minimum WISE single-exposure epochs
MIN_EPOCHS_PRE_GAP     = 5       # Minimum epochs from AllWISE (pre-hibernation)
MIN_EPOCHS_POST_GAP    = 5       # Minimum epochs from NEOWISE-R (post-hibernation)
MIN_REDSHIFT           = 0.002   # z < 0.002 = likely Galactic, not extragalactic AGN
MAX_REDSHIFT           = 5.0     # Beyond this WISE W1/W2 probe rest-frame UV, not IR torus
MAX_PROPER_MOTION_SIG  = 3.0     # Sigma threshold for star rejection via GAIA PM
MAX_RUWE               = 1.4     # GAIA astrometric quality (point source requirement)

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
GAIA_SEARCH_RADIUS_ARCSEC  = 1.5

# ---------------------------------------------------------------------------
# WISE gap: satellite was hibernating (do not interpolate across this)
# ---------------------------------------------------------------------------
WISE_HIBERNATION_MJD_START = 55593   # 2011-02-17
WISE_HIBERNATION_MJD_END   = 56141   # 2012-08-08

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
