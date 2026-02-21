# CLAGN Detection Pipeline — Research-Grade Claude Code Prompt
### Scientific basis: Ricci & Trakhtenbrot 2022 (arXiv:2211.05132)

---

## MISSION

Rebuild the entire CLAGN detection codebase from scratch as a **research-grade, modular Python package**. Assume all existing code is wrong or incomplete. The finished product must be capable of identifying Changing-Look AGN candidates at a level of scientific rigor suitable for an observational astronomy paper. The only photometric data sources are **WISE multi-epoch infrared** and **GAIA DR3 astrometric/optical**. Output the **top 6 CLAGN candidates** ranked by a physically motivated composite score.

This is not a data science exercise. Every statistical and algorithmic decision must be grounded in AGN physics.

---

## FOLDER STRUCTURE

Build a clean Python package, not a single script:

```
clagn/
├── main.py                  # CLI entry point — argparse, orchestrates all steps
├── config.py                # All constants, thresholds, WISE zero points, column names
├── ingestion/
│   ├── __init__.py
│   ├── catalog.py           # Load input catalog, validate redshifts, apply AGN pre-selection
│   ├── wise.py              # WISE AllWISE + NEOWISE-R multi-epoch queries and cleaning
│   └── gaia.py              # GAIA DR3 queries, proper motion filter, astrometric QA
├── models/
│   ├── __init__.py
│   ├── drw.py               # Full Damped Random Walk GP fitting with MCMC posterior
│   ├── structure_function.py # SF(Δt) computation with bootstrapped uncertainties
│   └── changepoint.py       # Bayesian structural break detection
├── scoring/
│   ├── __init__.py
│   └── clagn_score.py       # 7-component composite CLAGN scoring engine
├── output/
│   ├── __init__.py
│   ├── plots.py             # Publication-quality light curve and diagnostic plots
│   └── tables.py            # CSV/FITS output with all required columns
└── utils/
    ├── __init__.py
    ├── photometry.py        # Mag-to-flux conversion, sigma clipping, epoch weighting
    └── crossmatch.py        # Positional cross-matching utilities
```

---

## MODULE SPECIFICATIONS

---

### `config.py`

Define ALL scientific constants here. Never hardcode values in other modules.

```python
# WISE Vega zero points (Jarrett et al. 2011)
WISE_ZERO_POINTS = {
    'W1': 309.540,   # Jy
    'W2': 171.787,   # Jy
    'W3': 31.674,    # Jy
    'W4': 8.363      # Jy
}

# Quality thresholds
MIN_BASELINE_YEARS     = 10.0    # Minimum light curve baseline (Ricci+2022: CS transitions need decades)
MIN_EPOCHS             = 20      # Minimum WISE single-exposure epochs
MIN_REDSHIFT           = 0.002   # z < 0.002 = likely Galactic, not extragalactic AGN
MAX_REDSHIFT           = 5.0     # Beyond this WISE W1/W2 probe rest-frame UV, not IR torus
MAX_PROPER_MOTION_SIG  = 3.0     # Sigma threshold for star rejection via GAIA PM
MAX_RUWE               = 1.4     # GAIA astrometric quality (point source requirement)

# AGN WISE color selection (Stern et al. 2012, ApJ 753 30)
# W1-W2 > 0.8 (Vega) selects AGN with >95% reliability
WISE_AGN_COLOR_CUT     = 0.8     # W1-W2 Vega magnitudes

# DRW model priors (log-space, physically motivated for AGN)
DRW_LOG_TAU_MIN        = 1.0     # ln(days), tau > e^1 ~ 3 days
DRW_LOG_TAU_MAX        = 8.5     # ln(days), tau < e^8.5 ~ 5000 days
DRW_LOG_SIGMA_MIN      = -5.0    # ln(flux amplitude)
DRW_LOG_SIGMA_MAX      = 5.0

# Scoring weights (tuned to maximize separation from normal AGN variability)
SCORE_WEIGHTS = {
    'drw_nonstationarity': 3.0,   # Most important: deviation from stationary DRW
    'delta_mag_w1':        2.5,   # Raw flux change amplitude
    'changepoint_bic':     2.5,   # Bayesian evidence for structural break
    'sf_break':            2.0,   # Structure function slope flattening at long lags
    'color_evolution':     1.5,   # W1-W2 color change (dust temperature/accretion)
    'drw_sigma_excess':    1.5,   # Variability amplitude vs luminosity expectation
    'gaia_variability':    1.0,   # Supporting optical evidence from GAIA
}

# Cone search radii
WISE_SEARCH_RADIUS_ARCSEC  = 3.0
GAIA_SEARCH_RADIUS_ARCSEC  = 1.5

# WISE gap: satellite was hibernating (do not interpolate across this)
WISE_HIBERNATION_MJD_START = 55593   # 2011-02-17
WISE_HIBERNATION_MJD_END   = 56141   # 2012-08-08

# Sigma clipping for outlier rejection before DRW fit
SIGMA_CLIP_SIGMA  = 4.0
SIGMA_CLIP_ITERS  = 3

# Minimum flux change to be a serious CLAGN candidate (Ricci+2022: factor >= 2)
MIN_FLUX_RATIO_CHANGE = 2.0   # max_flux / min_flux (rolling 1-year medians)
MIN_MAG_CHANGE_W1     = 0.3   # mag — conservative lower bound; strong CLAGN show > 0.75 mag
```

---

### `ingestion/catalog.py`

**Redshift validation — apply immediately, log every removal:**
```python
def load_and_validate_catalog(filepath):
    """
    Load source catalog, apply all pre-flight quality filters.
    Supports CSV and FITS formats.
    Returns clean DataFrame with standardized column names.
    """
    # Load
    # Standardize columns: must have ra, dec, source_id, redshift
    
    n_start = len(df)
    
    # Step 1: Remove NaN redshifts
    df = df[df['redshift'].notna()]
    log(f"Removed {n_start - len(df)} sources with NaN redshift")
    
    # Step 2: Remove negative and zero redshifts
    df = df[df['redshift'] > MIN_REDSHIFT]
    log(f"Removed sources with z <= {MIN_REDSHIFT} (Galactic foreground)")
    
    # Step 3: Remove unphysical high-z
    df = df[df['redshift'] < MAX_REDSHIFT]
    log(f"Removed sources with z >= {MAX_REDSHIFT} (WISE W1/W2 probes UV at high-z, unreliable)")
    
    # Step 4: Require valid sky coordinates
    df = df[df['ra'].between(0, 360) & df['dec'].between(-90, 90)]
    
    log(f"Final catalog: {len(df)} sources after validation (started with {n_start})")
    return df
```

**AGN pre-selection using WISE colors:**
Before any variability analysis, confirm each source is an AGN using the Stern et al. (2012) WISE color cut. This dramatically reduces false positives from non-AGN variable sources:
```python
def apply_wise_agn_color_selection(df):
    """
    Apply W1-W2 > 0.8 mag (Vega) AGN color cut (Stern+2012).
    This selects AGN with >95% reliability and removes:
    - Normal galaxies (W1-W2 ~ 0)
    - Stars (W1-W2 < 0.5)
    - Starburst galaxies
    Sources without AllWISE colors are retained with a warning flag.
    """
```

---

### `ingestion/wise.py`

**This is the most critical module. Get every detail right.**

Query strategy — use TWO separate datasets and merge them to maximize baseline:
1. **AllWISE Multi-Epoch Photometry Table (MEPT)** — 2010-2011 (pre-hibernation), ~10 exposures per source
2. **NEOWISE-R Single Exposure Source Table** — 2013-present, ~100-200 exposures per source

Together these give 10+ year baselines for most sources. The gap (2011-2013) must be handled explicitly.

```python
def query_wise_lightcurve(ra, dec, source_id):
    """
    Query both AllWISE-MEPT and NEOWISE-R for a single source.
    Returns merged, cleaned, calibrated light curve as DataFrame.
    
    Columns returned: mjd, w1_mag, w1_err, w2_mag, w2_err,
                      w1_flux_mjy, w1_flux_err_mjy, w2_flux_mjy, w2_flux_err_mjy,
                      epoch_weight, dataset_flag ('allwise' or 'neowise')
    """
```

**WISE quality filtering — apply ALL of these, no exceptions:**
```python
WISE_QUALITY_FILTERS = {
    # NEOWISE-R filters
    'qi_fact':      lambda x: x == 1,           # Quality factor = 1 (good observation)
    'saa_sep':      lambda x: x > 5,            # > 5 deg from South Atlantic Anomaly
    'moon_masked':  lambda x: x == 0,           # Not moon-masked
    'cc_flags':     lambda x: x == '0000',      # No contamination/confusion flags
    'ph_qual_w1':   lambda x: x in ('A','B'),   # Only A or B quality detections
    'ph_qual_w2':   lambda x: x in ('A','B'),
    'w1snr':        lambda x: x > 5,            # SNR > 5 in W1
    'w2snr':        lambda x: x > 5,            # SNR > 5 in W2
    'nb':           lambda x: x <= 2,           # Not confused with nearby sources
}
```

**WISE gap handling:**
```python
def flag_wise_gap(mjd_array):
    """
    Flag epochs that fall within the WISE hibernation period (2011-02-17 to 2012-08-08).
    These should never exist in real data — if they do, something is wrong with the query.
    Raise a warning if any epochs fall in this window.
    """
    gap_mask = (mjd_array > WISE_HIBERNATION_MJD_START) & \
               (mjd_array < WISE_HIBERNATION_MJD_END)
    if gap_mask.any():
        warnings.warn(f"Found {gap_mask.sum()} epochs in WISE hibernation window — removing")
    return mjd_array[~gap_mask]
```

**Magnitude to flux conversion — always do this in flux space for DRW fitting:**
```python
def mag_to_flux_mjy(mag, mag_err, band):
    """
    Convert Vega magnitudes to flux density in mJy.
    Uses WISE zero points from config.
    Propagates errors correctly.
    
    flux_mjy = F0_jy * 10^(-mag/2.5) * 1000
    flux_err_mjy = flux_mjy * ln(10)/2.5 * mag_err
    """
    F0 = WISE_ZERO_POINTS[band]  # Jy
    flux_jy = F0 * 10**(-mag / 2.5)
    flux_mjy = flux_jy * 1000.0
    flux_err_mjy = flux_mjy * (np.log(10) / 2.5) * mag_err
    return flux_mjy, flux_err_mjy
```

**Epoch quality weighting:**
Not all WISE epochs are equally reliable. Assign per-epoch weights:
```python
def compute_epoch_weights(snr_w1, snr_w2, moon_sep, ecl_lat):
    """
    Compute inverse-variance weights for each WISE epoch, accounting for:
    - SNR (higher SNR = higher weight)
    - Moon proximity (closer to moon = downweight)
    - Ecliptic latitude (near ecliptic = more visits = not higher quality per visit)
    
    Returns normalized weights array.
    """
    snr_weight = 0.5 * (snr_w1 + snr_w2)
    moon_weight = np.clip((moon_sep - 10) / 90, 0.1, 1.0)
    return snr_weight * moon_weight
```

**Host galaxy contamination check at low redshift:**
```python
def check_host_contamination(z, w1_flux_mjy):
    """
    At z < 0.05, the WISE PSF (6 arcsec FWHM) can blend AGN and host galaxy light.
    Flag sources where host contamination may inflate apparent variability.
    
    At z < 0.05: flag as 'host_contamination_risk = True'
    This doesn't exclude the source but adds a warning to the output.
    
    Future improvement: use 2MASS extended source photometry to subtract host contribution.
    """
    risk = z < 0.05
    return risk
```

**Baseline validation:**
```python
def validate_baseline(mjd_array, source_id):
    """
    Compute and validate the temporal baseline.
    
    Rules:
    1. Baseline = (MJD_max - MJD_min) / 365.25
    2. Must be >= MIN_BASELINE_YEARS (10.0)
    3. Must span BOTH pre-gap (AllWISE) and post-gap (NEOWISE-R) epochs
       — a source with only NEOWISE-R data has a max baseline of ~2013-present
       which may or may not reach 10 years depending on the run date
    4. Must have >= MIN_EPOCHS total usable exposures
    
    Returns: baseline_years, passes_baseline (bool), n_epochs_pre_gap, n_epochs_post_gap
    """
```

---

### `ingestion/gaia.py`

```python
def query_gaia_dr3(ra, dec, source_id):
    """
    Query GAIA DR3 for astrometric and photometric properties.
    Uses astroquery.gaia with ADQL for efficiency.
    
    Returns dict with all fields needed for filtering and output.
    """
    query = f"""
    SELECT source_id, ra, dec, pmra, pmra_error, pmdec, pmdec_error,
           ruwe, astrometric_excess_noise, astrometric_excess_noise_sig,
           phot_g_mean_mag, phot_g_mean_flux_over_error,
           phot_variable_flag, non_single_star,
           classprob_dsc_combmod_quasar, classprob_dsc_combmod_galaxy
    FROM gaiadr3.gaia_source
    WHERE CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra}, {dec}, {GAIA_SEARCH_RADIUS_ARCSEC}/3600.0)
    ) = 1
    ORDER BY phot_g_mean_flux_over_error DESC
    """
```

**Proper motion significance filter — this is the star rejection step:**
```python
def compute_pm_significance(pmra, pmra_err, pmdec, pmdec_err):
    """
    Compute total proper motion significance in sigma.
    
    A genuine extragalactic AGN has zero proper motion.
    Stars at any distance have measurable proper motion.
    
    pm_total = sqrt(pmra^2 + pmdec^2)
    pm_err   = sqrt((pmra*pmra_err)^2 + (pmdec*pmdec_err)^2) / pm_total
    pm_sig   = pm_total / pm_err
    
    REJECT if pm_sig >= MAX_PROPER_MOTION_SIG (3.0 sigma)
    
    Note: handle NaN proper motions (GAIA may not measure PM for faint/crowded sources)
    — NaN PM means GAIA couldn't measure it, NOT that it's zero. Treat as unknown,
    do not auto-reject, but flag as 'gaia_pm_unknown = True'.
    """
```

**GAIA astrometric quality for point source confirmation:**
```python
def check_point_source_quality(ruwe, astrometric_excess_noise_sig):
    """
    Confirm the source is a point source (AGN) not an extended galaxy or blend.
    
    RUWE (Renormalised Unit Weight Error):
    - RUWE ~ 1.0: single point source, clean astrometry
    - RUWE > 1.4: extended, blended, or binary — REJECT
    
    astrometric_excess_noise_sig:
    - < 2: consistent with point source — ACCEPT
    - >= 2: excess astrometric noise, possible extended source — flag
    
    Returns: passes_quality (bool), quality_flag (str)
    """
```

**GAIA quasar probability:**
GAIA DR3 includes DSC (Discrete Source Classifier) quasar probabilities. Use this as an additional AGN confirmation:
```python
def check_gaia_quasar_classification(classprob_quasar):
    """
    GAIA DSC quasar probability > 0.5 provides strong independent AGN confirmation.
    Flag sources with classprob_quasar < 0.1 as potentially misclassified.
    This is advisory only — do not reject based on this alone.
    """
```

---

### `models/drw.py`

**This is the scientific core of the pipeline. Implement it rigorously.**

The Damped Random Walk (DRW) is an Ornstein-Uhlenbeck process — the standard model for AGN stochastic optical/IR variability (Kelly et al. 2009, Kozłowski et al. 2010, MacLeod et al. 2010). Its power spectral density is a Lorentzian, transitioning from pink noise (∝ f^-2) at high frequencies to white noise at low frequencies, with a characteristic break at f = 1/(2πτ).

**The DRW covariance kernel:**
```
k(dt) = σ_DRW² × exp(-|dt| / τ)
```
where:
- `τ` = characteristic damping timescale (days) — rest-frame: τ_rest = τ_obs / (1 + z)
- `σ_DRW` = amplitude of variability (same units as flux)

```python
def fit_drw_map(times, fluxes, flux_errors, z=0.0):
    """
    Fit DRW model via Maximum A Posteriori (MAP) optimization.
    Used as fast first-pass; results feed into MCMC for final candidates.
    
    IMPORTANT: Convert to rest-frame times before fitting:
        times_rest = times_obs / (1 + z)
    This is critical — observed timescales are stretched by (1+z).
    
    Parameters
    ----------
    times       : array, MJD (observer frame)
    fluxes      : array, flux density in mJy
    flux_errors : array, flux uncertainty in mJy
    z           : float, source redshift
    
    Returns
    -------
    tau_days    : float, rest-frame DRW timescale
    sigma_drw   : float, DRW amplitude
    log_like    : float, log-likelihood at MAP
    residuals   : array, standardized GP residuals (should be ~N(0,1) for good fit)
    pred_mean   : array, GP posterior mean at input times (for plotting)
    pred_std    : array, GP posterior std at input times (for plotting)
    """
    # Apply redshift correction
    times_rest = times / (1.0 + z)
    
    # Use celerite2 if available, otherwise fall back to numpy GP
    try:
        import celerite2
        return _fit_drw_celerite2(times_rest, fluxes, flux_errors)
    except ImportError:
        return _fit_drw_numpy(times_rest, fluxes, flux_errors)
```

```python
def fit_drw_mcmc(times, fluxes, flux_errors, z=0.0, n_walkers=32, n_steps=2000, n_burn=500):
    """
    Full MCMC posterior sampling of DRW parameters using emcee.
    Run this ONLY on the top candidates identified by MAP fitting.
    
    Priors (log-uniform, physically motivated):
        ln(τ) ~ Uniform(DRW_LOG_TAU_MIN, DRW_LOG_TAU_MAX)
        ln(σ) ~ Uniform(DRW_LOG_SIGMA_MIN, DRW_LOG_SIGMA_MAX)
    
    Returns
    -------
    tau_median, tau_16, tau_84    : float, DRW timescale with 1σ credible interval
    sigma_median, sigma_16, sigma_84 : float, DRW amplitude with 1σ credible interval
    samples                       : array (n_samples, 2), full posterior chain
    acceptance_fraction           : float, MCMC acceptance rate (should be 0.2-0.5)
    """
```

```python
def compute_drw_nonstationarity(times, fluxes, flux_errors, z, tau, sigma_drw):
    """
    Test whether the light curve is consistent with a STATIONARY DRW.
    
    CLAGN are by definition NON-STATIONARY — their mean flux level changes.
    A stationary DRW has constant mean and variance.
    
    Method: Split light curve into first half and second half (by time).
    Compute the mean flux in each half. Test whether the difference is
    significant relative to what a stationary DRW would predict.
    
    DRW variance at lag T: Var(T) = σ² × (1 - exp(-T/τ))
    For T >> τ: Var → σ² (the process variance)
    The expected scatter between the two half-means under DRW:
        σ_expected = σ_DRW × sqrt(2/N_half) × correction_factor(T_half, τ)
    
    Returns
    -------
    nonstationarity_sigma : float
        How many sigma the observed mean shift exceeds DRW expectation.
        > 3σ: strong CLAGN evidence
        > 5σ: very strong CLAGN evidence
    delta_mean_normalized : float
        (mean_late - mean_early) / sigma_expected
    """
```

**Fallback DRW using numpy (no external GP library needed):**
```python
def _fit_drw_numpy(times, fluxes, flux_errors):
    """
    Manual DRW GP implementation using numpy.
    Covariance matrix: K_ij = σ² × exp(-|t_i - t_j| / τ) + diag(err²)
    Log-likelihood: -0.5 × [y^T K^{-1} y + ln|K| + N×ln(2π)]
    
    Uses scipy.optimize.minimize with L-BFGS-B for MAP.
    Uses Cholesky decomposition for stable matrix inversion.
    
    NOTE: O(N³) complexity — for N > 500 epochs, subsample or use celerite2.
    """
    from scipy.optimize import minimize
    from scipy.linalg import cho_factor, cho_solve
    
    def log_likelihood(params):
        log_sigma, log_tau = params
        sigma = np.exp(log_sigma)
        tau   = np.exp(log_tau)
        
        # Prior bounds
        if not (DRW_LOG_TAU_MIN < log_tau < DRW_LOG_TAU_MAX):
            return 1e10
        if not (DRW_LOG_SIGMA_MIN < log_sigma < DRW_LOG_SIGMA_MAX):
            return 1e10
        
        dt = np.abs(times[:, None] - times[None, :])
        K = sigma**2 * np.exp(-dt / tau)
        K += np.diag(flux_errors**2 + 1e-12)
        
        try:
            c, low = cho_factor(K)
            alpha = cho_solve((c, low), fluxes)
            sign, logdet = np.linalg.slogdet(K)
            return 0.5 * (np.dot(fluxes, alpha) + logdet + len(fluxes) * np.log(2*np.pi))
        except np.linalg.LinAlgError:
            return 1e10
    
    # Grid initialization to avoid local minima
    best_ll, best_p0 = np.inf, None
    for log_sigma0 in np.linspace(-2, 2, 4):
        for log_tau0 in np.linspace(4, 7, 4):
            ll = log_likelihood([log_sigma0, log_tau0])
            if ll < best_ll:
                best_ll, best_p0 = ll, [log_sigma0, log_tau0]
    
    result = minimize(log_likelihood, best_p0, method='L-BFGS-B',
                      bounds=[(DRW_LOG_SIGMA_MIN, DRW_LOG_SIGMA_MAX),
                              (DRW_LOG_TAU_MIN, DRW_LOG_TAU_MAX)])
    ...
```

---

### `models/structure_function.py`

The Structure Function (SF) is a core diagnostic for AGN variability and CLAGN detection. It measures variability amplitude as a function of time lag. Normal AGN follow a power-law SF with a characteristic slope. CLAGN deviate by showing a **plateau or slope break** at long lags — the signature of a coherent, non-stochastic state change.

```python
def compute_structure_function(times, mags, mag_errors, n_lag_bins=20):
    """
    Compute the observed Structure Function SF(Δt) for a light curve.
    
    SF(Δt) = <(m(t+Δt) - m(t))²> - <2σ²>
    
    where the second term removes the contribution of measurement noise.
    
    Uses log-spaced lag bins from Δt_min = cadence to Δt_max = baseline.
    Bootstrap uncertainties: 1000 bootstrap resamples per lag bin.
    
    Parameters
    ----------
    times      : array, MJD
    mags       : array, magnitudes (NOT fluxes — SF is defined in mag space)
    mag_errors : array, magnitude uncertainties
    
    Returns
    -------
    lag_centers  : array, center of each lag bin (days)
    sf_values    : array, SF(Δt) values
    sf_errors    : array, bootstrap 1σ uncertainties on SF
    sf_slope     : float, power-law slope fitted to short lags (Δt < τ_DRW)
    sf_plateau   : float, SF amplitude at long lags (Δt > 3×τ_DRW)
    """
```

```python
def detect_sf_break(lag_centers, sf_values, sf_errors, tau_drw):
    """
    Detect a slope break in the structure function — the key CLAGN signature.
    
    Normal stochastic AGN: SF ∝ Δt^β with β ~ 0.3-0.5 at all lags,
    then flattening at Δt >> τ_DRW (stationary DRW plateau).
    
    CLAGN: SF shows excess power at long lags compared to DRW prediction,
    OR shows a sudden step change (δ-function in SF derivative) corresponding
    to the epoch of the state change.
    
    Method:
    1. Fit a broken power law to SF(Δt)
    2. Compare BIC to single power law
    3. Measure SF_observed(Δt_max) / SF_DRW_expected(Δt_max)
       — ratio >> 1 indicates excess long-lag variability (CLAGN)
    
    Returns
    -------
    sf_excess_ratio      : float, SF_obs / SF_DRW at longest lag
    break_detected       : bool
    break_lag_days       : float, lag at which break occurs (if detected)
    sf_break_significance: float, sigma significance of break
    """
```

---

### `models/changepoint.py`

```python
def bayesian_changepoint_detection(times, fluxes, flux_errors):
    """
    Rigorous Bayesian detection of a structural break in the light curve mean.
    
    Model comparison:
    - M0 (null): single stationary mean μ, variance σ²
    - M1 (CLAGN): two segments with means μ1, μ2; break at time t_break
    
    For each candidate break time t_break in (0.15*T, 0.85*T):
        Compute marginal log-likelihood of M1 analytically
        (conjugate Normal-inverse-Gamma prior on μ, σ²)
    
    BIC-corrected evidence: ΔBIC = BIC(M0) - BIC(M1)
        ΔBIC > 6:  strong evidence for structural break
        ΔBIC > 10: very strong evidence (Kass & Raftery 1995)
    
    Returns
    -------
    best_break_mjd     : float, MJD of most likely state change
    delta_bic          : float, BIC improvement from allowing a break
    break_significance : float, posterior probability that break is real
    pre_break_mean     : float, mean flux before break (mJy)
    post_break_mean    : float, mean flux after break (mJy)
    mean_ratio         : float, post/pre mean flux ratio (CLAGN show > 2)
    """
```

```python
def test_monotonic_trend(times, fluxes, flux_errors):
    """
    Test for a sustained monotonic flux trend — another CLAGN signature.
    
    Normal AGN: symmetric, mean-reverting variability (DRW)
    CLAGN turn-off: sustained decline over years
    CLAGN turn-on: sustained rise over years
    
    Uses Mann-Kendall trend test (non-parametric, robust to outliers)
    and Theil-Sen slope estimator.
    
    Returns
    -------
    mk_tau        : float, Mann-Kendall tau statistic (-1 to +1)
    mk_pvalue     : float, p-value for trend (< 0.01 = significant)
    trend_slope   : float, Theil-Sen slope (mJy/year)
    trend_dir     : str, 'rising', 'falling', or 'none'
    """
```

---

### `scoring/clagn_score.py`

```python
def compute_composite_clagn_score(source_record, lc_data, drw_results,
                                   sf_results, cp_results, gaia_data):
    """
    7-component composite CLAGN scoring engine.
    Each component is normalized to [0, 1] before weighting.
    Final score is a weighted sum.
    
    COMPONENTS:
    
    1. DRW Non-stationarity (weight 3.0)
       nonstationarity_sigma from drw.compute_drw_nonstationarity()
       Normalized: min(nonstationarity_sigma / 10.0, 1.0)
       Physical meaning: How inconsistent is the light curve with a
       stationary stochastic process? High = genuine state change.
    
    2. Delta Magnitude W1 (weight 2.5)
       max(|median_flux_early - median_flux_late|) / mean_flux
       Normalized by fractional change
       Physical meaning: Raw amplitude of long-term flux change.
       Ricci+2022: CS-AGN show factor >= 2 flux changes in WISE.
    
    3. Bayesian Changepoint Evidence (weight 2.5)
       delta_bic from changepoint.bayesian_changepoint_detection()
       Normalized: min(delta_bic / 20.0, 1.0)
       Physical meaning: Statistical evidence for a specific epoch
       of state change, not just general variability.
    
    4. Structure Function Break (weight 2.0)
       sf_excess_ratio from structure_function.detect_sf_break()
       Normalized: min((sf_excess_ratio - 1.0) / 4.0, 1.0)
       Physical meaning: Long-lag SF excess above DRW prediction
       indicates coherent, non-stochastic variability.
    
    5. W1-W2 Color Evolution (weight 1.5)
       |delta_color| = |(W1-W2)_late - (W1-W2)_early|
       Normalized: min(|delta_color| / 0.3, 1.0)
       Physical meaning: CS-AGN show color changes as hot dust responds
       to accretion state changes (Ricci+2022 §3.3.2).
       Color becoming bluer = turn-off (less hot dust)
       Color becoming redder = turn-on (more hot dust)
    
    6. DRW Sigma Excess vs Luminosity Expectation (weight 1.5)
       AGN variability anti-correlates with luminosity (Vanden Berk+2004).
       A source varying MORE than expected for its luminosity is anomalous.
       sigma_excess = sigma_DRW_observed / sigma_DRW_expected(L, z)
       Normalized: min((sigma_excess - 1.0) / 3.0, 1.0)
    
    7. GAIA Optical Variability (weight 1.0)
       Uses GAIA phot_variable_flag and G-band flux scatter.
       Binary: 1.0 if GAIA flagged VARIABLE, else 0.0
       Physical meaning: Correlated optical and IR variability
       is the hallmark of accretion-driven CS-AGN.
    
    Returns
    -------
    scores_dict : dict with all 7 component scores + composite
    composite   : float [0, 1], final CLAGN score
    label       : str, physical interpretation of candidate type
    """
```

```python
def classify_clagn_type(scores, drw_results, cp_results, lc_data):
    """
    Assign a physical interpretation label based on the Ricci & Trakhtenbrot
    (2022) CLAGN taxonomy.
    
    Classification logic:
    
    CS-AGN Turn-Off:
        - delta_mag_w1 > 0.5
        - post_break_mean < pre_break_mean (flux decreasing)
        - W1-W2 color becomes bluer (less hot dust, fading accretion)
        - DRW tau increases (longer correlation = slower variability)
        → Label: "CS-AGN Turn-Off (fading accretion, BLR likely disappearing)"
    
    CS-AGN Turn-On:
        - delta_mag_w1 > 0.5
        - post_break_mean > pre_break_mean (flux increasing)
        - W1-W2 color becomes redder (hotter dust, rising accretion)
        → Label: "CS-AGN Turn-On (rising accretion, BLR likely emerging)"
    
    Rapid Transition (possible TDE-in-AGN):
        - changepoint break duration < 2 years (estimated from BIC profile width)
        - delta_mag > 0.75
        → Label: "Rapid transition — possible TDE-in-AGN or magnetic disk event"
          (Ricci+2022 §3.6.2: TDEs in AGN can trigger CS-like events)
    
    CO-AGN candidate:
        - DRW tau < 100 days (short correlation = BLR cloud timescale)
        - Variability amplitude moderate (0.1-0.4 mag)
        - No sustained trend (Mann-Kendall non-significant)
        → Label: "CO-AGN candidate — possible BLR cloud eclipse"
    
    Unknown/Ambiguous:
        - Does not cleanly fit above criteria
        → Label: "CLAGN candidate — type ambiguous, spectroscopic followup required"
    """
```

```python
def apply_false_positive_rejection(candidates_df, lc_dict):
    """
    Apply false positive rejection logic before finalizing top 6.
    
    Reject or strongly downweight sources that show CLAGN-like photometry
    but are likely contaminants:
    
    1. Supernovae (SN) contamination:
       - SN rise in <60 days then decline over ~100-200 days
       - Flag: changepoint break duration < 100 days AND
               post_break_mean < pre_break_mean (after brief peak)
       - These sources have high delta_mag but are NOT CLAGN
    
    2. Stellar variability leakage (despite PM filter):
       - If GAIA classprob_quasar < 0.1 AND source is at |b| < 20 deg
         (Galactic latitude), apply a 0.5x penalty to composite score
    
    3. WISE artifacts:
       - Any source where variability is concentrated in a single WISE
         epoch (not confirmed by neighbors in time): downweight by 0.3x
       - Check: does removing the single most deviant epoch reduce
                delta_bic by > 50%? If yes, flag as artifact.
    
    4. Host galaxy contamination:
       - Source flagged as host_contamination_risk=True AND
         delta_mag < 0.5: downweight by 0.5x (marginal CLAGN may just be
         host galaxy variability)
    
    Returns updated candidates_df with contamination_flag and adjusted scores.
    """
```

---

### `output/plots.py`

**Publication-quality figures. Every plot must be camera-ready.**

```python
def plot_lightcurve_panel(source_id, times, w1_flux, w1_err, w2_flux, w2_err,
                           drw_pred_mean, drw_pred_std, drw_residuals,
                           w1_minus_w2, break_mjd, z, score, label, output_path):
    """
    4-panel publication-quality figure per CLAGN candidate.
    
    Panel 1 (top, largest): W1 and W2 flux vs MJD
        - W1 in blue circles, W2 in orange squares (with errorbars)
        - DRW posterior mean ± 1σ as shaded region
        - Vertical dashed line at best-fit changepoint break MJD
        - X-axis: both MJD (bottom) and calendar year (top)
        - Y-axis: Flux density (mJy)
        - Annotate AllWISE vs NEOWISE-R epochs with different markers
        - Mark WISE hibernation gap as gray shaded region
    
    Panel 2: W1-W2 color evolution
        - Color change with errorbars
        - Horizontal line at W1-W2 = 0.8 (AGN selection boundary)
        - Color direction (bluer/redder) annotated
    
    Panel 3: Standardized DRW residuals
        - Residuals = (flux - DRW_mean) / DRW_std
        - Horizontal lines at ±2σ and ±3σ
        - Should be ~N(0,1) for stationary DRW; CLAGN show outlier epochs
    
    Panel 4: Structure Function
        - SF(Δt) as filled circles with bootstrap error bars
        - DRW theoretical SF(Δt) = σ²(1 - exp(-Δt/τ)) as dashed line
        - Highlight lag range where SF excess is detected
    
    Title: "Source {source_id} | z={z:.3f} | Baseline={baseline:.1f} yr |
            Score={score:.3f} | {label}"
    
    Style: use matplotlib with seaborn-v0_8-paper style, 300 dpi,
           figure size (12, 14), tight layout.
    """
```

```python
def plot_summary_grid(all_candidates, output_path):
    """
    Single-page summary figure showing W1 light curves for all 6 candidates
    in a 2×3 grid. Good for quick visual inspection and paper figures.
    Annotate each panel with rank, score, and classification label.
    """
```

---

### `output/tables.py`

```python
def write_output_csv(candidates, output_path):
    """
    Write final candidates table with ALL required columns.
    
    Columns:
    rank, source_id, ra, dec, redshift, baseline_years,
    n_epochs_total, n_epochs_pre_gap, n_epochs_post_gap,
    w1_delta_mag, w2_delta_mag, w1_flux_ratio (max/min),
    w1_minus_w2_early, w1_minus_w2_late, delta_color,
    drw_tau_days, drw_tau_err_lo, drw_tau_err_hi,
    drw_sigma_mjy, drw_sigma_err_lo, drw_sigma_err_hi,
    drw_nonstationarity_sigma,
    sf_excess_ratio, sf_break_detected,
    changepoint_mjd, changepoint_delta_bic, changepoint_flux_ratio,
    mannkendall_pvalue, trend_direction,
    score_drw_nonstat, score_delta_mag, score_changepoint,
    score_sf_break, score_color, score_sigma_excess, score_gaia,
    composite_score,
    gaia_ruwe, gaia_pm_sig, gaia_variable_flag, gaia_quasar_prob,
    host_contamination_risk, contamination_flag,
    clagn_type_label
    """
```

---

### `main.py` — CLI Entry Point

```python
"""
Usage:
    python main.py --input catalog.csv --output ./results/ --top_n 6 --test_n 5

Arguments:
    --input     : Path to input catalog (CSV or FITS)
    --output    : Output directory (created if not exists)
    --top_n     : Number of top candidates to output (default: 6)
    --test_n    : If set, only process the first N sources (for debugging)
    --mcmc      : Run full MCMC on top candidates (slower, more rigorous)
    --resume    : Resume from checkpoint if previous run was interrupted
    --workers   : Number of parallel workers for multi-source processing (default: 4)
"""
```

**Checkpoint/resume system:**
```python
def save_checkpoint(source_id, result, checkpoint_dir):
    """Save per-source results to disk as we go — never lose work."""
    path = os.path.join(checkpoint_dir, f"{source_id}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(result, f)

def load_checkpoint(source_id, checkpoint_dir):
    """Load previously computed result. Returns None if not found."""
    path = os.path.join(checkpoint_dir, f"{source_id}.pkl")
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None
```

**Progress reporting:**
```python
# Every 10 sources, print a progress line:
# [047/500] Source J123456.7+654321 | z=0.342 | Baseline=11.3yr | Score=0.731 | ✓
# [048/500] Source J234567.8+123456 | z=0.089 | Rejected: baseline < 10yr
# [049/500] Source J345678.9+234567 | z=-0.002 | Rejected: negative redshift
```

---

## ROBUSTNESS REQUIREMENTS

1. **Never crash on a single bad source.** Every network call, file read, and numerical computation must be wrapped in try/except. Log failures with the source ID and reason, then continue to the next source.

2. **All DRW fitting must be done in FLUX space (mJy), never in magnitude space.** Magnitude errors are asymmetric and heteroscedastic; flux errors are approximately Gaussian. The DRW covariance kernel assumes Gaussian noise.

3. **Redshift time dilation correction is mandatory.** All DRW timescales must be computed and reported in the source rest frame: `τ_rest = τ_obs / (1 + z)`. Failing to do this causes systematic errors in τ for high-z sources.

4. **The WISE hibernation gap must never be interpolated across.** Treat pre-gap and post-gap data as two separate segments for changepoint detection purposes.

5. **Minimum data requirements before running any model:**
   - At least `MIN_EPOCHS = 20` usable exposures
   - At least `MIN_BASELINE_YEARS = 10.0` years of coverage
   - At least 5 epochs on each side of the WISE gap
   - If any requirement fails, log the reason and skip the source

6. **4σ outlier clipping before DRW fitting** using `astropy.stats.sigma_clip(sigma=4, maxiters=3)`. Remove clipped epochs entirely from the fit (do not replace with median).

7. **Handle WISE upper limits:** Epochs with `ph_qual = 'U'` are upper limits, not detections. They must be excluded from light curve fitting.

8. **Uncertainty propagation:** All reported quantities must include uncertainty estimates. No bare point estimates without errors in the final output.

9. **Parallelism:** Use `concurrent.futures.ProcessPoolExecutor` for processing multiple sources in parallel. Default 4 workers. Each worker is independent and writes its own checkpoint file.

10. **Final sanity checks on top 6 before output:**
    - `delta_mag_w1 > MIN_MAG_CHANGE_W1` (0.3 mag)
    - `flux_ratio_w1 > MIN_FLUX_RATIO_CHANGE` (factor 2)
    - `baseline_years >= MIN_BASELINE_YEARS` (10.0 yr)
    - `redshift > MIN_REDSHIFT`
    - `gaia_pm_sig < MAX_PROPER_MOTION_SIG` (or flagged as unknown)
    - If any candidate fails, replace with next ranked source and log the replacement.

---

## CROSS-VALIDATION AGAINST KNOWN CLAGN

Build a validation module that checks pipeline outputs against published CLAGN catalogs:

```python
KNOWN_CLAGN_CATALOG = [
    # From MacLeod+2016, Yang+2018, Graham+2020, Green+2022
    # Include RA, Dec, redshift, confirmed CLAGN type
    # Use this to compute recovery rate as a self-check
    {'name': 'Mrk 1018',     'ra': 32.100,  'dec': -0.369, 'z': 0.042, 'type': 'CS-AGN'},
    {'name': '1ES 1927+654', 'ra': 292.409, 'dec': 65.565, 'z': 0.011, 'type': 'CS-AGN'},
    {'name': 'NGC 2617',     'ra': 127.565, 'dec': -4.521, 'z': 0.014, 'type': 'CS-AGN'},
    # Add more from literature
]

def validate_against_known_clagn(all_scored_sources, known_catalog, search_radius=10.0):
    """
    Cross-match pipeline results against known CLAGN.
    Report:
    - Recovery rate: fraction of known CLAGN in input catalog that appear in top-N
    - Rank of each known CLAGN in the pipeline output
    - False positive rate estimate
    
    This is the primary accuracy metric for the pipeline.
    Log: 'Pipeline recovered X/Y known CLAGN in the input catalog'
    """
```

---

## EXPECTED PERFORMANCE

Based on the methodology in Ricci & Trakhtenbrot (2022) and the WISE CLAGN literature:

- **Completeness (true positive rate):** ~80-90% for genuine CS-AGN with >0.75 mag W1 changes over 10+ year baselines
- **Purity (1 - false positive rate):** ~85-95% after proper motion filtering and false positive rejection
- **Limiting cases:** Sources with delta_mag between 0.3-0.5 mag are the hardest to classify reliably — flag these as "marginal candidates" in the output
- **Known limitation:** Pipeline cannot distinguish CO-AGN from CS-AGN without spectroscopic data. All candidates should be marked for spectroscopic followup.

---

## FINAL INVOCATION

```bash
# Test mode (5 sources)
python main.py --input catalog.csv --output ./results/ --top_n 6 --test_n 5

# Full run with MCMC on top candidates
python main.py --input catalog.csv --output ./results/ --top_n 6 --mcmc

# Resume interrupted run
python main.py --input catalog.csv --output ./results/ --top_n 6 --resume
```

---

*Scientific basis: Ricci & Trakhtenbrot (2022) arXiv:2211.05132 | Kelly et al. (2009) ApJ 698 895 | Stern et al. (2012) ApJ 753 30 | MacLeod et al. (2016) MNRAS 457 389 | Yang et al. (2018) ApJ 862 109 | Graham et al. (2020) MNRAS 491 4925 | Vanden Berk et al. (2004) ApJ 601 692*