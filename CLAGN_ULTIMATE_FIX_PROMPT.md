# CLAGN Pipeline — Ultimate Scientific Fix Prompt

### Every real flaw identified from published literature. Fix all of them.

### Based on: WISE docs, Kozlowski+2017, Graham+2020, Sheng+2017, MacLeod+2010, Ricci+2022

---

## CONTEXT: WHY THIS PROMPT EXISTS

A review against published CLAGN methodology papers revealed **14 critical scientific flaws**
in the existing pipeline. Some produce wrong numbers silently. Some produce correct-looking
but physically meaningless output. At least two would cause a Caltech reviewer to reject
the results immediately.

Fix all 14 in order. Do not skip any. After every fix, verify the known CLAGN test
`python main.py --test_n 5` still recovers Mrk 1018 at rank 1.

---

## FLAW 1 (CRITICAL): delta_mag is Computed Wrong

### The problem

The pipeline computes `delta_mag` by taking:

```python
delta_mag = max(w1_mags) - min(w1_mags)   # WRONG
# or
delta_mag = w1_mag_late - w1_mag_early    # ALSO WRONG if using raw epochs
```

This is wrong for two reasons:

1. It uses raw single-epoch measurements — one bad epoch dominates the result
2. It treats the sign backwards — in WISE Vega mags, BRIGHTER = smaller number

### The correct method (from published papers: Sheng+2017, Graham+2020, Hon+2020)

All published CLAGN papers use **seasonal median flux** comparison, not raw epochs:

```python
def compute_delta_mag_correct(times, w1_flux_mjy, w1_flux_err_mjy, z):
    """
    Compute delta_mag the way published papers actually do it.

    Step 1: Bin light curve into 6-month WISE seasons
    Step 2: Compute weighted median flux per season (weight = 1/err^2)
    Step 3: Compare FIRST 2 seasons vs LAST 2 seasons
    Step 4: Convert flux ratio to delta_mag using Pogson formula

    SIGN CONVENTION (critical — must be consistent throughout):
    delta_mag > 0 means source got BRIGHTER  (flux increased)
    delta_mag < 0 means source got FAINTER   (flux decreased)
    delta_mag = -2.5 * log10(F_early / F_late)

    This matches: a turn-ON event (accretion increases) gives positive delta_mag
    This matches: a turn-OFF event (accretion decreases) gives negative delta_mag

    DO NOT use: max(mags) - min(mags)
    DO NOT use: single epoch comparison
    DO NOT use: magnitude subtraction instead of flux ratio
    """
    # Assign each epoch to a season (6-month window)
    season_id = np.floor((times - times.min()) / 182.5).astype(int)

    # Weighted median per season in flux space
    season_flux = {}
    for s in np.unique(season_id):
        mask = season_id == s
        if mask.sum() < 3:
            continue  # Skip seasons with fewer than 3 epochs
        weights = 1.0 / w1_flux_err_mjy[mask]**2
        season_flux[s] = np.average(w1_flux_mjy[mask], weights=weights)

    if len(season_flux) < 4:
        return None, None  # Not enough seasons

    seasons = sorted(season_flux.keys())

    # Early baseline: mean of first 2 seasons
    F_early = np.mean([season_flux[s] for s in seasons[:2]])
    # Late baseline: mean of last 2 seasons
    F_late  = np.mean([season_flux[s] for s in seasons[-2:]])

    if F_early <= 0 or F_late <= 0:
        return None, None

    # Flux ratio (> 1 = brightened, < 1 = faded)
    flux_ratio = F_late / F_early

    # Delta magnitude (positive = brightened, consistent with flux_ratio > 1)
    delta_mag = -2.5 * np.log10(F_early / F_late)
    # = +2.5 * log10(F_late / F_early)
    # Positive if late > early (turn-on)
    # Negative if late < early (turn-off)

    # Uncertainty: propagate from seasonal flux uncertainties
    delta_mag_err = (2.5 / np.log(10)) * np.sqrt(
        (1/F_early)**2 * (F_early * 0.05)**2 +
        (1/F_late)**2  * (F_late  * 0.05)**2
    )  # 5% systematic floor from WISE calibration

    return delta_mag, delta_mag_err, flux_ratio
```

### Fix in code

Find every occurrence of `delta_mag` computation in:

- `clagn/scoring/clagn_score.py`
- `clagn/ingestion/wise_complete.py`
- `clagn/models/drw.py`
- `main.py`

Replace ALL of them with the seasonal median flux method above.
Add a column `delta_mag_method = 'seasonal_median_flux'` to all output CSVs.

---

## FLAW 2 (CRITICAL): Magnitude to Flux Conversion Uses Wrong Zero Points or Direction

### The problem

WISE magnitudes are Vega system. The conversion is:

```
F_Jy = F0_Jy * 10^(-0.4 * mag_Vega)
```

The official WISE Vega zero points (Wright+2010, Table 1) are:

```
W1: F0 = 309.540 Jy
W2: F0 = 171.787 Jy
```

Any other value is wrong. The error propagation is:

```
dF_mJy = F_mJy * (0.4 * ln10) * dmag = F_mJy * 0.9210 * dmag
```

### The fix

```python
# In clagn/config.py — ensure EXACTLY these values
WISE_VEGA_ZERO_POINTS = {
    'W1': 309.540,   # Jy (Wright+2010 Table 1)
    'W2': 171.787,   # Jy (Wright+2010 Table 1)
    'W3': 31.674,    # Jy (for completeness)
    'W4': 8.363,     # Jy
}

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
    F0_mJy = WISE_VEGA_ZERO_POINTS[band] * 1000.0  # Jy -> mJy

    # Handle upper limits and bad values
    valid = (mag > 0) & (mag < 30) & np.isfinite(mag)
    flux_mjy = np.where(valid, F0_mJy * 10**(-0.4 * mag), np.nan)

    # Error propagation: partial derivative of F w.r.t. mag
    # dF/dmag = F * (-0.4 * ln10) = -F * 0.92103
    # |dF| = F * 0.92103 * |dmag|
    flux_err_mjy = np.where(
        valid & (mag_err > 0),
        flux_mjy * 0.92103 * mag_err,
        np.nan
    )

    return flux_mjy, flux_err_mjy

def flux_mjy_to_mag(flux_mjy, flux_err_mjy, band):
    """
    Convert flux density in mJy back to WISE Vega magnitude.
    Used for reporting delta_mag in final output.

    mag = -2.5 * log10(flux_mJy / F0_mJy)
    dmag = (2.5 / ln10) * dflux / flux = 1.08574 * dflux / flux
    """
    F0_mJy = WISE_VEGA_ZERO_POINTS[band] * 1000.0
    valid = (flux_mjy > 0) & np.isfinite(flux_mjy)
    mag = np.where(valid, -2.5 * np.log10(flux_mjy / F0_mJy), np.nan)
    mag_err = np.where(
        valid & (flux_err_mjy > 0),
        1.08574 * flux_err_mjy / flux_mjy,
        np.nan
    )
    return mag, mag_err
```

### Verification test

```python
# W1 mag = 13.0 should give F ≈ 3.099 mJy
F, dF = mag_to_flux_mjy(13.0, 0.02, 'W1')
assert 3.09 < F < 3.11, f"Wrong W1 flux: {F}"
assert 0.055 < dF < 0.060, f"Wrong W1 flux error: {dF}"

# W1 mag = 15.0 should give F ≈ 0.490 mJy
F2, _ = mag_to_flux_mjy(15.0, 0.02, 'W1')
assert 0.488 < F2 < 0.492, f"Wrong W1 flux at mag 15: {F2}"

print("Flux conversion test PASSED")
```

---

## FLAW 3 (CRITICAL): DRW is Fit in Wrong Space

### The problem

DRW fitting using Gaussian process likelihood assumes the RESIDUALS are Gaussian.
Magnitude residuals are NOT Gaussian — they are log-normal in flux space.

At SNR=10, a flux error of 10% corresponds to a magnitude error of ~0.11 mag.
At SNR=5, a flux error of 20% corresponds to a magnitude error of ~0.22 mag.
These are the SAME fractional error, but fitting in magnitude space treats them
as having different weights, which is incorrect.

**All published DRW papers fit in flux space (MacLeod+2010, Kelly+2009).**

### The fix

In `clagn/models/drw.py`, verify the fitting is in flux space:

```python
def fit_drw_map(times_obs, w1_flux_mjy, w1_flux_err_mjy, z,
                band='W1', min_epochs=15):
    """
    Fit DRW in flux space (mJy). NEVER in magnitude space.

    CRITICAL CHECKS before fitting:
    1. Input must be in flux units (mJy), not magnitudes
    2. Rest-frame time dilation must be applied to times
    3. Errors must be positive and finite
    4. No NaN values

    Parameters
    ----------
    times_obs : array
        Observation times in MJD (observer frame)
    w1_flux_mjy : array
        W1 flux density in mJy (already converted from Vega mag)
    w1_flux_err_mjy : array
        W1 flux uncertainty in mJy
    z : float
        Source redshift

    Returns
    -------
    dict with keys:
        tau_rest_days : rest-frame DRW timescale in days
        tau_err_lo, tau_err_hi : 1-sigma uncertainties
        sigma_drw_mjy : DRW variability amplitude in mJy (NOT magnitudes)
        sigma_drw_mag : DRW amplitude converted to magnitude equivalent
        log_likelihood : GP log likelihood at MAP solution
        n_epochs : number of epochs actually used
        baseline_years : observer-frame baseline used
        valid : bool — False if fit failed or is unreliable
        unreliable_reason : str or None
    """
    # --- Validate inputs ---
    assert w1_flux_mjy is not None, "Flux array is None"
    assert len(times_obs) == len(w1_flux_mjy) == len(w1_flux_err_mjy)

    # Require flux space input (sanity check: median flux should be in mJy range)
    median_val = np.nanmedian(w1_flux_mjy)
    assert 0.001 < median_val < 100000, (
        f"Median flux {median_val:.3f} is outside mJy range. "
        f"Did you accidentally pass magnitudes instead of fluxes?"
    )

    # Clean data
    valid = (
        np.isfinite(w1_flux_mjy) &
        np.isfinite(w1_flux_err_mjy) &
        (w1_flux_err_mjy > 0) &
        (w1_flux_mjy > 0)
    )
    t = times_obs[valid]
    f = w1_flux_mjy[valid]
    e = w1_flux_err_mjy[valid]

    if valid.sum() < min_epochs:
        return {'valid': False, 'unreliable_reason': f'too_few_epochs_{valid.sum()}'}

    # --- REST-FRAME TIME CORRECTION ---
    # This is mandatory. DRW timescales are physical (rest-frame) quantities.
    # tau_rest = tau_observed / (1 + z)
    # Equivalently: fit using rest-frame times directly.
    if z <= 0:
        return {'valid': False, 'unreliable_reason': 'invalid_redshift'}

    t_rest = t / (1.0 + z)  # Observer -> rest frame
    baseline_rest_days = t_rest.max() - t_rest.min()

    # --- DRW VALIDITY CHECK (Kozlowski+2017) ---
    # The baseline must be at least 10x the true DRW timescale for tau to
    # be reliably measured. We can't know tau in advance, so we check
    # AFTER fitting and flag if tau > baseline/10.

    # [fit happens here using celerite2 or fallback GP]
    # ...

    # --- POST-FIT VALIDITY CHECK ---
    # tau_rest_days: the fitted rest-frame timescale
    if tau_rest_days > baseline_rest_days / 10.0:
        result['unreliable_reason'] = (
            f'tau_unconstrained: tau={tau_rest_days:.0f}d > '
            f'baseline/10={baseline_rest_days/10:.0f}d (Kozlowski+2017)'
        )
        result['tau_reliable'] = False
    else:
        result['tau_reliable'] = True

    # --- SIGMA IN BOTH UNITS ---
    # sigma_drw is in mJy (flux space) — this is the primary value
    # Also convert to magnitude equivalent for reporting:
    # sigma_mag_equiv = 2.5 * log10(1 + sigma_mJy / median_flux_mJy)
    result['sigma_drw_mjy'] = sigma_drw  # from fit
    result['sigma_drw_mag_equiv'] = (
        2.5 * np.log10(1 + sigma_drw / np.median(f))
    )

    return result
```

---

## FLAW 4: NEOWISE W2 Systematic Offset Not Corrected

### The problem

From NEOWISE data release documentation:

> "The adjustment that was made on MJD=57000 is now known to have been slightly incorrect.
> As a result, the W2 residuals in the interval 57000 < MJD < 57071 are offset from the
> regular seasonal variations."

This introduces a fake jump in W2 light curves around MJD 57000 (early 2015).
Any CLAGN detected primarily from a W2 change near this date must be flagged.

### The fix

```python
# In clagn/config.py — add these known WISE photometric issues
WISE_KNOWN_SYSTEMATICS = [
    {
        'band': 'W2',
        'mjd_start': 57000,
        'mjd_end': 57071,
        'description': 'Incorrect ZP adjustment applied in W2 (NEOWISE docs)',
        'action': 'flag_epochs',  # Flag but do not remove
        'magnitude_offset': 0.01,  # ~0.01 mag systematic
    },
    {
        'band': 'W1',
        'mjd_start': 55400,  # End of AllWISE / start of NEOWISE-R crossover
        'mjd_end': 56200,
        'description': 'WISE hibernation gap — no data, not an artifact',
        'action': 'flag_gap',
    },
]

def flag_systematic_epochs(times, band):
    """
    Return boolean array: True = epoch affected by known WISE systematic.
    These epochs should be DOWN-WEIGHTED in DRW fitting, not removed.
    They are flagged in the output CSV for transparency.
    """
    flagged = np.zeros(len(times), dtype=bool)
    for sys in WISE_KNOWN_SYSTEMATICS:
        if sys['band'] == band and sys['action'] == 'flag_epochs':
            flagged |= (times >= sys['mjd_start']) & (times <= sys['mjd_end'])
    return flagged
```

---

## FLAW 5: DRW Baseline Requirement Misunderstood (Kozlowski+2017)

### The problem

The Kozlowski+2017 paper (arXiv:1611.08248) proves that DRW timescales are **unconstrained**
unless the light curve baseline is at least **10 times** the true DRW timescale.

The pipeline enforces a 10-year observer-frame baseline. This is correct for the baseline
requirement but the DRW fit result must also be checked AFTER fitting.

If the fitted tau_rest_days > baseline_rest_days / 10, the tau measurement is unreliable
and cannot be used for physical parameter estimation.

### The fix

After every DRW fit, add:

```python
def check_drw_reliability(tau_rest_days, baseline_obs_days, z):
    """
    Apply the Kozlowski+2017 reliability criterion.

    The baseline (in rest frame) must be >= 10 * tau_rest for tau to be
    reliably measured. Sources failing this test can still be CLAGN candidates
    based on other metrics, but their DRW tau should not be used for
    physics (M_BH estimation, disk timescales).

    Parameters
    ----------
    tau_rest_days : float
        Fitted DRW rest-frame timescale
    baseline_obs_days : float
        Observer-frame light curve baseline
    z : float
        Redshift

    Returns
    -------
    reliable : bool
    reason : str
    """
    baseline_rest_days = baseline_obs_days / (1 + z)

    if tau_rest_days > baseline_rest_days / 10.0:
        return False, (
            f"tau_rest={tau_rest_days:.0f}d > baseline_rest/10="
            f"{baseline_rest_days/10:.0f}d — Kozlowski+2017 criterion fails"
        )

    # Also check: tau should not be hitting prior bounds
    # If tau is very close to the prior minimum or maximum, it's unconstrained
    LOG_TAU_MIN = 1.0   # exp(1) ~ 2.7 days
    LOG_TAU_MAX = 8.5   # exp(8.5) ~ 4915 days
    log_tau = np.log(tau_rest_days)

    if log_tau < LOG_TAU_MIN + 0.3:
        return False, "tau at lower prior boundary — unconstrained"
    if log_tau > LOG_TAU_MAX - 0.3:
        return False, "tau at upper prior boundary — unconstrained"

    return True, "OK"
```

---

## FLAW 6: Host Galaxy Contamination Dilutes Variability (Not Corrected)

### The problem

At low redshift (z < 0.2), the host galaxy contributes substantially to WISE photometry.
WISE W1 PSF is 6.1 arcsec — it captures the entire inner galaxy disk.

From Kozlowski+2017:

> "The ratio of the recovered-to-input variability amplitude is directly proportional to
> the ratio of the AGN-to-the-total light."

If the host contributes 40% of W1 flux, the observed delta_mag is 40% smaller than
the true AGN delta_mag. The pipeline reports the DILUTED value, not the AGN value.

### The fix

```python
def estimate_host_contamination_fraction(z, w1_mean_flux_mjy, source_type='agn'):
    """
    Estimate fractional contribution of host galaxy to total WISE W1 flux.

    Method: at higher z, host galaxy is a smaller fraction of total WISE flux
    because: (1) AGN is intrinsically more luminous at higher z (selection)
             (2) host galaxy flux is redshifted out of W1 band

    Empirical calibration from Assef+2013 AGN SED decomposition:
    f_host ~ 0.5 * exp(-z / 0.15) for typical Seyfert luminosities

    This is a rough estimate. Flag sources with f_host > 0.3 as
    'host_contaminated' — their variability amplitudes are lower limits only.

    Returns
    -------
    f_host : float
        Estimated host fraction (0 = pure AGN, 1 = pure host galaxy)
    f_agn : float
        AGN fraction = 1 - f_host
    contamination_flag : str
        'severe' (f_host > 0.5), 'moderate' (0.3-0.5), 'low' (<0.3), 'negligible' (<0.1)
    """
    # Conservative estimate — errs on side of more contamination
    f_host = 0.5 * np.exp(-z / 0.15)
    f_host = np.clip(f_host, 0.0, 0.95)
    f_agn = 1.0 - f_host

    if f_host > 0.5:
        flag = 'severe_contamination'
    elif f_host > 0.3:
        flag = 'moderate_contamination'
    elif f_host > 0.1:
        flag = 'low_contamination'
    else:
        flag = 'negligible'

    return f_host, f_agn, flag

def correct_delta_mag_for_host(delta_mag_observed, f_agn):
    """
    Correct observed delta_mag for host galaxy dilution.

    If observed delta_mag = -2.5*log10(F_total_late/F_total_early)
    and F_total = F_AGN + F_host
    with F_host = constant (host doesn't vary)

    Then: delta_mag_AGN = -2.5*log10((F_agn_late)/(F_agn_early))
                        = -2.5*log10(1 + (flux_ratio_total - 1)/f_agn)

    This is larger in magnitude than delta_mag_observed.

    Parameters
    ----------
    delta_mag_observed : float
        Observed (diluted) delta_mag from total flux
    f_agn : float
        AGN fraction at W1 (0 to 1)

    Returns
    -------
    delta_mag_agn : float
        Host-corrected AGN-only delta_mag
    """
    if f_agn <= 0 or f_agn > 1:
        return delta_mag_observed

    flux_ratio_total = 10**(delta_mag_observed / 2.5)
    # Solve for AGN-only flux ratio
    flux_ratio_agn = 1 + (flux_ratio_total - 1) / f_agn

    if flux_ratio_agn <= 0:
        return delta_mag_observed  # Correction failed, return original

    return 2.5 * np.log10(flux_ratio_agn)
```

Report BOTH `delta_mag_observed` and `delta_mag_host_corrected` in output CSV.
Always state which one is used for CLAGN classification thresholds.
Use `delta_mag_observed` for detection (conservative) and `delta_mag_host_corrected`
for physical interpretation.

---

## FLAW 7: Sigma Clipping Done on Magnitudes, Not Fluxes

### The problem

If the pipeline applies sigma clipping to magnitude arrays before conversion to flux,
this is incorrect. Magnitudes have asymmetric errors (log scale), so a 3-sigma outlier
in magnitude space is not a 3-sigma outlier in flux space.

### The fix

```python
def sigma_clip_in_flux_space(times, fluxes, flux_errors, sigma=4.0, maxiters=5):
    """
    Sigma clip light curves in FLUX SPACE only.
    Never sigma clip in magnitude space.

    Why: magnitude errors are asymmetric (log scale).
    A magnitude outlier of +0.5 mag corresponds to flux 58% below median.
    A magnitude outlier of -0.5 mag corresponds to flux 58% ABOVE median.
    These are symmetric in magnitude but the flux excursions are equal.

    In flux space, the distribution is approximately Gaussian (central limit theorem)
    and sigma clipping is statistically valid.

    Parameters
    ----------
    times, fluxes, flux_errors : arrays (already in mJy)
    sigma : float
        Clipping threshold in standard deviations
    maxiters : int
        Maximum iterations

    Returns
    -------
    mask : boolean array, True = keep
    n_clipped : int
    """
    from astropy.stats import sigma_clip as astropy_sigma_clip

    # CRITICAL: clip on flux values, not magnitudes
    clipped = astropy_sigma_clip(fluxes, sigma=sigma, maxiters=maxiters,
                                  masked=True, copy=True)
    mask = ~clipped.mask
    n_clipped = clipped.mask.sum()

    return mask, n_clipped
```

---

## FLAW 8: Sign Convention for delta_mag is Inconsistent

### The problem

Throughout the codebase, `delta_mag` appears with different sign conventions:

- Some places: `delta_mag = mag_late - mag_early` (negative = turn-on = WRONG direction)
- Some places: `delta_mag = |mag_max - mag_min|` (always positive = loses direction info)
- Some places: flux ratio used instead (correct but inconsistent with other places)

A Caltech reviewer will immediately ask: "which direction is positive?"

### The fix: one universal standard

```python
# In clagn/config.py — add this documentation block

"""
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
(i.e., both turn-on and turn-off events qualify)
"""
```

Update ALL code to match this convention. Add an assertion in the output writer:

```python
# Sanity check: delta_mag and flux_ratio must be consistent
assert np.sign(result['delta_mag_w1']) == np.sign(np.log(result['flux_ratio_w1'])), \
    "delta_mag and flux_ratio have inconsistent signs — bug in computation"
```

---

## FLAW 9: GAIA Star Rejection Misses Parallax Test

### The problem

The current filter uses only `pm_sig < 3` (proper motion significance).
But some foreground stars have low proper motion and still contaminate the sample.
GAIA provides a direct distance measurement via parallax — this is the best star rejection test.

### The fix

```python
def is_galactic_source(gaia_row):
    """
    Test whether a GAIA source is a foreground Galactic object.
    Use BOTH proper motion AND parallax — either can identify a star.

    Rejection criteria (source is likely a foreground star if ANY is true):
    1. PM significance > 3: pm_sig = sqrt(pmra^2+pmdec^2)/sqrt(pmra_err^2+pmdec_err^2) > 3
    2. Parallax significance > 3: abs(parallax / parallax_error) > 3
    3. RUWE > 1.4: non-point-source morphology
    4. GAIA DSC quasar probability < 0.1: very unlikely to be a quasar

    A source passes (is kept as AGN candidate) only if NONE of these trigger.
    """
    reasons = []

    # Test 1: proper motion
    if gaia_row.pmra is not None and gaia_row.pmdec is not None:
        pm_sig = np.sqrt(gaia_row.pmra**2 + gaia_row.pmdec**2) / \
                 np.sqrt(gaia_row.pmra_error**2 + gaia_row.pmdec_error**2 + 1e-10)
        if pm_sig > 3.0:
            reasons.append(f'pm_sig={pm_sig:.1f}>3')

    # Test 2: parallax (BEST test — direct distance measurement)
    if gaia_row.parallax is not None and gaia_row.parallax_error is not None:
        plx_sig = abs(gaia_row.parallax) / (gaia_row.parallax_error + 1e-10)
        if plx_sig > 3.0:
            reasons.append(f'parallax_sig={plx_sig:.1f}>3 (nearby star)')

    # Test 3: RUWE
    if gaia_row.ruwe is not None and gaia_row.ruwe > 1.4:
        reasons.append(f'ruwe={gaia_row.ruwe:.2f}>1.4')

    # Test 4: GAIA quasar classification
    if hasattr(gaia_row, 'classprob_dsc_combmod_quasar'):
        if gaia_row.classprob_dsc_combmod_quasar < 0.1:
            reasons.append('gaia_qso_prob<0.1')

    is_galactic = len(reasons) > 0
    return is_galactic, reasons
```

---

## FLAW 10: Composite Score Components Are Correlated — Double Counting

### The problem

The 7-component score sums:

- `score_delta_mag` (from seasonal median flux comparison)
- `score_flux_ratio` (also from seasonal median flux comparison)
  These two measure the SAME quantity. A source with large delta_mag always has
  large flux_ratio. Summing them double-counts this evidence.

Similarly:

- `score_drw_nonstat` (DRW non-stationarity)
- `score_changepoint_bic` (Bayesian changepoint)
  Both detect the same state change. A real CLAGN triggers both.

### The fix: use maximum within correlated groups

```python
def compute_composite_score_v3(components):
    """
    Composite score v3: avoids double-counting correlated evidence.

    Component groups (take MAX within each group, not sum):

    Group A: Amplitude evidence (how much did it change?)
        - score_delta_mag
        - score_flux_ratio
        → A_score = max(score_delta_mag, score_flux_ratio)
        These measure the same thing. Use the stronger evidence only.

    Group B: Temporal structure evidence (when did it change?)
        - score_changepoint_bic
        - score_drw_nonstat
        - score_broken_drw_delta_bic
        → B_score = max of the three
        These all detect the same structural break.

    Group C: Color evidence (did the spectrum change?)
        - score_w1_w2_color_evolution
        Independent of Groups A and B.

    Group D: Statistical model evidence (is it unusual?)
        - score_sf_excess_ratio
        - score_drw_sigma_excess
        → D_score = max of the two
        Both measure excess variability relative to expectation.

    Group E: Astrometric evidence (is it a real AGN?)
        - score_gaia_variable
        - score_pm_filter_pass (binary: passed GAIA filter = 1.0)
        Independent.

    FINAL COMPOSITE:
    composite = (w_A * A_score + w_B * B_score + w_C * C_score
                 + w_D * D_score + w_E * E_score) / (w_A+w_B+w_C+w_D+w_E)

    Weights (calibrated so a confirmed CLAGN scores >= 0.5):
    w_A = 3.0  (amplitude is the most directly observable CLAGN signature)
    w_B = 3.0  (temporal structure is equally fundamental)
    w_C = 2.0  (color change is physically important but not always present)
    w_D = 1.5  (statistical excess is supporting evidence)
    w_E = 0.5  (GAIA flag is confirmatory, not diagnostic)
    """
    A_score = max(components.get('delta_mag', 0), components.get('flux_ratio', 0))
    B_score = max(components.get('changepoint_bic', 0),
                  components.get('drw_nonstat', 0),
                  components.get('broken_drw_bic', 0))
    C_score = components.get('color_evolution', 0)
    D_score = max(components.get('sf_excess', 0), components.get('drw_sigma_excess', 0))
    E_score = components.get('gaia_variable', 0)

    w_A, w_B, w_C, w_D, w_E = 3.0, 3.0, 2.0, 1.5, 0.5

    composite = (w_A*A_score + w_B*B_score + w_C*C_score +
                 w_D*D_score + w_E*E_score) / (w_A+w_B+w_C+w_D+w_E)

    return composite, {
        'A_amplitude': A_score,
        'B_temporal': B_score,
        'C_color': C_score,
        'D_statistical': D_score,
        'E_astrometric': E_score
    }
```

---

## FLAW 11: Physical Parameters Use Unreliable tau Values

### The problem

M_BH estimation uses the DRW scaling relation:

```
log(M_BH) = f(tau_rest, L_bol)
```

But if tau_rest is unreliable (Flaw 5), the M_BH estimate is meaningless.

### The fix

```python
def estimate_black_hole_mass(tau_rest_days, tau_reliable, l_bol_erg_s, z):
    """
    Estimate M_BH from DRW tau only if tau is reliable.

    If tau is unreliable (Kozlowski+2017 criterion fails):
    - Do NOT compute M_BH from tau
    - Return NaN with flag 'tau_unreliable'
    - Report only luminosity-based M_BH estimate if available

    Kelly+2009 scaling relation (Eq.6 in that paper):
    log(tau_rest/days) = 2.4 + 0.17*log(L_5100/1e44) + 0.013*(T-6000) + 0.038*log(M8)

    Inverted: log(M8) = (log(tau) - 2.4 - 0.17*log(L_5100/1e44)) / 0.038
    where M8 = M_BH / 1e8 M_sun

    Uncertainty: propagate from tau MCMC posterior, not just from point estimate.
    """
    if not tau_reliable:
        return {
            'logMBH': np.nan,
            'logMBH_err': np.nan,
            'method': 'tau_unreliable',
            'flag': 'Kozlowski+2017: tau > baseline/10, M_BH not computed'
        }

    # Bolometric correction: L_5100 from L_bol
    kappa_5100 = 10.3  # Richards+2006
    L_5100 = l_bol_erg_s / kappa_5100

    # Kelly+2009 Eq.6 inverted
    log_L44 = np.log10(L_5100 / 1e44)
    log_tau = np.log10(tau_rest_days)

    # T_eff for accretion disk; assume 6000K (no sensitivity for typical AGN)
    log_M8 = (log_tau - 2.4 - 0.17 * log_L44) / 0.038
    logMBH = log_M8 + 8.0  # log10(M_BH / M_sun)

    return {
        'logMBH': logMBH,
        'logMBH_err': 0.4,  # ~0.4 dex scatter in Kelly+2009 relation
        'method': 'DRW_Kelly2009',
        'flag': None
    }
```

---

## FLAW 12: Output CSV Reports Ambiguous Units

### The problem

The CSV output has columns like `delta_mag_w1` with no documented units,
no sign convention, and no distinction between seasonal-median-based and
single-epoch-based computations.

A reviewer looking at the CSV cannot know which method was used.

### The fix: rename all ambiguous columns

```python
# OLD column names → NEW column names with explicit meaning
COLUMN_RENAME_MAP = {
    # Amplitude columns
    'delta_mag_w1':    'delta_mag_w1_seasonal_pogson',  # = +2.5*log10(F_late/F_early), seasonal median
    'delta_mag_w2':    'delta_mag_w2_seasonal_pogson',
    'flux_ratio_w1':   'flux_ratio_w1_seasonal',        # = F_late_median / F_early_median

    # DRW columns
    'drw_tau':         'drw_tau_rest_days',              # rest-frame, Kelly+2009
    'drw_sigma':       'drw_sigma_flux_mjy',             # in mJy, NOT magnitudes
    'drw_sigma_mag':   'drw_sigma_mag_equiv',            # = 2.5*log10(1 + sigma_mJy/median_flux)

    # Validity flags (NEW — not in original)
    # tau_reliable: bool (Kozlowski+2017 criterion)
    # host_contamination_flag: 'severe/moderate/low/negligible'
    # delta_mag_host_corrected: corrected for host dilution
    # w2_systematic_flag: True if epochs near MJD 57000-57071
    # sign_convention: 'positive=brightening'
}
```

---

## FLAW 13: No Cross-Check Between W1 and W2

### The problem

Published CLAGN papers always check that W1 and W2 vary **together** (correlated).
A single-band detection is suspicious — it could be a detector artifact or
an asteroid crossing the field.

### The fix

```python
def check_w1_w2_coherence(times, w1_flux, w2_flux, w1_err, w2_err):
    """
    Test whether W1 and W2 variability is coherent (physically expected)
    or incoherent (suggests artifact or data problem).

    Method: Pearson correlation coefficient between W1 and W2 flux
    in the same time bins.

    Physical expectation:
    - Real AGN variability: W1 and W2 are highly correlated (r > 0.6)
      because both probe the same hot dust near the torus
    - CLAGN state change: W1 and W2 change in the SAME direction
      (both brighten for turn-on, both fade for turn-off)
    - Artifacts: W1 and W2 may be uncorrelated or anti-correlated

    Compute:
    1. Pearson r between W1 and W2 seasonal medians
    2. Sign consistency: delta_mag_W1 and delta_mag_W2 same sign?
    3. W1/W2 flux ratio change: if AGN drives the variability,
       this ratio should change predictably (color temperature)

    Returns
    -------
    coherence_score : float (0-1, higher = more coherent)
    w1_w2_pearson_r : float
    w1_w2_same_direction : bool
    coherence_flag : str ('coherent', 'marginal', 'incoherent')
    """
    # ... implementation ...
```

Add `w1_w2_coherence_score` to all output CSVs and to the composite score
as an INDEPENDENT component that penalizes incoherent detections.

---

## FLAW 14: Saturation Not Checked

### The problem

WISE W1 saturates at approximately W1 ~ 8.1 mag, W2 ~ 6.7 mag.
Sources brighter than these limits have unreliable photometry.
The WISE documentation states: profile-fit photometry is reliable only for
8.0 < W1 < 14.0 and 7.0 < W2 < 13.7.

NGC 2617 failed in the test run partly for this reason (W1 ~ 8).

### The fix

```python
# In clagn/config.py
WISE_RELIABLE_PHOTOMETRY_LIMITS = {
    'W1': {'bright': 8.0,  'faint': 14.5},  # mag, Vega
    'W2': {'bright': 6.7,  'faint': 13.7},
}

def check_wise_saturation(w1_mag_median, w2_mag_median):
    """
    Check whether median WISE magnitudes fall in reliable range.

    Returns
    -------
    w1_reliable : bool
    w2_reliable : bool
    saturation_flag : str or None
    """
    w1_ok = WISE_RELIABLE_PHOTOMETRY_LIMITS['W1']['bright'] < w1_mag_median < \
            WISE_RELIABLE_PHOTOMETRY_LIMITS['W1']['faint']
    w2_ok = WISE_RELIABLE_PHOTOMETRY_LIMITS['W2']['bright'] < w2_mag_median < \
            WISE_RELIABLE_PHOTOMETRY_LIMITS['W2']['faint']

    if not w1_ok:
        if w1_mag_median < WISE_RELIABLE_PHOTOMETRY_LIMITS['W1']['bright']:
            return False, w2_ok, f'W1={w1_mag_median:.1f} saturated (limit: 8.0 mag)'
        else:
            return False, w2_ok, f'W1={w1_mag_median:.1f} too faint (limit: 14.5 mag)'

    return True, w2_ok, None
```

---

## COMPLETE VERIFICATION SEQUENCE

After all fixes are applied, run these tests in order. **All must pass.**

```bash
# Test 1: Unit tests for all fixed functions
python -c "
import numpy as np
from clagn.ingestion.catalog import mag_to_flux_mjy

# Test flux conversion
F, dF = mag_to_flux_mjy(13.0, 0.02, 'W1')
assert abs(F - 3099.4) < 1.0, f'W1 flux wrong: {F}'  # 309.54 Jy * 1000 * 10^(-0.4*13)
print(f'W1 mag=13 -> {F:.1f} mJy (correct: ~3099 mJy)')

# Test sign convention
from clagn.models.variability import compute_delta_mag_correct
# Increasing flux should give positive delta_mag
times = np.linspace(55000, 60000, 50)
flux_increasing = 1.0 + 0.01 * np.arange(50)  # monotonically increasing
delta, _, ratio = compute_delta_mag_correct(times, flux_increasing,
                                             flux_increasing*0.05, z=0.1)
assert delta > 0, f'Increasing flux should give positive delta_mag, got {delta}'
print(f'Sign convention test: delta_mag={delta:.3f} (positive=correct)')

print('ALL UNIT TESTS PASSED')
"

# Test 2: Known CLAGN recovery (MUST still be 3/3)
python main.py --test_n 5
# Expected output:
# Mrk 1018    rank 1  score > 0.25  delta_mag_w1 > 0  (turn-off: faded)
# Mrk 590     rank 2  score > 0.15
# HE 1136     rank 3  score > 0.04
# 1ES 1927    rejected (too few WISE epochs — faint source)
# NGC 2617    rejected (W1 near saturation limit)

# Test 3: Check no negative fluxes in any light curve
python -c "
import glob, pickle
for f in glob.glob('results/checkpoints/*.pkl'):
    data = pickle.load(open(f,'rb'))
    if 'w1_flux_mjy' in data:
        assert (data['w1_flux_mjy'] > 0).all(), f'Negative flux in {f}'
print('No negative fluxes found')
"

# Test 4: Check sign convention is consistent in output CSV
python -c "
import pandas as pd, numpy as np
df = pd.read_csv('results/top_candidates.csv')
# delta_mag and flux_ratio must be consistent in sign
for _, row in df.iterrows():
    assert np.sign(row['delta_mag_w1_seasonal_pogson']) == \
           np.sign(np.log(row['flux_ratio_w1_seasonal'])), \
           f'Sign mismatch for {row[\"source_id\"]}'
print('Sign convention consistent in all candidates')
"
```

---

## SUMMARY OF ALL 14 FIXES

| #   | Flaw                                                | Severity     | File                                 |
| --- | --------------------------------------------------- | ------------ | ------------------------------------ |
| 1   | delta_mag uses raw epochs not seasonal medians      | **CRITICAL** | `wise_complete.py`, `clagn_score.py` |
| 2   | Wrong WISE Vega zero points or conversion direction | **CRITICAL** | `config.py`, `photometry.py`         |
| 3   | DRW fit in magnitude space not flux space           | **CRITICAL** | `drw.py`                             |
| 4   | NEOWISE W2 MJD 57000-57071 systematic not flagged   | High         | `config.py`, `wise_complete.py`      |
| 5   | DRW reliability not checked (Kozlowski+2017)        | High         | `drw.py`, `physics.py`               |
| 6   | Host galaxy contamination not corrected             | High         | `physics.py`, `clagn_score.py`       |
| 7   | Sigma clipping on magnitudes not flux               | Medium       | `drw.py`, `wise_complete.py`         |
| 8   | delta_mag sign convention inconsistent              | Medium       | All files                            |
| 9   | GAIA parallax not used for star rejection           | Medium       | `gaia_lightcurve.py`                 |
| 10  | Composite score double-counts correlated components | Medium       | `clagn_score.py`                     |
| 11  | M_BH computed from unreliable tau values            | Medium       | `physics.py`                         |
| 12  | Output CSV has ambiguous column names               | Low          | `tables.py`, `paper.py`              |
| 13  | No W1/W2 coherence cross-check                      | Low          | new function                         |
| 14  | WISE saturation not checked                         | Low          | `wise_complete.py`                   |

---

_References: Kozlowski+2017 (arXiv:1611.08248) | Wright+2010 (AJ 140 1868) | Kelly+2009 (ApJ 698 895) | MacLeod+2010 (ApJ 721 1014) | Sheng+2017 (ApJL 846 L7) | Graham+2020 (MNRAS 491 4925) | Ricci+2022 (arXiv:2211.05132)_
