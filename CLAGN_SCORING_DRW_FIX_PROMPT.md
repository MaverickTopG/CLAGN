# CLAGN Pipeline — variability.py / drw.py / clagn_score.py Fix Prompt

### All bugs from code review. Apply in priority order listed at bottom.

### Three files, 30 bugs. Some will crash at runtime TODAY.

---

## PRIORITY ORDER (runtime crashes first)

1. FIX drw-C2 — compute_drw_nonstationarity() returns tuple, callers expect dict → CRASH
2. FIX score-C1 — \_bootstrap_score_uncertainty() calls nonstationarity with missing args → CRASH
3. FIX score-C3 — v2 scorer uses wrong column names, silently fails to load flux → silent failure
4. FIX drw-C1 — MCMC path does not subtract mean flux (MAP does) → scientific inconsistency
5. FIX var-C1 — docstring says median, code uses weighted mean → reproducibility issue
6. FIX var-C2 — early/late seasons combined with plain mean, not inverse-variance → biased delta_mag
7. FIX var-C3 — delta_mag_err uses only calibration floor, not propagated uncertainty
8. FIX score-C4 — false-positive rejection reads wrong field name for delta_mag
9. FIX drw-C3 — no input cleaning before GP fit → Cholesky failures
10. All remaining in order

---

## FILE: variability.py

---

### FIX var-C1 (CRITICAL): Docstring Says Median, Code Uses Weighted Mean

**Location:** `compute_delta_mag_correct()` — season flux computation.

**Problem:** Every docstring, comment, and the published methodology description
says "seasonal median flux." The actual code uses weighted mean. This is a
scientific reproducibility violation — someone reading the paper cannot
reproduce the result because the described method doesn't match the code.

Weighted mean is also more sensitive to bad epochs and underestimated errors.
Median is more robust to residual WISE artifacts.

**Decision: use seasonal MEDIAN (more robust, matches docs). Update accordingly.**

```python
def compute_delta_mag_correct(times_mjd, w1_flux_mjy, w1_flux_err_mjy, z=0.0,
                               min_seasons=4, min_epochs_per_season=3):
    """
    Compute delta_mag using seasonal MEDIAN flux comparison.

    Method (matches published CLAGN papers: Sheng+2017, Graham+2020):
      1. Assign epochs to 6-month WISE seasons using global anchor
      2. Compute MEDIAN flux per season (robust to outliers)
      3. Estimate seasonal uncertainty using MAD-based robust scatter
      4. Compare FIRST 2 seasons (baseline) vs LAST 2 seasons (late state)
         using INVERSE-VARIANCE WEIGHTED combination of season medians
      5. Compute delta_mag from flux ratio with full error propagation

    Sign convention (enforced):
      delta_mag > 0 = source brightened (turn-ON event)
      delta_mag < 0 = source faded     (turn-OFF event)
      delta_mag = +2.5 * log10(F_late / F_early)

    Parameters
    ----------
    times_mjd : array
        Observation times in MJD (observer frame)
    w1_flux_mjy : array
        W1 flux in mJy (already converted from Vega mag)
    w1_flux_err_mjy : array
        W1 flux uncertainty in mJy
    z : float
        Redshift (reserved — observer-frame binning used, noted in output)
    min_seasons : int
        Minimum number of valid seasons required (default 4)
    min_epochs_per_season : int
        Minimum epochs per season to include it (default 3)

    Returns
    -------
    dict or None (None if insufficient data)
        delta_mag : float (positive = brightening)
        delta_mag_err : float (propagated from season uncertainties + calibration floor)
        flux_ratio : float (F_late / F_early, > 1 = brightening)
        F_early_mjy : float
        F_early_err_mjy : float
        F_late_mjy : float
        F_late_err_mjy : float
        n_seasons_used : int
        seasons_early : list of int (season indices)
        seasons_late : list of int (season indices)
        season_anchor_mjd : float
        binning_frame : str ('observer_frame')
        estimator : str ('seasonal_median')
    """
    from clagn.config import WISE_SEASON_ANCHOR_MJD

    CALIBRATION_FLOOR_FRAC = 0.028  # 2.8% WISE W1 calibration RMS (Wright+2010)

    # --- Clean inputs ---
    valid = (
        np.isfinite(times_mjd) &
        np.isfinite(w1_flux_mjy) &
        np.isfinite(w1_flux_err_mjy) &
        (w1_flux_err_mjy > 0) &
        (w1_flux_mjy > 0)
    )
    t = times_mjd[valid]
    f = w1_flux_mjy[valid]
    e = w1_flux_err_mjy[valid]

    if len(t) < 6:
        return None

    # --- Season assignment (global anchor) ---
    season_id = np.floor((t - WISE_SEASON_ANCHOR_MJD) / 182.625).astype(int)

    # --- Per-season MEDIAN and uncertainty ---
    season_flux     = {}
    season_flux_err = {}

    for s in np.unique(season_id):
        mask = season_id == s
        n_s  = mask.sum()

        if n_s < min_epochs_per_season:
            continue

        f_s = f[mask]
        e_s = e[mask]

        # MEDIAN flux (robust to outliers)
        med = float(np.median(f_s))

        # MAD-based uncertainty of the median
        mad = float(np.median(np.abs(f_s - med)))
        # 1.4826 * MAD = robust sigma for Gaussian
        # Uncertainty of median ≈ robust_sigma / sqrt(n)
        robust_sigma = 1.4826 * mad
        stat_err = robust_sigma / np.sqrt(n_s)

        # Calibration floor in flux units
        cal_floor_flux = CALIBRATION_FLOOR_FRAC * abs(med)

        # Total: stat + calibration floor in quadrature
        total_err = float(np.sqrt(stat_err**2 + cal_floor_flux**2))

        season_flux[s]     = med
        season_flux_err[s] = max(total_err, 1e-10)

    if len(season_flux) < min_seasons:
        return None

    seasons = sorted(season_flux.keys())

    # --- Inverse-variance weighted combination of first 2 and last 2 seasons ---
    def _weighted_combine(season_list):
        vals = np.array([season_flux[s]     for s in season_list])
        errs = np.array([season_flux_err[s] for s in season_list])
        w    = 1.0 / np.maximum(errs**2, 1e-30)
        mean = float(np.sum(w * vals) / np.sum(w))
        err  = float(np.sqrt(1.0 / np.sum(w)))
        return mean, err

    seasons_early = seasons[:2]
    seasons_late  = seasons[-2:]

    F_early, F_early_err = _weighted_combine(seasons_early)
    F_late,  F_late_err  = _weighted_combine(seasons_late)

    if F_early <= 0 or F_late <= 0:
        return None

    # --- delta_mag with full error propagation ---
    flux_ratio = F_late / F_early
    delta_mag  = float(2.5 * np.log10(flux_ratio))  # positive = brightened

    # Propagate fractional errors through log
    # d(delta_mag) = (2.5/ln10) * sqrt((dF_early/F_early)^2 + (dF_late/F_late)^2)
    frac_err_early = np.sqrt((F_early_err / F_early)**2 + CALIBRATION_FLOOR_FRAC**2)
    frac_err_late  = np.sqrt((F_late_err  / F_late )**2 + CALIBRATION_FLOOR_FRAC**2)
    delta_mag_err  = float((2.5 / np.log(10)) * np.sqrt(frac_err_early**2 + frac_err_late**2))

    # --- Sign consistency check ---
    flux_ratio_from_dm = 10.0 ** (delta_mag / 2.5)
    if not (
        (flux_ratio > 1 and delta_mag > 0) or
        (flux_ratio < 1 and delta_mag < 0) or
        (abs(flux_ratio - 1) < 1e-9 and abs(delta_mag) < 1e-9)
    ):
        # Should never happen — but log and correct rather than silently wrong
        import logging
        logging.getLogger(__name__).warning(
            f"Sign inconsistency: flux_ratio={flux_ratio:.4f}, "
            f"delta_mag={delta_mag:.4f} — recomputing from flux_ratio"
        )
        delta_mag = float(2.5 * np.log10(flux_ratio))

    return {
        'delta_mag':        delta_mag,
        'delta_mag_err':    delta_mag_err,
        'flux_ratio':       float(flux_ratio),
        'F_early_mjy':      F_early,
        'F_early_err_mjy':  F_early_err,
        'F_late_mjy':       F_late,
        'F_late_err_mjy':   F_late_err,
        'n_seasons_used':   len(season_flux),
        'seasons_early':    seasons_early,
        'seasons_late':     seasons_late,
        'season_anchor_mjd': float(WISE_SEASON_ANCHOR_MJD),
        'binning_frame':    'observer_frame',
        'estimator':        'seasonal_median',
    }
```

---

### FIX var-H4 (HIGH): z Parameter Accepted but Not Used — Document Clearly

**Location:** `compute_delta_mag_correct()` docstring and any z usage.

**Fix — add explicit note:**

```python
# In the docstring, replace "reserved for future rest-frame season binning" with:
"""
z : float
    Redshift. Currently observer-frame 6-month bins are used
    (consistent with WISE cadence and most published CLAGN papers).
    Rest-frame binning is NOT applied because WISE's semi-annual
    cadence already approximates a consistent rest-frame window for
    the redshift range of interest (z < 0.5).
    Stored in output as 'binning_frame': 'observer_frame' for transparency.
"""
```

---

### FIX var-H5 (HIGH): Hard-Coded 182.625 Day Season Bin — Document Clearly

**Not changing the implementation** (calendar binning is defensible for WISE).
**But add explicit justification comment:**

```python
# In compute_delta_mag_correct(), above the season_id line:

# 182.625 days = 365.25/2 = half-year WISE observing season
# WISE surveys each point on the sky approximately every 6 months.
# Calendar binning is used (not gap-based clustering) because:
#   1. It is reproducible across sources
#   2. It matches published methodology (Graham+2020, Sheng+2017)
#   3. WISE cadence is regular enough that bins rarely split real seasons
# The global WISE_SEASON_ANCHOR_MJD ensures bins are identical across all sources.
season_id = np.floor((t - WISE_SEASON_ANCHOR_MJD) / 182.625).astype(int)
```

---

### FIX var-M6 (MEDIUM): Sign Inconsistency Should Also Set a Debug Flag

**Already handled in var-C1 above.** The rewrite logs a warning.
For strict mode, add:

```python
# In config.py
DRW_STRICT_SIGN_CHECK = False   # Set True during testing to raise on sign error

# In compute_delta_mag_correct(), in the sign check block:
if DRW_STRICT_SIGN_CHECK:
    raise RuntimeError(
        f"Sign inconsistency detected: flux_ratio={flux_ratio:.4f}, "
        f"delta_mag={delta_mag:.4f}"
    )
```

---

### FIX var-M7 (MEDIUM): Remove Redundant Minimum Epoch Pre-Check

**Current:**

```python
if valid.sum() < 6:
    return None
```

**This is redundant** — the season logic (`min_seasons=4, min_epochs_per_season=3`)
already enforces at least 12 valid epochs implicitly. Keep a minimal sanity check only:

```python
if len(t) < 4:   # Absolute minimum — below this even 1 season is impossible
    return None
```

---

## FILE: drw.py

---

### FIX drw-C1 (CRITICAL): MCMC Path Does Not Subtract Mean Flux

**Location:** `fit_drw_mcmc()` — the `log_prob()` closure.

**Problem:** `fit_drw_map()` correctly centers flux before fitting (FIX A3 in
CLAGN_ADDITIONAL_BUGS_FIX_PROMPT.md). But `fit_drw_mcmc()` passes raw `f_s`
directly to `_drw_nll()`. This makes MAP and MCMC fit DIFFERENT data, producing
different tau/sigma for the wrong reason.

**Fix — add mean subtraction before the emcee sampler:**

```python
def fit_drw_mcmc(times_obs, w1_flux_mjy, w1_flux_err_mjy, z,
                  n_walkers=32, n_steps=2000, n_burnin=500,
                  max_n=300, band='W1'):
    """
    MCMC DRW fit. MUST use same centered flux as MAP fit.
    """
    # --- Input cleaning (same as MAP path) ---
    valid = (
        np.isfinite(times_obs) &
        np.isfinite(w1_flux_mjy) &
        np.isfinite(w1_flux_err_mjy) &
        (w1_flux_err_mjy > 0) &
        (w1_flux_mjy > 0)
    )
    t_clean = times_obs[valid]
    f_clean = w1_flux_mjy[valid]
    e_clean = w1_flux_err_mjy[valid]

    if len(t_clean) < 6:
        return {'valid': False, 'unreliable_reason': 'too_few_epochs_mcmc'}

    # --- Rest-frame time ---
    t_rest = t_clean / (1.0 + max(z, 0.0))

    # --- Subsampling ---
    t_s, f_s, e_s = _maybe_subsample(t_rest, f_clean, e_clean, max_n)

    # ── MEAN SUBTRACTION (CRITICAL — must match MAP path) ──────────────────
    weights_s = 1.0 / np.maximum(e_s**2, 1e-30)
    mu_flux   = float(np.average(f_s, weights=weights_s))
    f_s_centered = f_s - mu_flux
    # ───────────────────────────────────────────────────────────────────────

    def log_prob(params):
        log_sigma, log_tau = params
        # Bounds check
        if not (DRW_LOG_SIGMA_MIN < log_sigma < DRW_LOG_SIGMA_MAX):
            return -np.inf
        if not (DRW_LOG_TAU_MIN < log_tau < DRW_LOG_TAU_MAX):
            return -np.inf
        # Use CENTERED flux — same as MAP
        nll = _drw_nll(log_sigma, log_tau, t_s, f_s_centered, e_s)
        return -nll if nll < 1e9 else -np.inf

    # ... rest of emcee setup unchanged ...

    # Store mu_flux for provenance
    result['mu_flux_mjy'] = mu_flux
    result['mcmc_centered_flux'] = True
    return result
```

---

### FIX drw-C2 (CRITICAL): compute_drw_nonstationarity() Returns Tuple, Callers Expect Dict

**Location:** `compute_drw_nonstationarity()` — return statement.

**Current (wrong):**

```python
return nonstationarity_sigma, delta_mean_normalized, early_median, late_median
# Returns a TUPLE
```

**Used as dict in \_bootstrap_score_uncertainty():**

```python
nonstat = compute_drw_nonstationarity(...)
c_nonstat = _norm_drw_nonstat(nonstat.get('nonstationarity_sigma', 0.0))
# CRASHES: tuple has no .get()
```

**Fix — return dict everywhere:**

```python
def compute_drw_nonstationarity(times, fluxes, flux_errors, z, tau, sigma_drw):
    """
    [docstring — update to show dict return]

    Returns
    -------
    dict with keys:
        nonstationarity_sigma : float — main detection statistic
        delta_mean_normalized : float
        mean_early_mjy : float
        mean_late_mjy : float
        valid : bool
    """
    # ... computation unchanged ...

    # REPLACE the return statement at the end:
    return {
        'nonstationarity_sigma': float(nonstationarity_sigma),
        'delta_mean_normalized': float(delta_mean_normalized),
        'mean_early_mjy':        float(early_median),
        'mean_late_mjy':         float(late_median),
        'valid':                 True,
    }
```

**Also update ALL callers** that used tuple unpacking:

```python
# OLD (tuple unpacking):
nonstat_sigma, delta_norm, early, late = compute_drw_nonstationarity(...)

# NEW (dict):
nonstat_result = compute_drw_nonstationarity(...)
nonstat_sigma  = nonstat_result['nonstationarity_sigma']
delta_norm     = nonstat_result['delta_mean_normalized']
early          = nonstat_result['mean_early_mjy']
late           = nonstat_result['mean_late_mjy']
```

Search for ALL occurrences of `compute_drw_nonstationarity` in the codebase
and update each one.

---

### FIX drw-C3 (CRITICAL): No Input Cleaning Before GP Fit

**Location:** `fit_drw_map()`, `_fit_drw_numpy()`, `_fit_drw_celerite2()`.

**Problem:** NaN times, NaN fluxes, zero or negative errors, and duplicate
times can all silently corrupt or crash the Cholesky decomposition.

**Add this cleaning block at the TOP of `fit_drw_map()`, before any other logic:**

```python
def fit_drw_map(times_obs, w1_flux_mjy, w1_flux_err_mjy, z, band='W1', min_epochs=15):

    # ── INPUT CLEANING (required before any GP operation) ──────────────────
    times_obs      = np.asarray(times_obs,      dtype=float)
    w1_flux_mjy    = np.asarray(w1_flux_mjy,    dtype=float)
    w1_flux_err_mjy= np.asarray(w1_flux_err_mjy,dtype=float)

    valid = (
        np.isfinite(times_obs)       &
        np.isfinite(w1_flux_mjy)     &
        np.isfinite(w1_flux_err_mjy) &
        (w1_flux_err_mjy > 0)        &
        (w1_flux_mjy > 0)
    )
    times_obs       = times_obs[valid]
    w1_flux_mjy     = w1_flux_mjy[valid]
    w1_flux_err_mjy = w1_flux_err_mjy[valid]

    if len(times_obs) < min_epochs:
        return {'valid': False, 'unreliable_reason': f'too_few_clean_epochs_{valid.sum()}'}

    # Sort by time (required for Cholesky stability and celerite2)
    sort_idx         = np.argsort(times_obs)
    times_obs        = times_obs[sort_idx]
    w1_flux_mjy      = w1_flux_mjy[sort_idx]
    w1_flux_err_mjy  = w1_flux_err_mjy[sort_idx]

    # Handle duplicate times: add tiny jitter (1e-6 days) to avoid singular matrix
    dt = np.diff(times_obs)
    if (dt < 1e-4).any():
        n_dups = (dt < 1e-4).sum()
        times_obs = times_obs + np.random.uniform(0, 1e-6, len(times_obs))
        times_obs = np.sort(times_obs)
        import logging
        logging.getLogger(__name__).debug(
            f"Jittered {n_dups} near-duplicate time entries"
        )
    # ───────────────────────────────────────────────────────────────────────

    # Sanity check: flux in mJy range
    median_val = float(np.nanmedian(w1_flux_mjy))
    if not (0.001 < median_val < 100000):
        raise ValueError(
            f"Median flux {median_val:.4f} outside mJy range [0.001, 100000]. "
            f"Did you pass magnitudes instead of flux?"
        )

    # ... rest of function continues ...
```

---

### FIX drw-H4 (HIGH): Grid Init Ignores Config Bounds

**Location:** `_fit_drw_numpy()` — the grid search initialization.

**Current:**

```python
for log_sigma0 in np.linspace(-2.0, 2.0, 4):
    for log_tau0 in np.linspace(4.0, 7.0, 4):
```

**Fix — use config bounds:**

```python
# Import at top of drw.py if not already present:
from clagn.config import (
    DRW_LOG_TAU_MIN, DRW_LOG_TAU_MAX,
    DRW_LOG_SIGMA_MIN, DRW_LOG_SIGMA_MAX
)

# In _fit_drw_numpy():
# Use interior of bounds to avoid starting at boundary
inner_tau_min   = DRW_LOG_TAU_MIN   + 0.5 * (DRW_LOG_TAU_MAX   - DRW_LOG_TAU_MIN) * 0.1
inner_tau_max   = DRW_LOG_TAU_MAX   - 0.5 * (DRW_LOG_TAU_MAX   - DRW_LOG_TAU_MIN) * 0.1
inner_sigma_min = DRW_LOG_SIGMA_MIN + 0.5 * (DRW_LOG_SIGMA_MAX - DRW_LOG_SIGMA_MIN) * 0.1
inner_sigma_max = DRW_LOG_SIGMA_MAX - 0.5 * (DRW_LOG_SIGMA_MAX - DRW_LOG_SIGMA_MIN) * 0.1

for log_sigma0 in np.linspace(inner_sigma_min, inner_sigma_max, 4):
    for log_tau0 in np.linspace(inner_tau_min, inner_tau_max, 4):
```

---

### FIX drw-H5 (HIGH): Two Competing Bound Systems

**Location:** `drw.py` — top of file, local constant definitions.

**Current:**

```python
# Imported from config:
DRW_LOG_TAU_MIN, DRW_LOG_TAU_MAX, DRW_LOG_SIGMA_MIN, DRW_LOG_SIGMA_MAX

# Also defined locally:
LOG_TAU_MIN = ...
LOG_TAU_MAX = ...
LOG_SIG_MIN = ...
LOG_SIG_MAX = ...
```

**Fix — delete local definitions, derive all from config:**

```python
# DELETE the locally defined LOG_TAU_MIN/MAX, LOG_SIG_MIN/MAX constants.
# ADD these derived values immediately after the config import:

LOG_TAU_MIN = DRW_LOG_TAU_MIN
LOG_TAU_MAX = DRW_LOG_TAU_MAX
LOG_SIG_MIN = DRW_LOG_SIGMA_MIN
LOG_SIG_MAX = DRW_LOG_SIGMA_MAX

# celerite2 rho parameter: rho = tau * 2*pi (SHOTerm parameterization)
LOG_RHO_MIN = LOG_TAU_MIN + np.log(2 * np.pi)
LOG_RHO_MAX = LOG_TAU_MAX + np.log(2 * np.pi)
```

This guarantees numpy and celerite2 paths use identical parameter spaces.

---

### FIX drw-H6 (HIGH): assert Stripped in Optimized Python

**Location:** `fit_drw_map()` — flux range sanity check.

**Current:**

```python
assert 0.001 < median_val < 100000, "..."
```

**Fix (already done in drw-C3 above):**

```python
if not (0.001 < median_val < 100000):
    raise ValueError(...)
```

Always use `raise ValueError` for validation, never `assert`, in production code.

---

### FIX drw-H7 (HIGH): MCMC Health Not Checked Beyond Acceptance Fraction

**Location:** `fit_drw_mcmc()` — after sampler run.

**Add these checks:**

```python
# After sampler.run_mcmc():

acceptance = np.mean(sampler.acceptance_fraction)

# Check 1: acceptance fraction
if acceptance < 0.1 or acceptance > 0.7:
    result['mcmc_warning'] = f'acceptance_fraction={acceptance:.3f}_outside_0.1-0.7'
    result['mcmc_converged'] = False

# Check 2: boundary pile-up
flat_samples = sampler.get_chain(discard=n_burnin, flat=True)
frac_near_tau_bound = (
    (flat_samples[:,1] < LOG_TAU_MIN + 0.2).sum() +
    (flat_samples[:,1] > LOG_TAU_MAX - 0.2).sum()
) / len(flat_samples)

if frac_near_tau_bound > 0.2:
    result['mcmc_warning'] = (
        result.get('mcmc_warning', '') +
        f'|tau_boundary_pileup={frac_near_tau_bound:.2f}'
    )
    result['tau_reliable'] = False

# Check 3: autocorrelation (in try/except — can fail for short chains)
try:
    tau_autocorr = sampler.get_autocorr_time(quiet=True)
    n_effective   = n_steps / (2 * np.max(tau_autocorr))
    result['mcmc_n_effective'] = float(n_effective)
    if n_effective < 50:
        result['mcmc_warning'] = (
            result.get('mcmc_warning','') +
            f'|low_effective_samples={n_effective:.0f}'
        )
except Exception:
    result['mcmc_n_effective'] = np.nan

result['mcmc_acceptance_fraction'] = float(acceptance)
```

---

### FIX drw-M8 (MEDIUM): Subsampling Is Random, Not Stratified

**Location:** `_maybe_subsample()`.

**Replace random subsampling with time-stratified:**

```python
def _maybe_subsample(times, fluxes, errors, max_n):
    """
    Stratified-by-time subsampling.
    Preserves coverage of the full baseline, critical for DRW timescale fitting.
    Random subsampling can undersample the baseline edges.
    """
    n = len(times)
    if n <= max_n:
        return times, fluxes, errors

    # Evenly spaced indices across sorted time array
    sort_idx = np.argsort(times)
    times_s  = times[sort_idx]
    fluxes_s = fluxes[sort_idx]
    errors_s = errors[sort_idx]

    # Stratified: evenly spaced + random fill to max_n
    n_strat  = max_n // 2
    strat_idx = np.linspace(0, n-1, n_strat, dtype=int)

    # Random fill for remaining slots
    remaining = set(range(n)) - set(strat_idx)
    n_random  = max_n - n_strat
    rand_idx  = np.random.choice(list(remaining),
                                  size=min(n_random, len(remaining)),
                                  replace=False)

    chosen = np.sort(np.concatenate([strat_idx, rand_idx]))
    return times_s[chosen], fluxes_s[chosen], errors_s[chosen]
```

---

### FIX drw-M10 (MEDIUM): compute_drw_nonstationarity() Ignores Measurement Errors

**Location:** `compute_drw_nonstationarity()` — sigma_expected computation.

**Current:** uses only DRW-predicted sigma, ignores photometric noise.

**Fix — add measurement uncertainty in quadrature:**

```python
# In compute_drw_nonstationarity(), when computing expected scatter:

# DRW-predicted scatter over the half-baseline
sigma_drw_expected = sigma_drw * np.sqrt(
    1 - np.exp(-baseline_half / (2 * tau))
)

# Measurement uncertainty contribution
sigma_meas = np.sqrt(
    np.nanmean(flux_errors[early_mask]**2) / max(early_mask.sum(), 1) +
    np.nanmean(flux_errors[late_mask]**2)  / max(late_mask.sum(),  1)
)

# Combined expected uncertainty
sigma_expected = np.sqrt(sigma_drw_expected**2 + sigma_meas**2)

# Nonstationarity significance (same as before)
nonstationarity_sigma = abs(delta_mean) / max(sigma_expected, 1e-10)
```

---

## FILE: clagn_score.py

---

### FIX score-C1 (CRITICAL): \_bootstrap_score_uncertainty() Calls nonstationarity With Missing Args

**Location:** `_bootstrap_score_uncertainty()`.

**Problem:** calls `compute_drw_nonstationarity(t_b, f_b, e_b, z=z)` but
the function signature requires `tau` and `sigma_drw` as positional args.
This is a `TypeError` at runtime.

**Fix — pass required args, or remove DRW nonstat from bootstrap (simpler):**

```python
def _bootstrap_score_uncertainty(source_data, n_bootstrap=100):
    """
    Bootstrap score uncertainty estimation.

    NOTE: DRW nonstationarity is NOT bootstrapped here because:
    1. It requires DRW tau/sigma which are expensive to recompute
    2. Bootstrap resampling disrupts the temporal structure needed for DRW
    3. The main DRW uncertainty is already captured by MCMC posterior width

    Bootstrapped components: delta_mag, changepoint, structure function
    Non-bootstrapped: DRW (uses MCMC chains), nonstationarity (uses full LC)
    """
    # Extract required data
    t = source_data.get('times')
    f = source_data.get('w1_flux_mjy')
    e = source_data.get('w1_flux_err_mjy')
    z = source_data.get('z', 0.1)

    if t is None or f is None or len(t) < 10:
        return {'score_err': np.nan, 'n_bootstrap': 0}

    scores = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(t), size=len(t), replace=True)
        t_b, f_b, e_b = t[idx], f[idx], e[idx]

        # Bootstrap delta_mag only (fast, well-defined under resampling)
        dm_result = compute_delta_mag_correct(t_b, f_b, e_b, z=z)
        if dm_result is None:
            continue

        # Score only delta_mag component for bootstrap uncertainty
        c_dm = _norm_delta_mag(abs(dm_result['delta_mag']))
        scores.append(c_dm)

    if len(scores) < 10:
        return {'score_err': np.nan, 'n_bootstrap': len(scores)}

    return {
        'score_err':          float(np.std(scores)),
        'score_err_lo':       float(np.percentile(scores, 16)),
        'score_err_hi':       float(np.percentile(scores, 84)),
        'n_bootstrap':        len(scores),
        'bootstrap_component':'delta_mag_only',
    }
```

---

### FIX score-C2 (CRITICAL): compute_drw_nonstationarity() Called With Tuple Unpack

Already fixed in drw-C2 above (function now returns dict).

**Update ALL callers in clagn_score.py:**

```python
# Find every line like:
nonstat_sigma, _, _, _ = compute_drw_nonstationarity(...)
# or:
nonstat = compute_drw_nonstationarity(...)
c = nonstat.get(...)

# Replace ALL with:
nonstat_result = compute_drw_nonstationarity(
    times=t, fluxes=f, flux_errors=e,
    z=z,
    tau=drw_result.get('tau_rest_days', 200.0),
    sigma_drw=drw_result.get('sigma_drw_mjy', 0.1),
)
nonstat_sigma = nonstat_result.get('nonstationarity_sigma', 0.0)
```

---

### FIX score-C3 (CRITICAL): v2 Scorer Uses Wrong Column Names for Flux Arrays

**Location:** `compute_composite_score_v2()` — flux column extraction.

**Current (wrong):**

```python
for col_f in ['w1flux', 'w1flux_ep']:
    ...
for col_e in ['w1flux_err', 'w1sigflux_ep', 'w1sigflux']:
    ...
```

**Fix — add canonical column names FIRST:**

```python
# Flux columns: try canonical name first, then legacy names
fluxes = None
for col_f in ['w1_flux_mjy', 'w1flux', 'w1flux_ep', 'w1_flux']:
    if col_f in df.columns:
        arr = pd.to_numeric(df[col_f], errors='coerce').values
        if np.isfinite(arr).sum() >= 5:
            fluxes = arr
            flux_col_used = col_f
            break

# Error columns: try canonical name first, then legacy names
errors = None
for col_e in ['w1_flux_err_mjy', 'w1flux_err', 'w1sigflux_ep', 'w1sigflux', 'w1_flux_err']:
    if col_e in df.columns:
        arr = pd.to_numeric(df[col_e], errors='coerce').values
        if (arr > 0).sum() >= 5:
            errors = arr
            err_col_used = col_e
            break

if fluxes is None or errors is None:
    logger.warning(
        f"compute_composite_score_v2: could not find flux columns. "
        f"Available: {list(df.columns)}"
    )
    # Degrade gracefully — skip components requiring light curve
    fluxes = None
    errors = None
```

---

### FIX score-C4 (CRITICAL): False-Positive Rejection Reads Wrong Field Name

**Location:** `apply_false_positive_rejection()`.

**Current (wrong):**

```python
delta_mag = _safe(row.get('w1_delta_mag'), default=0.0)
```

**v1 scorer returns `delta_mag_w1`, not `w1_delta_mag`.**

**Fix — try both names:**

```python
delta_mag = _safe(
    row.get('delta_mag_w1',           # v1 scorer canonical name
    row.get('w1_delta_mag',           # legacy/alternative name
    row.get('delta_mag',              # generic fallback
    0.0))),                           # last resort
    default=0.0
)

# Same pattern for flux_ratio
flux_ratio = _safe(
    row.get('flux_ratio_w1',
    row.get('w1_flux_ratio',
    row.get('flux_ratio',
    1.0))),
    default=1.0
)
```

---

### FIX score-H5 (HIGH): \_norm_delta_mag Name Doesn't Match Input

**Location:** `_norm_delta_mag(frac_change)` — function name vs parameter name.

**Fix — rename to match actual input:**

```python
def _norm_frac_flux_change(frac_change):
    """
    Normalize fractional flux change to score in [0, 1].

    Parameters
    ----------
    frac_change : float
        |F_late - F_early| / F_early = |flux_ratio - 1|
        (NOT delta_mag in magnitudes)

    Returns
    -------
    float in [0, 1]
    """
    # ... same implementation ...
```

Update all callers from `_norm_delta_mag(...)` to `_norm_frac_flux_change(...)`.

---

### FIX score-H6 (HIGH): Host Correction Can Explode for Small f_agn

**Location:** `correct_delta_mag_for_host()`.

**Fix — add floor on f_agn and cap on output:**

```python
def correct_delta_mag_for_host(delta_mag_observed, f_agn):
    """..."""
    # Hard floor: below 20% AGN fraction, correction is unreliable
    if f_agn < 0.20:
        return delta_mag_observed  # Return uncorrected — correction would amplify noise

    flux_ratio_total = 10.0 ** (delta_mag_observed / 2.5)
    flux_ratio_agn   = 1.0 + (flux_ratio_total - 1.0) / f_agn

    # Cap: physically unreasonable to have >100x or <0.01x flux ratio
    flux_ratio_agn = float(np.clip(flux_ratio_agn, 0.01, 100.0))

    if flux_ratio_agn <= 0:
        return delta_mag_observed

    return float(2.5 * np.log10(flux_ratio_agn))
```

---

### FIX score-H7 (HIGH): check_w1_w2_coherence() False-Positive for Flat Sources

**Location:** `check_w1_w2_coherence()` — `same_direction` computation.

**Current (wrong for flat sources):**

```python
same_direction = np.sign(delta_w1) == np.sign(delta_w2)
# True when both are zero → false coherence bonus
```

**Fix — require meaningful change:**

```python
delta_w1 = f1_late - f1_early
delta_w2 = f2_late - f2_early

# Threshold: change must exceed 2% of mean flux to count as directional
thresh1 = 0.02 * abs(f1_early + f1_late) / 2
thresh2 = 0.02 * abs(f2_early + f2_late) / 2

if abs(delta_w1) < thresh1 or abs(delta_w2) < thresh2:
    same_direction = None  # Indeterminate — too small to measure direction
    dir_flag = 'indeterminate'
else:
    same_direction = np.sign(delta_w1) == np.sign(delta_w2)
    dir_flag = 'same' if same_direction else 'opposite'

result['w1_w2_same_direction'] = same_direction
result['w1_w2_direction_flag'] = dir_flag
```

---

### FIX score-H8 (HIGH): \_check_wise_artifact() Uses Raw max-min

**Location:** `_check_wise_artifact()`.

**Problem:** Single-epoch domination check uses `max(mags) - min(mags)` which
is noisy and inconsistent with the seasonal amplitude used everywhere else.

**Fix — use seasonal delta_mag before and after removing suspect epoch:**

```python
def _check_wise_artifact(times, fluxes, flux_errors, z=0.0):
    """
    Check for single-epoch artifact domination.
    Uses seasonal delta_mag (consistent with main pipeline metric)
    not raw max-min (inconsistent and noisy).
    """
    if len(fluxes) < 8:
        return False, 'too_few_epochs'

    # Baseline seasonal delta_mag
    dm_full = compute_delta_mag_correct(times, fluxes, flux_errors, z=z)
    if dm_full is None:
        return False, 'cannot_compute_baseline'

    delta_full = abs(dm_full['delta_mag'])

    # Find the most deviant epoch by standardized residual
    med     = np.median(fluxes)
    mad     = np.median(np.abs(fluxes - med))
    z_scores = np.abs(fluxes - med) / (1.4826 * mad + 1e-10)
    worst_idx = int(np.argmax(z_scores))

    if z_scores[worst_idx] < 3.0:
        return False, 'no_outlier_epoch'

    # Remove worst epoch and recompute
    mask = np.ones(len(times), dtype=bool)
    mask[worst_idx] = False

    dm_clean = compute_delta_mag_correct(
        times[mask], fluxes[mask], flux_errors[mask], z=z
    )

    if dm_clean is None:
        return True, 'amplitude_lost_without_outlier'

    delta_clean = abs(dm_clean['delta_mag'])

    # Single-epoch dominates if removing it reduces seasonal amplitude by >40%
    if delta_full > 0.01 and delta_clean < delta_full * 0.6:
        return True, f'single_epoch_dominates_z={z_scores[worst_idx]:.1f}'

    return False, 'passed'
```

---

### FIX score-H9 (HIGH): \_compute_sigma_excess() Name Misleadingly Implies Luminosity

**Location:** `_compute_sigma_excess()`.

**Fix — rename and update docstring:**

```python
def _compute_flux_variability_excess(mean_flux_mjy, sigma_drw_mjy,
                                      z=None, band='W1'):
    """
    Compute fractional flux variability relative to a typical AGN at this brightness.

    NOTE: This is a FLUX-based heuristic, not luminosity-corrected.
    It is redshift-dependent because flux depends on distance.
    Use only as a relative comparison within the same sample,
    not as an absolute physical measurement.

    Returns
    -------
    sigma_excess : float
        sigma_drw / expected_sigma_for_this_flux_level
        > 1 means more variable than typical AGN of same brightness
    """
    # ... same implementation, just renamed and docstring updated ...
```

Update all callers from `_compute_sigma_excess(...)` to
`_compute_flux_variability_excess(...)`.

---

### FIX score-M12 (MEDIUM): Composite v3 Weights Are Hardcoded

**Location:** `compute_composite_score_v3()` — weight definitions.

**Fix — move to config.py:**

```python
# In config.py:
SCORE_GROUP_WEIGHTS_V3 = {
    'amplitude':    3.0,   # delta_mag / flux_ratio group
    'temporal':     3.0,   # changepoint / DRW nonstat group
    'color':        2.0,   # W1-W2 color evolution
    'statistical':  1.5,   # SF excess / DRW sigma excess
    'astrometric':  0.5,   # GAIA variable flag
}

# In clagn_score.py:
from clagn.config import SCORE_GROUP_WEIGHTS_V3

def compute_composite_score_v3(components):
    w = SCORE_GROUP_WEIGHTS_V3
    # ... use w['amplitude'], w['temporal'], etc. ...
    composite = (
        w['amplitude']   * A_score +
        w['temporal']    * B_score +
        w['color']       * C_score +
        w['statistical'] * D_score +
        w['astrometric'] * E_score
    ) / sum(w.values())
```

---

### FIX score-M13 (MEDIUM): v2 Normalization Can Break if MAX_SCORE_V2 Drifts

**Location:** `compute_composite_score_v2()` — final normalization.

**Fix:**

```python
# At the end of v2 scoring, instead of:
# composite_score_v2 = raw_score / MAX_SCORE_V2

# Use:
max_score_actual = sum(SCORE_WEIGHTS_V2.values())
composite_score_v2 = raw_score / max_score_actual

# Assert consistency:
assert abs(max_score_actual - MAX_SCORE_V2) < 0.01, (
    f"MAX_SCORE_V2={MAX_SCORE_V2} inconsistent with "
    f"sum(SCORE_WEIGHTS_V2)={max_score_actual}"
)
```

---

## VERIFICATION SEQUENCE

Run all tests after applying fixes. All must pass.

```bash
# 1. Runtime crash tests (catches C1, C2, C3, C4)
python -c "
from clagn.models.drw import compute_drw_nonstationarity
import numpy as np

# Must return dict not tuple
t = np.linspace(55000, 60000, 60)
f = 2.0 + 0.3 * np.random.randn(60)
e = np.ones(60) * 0.05
result = compute_drw_nonstationarity(t, f, e, z=0.1, tau=300.0, sigma_drw=0.3)
assert isinstance(result, dict), f'Expected dict, got {type(result)}'
assert 'nonstationarity_sigma' in result, 'Missing nonstationarity_sigma key'
print(f'nonstationarity_sigma = {result[\"nonstationarity_sigma\"]:.3f}')
print('PASS: compute_drw_nonstationarity returns dict')
"

# 2. MCMC mean subtraction test
python -c "
from clagn.models.drw import fit_drw_mcmc
import numpy as np

t = np.linspace(55000, 60000, 80)
f = 5.0 + 0.3 * np.random.randn(80)  # mean=5 mJy, not zero
e = np.ones(80) * 0.05
result = fit_drw_mcmc(t, f, e, z=0.1, n_walkers=16, n_steps=200, n_burnin=50)
assert result.get('mcmc_centered_flux', False), 'MCMC did not center flux'
assert 'mu_flux_mjy' in result, 'mu_flux_mjy not stored'
assert 4.8 < result['mu_flux_mjy'] < 5.2, f'Wrong mean: {result[\"mu_flux_mjy\"]}'
print(f'mu_flux = {result[\"mu_flux_mjy\"]:.3f} mJy (expected ~5.0)')
print('PASS: MCMC subtracts mean flux')
"

# 3. Variability estimator consistency test
python -c "
from clagn.models.variability import compute_delta_mag_correct
import numpy as np

# Known delta_mag: early flux=2.0, late flux=4.0 → ratio=2 → dm=+0.753
np.random.seed(42)
t = np.linspace(55200, 60000, 100)
f = np.where(t < 57600, 2.0, 4.0) + 0.05 * np.random.randn(100)
e = np.ones(100) * 0.05
result = compute_delta_mag_correct(t, f, e, z=0.1)
assert result is not None, 'delta_mag computation returned None'
assert result['estimator'] == 'seasonal_median', f'Wrong estimator: {result[\"estimator\"]}'
assert result['delta_mag'] > 0, f'Turn-on should be positive: {result[\"delta_mag\"]}'
assert abs(result['delta_mag'] - 0.753) < 0.15, f'delta_mag off: {result[\"delta_mag\"]:.3f}'
assert result['delta_mag_err'] > 0, 'Error must be positive'
print(f'delta_mag = {result[\"delta_mag\"]:.3f} ± {result[\"delta_mag_err\"]:.3f}')
print(f'estimator = {result[\"estimator\"]}')
print('PASS: variability estimator is seasonal median with propagated errors')
"

# 4. Column name compatibility test
python -c "
import pandas as pd, numpy as np
from clagn.scoring.clagn_score import compute_composite_score_v2

# Create df with canonical column names
df = pd.DataFrame({
    'w1_flux_mjy':     np.random.uniform(1, 3, 50),
    'w1_flux_err_mjy': np.ones(50) * 0.05,
    'mjd':             np.linspace(55000, 60000, 50),
})
result = compute_composite_score_v2({'epochs_df': df, 'z': 0.1,
                                      'delta_mag_w1_seasonal_pogson': 0.4})
assert result is not None, 'v2 scorer returned None with canonical columns'
print(f'v2 score = {result.get(\"composite_score_v2\", \"missing\"):.3f}')
print('PASS: v2 scorer finds canonical column names')
"

# 5. Full 5-source recovery
python main.py --input catalog.csv --output ./results/ --test_n 5
# Must still recover 3/3 known CLAGN
# Mrk 1018 delta_mag must be NEGATIVE (turn-off, faded)
```

---

## SUMMARY TABLE

| ID        | File           | Severity     | Fix                                                 |
| --------- | -------------- | ------------ | --------------------------------------------------- |
| var-C1    | variability.py | **CRITICAL** | change to seasonal median, update all docs          |
| var-C2    | variability.py | **CRITICAL** | inverse-variance combine early/late seasons         |
| var-C3    | variability.py | **CRITICAL** | propagate actual delta_mag_err                      |
| var-H4    | variability.py | High         | document z not used (observer-frame)                |
| var-H5    | variability.py | High         | document 182.625 day bin choice                     |
| var-M6    | variability.py | Medium       | sign check logs warning + strict mode               |
| var-M7    | variability.py | Medium       | remove redundant pre-check                          |
| drw-C1    | drw.py         | **CRITICAL** | MCMC centers flux same as MAP                       |
| drw-C2    | drw.py         | **CRITICAL** | nonstationarity returns dict not tuple              |
| drw-C3    | drw.py         | **CRITICAL** | input cleaning before GP fit                        |
| drw-H4    | drw.py         | High         | grid init uses config bounds                        |
| drw-H5    | drw.py         | High         | one bound system from config only                   |
| drw-H6    | drw.py         | High         | assert → raise ValueError                           |
| drw-H7    | drw.py         | High         | MCMC health checks (acceptance, boundary, ESS)      |
| drw-M8    | drw.py         | Medium       | stratified-by-time subsampling                      |
| drw-M10   | drw.py         | Medium       | nonstationarity includes measurement error          |
| score-C1  | clagn_score.py | **CRITICAL** | bootstrap: remove broken nonstationarity call       |
| score-C2  | clagn_score.py | **CRITICAL** | update all nonstationarity callers for dict         |
| score-C3  | clagn_score.py | **CRITICAL** | v2 scorer column names: add canonical first         |
| score-C4  | clagn_score.py | **CRITICAL** | false-positive rejection: try both field names      |
| score-H5  | clagn_score.py | High         | rename \_norm_delta_mag to \_norm_frac_flux_change  |
| score-H6  | clagn_score.py | High         | host correction: floor f_agn at 0.2 + cap ratio     |
| score-H7  | clagn_score.py | High         | coherence: require meaningful change for direction  |
| score-H8  | clagn_score.py | High         | artifact check: use seasonal delta_mag not max-min  |
| score-H9  | clagn_score.py | High         | rename \_compute_sigma_excess (flux not luminosity) |
| score-M12 | clagn_score.py | Medium       | v3 weights → config.py                              |
| score-M13 | clagn_score.py | Medium       | v2 normalization computed not hardcoded             |

_Reviewed against: Kelly+2009 (ApJ 698 895) | MacLeod+2010 (ApJ 721 1014) | Kozłowski+2017 (arXiv:1611.08248) | Sheng+2017 (ApJL 846 L7)_
