# CLAGN Pipeline — Additional Bug Fix Prompt

### All bugs identified in code review. Apply after CLAGN_ULTIMATE_FIX_PROMPT.md.

### Severity: A = critical/high, B = medium scientific, C = reproducibility

---

## IMPORTANT: UNIT ERROR IN PREVIOUS FIX PROMPT

Before anything else, fix an error introduced by the previous fix prompt
(CLAGN_ULTIMATE_FIX_PROMPT.md Flaw 2 verification test). The example value
was wrong by a factor of ~1600.

**Correct calculation:**

```
W1 zero point = 309.540 Jy
At mag = 13.0:
F = 309.540 Jy × 10^(-0.4 × 13.0)
  = 309.540 × 6.310×10⁻⁶
  = 0.001953 Jy
  = 1.953 mJy   ← CORRECT VALUE
```

The previous prompt had `assert 3.09 < F < 3.11` which is wrong.
Replace that entire verification block with:

```python
# tests/test_photometry_units.py  (create this file)
import numpy as np
from clagn.ingestion.catalog import mag_to_flux_mjy, flux_mjy_to_mag

def test_w1_flux_conversion():
    """Values computed from official WISE formula: F = F0 * 10^(-0.4*mag)"""
    # W1 mag=13 → F = 309.540e3 mJy * 10^(-0.4*13) = 1.953 mJy
    F, dF = mag_to_flux_mjy(13.0, 0.02, 'W1')
    assert 1.90 < F < 2.00, f"W1 mag=13 flux wrong: {F:.4f} mJy (expected ~1.953)"
    assert 0.034 < dF < 0.040, f"W1 mag=13 flux error wrong: {dF:.4f} mJy"

def test_w2_flux_conversion():
    # W2 mag=13 → F = 171.787e3 mJy * 10^(-0.4*13) = 1.083 mJy
    F, dF = mag_to_flux_mjy(13.0, 0.02, 'W2')
    assert 1.05 < F < 1.12, f"W2 mag=13 flux wrong: {F:.4f} mJy (expected ~1.083)"

def test_roundtrip():
    """Convert mag→flux→mag and get back original value"""
    mag_in = 14.5
    F, dF = mag_to_flux_mjy(mag_in, 0.05, 'W1')
    mag_out, dmag_out = flux_mjy_to_mag(F, dF, 'W1')
    assert abs(mag_out - mag_in) < 1e-6, f"Roundtrip failed: {mag_in} → {mag_out}"

def test_brighter_magnitude_larger_flux():
    """Brighter (smaller mag number) must give larger flux"""
    F_bright, _ = mag_to_flux_mjy(12.0, 0.02, 'W1')
    F_faint,  _ = mag_to_flux_mjy(14.0, 0.02, 'W1')
    assert F_bright > F_faint, "Brighter source must have larger flux"

if __name__ == '__main__':
    test_w1_flux_conversion()
    test_w2_flux_conversion()
    test_roundtrip()
    test_brighter_magnitude_larger_flux()
    print("ALL PHOTOMETRY UNIT TESTS PASSED")
    print(f"  W1 mag=13 → {mag_to_flux_mjy(13.0, 0.02, 'W1')[0]:.4f} mJy")
    print(f"  W2 mag=13 → {mag_to_flux_mjy(13.0, 0.02, 'W2')[0]:.4f} mJy")
```

Run this first: `python tests/test_photometry_units.py`
All 4 tests must pass before proceeding.

---

## FLAW A2 (HIGH): WISE Season Binning Uses Source-Relative Anchor

### Problem

```python
# CURRENT (wrong):
season_id = np.floor((times - times.min()) / 182.625).astype(int)
```

This uses the first epoch of each source as the season anchor. Two sources
observed 3 months apart get different season boundaries. Results are not
reproducible or comparable across sources.

### Fix

```python
# In clagn/config.py — add this constant
WISE_SEASON_ANCHOR_MJD = 55200.0
# MJD 55200 = 2010-01-14, approximate start of WISE all-sky survey.
# All sources use this same anchor so season boundaries are identical.
# Season 0: MJD 55200–55382, Season 1: MJD 55383–55565, etc.

# In every function that bins WISE epochs into seasons:
# REPLACE:
#   season_id = np.floor((times - times.min()) / 182.625).astype(int)
# WITH:
season_id = np.floor(
    (times_mjd - WISE_SEASON_ANCHOR_MJD) / 182.625
).astype(int)
```

Files to update:

- `clagn/ingestion/wise_complete.py` — `compute_wise_seasonal_structure()`
- `clagn/models/drw.py` — any seasonal binning
- `clagn/scoring/clagn_score.py` — delta_mag seasonal computation
- `clagn/analysis/population.py` — any seasonal statistics

Add to output CSV: `wise_season_anchor_mjd = 55200.0` as a fixed metadata column.

---

## FLAW A3 (CRITICAL): DRW Fallback Does Not Subtract Mean Flux

### Problem

The NumPy fallback DRW path fits the GP likelihood on raw flux values.
The GP covariance model describes FLUCTUATIONS around a mean. If the mean
is not subtracted first, the optimizer partly explains the large DC offset
as stochastic variability, biasing both τ and σ.

### Fix

In `clagn/models/drw.py`, find the NumPy fallback `_drw_nll` function and
apply this fix:

```python
def fit_drw_numpy_fallback(times_rest, fluxes_mjy, flux_errors_mjy):
    """
    NumPy/scipy fallback DRW fit.
    CRITICAL: must subtract weighted mean before computing likelihood.
    """
    # --- MEAN SUBTRACTION (required for valid GP likelihood) ---
    weights = 1.0 / flux_errors_mjy**2
    mu_flux = np.average(fluxes_mjy, weights=weights)
    f_centered = fluxes_mjy - mu_flux
    # f_centered now has zero weighted mean — GP residuals are valid

    def _drw_nll(params):
        log_tau, log_sigma = params
        tau   = np.exp(log_tau)
        sigma = np.exp(log_sigma)

        n = len(times_rest)
        dt = np.abs(times_rest[:, None] - times_rest[None, :])

        # Ornstein-Uhlenbeck covariance: k(dt) = sigma^2 * exp(-|dt|/tau)
        K = sigma**2 * np.exp(-dt / tau)
        K += np.diag(flux_errors_mjy**2)   # Add measurement noise

        try:
            L = np.linalg.cholesky(K + 1e-10 * np.eye(n))
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, f_centered))
            log_det = 2 * np.sum(np.log(np.diag(L)))
            nll = 0.5 * (f_centered @ alpha + log_det + n * np.log(2*np.pi))
            return nll
        except np.linalg.LinAlgError:
            return 1e10

    # ... optimization continues ...

    # Return mu_flux for provenance
    result['mu_flux_mjy'] = mu_flux
    result['drw_fit_on_centered_flux'] = True
    return result
```

After this fix, verify: fit a constant light curve (all values = 5.0 mJy ± 0.1).
The fitted σ_DRW should be near zero, not near some large value driven by the mean.

---

## FLAW A4 (HIGH): celerite2 τ Conversion Must Be Validated

### Problem

The celerite2 SHOTerm parameterization uses `rho` (underdamped period) and
the conversion `tau = rho / (2*pi)` may not be the correct mapping to the
DRW (Ornstein-Uhlenbeck) timescale τ.

The DRW covariance is: `k(dt) = σ² exp(-|dt|/τ)`
The celerite2 SHOTerm with Q=0.5 has: `k(dt) ∝ exp(-dt/rho) * (1 + dt/rho)`
These are NOT the same kernel. The SHOTerm Q=0.5 approximates but does not
exactly reproduce the OU/DRW process.

### Fix

```python
def validate_celerite2_tau_mapping():
    """
    Unit test: simulate a DRW with known tau, fit with celerite2,
    check that recovered tau is within 30% of injected value.

    Run this once and report the calibration factor.
    """
    import numpy as np
    from scipy.stats import norm

    np.random.seed(42)
    tau_true = 300.0  # days
    sigma_true = 0.5  # mJy
    n = 100

    # Simulate DRW using exact OU process
    times = np.sort(np.random.uniform(55000, 60000, n))
    flux = np.zeros(n)
    flux[0] = sigma_true * norm.rvs()
    for i in range(1, n):
        dt = times[i] - times[i-1]
        e_factor = np.exp(-dt / tau_true)
        flux[i] = (e_factor * flux[i-1] +
                   np.sqrt(1 - e_factor**2) * sigma_true * norm.rvs())

    errors = np.ones(n) * 0.05
    flux += errors * norm.rvs(size=n)

    # Fit with celerite2
    from clagn.models.drw import fit_drw_map
    result = fit_drw_map(times, flux + 2.0, errors, z=0.0)

    tau_recovered = result['tau_rest_days']
    ratio = tau_recovered / tau_true

    print(f"Injected tau: {tau_true:.0f} d")
    print(f"Recovered tau: {tau_recovered:.0f} d")
    print(f"Ratio: {ratio:.3f} (acceptable range: 0.5 – 2.0)")

    if 0.5 < ratio < 2.0:
        print("PASS: tau recovery within acceptable range")
    else:
        print("FAIL: tau mapping may be incorrect — review celerite2 parameterization")
        print("Fix: adjust tau = rho * CALIBRATION_FACTOR in celerite2 path")

    return ratio
```

Add this as `tests/test_drw_recovery.py` and run it. If ratio is outside 0.5–2.0,
the celerite2 path needs a calibration correction factor or must be replaced
with the numpy fallback as the primary path.

---

## FLAW A5 (HIGH): No Parameter Bounds in celerite2 Optimization

### Problem

The celerite2 path uses unconstrained minimization. The NumPy fallback correctly
uses bounds. This asymmetry means the two paths can return inconsistent results.

### Fix

In `clagn/models/drw.py`, find the celerite2 optimization call and add bounds:

```python
from scipy.optimize import minimize

# These bounds match the NumPy fallback — must be identical in both paths
LOG_TAU_MIN = np.log(2.0)       # tau_min = 2 days
LOG_TAU_MAX = np.log(5000.0)    # tau_max = 5000 days
LOG_SIG_MIN = np.log(1e-4)      # sigma_min = 0.0001 mJy
LOG_SIG_MAX = np.log(1e4)       # sigma_max = 10000 mJy

bounds = [(LOG_TAU_MIN, LOG_TAU_MAX),
          (LOG_SIG_MIN, LOG_SIG_MAX)]

result = minimize(
    nll,
    x0=[np.log(200.0), np.log(np.std(f_centered) + 1e-6)],
    method='L-BFGS-B',
    bounds=bounds,           # ADD THIS
    options={'maxiter': 1000, 'ftol': 1e-9}
)

# Post-fit: check if solution is at a boundary
log_tau_fit, log_sig_fit = result.x
at_boundary = (
    log_tau_fit < LOG_TAU_MIN + 0.1 or
    log_tau_fit > LOG_TAU_MAX - 0.1 or
    log_sig_fit < LOG_SIG_MIN + 0.1 or
    log_sig_fit > LOG_SIG_MAX - 0.1
)
if at_boundary:
    drw_result['tau_reliable'] = False
    drw_result['unreliable_reason'] = 'solution_at_parameter_boundary'
```

---

## FLAW A6 (HIGH): GAIA Parallax Not Used for Star Rejection

_(This was also in CLAGN_ULTIMATE_FIX_PROMPT.md Flaw 9 — confirming it is needed)_

### Fix

In `clagn/ingestion/gaia_lightcurve.py` or `clagn/ingestion/gaia.py`,
add `parallax` and `parallax_error` to the ADQL query and rejection logic:

```python
# Add to GAIA ADQL query SELECT columns:
# parallax, parallax_error

def is_foreground_star(gaia_row):
    """
    Reject if ANY of these three criteria trigger.
    All three must be checked — not just PM.
    """
    reasons = []

    # 1. Proper motion
    try:
        pm = np.sqrt(gaia_row['pmra']**2 + gaia_row['pmdec']**2)
        pm_err = np.sqrt(gaia_row['pmra_error']**2 + gaia_row['pmdec_error']**2)
        pm_sig = pm / (pm_err + 1e-10)
        if pm_sig > 3.0:
            reasons.append(f'pm_sig={pm_sig:.1f}')
    except (KeyError, TypeError):
        pass  # PM not available — do not penalize

    # 2. Parallax (direct distance measurement — strongest test)
    try:
        plx = gaia_row['parallax']
        plx_err = gaia_row['parallax_error']
        if plx is not None and plx_err is not None and plx_err > 0:
            plx_sig = abs(plx) / plx_err
            if plx_sig > 3.0:
                reasons.append(f'parallax_sig={plx_sig:.1f}')
    except (KeyError, TypeError):
        pass  # Parallax not available

    # 3. RUWE
    try:
        if gaia_row['ruwe'] > 1.4:
            reasons.append(f'ruwe={gaia_row["ruwe"]:.2f}')
    except (KeyError, TypeError):
        pass

    return len(reasons) > 0, reasons

# Update output CSV:
# gaia_rejection_reason: comma-separated list or None
# gaia_parallax_sig: float or NaN
```

---

## FLAW A7 (HIGH): W2 Quality Filtering Weaker Than W1

### Problem

W1 quality cuts include: SNR, chi2, saturation, cc_flags, ph_qual.
W2 cuts may be missing some of these, causing inconsistent band quality.

### Fix

In `clagn/ingestion/wise_complete.py`, ensure symmetric quality masks:

```python
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
        (table['w1snr']    >= 5.0)   &
        (table['w1rchi2']  <  5.0)   &
        (table['w1sat']    == 0)      &
        (table['w1sigmpro'] > 0)      &
        np.array([c[0] in ('0','H','h')
                  for c in table['cc_flags']])  # W1 cc_flag
    )

    # W2 quality criteria — SAME stringency as W1
    mask_w2 = (
        (table['w2snr']    >= 5.0)   &    # Same SNR threshold
        (table['w2rchi2']  <  5.0)   &    # Same chi2 threshold
        (table['w2sat']    == 0)      &    # Saturation check
        (table['w2sigmpro'] > 0)      &    # Valid uncertainty
        np.array([c[1] in ('0','H','h')
                  for c in table['cc_flags']])  # W2 cc_flag (char index 1)
    )

    mask_joint = mask_w1 & mask_w2

    return mask_w1, mask_w2, mask_joint
```

Update all downstream functions to use the appropriate mask:

- DRW fitting on W1: use `mask_w1`
- DRW fitting on W2: use `mask_w2`
- W1-W2 color computation: use `mask_joint`
- W1-W2 coherence test: use `mask_joint`

Add to output CSV:

- `n_epochs_w1_good`
- `n_epochs_w2_good`
- `n_epochs_joint_good`

---

## FLAW B1 (MEDIUM): Delta-mag Uncertainty Uses Hardcoded Floor Only

### Fix

Replace the hardcoded 5% floor with empirical seasonal scatter:

```python
def compute_seasonal_flux_and_uncertainty(times, fluxes, errors, season_id):
    """
    Per season: compute weighted mean AND uncertainty of weighted mean.
    Combine with calibration floor in quadrature.

    Returns per-season: mean, total_err, n_epochs, intra_season_scatter
    """
    WISE_CALIBRATION_FLOOR_FRACTION = 0.028  # 2.8% from WISE docs (W1 RMS)

    results = {}
    for s in np.unique(season_id):
        mask = season_id == s
        if mask.sum() < 2:
            continue

        f_s = fluxes[mask]
        e_s = errors[mask]
        w_s = 1.0 / e_s**2

        # Weighted mean
        mean_flux = np.average(f_s, weights=w_s)

        # Formal uncertainty of weighted mean
        formal_err = 1.0 / np.sqrt(w_s.sum())

        # Intra-season scatter (may exceed formal error due to real variability)
        scatter = np.sqrt(np.average((f_s - mean_flux)**2, weights=w_s))

        # Calibration floor
        cal_floor = WISE_CALIBRATION_FLOOR_FRACTION * mean_flux

        # Total uncertainty: max of formal, scatter/sqrt(N), floor
        total_err = np.sqrt(
            max(formal_err, scatter / np.sqrt(mask.sum()))**2 +
            cal_floor**2
        )

        results[s] = {
            'mean_flux': mean_flux,
            'total_err': total_err,
            'n_epochs': mask.sum(),
            'intra_season_scatter': scatter,
            'formal_err': formal_err,
        }

    return results
```

---

## FLAW B2 (MEDIUM): Saturation Check Uses Global Median Only

### Fix

```python
def check_wise_saturation_complete(w1_mags, w2_mags, times):
    """
    Check saturation on per-epoch basis, not just median.
    A source entering saturation during a bright state is the most
    dangerous case — exactly the epochs most relevant to CLAGN detection.
    """
    W1_SAT_LIMIT = 8.0   # mag, Vega (from WISE docs)
    W2_SAT_LIMIT = 6.7

    w1_sat_mask = w1_mags < W1_SAT_LIMIT
    w2_sat_mask = w2_mags < W2_SAT_LIMIT

    return {
        'w1_any_saturated': bool(w1_sat_mask.any()),
        'w2_any_saturated': bool(w2_sat_mask.any()),
        'n_w1_saturated_epochs': int(w1_sat_mask.sum()),
        'n_w2_saturated_epochs': int(w2_sat_mask.sum()),
        'w1_min_mag': float(w1_mags.min()),  # Brightest observed
        'w2_min_mag': float(w2_mags.min()),
        'w1_median_mag': float(np.median(w1_mags)),
        'saturation_flag': (
            'saturated' if (w1_sat_mask.any() or w2_sat_mask.any())
            else 'clear'
        )
    }
```

---

## FLAW B3 (MEDIUM): W1/W2 Coherence Not Uncertainty-Weighted

### Fix

```python
def compute_w1_w2_coherence(w1_seasonal, w2_seasonal):
    """
    Compute coherence using only joint valid seasons,
    weighted by combined seasonal uncertainty.

    w1_seasonal, w2_seasonal: dicts from compute_seasonal_flux_and_uncertainty()
    """
    # Find seasons valid in BOTH bands
    joint_seasons = sorted(
        set(w1_seasonal.keys()) & set(w2_seasonal.keys())
    )

    if len(joint_seasons) < 4:
        return {
            'w1_w2_pearson_r': np.nan,
            'n_joint_seasons': len(joint_seasons),
            'coherence_flag': 'insufficient_joint_seasons',
            'coherence_score': 0.5  # Neutral — cannot determine
        }

    w1_vals = np.array([w1_seasonal[s]['mean_flux'] for s in joint_seasons])
    w2_vals = np.array([w2_seasonal[s]['mean_flux'] for s in joint_seasons])
    w1_errs = np.array([w1_seasonal[s]['total_err'] for s in joint_seasons])
    w2_errs = np.array([w2_seasonal[s]['total_err'] for s in joint_seasons])

    # Combined weight per season
    combined_weights = 1.0 / (w1_errs**2 + w2_errs**2)
    combined_weights /= combined_weights.sum()

    # Weighted Pearson r
    w1_wmean = np.average(w1_vals, weights=combined_weights)
    w2_wmean = np.average(w2_vals, weights=combined_weights)

    cov = np.sum(combined_weights * (w1_vals - w1_wmean) * (w2_vals - w2_wmean))
    std1 = np.sqrt(np.sum(combined_weights * (w1_vals - w1_wmean)**2))
    std2 = np.sqrt(np.sum(combined_weights * (w2_vals - w2_wmean)**2))

    r = cov / (std1 * std2 + 1e-10)

    # Sign consistency: do W1 and W2 change in the same direction?
    delta_w1 = w1_vals[-1] - w1_vals[0]
    delta_w2 = w2_vals[-1] - w2_vals[0]
    same_direction = np.sign(delta_w1) == np.sign(delta_w2)

    # Coherence score
    coherence_score = max(0, r) * (1.0 if same_direction else 0.5)

    if r > 0.6 and same_direction:
        flag = 'coherent'
    elif r > 0.3:
        flag = 'marginal'
    else:
        flag = 'incoherent'

    return {
        'w1_w2_pearson_r': float(r),
        'n_joint_seasons': len(joint_seasons),
        'w1_w2_same_direction': bool(same_direction),
        'coherence_flag': flag,
        'coherence_score': float(coherence_score),
        'coherence_weighted': True  # Provenance marker
    }
```

---

## FLAW B4 (MEDIUM): Host Correction Must Be Labeled as Estimate

### Fix — no code change, only labeling and output change

```python
# Rename these output columns:
# delta_mag_host_corrected      → delta_mag_host_corrected_estimate
# host_contamination_fraction   → host_fraction_prior_estimate

# Add to output CSV:
# host_model = 'redshift_prior_heuristic_Assef2013'
# host_correction_note = 'Rough prior; not source-specific SED decomposition'

# For CLAGN ranking: always use delta_mag_observed (conservative)
# For physical interpretation: report both, label estimate clearly
```

---

## FLAW B5 (MEDIUM): Unreliable tau Must Hard-Block Physics Everywhere

### Fix — audit and add assertions

In `clagn/analysis/physics.py`, find every use of `tau_rest_days` and add:

```python
def _require_reliable_tau(drw_result, context='unknown'):
    """
    Call this before any physical computation that uses DRW tau.
    Raises ValueError if tau is not reliable.
    """
    if not drw_result.get('tau_reliable', False):
        reason = drw_result.get('unreliable_reason', 'unknown')
        raise ValueError(
            f"Attempted to use unreliable DRW tau in {context}. "
            f"Reason: {reason}. "
            f"This computation must be skipped for this source."
        )

# Usage: wrap every physics function that uses tau:
def estimate_black_hole_mass(drw_result, l_bol, z):
    _require_reliable_tau(drw_result, context='BH_mass_estimation')
    tau = drw_result['tau_rest_days']
    # ... rest of computation ...
```

---

## FLAW C1 (MEDIUM): No Provenance Metadata Sidecar

### Fix

Create `clagn/utils/provenance.py`:

```python
import json
import datetime
import subprocess
import numpy as np
from clagn import config

def get_git_hash():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return 'unknown'

def write_provenance_sidecar(output_csv_path, pipeline_config):
    """
    Write a JSON sidecar next to every output CSV.

    Example output: top_candidates.csv → top_candidates_provenance.json
    """
    provenance = {
        'generated_at': datetime.datetime.utcnow().isoformat() + 'Z',
        'git_commit': get_git_hash(),
        'python_version': __import__('sys').version,

        # All threshold values that affect results
        'thresholds': {
            'min_baseline_years': pipeline_config.get('min_baseline_years', 10.0),
            'min_epochs': pipeline_config.get('min_epochs', 20),
            'min_delta_mag': pipeline_config.get('min_delta_mag', 0.3),
            'min_changepoint_bic': pipeline_config.get('min_changepoint_bic', 6.0),
            'min_drw_nonstat_sigma': pipeline_config.get('min_drw_nonstat_sigma', 2.0),
            'gaia_max_pm_sig': 3.0,
            'gaia_max_parallax_sig': 3.0,
            'gaia_max_ruwe': 1.4,
            'wise_min_snr': 5.0,
            'wise_season_anchor_mjd': config.WISE_SEASON_ANCHOR_MJD,
        },

        # All physical constants used
        'constants': {
            'wise_w1_zero_point_jy': config.WISE_VEGA_ZERO_POINTS['W1'],
            'wise_w2_zero_point_jy': config.WISE_VEGA_ZERO_POINTS['W2'],
            'wise_hibernation_start_mjd': 55593,
            'wise_hibernation_end_mjd': 56141,
        },

        # Scoring weights used
        'scoring_weights': {
            'w_amplitude': 3.0,
            'w_temporal': 3.0,
            'w_color': 2.0,
            'w_statistical': 1.5,
            'w_astrometric': 0.5,
            'scoring_version': 'v3_no_double_counting',
        },

        # DRW configuration
        'drw': {
            'backend': 'celerite2_with_numpy_fallback',
            'fit_in_flux_space': True,
            'mean_subtracted_before_fit': True,
            'rest_frame_correction_applied': True,
            'kozlowski2017_reliability_check': True,
        },

        # Delta-mag method
        'delta_mag': {
            'method': 'seasonal_weighted_mean_flux_comparison',
            'season_anchor_mjd': config.WISE_SEASON_ANCHOR_MJD,
            'sign_convention': 'positive_is_brightening',
            'early_seasons': 'first_2',
            'late_seasons': 'last_2',
        }
    }

    sidecar_path = str(output_csv_path).replace('.csv', '_provenance.json')
    with open(sidecar_path, 'w') as f:
        json.dump(provenance, f, indent=2)

    return sidecar_path
```

Call `write_provenance_sidecar()` in `main.py` after writing every CSV output.

---

## FLAW C2 (LOW): Unit Example Values Were Wrong in Previous Prompt

Fixed at the top of this document. After applying the fix, run:

```bash
python tests/test_photometry_units.py
```

All 4 tests must pass. If any fail, the flux conversion function is broken.

---

## FLAW C3 (LOW): mag_to_flux_mjy Breaks on Scalar Input

### Fix

```python
def mag_to_flux_mjy(mag, mag_err, band):
    """[existing docstring]"""
    # Normalize to arrays — handle scalar inputs safely
    scalar_input = np.isscalar(mag)
    mag     = np.atleast_1d(np.asarray(mag,     dtype=float))
    mag_err = np.atleast_1d(np.asarray(mag_err, dtype=float))

    F0_mJy = WISE_VEGA_ZERO_POINTS[band] * 1000.0
    valid = (mag > 0) & (mag < 30) & np.isfinite(mag)

    flux_mjy = np.where(valid, F0_mJy * 10**(-0.4 * mag), np.nan)
    flux_err_mjy = np.where(
        valid & (mag_err > 0),
        flux_mjy * 0.92103 * mag_err,
        np.nan
    )

    # Return scalar if scalar was given
    if scalar_input:
        return float(flux_mjy[0]), float(flux_err_mjy[0])
    return flux_mjy, flux_err_mjy
```

---

## FLAW C4 (LOW): Terminology Cleanup

Search and replace these terms throughout all code and docstrings:

| Find                        | Replace with                                                 |
| --------------------------- | ------------------------------------------------------------ |
| "seasonal median"           | "seasonal weighted mean" (unless actual median is used)      |
| "delta mag"                 | "delta_mag (positive=brightening, Pogson scale)"             |
| "tau" without qualifier     | "tau_rest_days (rest-frame, observer/1+z)"                   |
| "sigma_DRW" without units   | "sigma_DRW_mjy (flux space, mJy)"                            |
| "reliable" without citation | "tau_reliable (Kozłowski+2017 criterion: baseline ≥ 10×tau)" |
| "host corrected"            | "host_corrected_estimate (redshift-prior heuristic)"         |

---

## COMPLETE VERIFICATION SEQUENCE

Run ALL of these after applying all fixes. Every test must pass.

```bash
# 1. Photometry unit tests (catches the C2 unit error)
python tests/test_photometry_units.py

# 2. DRW recovery test (catches A4 tau mapping error)
python tests/test_drw_recovery.py

# 3. Season anchor test (catches A2)
python -c "
from clagn.ingestion.wise_complete import compute_wise_seasonal_structure
from clagn.config import WISE_SEASON_ANCHOR_MJD
import numpy as np
# Two sources observed 90 days apart should have same season boundaries
t1 = np.array([55200., 55380., 55560.])
t2 = t1 + 90.0
s1 = np.floor((t1 - WISE_SEASON_ANCHOR_MJD) / 182.625).astype(int)
s2 = np.floor((t2 - WISE_SEASON_ANCHOR_MJD) / 182.625).astype(int)
# Season boundaries are fixed to anchor, not source-relative
print('Season IDs source 1:', s1)
print('Season IDs source 2:', s2)
print('Anchor is global:', WISE_SEASON_ANCHOR_MJD)
print('PASS: season anchor test')
"

# 4. Mean subtraction test (catches A3)
python -c "
from clagn.models.drw import fit_drw_numpy_fallback
import numpy as np
np.random.seed(0)
# Constant light curve — DRW sigma should be near zero
t = np.linspace(55000, 60000, 50)
f = np.ones(50) * 5.0 + np.random.randn(50) * 0.05
e = np.ones(50) * 0.05
result = fit_drw_numpy_fallback(t, f, e)
print(f'sigma_DRW on constant LC: {result[\"sigma_drw_mjy\"]:.4f} mJy')
print(f'mu_flux: {result[\"mu_flux_mjy\"]:.4f} mJy (should be ~5.0)')
assert result['drw_fit_on_centered_flux'] == True, 'Mean subtraction not applied'
assert result['mu_flux_mjy'] > 4.9, 'Mean flux wrong'
print('PASS: mean subtraction test')
"

# 5. Known CLAGN recovery (must still be 3/3)
python main.py --input catalog.csv --output ./results/ --test_n 5
# Expected:
# Mrk 1018    rank 1   delta_mag < 0  (turn-off, faded)
# Mrk 590     rank 2
# HE 1136     rank 3

# 6. Provenance sidecar exists after run
python -c "
import os
assert os.path.exists('results/top_candidates_provenance.json'), \
    'Provenance sidecar not written'
import json
p = json.load(open('results/top_candidates_provenance.json'))
assert p['delta_mag']['sign_convention'] == 'positive_is_brightening'
assert p['drw']['fit_in_flux_space'] == True
assert p['drw']['mean_subtracted_before_fit'] == True
print('PASS: provenance sidecar test')
print('git commit:', p['git_commit'])
"
```

---

## SUMMARY

| ID  | Flaw                                                 | Severity     | Primary File                     |
| --- | ---------------------------------------------------- | ------------ | -------------------------------- |
| —   | Unit error in previous prompt (W1 mag=13 ≠ 3099 mJy) | **CRITICAL** | `tests/test_photometry_units.py` |
| A2  | Season binning source-relative not global            | High         | `config.py`, `wise_complete.py`  |
| A3  | DRW fallback does not subtract mean flux             | **CRITICAL** | `models/drw.py`                  |
| A4  | celerite2 tau mapping unvalidated                    | High         | `models/drw.py`, `tests/`        |
| A5  | No bounds in celerite2 optimization                  | High         | `models/drw.py`                  |
| A6  | GAIA parallax not used for star rejection            | High         | `ingestion/gaia.py`              |
| A7  | W2 quality filtering weaker than W1                  | High         | `ingestion/wise_complete.py`     |
| B1  | Delta-mag uncertainty hardcoded floor only           | Medium       | `wise_complete.py`               |
| B2  | Saturation check uses global median only             | Medium       | `wise_complete.py`               |
| B3  | W1/W2 coherence not uncertainty-weighted             | Medium       | `scoring/clagn_score.py`         |
| B4  | Host correction not labeled as estimate              | Medium       | `analysis/physics.py`, CSVs      |
| B5  | Unreliable tau not hard-blocked downstream           | Medium       | `analysis/physics.py`            |
| C1  | No provenance metadata sidecar                       | Medium       | `utils/provenance.py` (new)      |
| C2  | Unit example values wrong in previous prompt         | Low          | `tests/test_photometry_units.py` |
| C3  | Scalar input breaks mag_to_flux_mjy                  | Low          | `ingestion/catalog.py`           |
| C4  | Terminology drift in docs/code                       | Low          | all files                        |

_Identified by code review. References: Kozłowski+2017 (arXiv:1611.08248) | Wright+2010 (AJ 140 1868)_
