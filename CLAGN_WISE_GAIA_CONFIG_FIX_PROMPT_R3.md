# CLAGN Pipeline — Master Scientific Validation Standard

### The standard required to publish at a level no referee can reject

### This document defines WHAT must be true, not just what must run.

### Every section has a PASS/FAIL criterion. Nothing is optional.

---

## PHILOSOPHY

Engineering sophistication is necessary but not sufficient.
A pipeline that runs perfectly on bad assumptions produces wrong science faster.

This document defines the difference between:

- "We built a pipeline that finds things that look like CLAGN" (not publishable)
- "We present a validated selection of changing-state AGN candidates with
  measured purity, completeness, and spectroscopic confirmation of a subset"
  (publishable at ApJ/MNRAS level)

Every section below has a BINARY outcome: PASS or FAIL.
A paper cannot be submitted until every section is PASS.

---

## PART 1: SCIENTIFIC DEFINITION LOCK

### 1.1 — Define "Changing-Look" Operationally

**Why:** "Big variability" ≠ changing-look. Referees will ask this on page 1.

**Required deliverable:** A written definition section (goes into paper Section 2)
stating EXACTLY what this pipeline selects and what it does not claim.

```
CONFIRMED CLAGN (Tier 1):
  - Multi-epoch spectroscopy showing broad line appearance/disappearance
  - Seyfert type transition (1→2, 2→1, or intermediate)
  - Reference: Tohline & Osterbrock 1976, LaMassa+2015, etc.
  - THIS PIPELINE CANNOT PRODUCE TIER 1 WITHOUT SPECTROSCOPY

STRONG PHOTOMETRIC CANDIDATES (Tier 2):
  - IR amplitude |delta_mag| > threshold (calibrated from injection-recovery)
  - Multi-wavelength corroboration (optical variability, color change)
  - Stellar contamination excluded (Gaia)
  - Artifact contamination excluded (quality flags, coherence)
  - NOT yet confirmed changing-look — confirmed changing-state CANDIDATE

LIKELY CONTAMINANTS (Tier 3):
  - SNe in host galaxies
  - Blazars
  - Dust obscuration events
  - Instrumental artifacts
  - These must be explicitly enumerated and removed

AMBIGUOUS / REQUIRES FOLLOW-UP (Tier 4):
  - Passes photometric cuts but lacks external corroboration
  - Enters ranked list with explicit uncertainty flag
```

**PASS criterion:** This table exists in a `definitions.md` file AND is
reproduced verbatim in the paper methods section.

---

## PART 2: BENCHMARK DATASET

### 2.1 — Positive Set: Known CLAGN

**Why:** Without known positives, "recovery" is meaningless.

**Required:**

```python
# File: data/benchmark/known_clagn.csv
# Required columns:
KNOWN_CLAGN_REQUIRED_COLS = [
    'name',              # source name
    'ra',                # degrees
    'dec',               # degrees
    'redshift',          # spectroscopic z
    'transition_type',   # turn_on / turn_off / unknown
    'transition_epoch_mjd',  # approximate MJD of transition (or NaN)
    'delta_mag_reported',    # literature value if available
    'reference',         # ADS bibcode or DOI
    'tier',              # 1=spectroscopic, 2=photometric+multiwave
]

# Minimum required sources:
MIN_KNOWN_CLAGN = 30
# Sources to include (at minimum):
REQUIRED_BENCHMARK_SOURCES = [
    'Mrk 1018',    # canonical turn-off (Cohen+1986, McElroy+2016)
    'Mrk 590',     # canonical turn-off (Denney+2014)
    'NGC 2617',    # turn-on (Shappee+2014)
    'HE 1136-2304',# turn-on (Parker+2016)
    'NGC 1566',    # complex (Oknyansky+2019)
    # ... add all available from literature lists
]
```

**Literature sources to mine for benchmark:**

- Graham+2020 (ZTF CLAGN catalog)
- MacLeod+2019 (SDSS repeat spectra)
- Yang+2018 (SDSS Stripe 82)
- Hon+2022 (WISE-selected)
- Sheng+2020 (MIR CLAGN)
- Green+2022 (SDSS+BOSS)
- Lopez-Navas+2022

**PASS criterion:** `data/benchmark/known_clagn.csv` exists with >= 30 sources,
all required columns present, all sources have literature references.

---

### 2.2 — Hard Negative Set: Known Contaminants

**Why:** Your false positive rate is unmeasurable without known negatives.

```python
# File: data/benchmark/known_contaminants.csv
CONTAMINANT_REQUIRED_COLS = [
    'name',
    'ra', 'dec',
    'redshift',
    'contaminant_type',  # sn_in_host / blazar / dust_event / variable_star / artifact
    'reference',
    'notes',
]

CONTAMINANT_TYPES_REQUIRED = [
    'sn_in_host',      # min 10 examples — crossmatch with ASAS-SN, TNS
    'blazar',          # min 10 examples — from ROMABZCAT, BZCAT
    'variable_star',   # min 10 examples — from Gaia variable catalog
    'normal_agn',      # min 50 examples — from SDSS QSO catalog, no known transition
]
```

**PASS criterion:** >= 80 total contaminants across all four types,
with references. Pipeline must reject >= 80% of each type.

---

### 2.3 — Control Set: Normal AGN (No Known Transition)

**Why:** Measures false positive rate on the bulk AGN population.

```python
# File: data/benchmark/normal_agn_control.csv
# 200+ SDSS-confirmed QSOs with WISE coverage
# None should appear in your top candidates
# If >5% of normal AGN score above your operating threshold → threshold too low

MIN_CONTROL_AGN = 200
```

**PASS criterion:** False positive rate on control AGN < 5% at chosen
operating threshold. This threshold must be stated in the paper.

---

## PART 3: PRECISION-RECALL CHARACTERIZATION

### 3.1 — Purity and Completeness Curves

**Why:** "We recover 50 known objects" is not science. "We achieve 85% completeness
at 70% purity at our chosen operating threshold" is science.

**Required implementation:**

```python
# File: validation/precision_recall.py

def compute_precision_recall_curve(pipeline_scores, benchmark_labels,
                                    score_col='composite_score_v2',
                                    label_col='is_clagn'):
    """
    Compute precision-recall curve across all score thresholds.

    Parameters
    ----------
    pipeline_scores : DataFrame — output of pipeline on benchmark set
    benchmark_labels : DataFrame — known_clagn.csv + known_contaminants.csv

    Returns
    -------
    dict:
        thresholds : array
        precision   : array (= purity)
        recall      : array (= completeness)
        f1          : array
        auc_pr      : float — area under PR curve
        operating_point : dict — chosen threshold + justification
    """
    from sklearn.metrics import precision_recall_curve, auc

    # Merge pipeline scores with ground truth labels
    merged = pipeline_scores.merge(benchmark_labels, on=['ra','dec'], how='inner')

    y_true  = merged[label_col].astype(int).values
    y_score = merged[score_col].values

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    auc_pr = auc(recall, precision)

    # F1 at each threshold
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    # Recommended operating point: maximize F1
    best_idx = np.argmax(f1)

    return {
        'thresholds':      thresholds,
        'precision':       precision,
        'recall':          recall,
        'f1':              f1,
        'auc_pr':          float(auc_pr),
        'operating_point': {
            'threshold':   float(thresholds[best_idx]),
            'precision':   float(precision[best_idx]),
            'recall':      float(recall[best_idx]),
            'f1':          float(f1[best_idx]),
        },
    }


def run_full_benchmark(pipeline, benchmark_dir='data/benchmark/'):
    """
    Run pipeline on entire benchmark and produce all validation metrics.
    This is the single function that generates the validation section of the paper.
    """
    known_clagn      = pd.read_csv(f'{benchmark_dir}/known_clagn.csv')
    contaminants     = pd.read_csv(f'{benchmark_dir}/known_contaminants.csv')
    control_agn      = pd.read_csv(f'{benchmark_dir}/normal_agn_control.csv')

    results = {}

    # Run pipeline on each set
    for name, df in [('known_clagn', known_clagn),
                      ('contaminants', contaminants),
                      ('control_agn',  control_agn)]:
        scores = pipeline.run(df)
        results[name] = scores

    # Precision-recall
    all_sources = pd.concat([
        known_clagn.assign(is_clagn=1),
        contaminants.assign(is_clagn=0),
        control_agn.assign(is_clagn=0),
    ])
    all_scores = pd.concat([results['known_clagn'],
                             results['contaminants'],
                             results['control_agn']])

    pr = compute_precision_recall_curve(all_scores, all_sources)

    # Contaminant rejection rates
    for ctype in ['sn_in_host', 'blazar', 'variable_star']:
        ct_subset = contaminants[contaminants['contaminant_type'] == ctype]
        ct_scores = results['contaminants'][
            results['contaminants']['name'].isin(ct_subset['name'])
        ]
        reject_rate = (ct_scores['composite_score_v2'] < pr['operating_point']['threshold']).mean()
        results[f'{ctype}_rejection_rate'] = float(reject_rate)

    return results, pr
```

**Required figures (must appear in paper):**

1. Precision-recall curve with operating point marked
2. Score distribution: known CLAGN vs contaminants vs control AGN (histogram)
3. Recovery rate vs delta_mag (how faint a transition can you detect?)
4. Recovery rate vs baseline length

**PASS criterion:**

- AUC-PR > 0.7
- At operating threshold: precision > 0.6, recall > 0.6
- SN rejection rate > 80%
- Blazar rejection rate > 70%
- False positive rate on control AGN < 5%

---

## PART 4: INJECTION-RECOVERY

### 4.1 — Synthetic Transition Injection

**Why:** This is the only way to know your detection threshold is real,
not a side effect of your specific known-CLAGN sample.

**This is the single highest-impact validation you can add.**

```python
# File: validation/injection_recovery.py

# Transition morphologies to inject (must test ALL four)
INJECTION_MORPHOLOGIES = {
    'step':      'instantaneous state change — most CLAGN look like this',
    'ramp':      'linear transition over N days — some slow CLAGN',
    'gaussian':  'temporary excursion — peak + return (SN-like / eclipse)',
    'two_step':  'partial transition followed by second change',
}

# Parameter grid (must test FULL grid — not just easy cases)
INJECTION_GRID = {
    'delta_mag':       [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5],  # magnitudes
    'transition_epoch_frac': [0.1, 0.3, 0.5, 0.7, 0.9],        # fraction of baseline
    'ramp_duration_days':    [30, 90, 180, 365],                 # for ramp morphology
    'morphology':      list(INJECTION_MORPHOLOGIES.keys()),
}

# Host light curves to inject into (use your control AGN set)
# Each combination tested on N_REALIZATIONS host LCs to average over noise
N_REALIZATIONS = 20   # per grid point — minimum
N_HOSTS        = 50   # different host light curves per grid point


def inject_transition(times_mjd, flux_mjy, flux_err_mjy,
                       delta_mag, transition_mjd, morphology='step',
                       ramp_duration=90.0):
    """
    Inject a synthetic flux transition into a real WISE light curve.

    Parameters
    ----------
    delta_mag : float (signed — positive = turn-on, negative = turn-off)
    transition_mjd : float — epoch of transition
    morphology : str — 'step', 'ramp', 'gaussian', 'two_step'
    ramp_duration : float — days for ramp to complete

    Returns
    -------
    flux_injected : array — original flux + injected transition
    truth : dict — true injected parameters for recovery comparison
    """
    flux_ratio = 10.0 ** (delta_mag / 2.5)

    if morphology == 'step':
        multiplier = np.where(times_mjd >= transition_mjd, flux_ratio, 1.0)

    elif morphology == 'ramp':
        t_end = transition_mjd + ramp_duration
        frac  = np.clip((times_mjd - transition_mjd) / ramp_duration, 0, 1)
        multiplier = 1.0 + frac * (flux_ratio - 1.0)

    elif morphology == 'gaussian':
        # Temporary excursion: peak at transition_mjd, return to baseline
        sigma_days = ramp_duration / 2.355   # FWHM = ramp_duration
        frac = np.exp(-0.5 * ((times_mjd - transition_mjd) / sigma_days)**2)
        multiplier = 1.0 + frac * (flux_ratio - 1.0)

    elif morphology == 'two_step':
        # First partial transition, then second
        t2 = transition_mjd + ramp_duration
        m1 = np.where(times_mjd >= transition_mjd, np.sqrt(flux_ratio), 1.0)
        m2 = np.where(times_mjd >= t2, np.sqrt(flux_ratio), 1.0)
        multiplier = m1 * m2

    else:
        raise ValueError(f"Unknown morphology: {morphology}")

    flux_injected = flux_mjy * multiplier

    truth = {
        'delta_mag_injected':    delta_mag,
        'transition_mjd':        transition_mjd,
        'morphology':            morphology,
        'ramp_duration_days':    ramp_duration,
        'flux_ratio_injected':   float(flux_ratio),
    }

    return flux_injected, truth


def run_injection_recovery(pipeline, control_agn_lcs, grid=None, n_real=20):
    """
    Full injection-recovery campaign.

    For every grid point (delta_mag, epoch, morphology):
      1. Inject into n_real host light curves
      2. Run pipeline
      3. Record: recovered (score > threshold), recovered_delta_mag, recovered_epoch

    Returns efficiency table: rows = grid points, cols = recovery metrics
    """
    if grid is None:
        grid = INJECTION_GRID

    results = []

    for morph in grid['morphology']:
        for dm in grid['delta_mag']:
            for epoch_frac in grid['transition_epoch_frac']:

                recovered = []
                dm_recovered = []

                for host_lc in control_agn_lcs[:n_real]:
                    t = host_lc['mjd'].values
                    f = host_lc['w1_flux_mjy'].values
                    e = host_lc['w1_flux_err_mjy'].values

                    t_trans = t.min() + epoch_frac * (t.max() - t.min())

                    f_inj, truth = inject_transition(
                        t, f, e,
                        delta_mag=dm,
                        transition_mjd=t_trans,
                        morphology=morph,
                    )

                    # Run pipeline on injected light curve
                    score_result = pipeline.score_single(
                        times=t, flux=f_inj, flux_err=e,
                        z=host_lc.get('z', 0.1)
                    )

                    score = score_result.get('composite_score_v2', 0.0)
                    recovered.append(score >= pipeline.operating_threshold)
                    dm_recovered.append(score_result.get('delta_mag', 0.0))

                results.append({
                    'morphology':          morph,
                    'delta_mag_injected':  dm,
                    'transition_epoch_frac': epoch_frac,
                    'recovery_rate':       float(np.mean(recovered)),
                    'n_realizations':      len(recovered),
                    'mean_dm_recovered':   float(np.nanmean(dm_recovered)),
                    'dm_bias':             float(np.nanmean(dm_recovered) - dm),
                })

    return pd.DataFrame(results)


def compute_detection_threshold_from_injection(efficiency_df, target_completeness=0.90):
    """
    Find the minimum delta_mag at which recovery_rate >= target_completeness.
    This is your scientifically justified detection threshold.
    Replace arbitrary MIN_DELTA_MAG in config with this value.
    """
    step_eff = efficiency_df[efficiency_df['morphology'] == 'step'].copy()
    step_eff = step_eff.groupby('delta_mag_injected')['recovery_rate'].mean()

    above_threshold = step_eff[step_eff >= target_completeness]
    if above_threshold.empty:
        return np.nan, 'threshold_not_reached'

    min_dm = float(above_threshold.index.min())
    return min_dm, f'injection_recovery_completeness={target_completeness}'
```

**Required output figures:**

1. Efficiency heatmap: recovery rate vs (delta_mag, transition_epoch_frac)
   - One panel per morphology (4 panels)
2. Recovery rate vs delta_mag curve (integrated over all epochs)
   - Mark the 50%, 80%, 90% completeness thresholds
3. Amplitude bias plot: recovered delta_mag vs injected delta_mag
4. Efficiency vs baseline length (short baselines = harder)

**PASS criteria:**

- Recovery > 90% for step-function transitions with |delta_mag| > 0.5 mag
- Recovery > 50% for ramp transitions with |delta_mag| > 0.5 mag
- Amplitude bias < 20% for detectable transitions
- Detection threshold derived from injection-recovery (not arbitrary)
- All 4 morphologies tested

---

## PART 5: CONTAMINANT WAR PLAN

### 5.1 — Supernova Rejection

**Why:** Nuclear SNe in host galaxies produce exactly the IR light curve
signature you are hunting. This is your #1 false positive source.

```python
# File: validation/sn_rejection.py

SN_REJECTION_CRITERIA = {
    'duration':
        # SNe are transient: rise < 50d, fade < 200d total
        # CLAGN transitions persist or remain elevated
        # Requirement: flag if >70% of elevated flux within single season
        'single_season_dominance_flag',

    'recurrence':
        # CLAGN state changes can recur (NGC 1566, Mrk 590)
        # SNe do not recur in same nucleus
        # Requirement: check if transition is isolated vs. persistent/recurrent
        'transition_recurrence_check',

    'color':
        # SN dust echoes: W1-W2 can rise then fall
        # AGN CLAGNs: W1-W2 change is tied to accretion state
        # Requirement: flag if color change reverses within 1 year
        'color_reversal_timescale_flag',

    'catalog_crossmatch':
        # Crossmatch all candidates with:
        # - ASAS-SN transient catalog (asas-sn.osu.edu)
        # - TNS (wis-tns.org)
        # - ALeRCE ZTF broker
        # Any known SN within 2" and within 2 years of IR transition = reject
        'transient_catalog_xmatch',
}

def check_sn_morphology(times_mjd, flux_mjy, season_medians):
    """
    Flag light curves that match SN-like morphology.

    SN signature:
    - Single elevated season with return to baseline within 2 seasons
    - Asymmetric: fast rise, slow decay (for type II)
    - No sustained elevation
    """
    seasons = sorted(season_medians.keys())
    if len(seasons) < 4:
        return {'sn_morphology_flag': False, 'reason': 'insufficient_seasons'}

    baseline_flux = np.median([season_medians[s] for s in seasons[:2]])
    peak_flux     = max(season_medians.values())
    peak_season   = max(season_medians, key=season_medians.get)
    peak_idx      = seasons.index(peak_season)

    # Check if elevated flux collapses within 2 seasons of peak
    post_peak = seasons[peak_idx+1:peak_idx+3] if peak_idx+1 < len(seasons) else []
    if post_peak:
        post_flux = np.mean([season_medians[s] for s in post_peak])
        collapse_frac = (peak_flux - post_flux) / max(peak_flux - baseline_flux, 1e-10)
        sn_like = collapse_frac > 0.5 and peak_idx > 0  # rises then falls quickly
    else:
        sn_like = False
        collapse_frac = np.nan

    return {
        'sn_morphology_flag':  sn_like,
        'sn_collapse_fraction':float(collapse_frac) if np.isfinite(collapse_frac) else np.nan,
        'sn_peak_season':      peak_season,
        'sn_peak_flux_mjy':    float(peak_flux),
    }


def crossmatch_transient_catalogs(ra, dec, transition_mjd, search_radius_arcsec=3.0):
    """
    Crossmatch with public transient catalogs.
    Returns list of matching transients within radius and time window.

    Catalogs to query:
    - ASAS-SN: https://www.astronomy.ohio-state.edu/asassn/transients.html
    - TNS: https://wis-tns.org
    - If network unavailable: use locally cached versions
    """
    matches = []

    # TNS query (if available)
    # ASAS-SN query (if available)
    # Return matches within 3 arcsec AND within 2 years of transition_mjd

    return {
        'n_transient_matches': len(matches),
        'transient_matches':   matches,
        'sn_catalog_checked':  True,
    }
```

---

### 5.2 — Blazar Rejection

```python
# File: validation/blazar_rejection.py

BLAZAR_REJECTION_CRITERIA = {
    'radio_crossmatch':
        # Blazars are radio-loud AGN
        # Crossmatch with FIRST (1.4 GHz), NVSS, VLASS
        # Radio detection within 3" → blazar candidate
        # Radio flux > 1 mJy at 1.4 GHz → strong blazar indicator
        'radio_catalog_xmatch',

    'wise_color':
        # Blazar WISE colors: W1-W2 > 0.8 AND W2-W3 < 2.2
        # (Massaro+2011 gamma-AGN selection box)
        # These overlap with your CLAGN color selection region
        'wise_color_blazar_locus',

    'variability_structure':
        # Blazars: highly stochastic, no characteristic timescale
        # DRW tau typically very short (<100 days) or unconstrained
        # CLAGN: structured transition with characteristic timescale
        'drw_tau_blazar_flag',
}

BLAZAR_CATALOGS_TO_CHECK = [
    'ROMABZCAT',   # Roma BZCAT — complete blazar catalog
    'BZCAT5',      # 5th edition
    '3LAC',        # Fermi LAT AGN catalog (gamma-loud blazars)
    'FIRST',       # Radio: 1.4 GHz VLA survey
    'NVSS',        # Radio: 1.4 GHz all-sky
]

def check_blazar_colors(w1_mag, w2_mag, w3_mag=None):
    """
    Check if WISE colors fall in blazar locus.
    Massaro+2011: W1-W2 vs W2-W3 color-color selection.
    """
    w1w2 = w1_mag - w2_mag if np.isfinite(w1_mag) and np.isfinite(w2_mag) else np.nan

    # Simple blazar color flag: W1-W2 > 0.5 is suspicious
    # (normal AGN CLAGN should have W1-W2 ~ 0.2-0.8)
    # True blazar flag requires W2-W3 as well

    blazar_color_flag = np.isfinite(w1w2) and w1w2 > 0.8

    return {
        'w1_w2_color':        float(w1w2) if np.isfinite(w1w2) else np.nan,
        'blazar_color_flag':  blazar_color_flag,
        'blazar_color_method':'Massaro+2011_partial',
    }
```

---

### 5.3 — Dust Obscuration Event Discrimination

```python
DUST_DISCRIMINATION_CRITERIA = {
    'color_behavior':
        # Dust obscuration: W1-W2 INCREASES during fading (source redder when faint)
        # CLAGN turn-off: W1-W2 can decrease or stay flat (AGN contribution drops)
        # This is the strongest photometric discriminant
        'color_vs_flux_correlation',

    'timescale':
        # AGN dust torus light-crossing time: weeks-months
        # Nuclear dust transients: variable, can be fast
        # Use DRW tau as proxy
        'drw_tau_check',
}

def check_color_flux_correlation(w1_seasonal, w2_seasonal):
    """
    Compute correlation between W1-W2 color and W1 flux.

    Dust obscuration: brighter = bluer (negative correlation color vs mag)
    CLAGN: color change reflects accretion state (weaker or positive correlation)

    Not a perfect discriminant but adds information.
    """
    seasons = sorted(set(w1_seasonal.keys()) & set(w2_seasonal.keys()))
    if len(seasons) < 3:
        return {'color_flux_correlation': np.nan}

    w1v    = np.array([w1_seasonal[s] for s in seasons])
    color  = np.array([w1_seasonal[s] - w2_seasonal[s] for s in seasons
                       if s in w2_seasonal])

    if len(color) < 3:
        return {'color_flux_correlation': np.nan}

    r = float(np.corrcoef(w1v[:len(color)], color)[0,1])

    return {
        'color_flux_correlation':        r,
        'dust_obscuration_flag':         r > 0.7,  # strong positive = dust
        'color_flux_correlation_method': 'seasonal_median_pearson',
    }
```

---

## PART 6: GAIA VALIDATION

### 6.1 — Gaia Confusion Matrix on Known AGN

**Why:** Prove your Gaia cuts don't reject real AGN.

```python
# File: validation/gaia_validation.py

def compute_gaia_confusion_matrix(known_agn_list, pipeline):
    """
    For each known AGN in benchmark:
    - Run Gaia matching
    - Record: matched/not, rejected/passed, reason

    Compute:
    - True positive rate (known AGN that pass Gaia filter)
    - False negative rate (known AGN incorrectly rejected)
    - For rejected: what was the rejection reason?
    """
    results = []
    for source in known_agn_list:
        gaia_result = pipeline.run_gaia(source['ra'], source['dec'])
        results.append({
            'name':             source['name'],
            'gaia_matched':     gaia_result.get('gaia_matched', False),
            'reject_as_star':   gaia_result.get('reject_as_star', False),
            'rejection_reason': gaia_result.get('rejection_reason', 'none'),
            'pm_sig':           gaia_result.get('pm_sig', np.nan),
            'parallax_sig':     gaia_result.get('parallax_sig', np.nan),
            'ruwe':             gaia_result.get('ruwe', np.nan),
        })

    df = pd.DataFrame(results)

    # Known AGN should NOT be rejected as stars
    false_negative_rate = df['reject_as_star'].mean()

    if false_negative_rate > 0.10:
        raise ValueError(
            f"Gaia filter rejects {false_negative_rate:.1%} of known AGN — "
            f"threshold too aggressive. Max allowed: 10%."
        )

    return {
        'n_known_agn_tested':   len(df),
        'false_negative_rate':  float(false_negative_rate),
        'rejection_reasons':    df[df['reject_as_star']]['rejection_reason'].value_counts().to_dict(),
        'pass_criterion':       false_negative_rate <= 0.10,
    }
```

**PASS criterion:** Gaia filter incorrectly rejects < 10% of known AGN.

---

## PART 7: SPECTROSCOPIC VALIDATION

### 7.1 — Archival Spectroscopy Crossmatch

**This is what converts candidates into science.**

```python
# File: validation/spectroscopy.py

SPECTROSCOPIC_ARCHIVES_TO_CHECK = [
    {
        'name':    'SDSS',
        'url':     'https://skyserver.sdss.org/dr17/',
        'method':  'CasJobs or SkyServer radius search',
        'radius':  3.0,  # arcsec
        'value':   'high — many AGN spectra, multi-epoch for repeat obs',
    },
    {
        'name':    'LAMOST',
        'url':     'http://www.lamost.org/dr9/',
        'method':  'LAMOST DR9 value-added catalog',
        'radius':  3.0,
        'value':   'medium — wide field, lower resolution',
    },
    {
        'name':    'DESI',
        'url':     'https://data.desi.lbl.gov/',
        'method':  'EDR or DR1 if available',
        'radius':  1.5,
        'value':   'high — deep, many redshifts',
    },
    {
        'name':    'NED',
        'url':     'https://ned.ipac.caltech.edu/',
        'method':  'NED cone search for spectra',
        'radius':  3.0,
        'value':   'aggregator — catches many archives',
    },
    {
        'name':    '6dFGS',
        'url':     'http://www.6dfgs.net/',
        'method':  'Southern hemisphere coverage',
        'radius':  3.0,
        'value':   'medium',
    },
]

SPECTROSCOPIC_EVIDENCE_TIERS = {
    'tier_1_confirmed': {
        'criteria': [
            'Multi-epoch spectra showing broad line appearance OR disappearance',
            'Seyfert type change documented',
        ],
        'label': 'Spectroscopically confirmed CLAGN',
        'weight_in_paper': 'Primary result',
    },
    'tier_2_strong': {
        'criteria': [
            'Single spectrum showing broad lines consistent with AGN state',
            'Redshift confirmed spectroscopically',
            'No evidence of star/SN in spectrum',
        ],
        'label': 'Spectroscopically supported CLAGN candidate',
        'weight_in_paper': 'Secondary result',
    },
    'tier_3_weak': {
        'criteria': [
            'Redshift from archival photometric redshift only',
            'No spectral typing available',
        ],
        'label': 'Photometric candidate (unconfirmed)',
        'weight_in_paper': 'Appendix / catalog only',
    },
}

def classify_spectroscopic_evidence(candidate, spec_matches):
    """
    Given a candidate and any matching spectra, assign evidence tier.
    """
    if not spec_matches:
        return {'spectroscopic_tier': 'tier_3_weak', 'n_spectra': 0}

    # Check for multi-epoch spectra
    n_spec = len(spec_matches)
    has_broad_lines = any(s.get('broad_lines_present') for s in spec_matches)
    has_type_change = any(s.get('seyfert_type_change') for s in spec_matches)
    z_confirmed     = any(s.get('z_spectroscopic') for s in spec_matches)

    if has_type_change or (n_spec >= 2 and has_broad_lines):
        tier = 'tier_1_confirmed'
    elif has_broad_lines and z_confirmed:
        tier = 'tier_2_strong'
    else:
        tier = 'tier_3_weak'

    return {
        'spectroscopic_tier':    tier,
        'n_spectra':             n_spec,
        'has_broad_lines':       has_broad_lines,
        'has_type_change':       has_type_change,
        'z_spectroscopic':       z_confirmed,
        'spectroscopic_archives_checked': [s['archive'] for s in spec_matches],
    }
```

**PASS criterion for paper submission:**

- Top 20 candidates crossmatched with ALL spectroscopic archives
- At least 3 candidates with Tier 1 or Tier 2 spectroscopic evidence
- All Tier 1 candidates described in paper body with figures
- All Tier 2-3 candidates in appendix table

---

## PART 8: REPRODUCIBILITY STANDARD

### 8.1 — Single-Command Reproduction

```bash
# This command must reproduce ALL results in the paper from scratch:
python reproduce_all.py \
    --data_dir data/ \
    --output_dir results/ \
    --benchmark_dir data/benchmark/ \
    --random_seed 42 \
    --n_injection_realizations 20

# Expected outputs:
# results/candidates/candidate_table.csv
# results/validation/precision_recall.png
# results/validation/injection_recovery_heatmap.png
# results/validation/score_distributions.png
# results/paper_figures/*.pdf
# results/provenance/pipeline_version.json
```

### 8.2 — Provenance Sidecar (Required for Every Output)

```python
# Every CSV output must have an accompanying JSON:
# candidate_table.csv → candidate_table_provenance.json

PROVENANCE_REQUIRED_FIELDS = {
    'pipeline_version':    'git describe --tags',
    'git_hash':            'git rev-parse HEAD',
    'run_timestamp':       'ISO 8601 UTC',
    'config_snapshot':     'entire config.py contents as dict',
    'python_version':      'sys.version',
    'key_package_versions':['numpy', 'scipy', 'astropy', 'celerite2', 'emcee'],
    'random_seed':         'value used',
    'n_sources_input':     'integer',
    'n_sources_output':    'integer',
    'operating_threshold': 'value + justification',
    'wise_tables_queried': 'list of table names + query dates',
    'irsa_api_version':    'if available',
}

def write_provenance(output_path, config, results_summary):
    """Write provenance sidecar JSON next to every output file."""
    import subprocess, sys, json
    from datetime import datetime, timezone

    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = 'unavailable'

    prov = {
        'git_hash':         git_hash,
        'run_timestamp':    datetime.now(timezone.utc).isoformat(),
        'python_version':   sys.version,
        'config':           {k: v for k, v in vars(config).items()
                             if not k.startswith('_')},
        'results_summary':  results_summary,
    }

    prov_path = output_path.replace('.csv', '_provenance.json')
    with open(prov_path, 'w') as f:
        json.dump(prov, f, indent=2, default=str)
```

### 8.3 — Fixed Random Seeds Everywhere

```python
# In main.py, at the very top:
GLOBAL_RANDOM_SEED = 42
np.random.seed(GLOBAL_RANDOM_SEED)
import random; random.seed(GLOBAL_RANDOM_SEED)

# In every function using randomness:
def fit_drw_mcmc(..., seed=None):
    if seed is not None:
        np.random.seed(seed)
    # ...
```

### 8.4 — Environment Lock

```bash
# Required files in repo root:
# environment.yml  (conda)
# requirements.txt (pip with pinned versions)

# Generate with:
conda env export > environment.yml
pip freeze > requirements.txt
```

---

## PART 9: SENSITIVITY ANALYSIS

### 9.1 — Threshold Sensitivity

**Every threshold that affects the final candidate count must be tested.**

```python
# File: validation/sensitivity_analysis.py

THRESHOLDS_TO_TEST = {
    'MIN_BASELINE_YEARS':      [5.0, 7.0, 10.0],
    'MIN_REDSHIFT':            [0.002, 0.005, 0.01],
    'SIGMA_CLIP_SIGMA':        [3.5, 4.0, 5.0],
    'MIN_DELTA_MAG':           [0.2, 0.3, 0.4, 0.5],
    'MIN_EPOCHS_PER_SEASON':   [2, 3, 5],
    'MIN_SEASONS_REQUIRED':    [3, 4, 5],
    'GAIA_SEARCH_RADIUS_ARCSEC':[1.0, 1.5, 2.0],
    'MAX_PARALLAX_SIG':        [2.0, 3.0, 5.0],
    'operating_threshold':     [0.3, 0.4, 0.5, 0.6],
}

def run_sensitivity_sweep(pipeline, benchmark, param_name, values):
    """
    Run benchmark at each value of one parameter.
    Report: n_candidates, precision, recall, AUC-PR.
    """
    results = []
    for v in values:
        setattr(pipeline.config, param_name, v)
        scores, pr = run_full_benchmark(pipeline, benchmark)
        results.append({
            'param':        param_name,
            'value':        v,
            'n_candidates': scores.get('n_candidates', 0),
            'precision':    pr['operating_point']['precision'],
            'recall':       pr['operating_point']['recall'],
            'auc_pr':       pr['auc_pr'],
        })
    return pd.DataFrame(results)
```

**PASS criterion:** Precision-recall AUC does not change by > 0.1 when
any single threshold varies across its test range. If it does change
by > 0.1, that threshold is a fragile hyperparameter and must be
either justified more carefully or removed.

---

## PART 10: PAPER CLAIMS DISCIPLINE

### 10.1 — Claim Hierarchy (What You Are Allowed to Say)

```
ALLOWED CLAIMS (supported by this pipeline):

  "We present a photometric selection pipeline for changing-state AGN
   candidates based on WISE/NEOWISE multi-epoch photometry."

  "We achieve X% completeness and Y% purity at our operating threshold,
   as measured on a literature benchmark of Z known CLAGN."

  "We detect synthetic transitions with |Δmag| > 0.5 at >90% efficiency
   for step-like morphologies (injection-recovery)."

  "We present a ranked catalog of N candidates with composite scores,
   multiwavelength diagnostics, and Gaia stellar contamination assessment."

  "A subset of M candidates have archival spectroscopic support
   [Tier 1: K confirmed, Tier 2: L supported]."


NOT ALLOWED WITHOUT SPECTROSCOPY:

  "We discover N new changing-look AGN."
  [Must say: "N new changing-look AGN candidates"]

  "The BLR of source X is disappearing."
  [Must say: "Source X shows IR variability consistent with a fading
   accretion state, possibly associated with BLR changes"]

  "Our pipeline detects changing-look AGN."
  [Must say: "Our pipeline selects changing-state AGN candidates"]


REQUIRES CITATION (cannot be stated as original):

  The definition of changing-look AGN → cite LaMassa+2015 or earlier
  DRW as AGN variability model → cite Kelly+2009
  WISE photometric system → cite Wright+2010
  Seasonal amplitude method → cite Sheng+2017 or Graham+2020
```

---

## PART 11: KNOWN LIMITATIONS SECTION

**This section must appear in the paper. Reviewers will add it themselves
if you don't. Better to write it yourself and control the framing.**

```
Required subsections in "Limitations" or "Caveats":

11.1 Selection Function
  - We are sensitive only to transitions of |Δmag| > X (from injection-recovery)
  - We are insensitive to transitions shorter than ~1 WISE season (6 months)
  - We are insensitive to transitions longer than our baseline (~12 years)
  - Coverage gaps: WISE hibernation 2011-2013 creates a blind period

11.2 Contamination
  - Nuclear SNe are our dominant photometric contaminant
  - We estimate residual SN contamination rate of X% (from control sample)
  - Blazars: estimated contamination Y% after radio crossmatch filter
  - Dust obscuration events: cannot be fully distinguished photometrically

11.3 Redshift Dependence
  - Host galaxy contamination increases at low z (z < 0.05)
  - Completeness decreases at high z (z > 0.5) due to flux limit
  - No K-correction applied to WISE photometry

11.4 WISE-Specific Limitations
  - Spatial resolution ~6 arcsec — cannot resolve nuclear vs. off-nuclear events
  - Only two IR bands — limited SED information
  - WISE calibration uncertainty ~2.8% limits amplitude precision below ~0.03 mag

11.5 DRW Model Limitations
  - DRW is a stationary GP — misspecified for true CLAGN transitions
  - τ measurement unreliable when baseline < 10τ
  - DRW nonstationarity statistic is heuristically calibrated
    (simulation calibration provided for top candidates only)
```

---

## MASTER CHECKLIST — GATE FOR PAPER SUBMISSION

All items must be TRUE before submitting to any journal.

### Science Definition

- [ ] `definitions.md` exists with Tier 1-4 classification written out
- [ ] All paper claims use candidate/photometric language for unconfirmed sources

### Benchmark Dataset

- [ ] `known_clagn.csv` with >= 30 sources and references
- [ ] `known_contaminants.csv` with >= 80 sources across 4 types
- [ ] `normal_agn_control.csv` with >= 200 sources

### Pipeline Validation

- [ ] Precision-recall curve computed on full benchmark
- [ ] AUC-PR > 0.7
- [ ] Precision > 0.6 AND recall > 0.6 at operating threshold
- [ ] SN rejection rate > 80% on SN contaminant set
- [ ] Blazar rejection rate > 70% on blazar set
- [ ] False positive rate < 5% on normal AGN control

### Injection-Recovery

- [ ] All 4 morphologies tested (step, ramp, gaussian, two-step)
- [ ] Full parameter grid completed (>= 7 delta_mag values × 5 epochs)
- [ ] N >= 20 realizations per grid point
- [ ] Efficiency heatmap figure exists
- [ ] Detection threshold derived from injection-recovery (not arbitrary)
- [ ] Recovery > 90% for step |Δmag| > 0.5

### Contaminant War Plan

- [ ] SN morphology filter implemented and tested
- [ ] Transient catalog crossmatch run (ASAS-SN and/or TNS)
- [ ] Blazar radio crossmatch run (FIRST and/or NVSS)
- [ ] Dust obscuration color-flux correlation check implemented

### Gaia Validation

- [ ] Confusion matrix on known AGN computed
- [ ] Gaia false negative rate < 10% on known AGN

### Spectroscopic Validation

- [ ] Top 20 candidates crossmatched with SDSS, LAMOST, DESI, NED
- [ ] > = 3 candidates with Tier 1 or Tier 2 spectroscopic evidence
- [ ] Spectroscopic evidence table in paper

### Reproducibility

- [ ] Single command reproduces all results
- [ ] `environment.yml` and `requirements.txt` committed to repo
- [ ] Random seeds fixed and documented
- [ ] Provenance JSON written next to every output CSV
- [ ] All figures generated by scripts (no manual steps)

### Sensitivity Analysis

- [ ] All thresholds in THRESHOLDS_TO_TEST swept
- [ ] AUC-PR stable (< 0.1 change) across threshold ranges
- [ ] Fragile thresholds identified and justified

### Paper Text

- [ ] Definition section present (Sec 2)
- [ ] Selection function described with injection-recovery numbers
- [ ] Known limitations section present
- [ ] All threshold values cited as empirical or given literature reference
- [ ] Claim hierarchy respected (no unqualified "discovery" language)

---

## TIMELINE ESTIMATE

| Task                                                 | Estimated effort                   |
| ---------------------------------------------------- | ---------------------------------- |
| Build benchmark dataset (known CLAGN + contaminants) | 1-2 weeks                          |
| Precision-recall validation on benchmark             | 2-3 days                           |
| Injection-recovery campaign (full grid)              | 3-5 days compute + 2 days analysis |
| Contaminant war plan (SN + blazar filters)           | 1 week                             |
| Spectroscopic crossmatch (archival)                  | 3-5 days                           |
| Sensitivity analysis                                 | 2-3 days                           |
| Reproducibility packaging                            | 2-3 days                           |
| Paper writing (methods + validation sections)        | 2-3 weeks                          |
| **Total**                                            | **~8-10 weeks focused effort**     |

This is honest. A pipeline paper at this standard takes this long.
The engineering is done. The science validation is what remains.

```

```
