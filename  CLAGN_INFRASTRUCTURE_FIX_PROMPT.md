# CLAGN Pipeline — Infrastructure, Orchestration & Statistical Validity Fix Prompt
### Areas NOT covered in previous 5 fix prompts.
### Targets: main.py / run_pipeline.py, cache layer, evaluation.py, catalog.py, config.py
### 22 new issues. These are distinct from all previous fix prompts.

---

## SCOPE CLARIFICATION

Previous prompts covered:
- wise_complete.py (baseline anchor, sigma clipping, epoch weights, schema safety)
- gaia.py (GAIA match ordering, quality flags, saturation)
- variability.py (seasonal median, gap clustering, amplitude metrics)
- drw.py (mean subtraction, nonstationarity dict return, input cleaning)
- clagn_score.py (bootstrap, column names, unit mismatch, host correction)

This prompt covers everything else:
- Pipeline orchestration (main.py / run_pipeline.py)
- Caching and idempotency
- Evaluation infrastructure
- Catalog ingestion
- Cross-validation and threshold selection
- Statistical reporting

---

## FILE: main.py / run_pipeline.py

---

### FIX infra-C1 (CRITICAL): No Random Seed Propagation to All Stochastic Steps

**Problem:** `random_seed: 42` is set in pipeline_policy.yaml but it is only passed to
some functions. The DRW MCMC sampler (emcee), the stratified subsampling
`_maybe_subsample()`, the block bootstrap, and scikit-learn functions each
maintain their own RNG state. A run is not reproducible unless ALL of these
receive the same seed.

**Fix — add global seed initialization at pipeline entry point:**

```python
# In main.py / run_pipeline.py, at the very top of __main__ or run():

import random
import numpy as np

def initialize_global_seed(seed: int):
    """
    Set all RNG seeds. Must be called ONCE before any stochastic operation.
    
    Covers:
    - numpy (used in variability, drw, clagn_score)
    - Python random (used in some utility functions)
    - emcee uses numpy under the hood
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # If tensorflow or torch are ever added:
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    
    import logging
    logging.getLogger(__name__).info(f"Global RNG seed set to {seed}")

# Call at entry:
seed = policy.get('random_seed', 42)
initialize_global_seed(seed)
```

**Also: pass seed explicitly to every function that accepts it:**
```python
# _maybe_subsample:
np.random.seed(seed)  # or pass seed parameter

# emcee sampler:
sampler = emcee.EnsembleSampler(...)
sampler.run_mcmc(p0, n_steps, rng=np.random.default_rng(seed))

# block bootstrap:
rng = np.random.default_rng(seed)
resampled_seasons = rng.choice(seasons, size=len(seasons), replace=True)
```

---

### FIX infra-C2 (CRITICAL): No Atomic Output Writing — Partial Runs Leave Corrupt State

**Problem:** If the pipeline crashes mid-run (OOM, network timeout, keyboard interrupt),
it leaves partial output files. The next run either overwrites them silently or
reads partial data as if it were complete. This makes debugging very hard.

**Fix — write to temp files, rename atomically:**

```python
import tempfile
import os
import shutil

def write_results_atomic(results_df, output_path):
    """
    Write results to a temp file in the same directory, then atomically rename.
    If the write fails, output_path is not modified.
    """
    output_dir = os.path.dirname(os.path.abspath(output_path))
    
    # Write to temp file in same directory (same filesystem = atomic rename)
    with tempfile.NamedTemporaryFile(
        mode='w', dir=output_dir, suffix='.tmp', delete=False
    ) as f:
        tmp_path = f.name
        results_df.to_csv(f, index=False)
    
    # Atomic rename
    os.replace(tmp_path, output_path)
    
# Similarly for JSON outputs:
def write_json_atomic(data, output_path):
    import json
    output_dir = os.path.dirname(os.path.abspath(output_path))
    with tempfile.NamedTemporaryFile(
        mode='w', dir=output_dir, suffix='.tmp', delete=False
    ) as f:
        tmp_path = f.name
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp_path, output_path)
```

---

### FIX infra-C3 (CRITICAL): Pipeline Version Not Recorded in Output

**Problem:** Output CSVs and JSON cards contain no record of which code version,
which config, and which random seed produced them. You cannot reproduce a result
from an output file alone.

**Fix — add provenance header to every output:**

```python
import subprocess
import hashlib
import json
from datetime import datetime, timezone

def get_pipeline_provenance(config_path: str, seed: int) -> dict:
    """
    Capture everything needed to reproduce this exact run.
    """
    # Git commit hash
    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        git_dirty = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = 'unknown'
        git_dirty = 'unknown'
    
    # Config hash
    with open(config_path, 'rb') as f:
        config_hash = hashlib.sha256(f.read()).hexdigest()[:16]
    
    return {
        'pipeline_version':   git_hash,
        'pipeline_dirty':     bool(git_dirty),  # True = uncommitted changes
        'config_path':        config_path,
        'config_hash':        config_hash,
        'random_seed':        seed,
        'timestamp_utc':      datetime.now(timezone.utc).isoformat(),
        'python_version':     __import__('sys').version,
    }

# Embed in output CSV as a comment header:
def write_csv_with_provenance(df, output_path, provenance):
    with open(output_path, 'w') as f:
        f.write(f"# CLAGN Pipeline Output\n")
        for k, v in provenance.items():
            f.write(f"# {k}: {v}\n")
        f.write("#\n")
        df.to_csv(f, index=False)

# Embed in metrics.json:
metrics = {**evaluation_results, **provenance}
write_json_atomic(metrics, 'results/metrics.json')
```

---

### FIX infra-H4 (HIGH): No Checkpoint/Resume for Long Runs

**Problem:** Running make_cards on 500K sources takes hours. If it crashes at
source 400K, the entire run must restart from scratch.

**Fix — add per-source checkpointing:**

```python
import os
import json

def process_with_checkpoint(source_ids, process_fn, checkpoint_dir, force=False):
    """
    Process sources with per-source checkpointing.
    Already-completed sources are skipped on resume.
    
    Parameters
    ----------
    source_ids : list
    process_fn : callable(source_id) -> dict result
    checkpoint_dir : str — directory to store per-source JSON results
    force : bool — if True, reprocess even if checkpoint exists
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    results = {}
    skipped = 0
    
    for sid in source_ids:
        checkpoint_path = os.path.join(checkpoint_dir, f"{sid}.json")
        
        if not force and os.path.exists(checkpoint_path):
            # Load from checkpoint
            with open(checkpoint_path) as f:
                results[sid] = json.load(f)
            skipped += 1
            continue
        
        try:
            result = process_fn(sid)
            # Write checkpoint atomically
            with tempfile.NamedTemporaryFile(
                mode='w', dir=checkpoint_dir, suffix='.tmp', delete=False
            ) as f:
                tmp = f.name
                json.dump(result, f, default=str)
            os.replace(tmp, checkpoint_path)
            results[sid] = result
            
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                f"Source {sid} failed: {type(e).__name__}: {e}"
            )
            results[sid] = {'valid': False, 'error': str(e), 'source_id': sid}
    
    return results, skipped
```

---

### FIX infra-H5 (HIGH): Parallel Processing Has No Rate Limiting for IRSA Queries

**Problem:** If the pipeline uses multiprocessing or threading to process sources
in parallel, each worker fires IRSA API queries independently. IRSA has rate
limits. Concurrent queries will result in 429 errors and silent data gaps.

**Fix — add a shared rate limiter:**

```python
import threading
import time

class IRSARateLimiter:
    """
    Token bucket rate limiter for IRSA API queries.
    Default: 10 queries/second max (conservative for shared API).
    """
    def __init__(self, max_rate=10.0):
        self.max_rate = max_rate
        self.min_interval = 1.0 / max_rate
        self._lock = threading.Lock()
        self._last_call = 0.0
    
    def acquire(self):
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last_call = time.monotonic()

# Global singleton:
_irsa_limiter = IRSARateLimiter(max_rate=5.0)  # conservative

# Wrap every IRSA call:
def irsa_query_rate_limited(url, params, **kwargs):
    _irsa_limiter.acquire()
    return requests.get(url, params=params, **kwargs)
```

---

## FILE: evaluation.py (or wherever benchmark evaluation lives)

---

### FIX eval-C1 (CRITICAL): Threshold Selection Is Done on the Test Set

**Problem:** The session transcripts imply `t_recall = 0.3493918746` was selected
by observing performance on the full benchmark or dev set without proper
cross-validation. If this threshold was selected to maximize any metric on the
dev set, it is overfit to the dev set and the test set result (0% recall) is
the true generalization estimate.

**Fix — implement proper k-fold threshold selection:**

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import numpy as np

def select_threshold_cv(scores, labels, n_splits=5, seed=42):
    """
    Select decision threshold using cross-validation on the DEV set only.
    Never touch the test set during threshold selection.
    
    Parameters
    ----------
    scores : array — pipeline scores for each source
    labels : array — binary (1=CLAGN, 0=other)
    
    Returns
    -------
    dict:
        threshold : float — selected threshold
        cv_recall_mean : float — mean CV recall at threshold
        cv_recall_std : float
        cv_balanced_acc_mean : float
        threshold_grid_results : list of dicts
    """
    thresholds = np.linspace(
        np.percentile(scores, 5),
        np.percentile(scores, 95),
        50
    )
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    threshold_results = []
    
    for t in thresholds:
        fold_recalls = []
        fold_bal_accs = []
        fold_blazar_fprs = []
        
        for train_idx, val_idx in skf.split(scores, labels):
            val_scores = scores[val_idx]
            val_labels = labels[val_idx]
            
            preds = (val_scores >= t).astype(int)
            
            recall = (
                (preds[val_labels == 1] == 1).sum() /
                max((val_labels == 1).sum(), 1)
            )
            bal_acc = balanced_accuracy_score(val_labels, preds)
            
            fold_recalls.append(recall)
            fold_bal_accs.append(bal_acc)
        
        threshold_results.append({
            'threshold': float(t),
            'recall_mean': float(np.mean(fold_recalls)),
            'recall_std': float(np.std(fold_recalls)),
            'balanced_acc_mean': float(np.mean(fold_bal_accs)),
        })
    
    # Select threshold that satisfies recall >= 0.80 with max balanced_acc
    valid = [r for r in threshold_results if r['recall_mean'] >= 0.80]
    if not valid:
        # Relax: take threshold with best recall
        best = max(threshold_results, key=lambda r: r['recall_mean'])
    else:
        best = max(valid, key=lambda r: r['balanced_acc_mean'])
    
    return {
        'threshold':              best['threshold'],
        'cv_recall_mean':         best['recall_mean'],
        'cv_recall_std':          best['recall_std'],
        'cv_balanced_acc_mean':   best['balanced_acc_mean'],
        'threshold_grid_results': threshold_results,
        'selection_method':       '5fold_cv_max_balanced_acc_at_recall_0.80',
    }
```

**After threshold selection:**
```python
# Evaluate on TEST SET exactly once, at the end
threshold = cv_result['threshold']
test_preds = (test_scores >= threshold).astype(int)
test_recall = ...
# Report this as the final result. Never retune based on test set results.
```

---

### FIX eval-C2 (CRITICAL): Per-Class Rejection Breakdown Is Not Computed

**Problem:** You report overall accuracy and overall recall. You do not report
what fraction of each contaminant class is being incorrectly passed as CLAGN.

For a survey paper, referees will specifically ask:
- What fraction of your SNe are passed as CLAGN?
- What fraction of your blazars are passed?

**Fix — add per-class confusion analysis:**

```python
def compute_per_class_metrics(scores, true_labels, threshold, classes=None):
    """
    For each class, compute:
    - True positive rate (if CLAGN)
    - False positive rate (if contaminant)
    - Most common rejection reason
    """
    if classes is None:
        classes = ['CLAGN', 'SN', 'blazar', 'normal_agn']
    
    preds_binary = (scores >= threshold).astype(int)
    
    results = {}
    for cls in classes:
        mask = (true_labels == cls)
        n = mask.sum()
        if n == 0:
            continue
        
        n_pass = preds_binary[mask].sum()
        rate = n_pass / n
        
        if cls == 'CLAGN':
            results[cls] = {
                'n': int(n),
                'n_recovered': int(n_pass),
                'recall': float(rate),
                'n_missed': int(n - n_pass),
            }
        else:
            results[cls] = {
                'n': int(n),
                'n_false_positive': int(n_pass),
                'false_positive_rate': float(rate),
                'n_correctly_rejected': int(n - n_pass),
            }
    
    return results

# Example required output:
# CLAGN:      n=46,  recall=0.83,       n_missed=8
# SN:         n=62,  FPR=0.06,          n_false_positive=4
# blazar:     n=35,  FPR=0.09,          n_false_positive=3
# normal_agn: n=65,  FPR=0.03,          n_false_positive=2
```

---

### FIX eval-H3 (HIGH): No Sensitivity Analysis on Thresholds

**Problem:** The pipeline has many thresholds: `r_min`, `peak_frac`, `min_seasons`,
`t_recall`, `min_clean_points_total`. You do not know how sensitive your results
are to each one. A reviewer will ask: "What happens if you change r_min by 10%?"

**Fix — add threshold sensitivity sweep:**

```python
def threshold_sensitivity_analysis(base_params, cards_dict, benchmark_df,
                                     perturbation_frac=0.1):
    """
    For each parameter, vary it by ±10% while holding all others fixed.
    Report the change in CLAGN recall and balanced accuracy.
    
    This becomes Table 3 in the paper.
    """
    base_metrics = evaluate_gates(cards_dict, benchmark_df, base_params)
    
    results = []
    for param_name, base_val in base_params.items():
        for direction, multiplier in [('up', 1 + perturbation_frac),
                                       ('down', 1 - perturbation_frac)]:
            perturbed = dict(base_params)
            perturbed[param_name] = base_val * multiplier
            
            perturbed_metrics = evaluate_gates(cards_dict, benchmark_df, perturbed)
            
            results.append({
                'parameter':          param_name,
                'direction':          direction,
                'base_value':         base_val,
                'perturbed_value':    perturbed[param_name],
                'delta_recall':       perturbed_metrics['recall'] - base_metrics['recall'],
                'delta_balanced_acc': perturbed_metrics['balanced_accuracy'] - base_metrics['balanced_accuracy'],
                'delta_blazar_fpr':   perturbed_metrics['blazar_fpr'] - base_metrics['blazar_fpr'],
            })
    
    return pd.DataFrame(results).sort_values('delta_recall', key=abs, ascending=False)
```

---

### FIX eval-H4 (HIGH): AUC-ROC Not Computed

**Problem:** The pipeline reports threshold-dependent metrics (recall, precision,
accuracy). The threshold-independent metric — AUC-ROC — is not computed anywhere.
AUC-ROC is the standard discrimination metric for a classifier paper and is
required by most astrophysics journal referees.

**Fix:**

```python
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

def compute_roc_metrics(scores, true_labels_binary):
    """
    Parameters
    ----------
    scores : array — pipeline scores (higher = more likely CLAGN)
    true_labels_binary : array — 1=CLAGN, 0=other
    
    Returns
    -------
    dict with AUC-ROC, AUC-PR, and curve data for plotting
    """
    auc_roc = float(roc_auc_score(true_labels_binary, scores))
    auc_pr  = float(average_precision_score(true_labels_binary, scores))
    
    fpr, tpr, thresholds = roc_curve(true_labels_binary, scores)
    
    # Operating point: find threshold where recall=0.80
    idx_80 = np.argmin(np.abs(tpr - 0.80))
    
    return {
        'auc_roc':           auc_roc,
        'auc_pr':            auc_pr,
        'fpr_at_recall_80':  float(fpr[idx_80]),
        'threshold_at_recall_80': float(thresholds[idx_80]),
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
        }
    }
```

**Minimum acceptable:** AUC-ROC ≥ 0.85 for a publishable classifier.

---

### FIX eval-H5 (HIGH): No Precision-Recall Curve or Precision@K

**Problem:** For rare-class detection (CLAGN are ~10% of sources), the
precision-recall curve is more informative than ROC. Precision@K is the
metric that directly answers: "If I observe the top K candidates, how many
are real?"

**Fix:**

```python
def compute_precision_at_k(scores_df, true_labels, k_values=None):
    """
    Compute precision and recall at K ranked candidates.
    """
    if k_values is None:
        k_values = [5, 10, 20, 50, 100]
    
    n_clagn_total = (true_labels == 'CLAGN').sum()
    
    # Sort by score descending
    sorted_idx = np.argsort(scores_df['score'].values)[::-1]
    sorted_labels = true_labels.values[sorted_idx]
    
    results = {}
    for k in k_values:
        top_k = sorted_labels[:k]
        n_correct = (top_k == 'CLAGN').sum()
        results[f'precision_at_{k}'] = float(n_correct / k)
        results[f'recall_at_{k}']    = float(n_correct / n_clagn_total)
    
    return results

# REQUIRED TARGETS:
# precision_at_20 >= 0.50 (at least 10 of top 20 are real CLAGN)
# precision_at_50 >= 0.40
```

---

## FILE: catalog.py / data ingestion

---

### FIX cat-H1 (HIGH): Redshift Validation Is Missing

**Problem:** Redshift `z` is passed through the pipeline to DRW (rest-frame time
correction), delta_mag, and host correction. Invalid or missing redshifts
silently corrupt these computations.

**Fix — validate redshifts at ingestion:**

```python
def validate_and_clean_redshifts(catalog_df, z_col='z'):
    """
    Validate redshifts at catalog ingestion. Flag problematic sources.
    
    Rules:
    - z must be finite and >= 0
    - z > 5.0 is almost certainly wrong for WISE sources (flag, don't reject)
    - z == 0.0 exactly is suspicious (photometric z? missing?)
    - z < 0.005 (very nearby) changes DRW rest-frame correction negligibly
    
    Returns df with z_clean and z_flag columns added.
    """
    z = pd.to_numeric(catalog_df[z_col], errors='coerce')
    
    flags = pd.Series('ok', index=catalog_df.index)
    flags[z.isna() | ~np.isfinite(z)]   = 'missing'
    flags[z < 0]                          = 'negative'
    flags[z == 0.0]                       = 'zero_suspicious'
    flags[z > 5.0]                        = 'implausible_high'
    flags[(z > 0) & (z < 0.005)]         = 'very_nearby'
    
    # Replace invalid z with median of valid values (not zero — that corrupts DRW)
    z_median = float(z[(flags == 'ok') | (flags == 'very_nearby')].median())
    z_clean  = z.copy()
    z_clean[flags.isin(['missing', 'negative', 'implausible_high'])] = z_median
    z_clean[flags == 'zero_suspicious'] = z_median  # or 0.1 as conservative default
    
    catalog_df = catalog_df.copy()
    catalog_df['z_clean'] = z_clean
    catalog_df['z_flag']  = flags
    
    n_bad = (flags != 'ok').sum()
    if n_bad > 0:
        import logging
        logging.getLogger(__name__).warning(
            f"z validation: {n_bad}/{len(catalog_df)} sources have z issues. "
            f"Breakdown: {flags.value_counts().to_dict()}"
        )
    
    return catalog_df

# Use z_clean throughout pipeline, NOT the raw z column
```

---

### FIX cat-H2 (HIGH): Coordinate Validation Missing — NaN RA/Dec Will Crash GAIA Query

**Problem:** RA/Dec are passed directly to the GAIA ADQL query. If a source has
NaN or out-of-range coordinates, the query either crashes or returns wrong results.

**Fix:**

```python
def validate_coordinates(catalog_df, ra_col='ra', dec_col='dec'):
    """
    Validate RA/Dec at ingestion. Reject sources with invalid coordinates.
    """
    ra  = pd.to_numeric(catalog_df[ra_col],  errors='coerce')
    dec = pd.to_numeric(catalog_df[dec_col], errors='coerce')
    
    valid = (
        ra.notna()  & dec.notna()  &
        (ra  >= 0)  & (ra  < 360)  &
        (dec >= -90) & (dec <= 90)
    )
    
    n_invalid = (~valid).sum()
    if n_invalid > 0:
        import logging
        logging.getLogger(__name__).error(
            f"Coordinate validation: {n_invalid} sources have invalid RA/Dec. "
            f"These sources will be skipped. "
            f"Source IDs: {catalog_df.loc[~valid, 'source_id'].tolist()[:10]}"
        )
    
    # Add flag, do NOT silently drop — caller decides what to do
    catalog_df = catalog_df.copy()
    catalog_df['coord_valid'] = valid
    return catalog_df

# In pipeline: skip sources where coord_valid == False and log them
```

---

### FIX cat-M3 (MEDIUM): No Duplicate Source Detection

**Problem:** If the input catalog has duplicate source IDs or near-duplicate
coordinates (same source entered twice under different IDs), the pipeline
processes each independently and may produce inconsistent results.

**Fix:**

```python
def check_catalog_duplicates(catalog_df, ra_col='ra', dec_col='dec',
                               id_col='source_id', match_radius_arcsec=2.0):
    """
    Detect exact ID duplicates and coordinate near-duplicates.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    
    # Exact ID duplicates
    id_dups = catalog_df[id_col].duplicated()
    if id_dups.any():
        import logging
        logging.getLogger(__name__).warning(
            f"Duplicate source_ids found: {catalog_df.loc[id_dups, id_col].tolist()}"
        )
    
    # Coordinate near-duplicates (within 2 arcsec)
    coords = SkyCoord(
        ra=catalog_df[ra_col].values * u.deg,
        dec=catalog_df[dec_col].values * u.deg
    )
    idx, d2d, _ = coords.match_to_catalog_sky(coords, nthneighbor=2)
    near_dup_mask = d2d < match_radius_arcsec * u.arcsec
    
    if near_dup_mask.any():
        import logging
        logging.getLogger(__name__).warning(
            f"{near_dup_mask.sum()} sources have a neighbor within "
            f"{match_radius_arcsec} arcsec — possible coordinate duplicates."
        )
    
    return {
        'n_id_duplicates':    int(id_dups.sum()),
        'n_coord_duplicates': int(near_dup_mask.sum()),
    }
```

---

## FILE: config.py

---

### FIX cfg-H1 (HIGH): No Config Schema Validation at Startup

**Problem:** `pipeline_policy.yaml` is loaded with `yaml.safe_load()` and
fields are accessed with `.get()` throughout the codebase. A typo in the YAML
(e.g. `r_minn: 2.0` instead of `r_min: 2.0`) silently uses the default value
and produces wrong results with no error.

**Fix — add schema validation at startup:**

```python
# Install: pip install pydantic

from pydantic import BaseModel, Field, validator
from typing import Optional, List

class GateConfig(BaseModel):
    enabled: bool = True
    r_min: float = Field(2.0, ge=0.5, le=10.0)
    eps: float = Field(0.001, ge=0.0)

class TransientSpikeConfig(BaseModel):
    enabled: bool = True
    peak_frac: float = Field(0.6, ge=0.0, le=1.0)
    adjacency_frac: float = Field(0.4, ge=0.0, le=1.0)
    min_seasons: int = Field(4, ge=2, le=20)

class QAConfig(BaseModel):
    min_qual_frame: float = 1.0
    cc_flag_clean_only: bool = True
    min_points_per_season: int = Field(2, ge=1)
    min_clean_points_total: int = Field(10, ge=4)
    min_season_bins_total: int = Field(4, ge=2)

class PipelinePolicy(BaseModel):
    t_recall: float = Field(..., ge=0.0, le=1.0)
    strict_mode_default: bool = True
    random_seed: int = 42
    qa: QAConfig = QAConfig()
    state_change_ratio: GateConfig = GateConfig()
    transient_spike: TransientSpikeConfig = TransientSpikeConfig()
    
    @validator('t_recall')
    def t_recall_must_be_reasonable(cls, v):
        if v < 0.01 or v > 0.99:
            raise ValueError(f"t_recall={v} is outside reasonable range [0.01, 0.99]")
        return v

# At pipeline startup:
import yaml
from pydantic import ValidationError

def load_policy(config_path: str) -> PipelinePolicy:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    try:
        policy = PipelinePolicy(**raw)
    except ValidationError as e:
        raise ValueError(
            f"pipeline_policy.yaml validation failed:\n{e}\n"
            f"Check for typos in field names."
        )
    return policy
```

---

### FIX cfg-M2 (MEDIUM): WISE Flux Conversion Constants Are Duplicated

**Problem:** The Vega-to-mJy zero-point conversion for W1 and W2
(`F_nu_0_W1 = 309.54 Jy`, `F_nu_0_W2 = 171.787 Jy`) appears in at least
three places: wise_complete.py, config.py, and possibly clagn_score.py.
A correction to the zero-point (the WISE team has issued minor updates)
would require updating multiple files.

**Fix — single source of truth in config.py:**

```python
# In config.py ONLY:

# WISE Vega zero-points (Jarrett+2011, Table 1; Wright+2010)
# W1: 3.368 micron, W2: 4.618 micron
WISE_W1_VEGA_FLUX_ZERO_POINT_JY = 309.540   # Jy
WISE_W2_VEGA_FLUX_ZERO_POINT_JY = 171.787   # Jy
WISE_W3_VEGA_FLUX_ZERO_POINT_JY =  31.674   # Jy (for completeness)
WISE_W4_VEGA_FLUX_ZERO_POINT_JY =   8.363   # Jy

# Conversion: flux_mJy = 10^(-mag/2.5) * zero_point_Jy * 1000
# Example: mag=10.0, W1: flux = 10^(-4) * 309540 mJy = 30.954 mJy

def wise_mag_to_flux_mjy(mag, band='W1'):
    """Convert WISE Vega magnitude to flux in mJy."""
    zero_points = {
        'W1': WISE_W1_VEGA_FLUX_ZERO_POINT_JY,
        'W2': WISE_W2_VEGA_FLUX_ZERO_POINT_JY,
        'W3': WISE_W3_VEGA_FLUX_ZERO_POINT_JY,
        'W4': WISE_W4_VEGA_FLUX_ZERO_POINT_JY,
    }
    zp = zero_points[band]
    return 10.0 ** (-mag / 2.5) * zp * 1000.0  # mJy

# DELETE all other definitions of these zero-points elsewhere in the codebase
# USE this function for ALL magnitude-to-flux conversions
```

---

### FIX cfg-M3 (MEDIUM): No Logging Configuration — All Workers Log to stdout

**Problem:** There is no centralized logging configuration. Each module
creates its own logger with `logging.getLogger(__name__)`. For a long pipeline
run, all log messages go to stdout mixed together with no timestamps,
no log levels, no per-run log files, and no way to diagnose failures after the fact.

**Fix — add logging setup at pipeline entry:**

```python
import logging
import os
from datetime import datetime

def setup_logging(output_dir: str, level: str = 'INFO', run_id: str = None):
    """
    Configure logging for the full pipeline run.
    Outputs to both console (WARNING+) and file (DEBUG+).
    """
    if run_id is None:
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    log_path = os.path.join(output_dir, f'pipeline_{run_id}.log')
    os.makedirs(output_dir, exist_ok=True)
    
    # Root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    
    # Console handler: WARNING and above only (not too noisy)
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    
    # File handler: DEBUG and above (full detail)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(name)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S'
    ))
    
    root.addHandler(console)
    root.addHandler(file_handler)
    
    logging.getLogger(__name__).info(
        f"Pipeline logging initialized. Log file: {log_path}"
    )
    
    return log_path

# At pipeline entry:
log_path = setup_logging(output_dir=args.output, run_id=run_id)
```

---

## VERIFICATION SEQUENCE

```bash
# 1. Seed reproducibility: two runs with same seed must produce identical results
python run_pipeline.py --config configs/pipeline_policy.yaml \
    --input data/benchmark_dev.csv --output run_a/ --seed 42
python run_pipeline.py --config configs/pipeline_policy.yaml \
    --input data/benchmark_dev.csv --output run_b/ --seed 42

python -c "
import pandas as pd
a = pd.read_csv('run_a/scores.csv').sort_values('source_id')
b = pd.read_csv('run_b/scores.csv').sort_values('source_id')
assert (a['score'].round(8) == b['score'].round(8)).all(), 'Scores differ between runs!'
print('PASS: pipeline is reproducible with same seed')
"

# 2. Atomic write: no partial files after interruption
# (Run pipeline, kill it mid-run, verify no .tmp files remain)

# 3. Provenance: output contains git hash and config hash
python -c "
import json
with open('run_a/metrics.json') as f:
    m = json.load(f)
assert 'pipeline_version' in m, 'Missing pipeline_version'
assert 'config_hash' in m, 'Missing config_hash'
assert 'random_seed' in m, 'Missing random_seed'
print(f'Pipeline version: {m[\"pipeline_version\"]}')
print(f'Config hash: {m[\"config_hash\"]}')
print('PASS: provenance recorded')
"

# 4. Config validation catches typos
python -c "
import yaml
from clagn.config import load_policy
# Write bad config
with open('/tmp/bad_policy.yaml', 'w') as f:
    yaml.dump({'r_minn': 2.0, 't_recall': 0.35}, f)  # typo: r_minn
try:
    load_policy('/tmp/bad_policy.yaml')
    print('FAIL: should have raised on unknown field r_minn')
except Exception as e:
    print(f'PASS: caught bad config: {e}')
"

# 5. Per-class metrics are computed
python -c "
import json
with open('run_a/metrics.json') as f:
    m = json.load(f)
required = ['clagn_recall', 'sn_fpr', 'blazar_fpr', 'auc_roc', 'precision_at_20']
for k in required:
    assert k in m, f'Missing metric: {k}'
    print(f'{k}: {m[k]:.3f}')
print('PASS: all required metrics present')
"

# 6. CV threshold selection runs without touching test set
# (verify by code inspection: select_threshold_cv called only on dev data)

# 7. Full benchmark with all metrics
python run_pipeline.py \
    --config configs/pipeline_policy.yaml \
    --input data/benchmark_master.csv \
    --output results/final/ \
    --seed 42 \
    --evaluate

cat results/final/metrics.json
# Must contain: clagn_recall >= 0.80, blazar_fpr <= 0.10, auc_roc >= 0.80
```

---

## SUMMARY TABLE

| ID | File | Severity | Fix |
|----|------|----------|-----|
| infra-C1 | main.py | **CRITICAL** | global seed propagated to ALL stochastic steps |
| infra-C2 | main.py | **CRITICAL** | atomic output writes — no partial files on crash |
| infra-C3 | main.py | **CRITICAL** | provenance (git hash, config hash, seed) in every output |
| infra-H4 | main.py | High | per-source checkpointing for long runs |
| infra-H5 | main.py | High | IRSA rate limiter for parallel queries |
| eval-C1 | evaluation.py | **CRITICAL** | threshold selection uses CV not test set |
| eval-C2 | evaluation.py | **CRITICAL** | per-class FPR breakdown (SN, blazar, normal AGN) |
| eval-H3 | evaluation.py | High | threshold sensitivity analysis ±10% |
| eval-H4 | evaluation.py | High | AUC-ROC computed and reported |
| eval-H5 | evaluation.py | High | Precision@K and precision-recall curve |
| cat-H1 | catalog.py | High | redshift validation at ingestion |
| cat-H2 | catalog.py | High | RA/Dec validation before GAIA query |
| cat-M3 | catalog.py | Medium | duplicate source detection |
| cfg-H1 | config.py | High | pydantic schema validation — typos caught at startup |
| cfg-M2 | config.py | Medium | WISE flux zero-points single source of truth |
| cfg-M3 | config.py | Medium | centralized logging to file per run |

*These 16 issues are entirely distinct from the 96 bugs in the previous five fix prompts.*
*Apply AFTER the previous fix prompts have been applied and verified.*