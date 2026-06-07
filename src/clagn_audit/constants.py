from __future__ import annotations

from typing import Final

VALID_STATUSES: Final[tuple[str, ...]] = (
    'OK',
    'TOO_FEW_POINTS',
    'NO_DATA',
    'ERROR',
    'NAME_RESOLVE_FAIL',
    'UNKNOWN',
)

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    'source_id',
    'canonical_id',
    'score',
    'status',
    'n_points',
    'baseline_days',
    'band_coverage',
)

OPTIONAL_COLUMNS: Final[tuple[str, ...]] = (
    "vf3_optional_quality_flag",
    'baseline_years',
    'delta_mag',
    'delta_mag_method',
    'rejection_reason',
)

NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    'score',
    'n_points',
    'baseline_days',
    'baseline_years',
    'delta_mag',
)

DEFAULT_DPI: Final[int] = 180
DEFAULT_FIGSIZE: Final[tuple[int, int]] = (6, 4)
DEFAULT_HIST_BINS: Final[int] = 30
DEFAULT_BIAS_BINS: Final[int] = 6

SORT_COLUMNS_DEFAULT: Final[tuple[str, ...]] = ('canonical_id', 'source_id')

CASE_MINING_COLUMNS: Final[tuple[str, ...]] = (
    'source_id',
    'source_id_normalized',
    'canonical_id',
    'source_id_matches_canonical',
    'id_mismatch_reason',
    'is_ambiguous_id',
    'score',
    'status',
    'n_points',
    'baseline_days',
    'baseline_years',
    'baseline_years_recomputed',
    'baseline_years_diff',
    'baseline_years_consistent',
    'band_coverage',
    'delta_mag',
    'rejection_reason',
    'strict_quality',
)

SCORE_DEFINITION_TEMPLATE: Final[str] = """# Score Definition (Template)

## 1. What the Current Score Is

The current `score` is treated as a **ranking statistic**, not a calibrated probability.
It is used to rank CLAGN candidates for triage, inspection, and follow-up prioritization.

## 2. What the Score Is Not

- It is **not** a posterior probability that a source is a CLAGN.
- It is **not** a direct estimate of a physical parameter (e.g., accretion rate change).
- It should **not** be interpreted probabilistically unless calibrated against independent labels.

## 3. Current Usage

- Rank candidates for manual review.
- Apply quality flags and coverage filters.
- Select subsets for crossmatch, light-curve inspection, and spectroscopy prioritization.

## 4. Calibration Roadmap (Future Work)

- Injection-recovery characterization across cadence, amplitude, and baseline.
- Reliability calibration using independent labeled benchmarks.
- Probability calibration (e.g., isotonic regression or Platt scaling) on a dev set only.
- Locked threshold evaluation on a held-out test set with frozen benchmark hashes.

## 5. Reporting Rules

- Do not describe the current score as a probability.
- Report thresholds together with validation context and selection effects.
- Report uncertainty / confidence intervals for performance metrics when labels are available.

## 6. Versioning Note

Record the scoring pipeline version, configuration hash, and benchmark hash in all derived results.
"""

RUNBOOK_TEMPLATE: Final[str] = """# Top-Tier Analysis Runbook

## Purpose

This analysis layer audits an existing CLAGN candidate results table and produces reproducible quality reports, bias diagnostics, and case-mining outputs.

## Run

```bash
python -m analysis.run_all --input data/results.csv --outdir outputs/
```

## Key Outputs

- `outputs/cleaned_results.csv`: typed and enriched table with derived columns
- `outputs/quality_report.md`: status/coverage summary + correlations
- `outputs/bias_audit.md`: proxy rejection-bias analysis
- `outputs/top_candidates.csv`: case-mining tables for manual follow-up
- `outputs/*.png`: reproducible figures
- `outputs/run_manifest.json`: run metadata and deterministic settings

## Interpretation Notes

- `score` is a ranking statistic, not a calibrated probability.
- `strict_quality` is a conservative data-quality flag for follow-up prioritization.
- Bias audit here is proxy-only and does not establish causality.
"""
