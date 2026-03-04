# CLAGN Candidate Cards (Offline WISE-First)

This pipeline builds reproducible 1-page candidate cards (Markdown, optional PDF) for the top-N CLAGN candidates from a `results.csv` table.

## Files
- `make_cards.py` — main CLI
- `scripts/merge_master_with_cards.py` — merge `summary_table.csv` fields into benchmark master
- `scripts/evaluate_real_clagn.py` — stage-wise imbalanced evaluation and acceptance checks
- `wise_processing.py` — WISE schema normalization, QA filtering, seasonal binning, plotting
- `catalog_match.py` — local catalog matching (known blazars, known CLAGN, optional Gaia)
- `reporting.py` — Markdown rendering helpers
- `configs/pipeline_policy.yaml` — strict policy (thresholds, QA limits, acceptance targets)

## Requirements
Python + minimal packages:
- `numpy`
- `pandas`
- `matplotlib`

Optional:
- `pandoc` (only if you use `--pdf`)

Install (example):
```bash
pip install numpy pandas matplotlib
```

## Required Inputs
- `results.csv` with columns:
  - `source_id, canonical_id, score, status, n_points, baseline_days, band_coverage`
  - optional: `baseline_years, delta_mag, delta_mag_method, rejection_reason, ra, dec, class, y_true`
- Local WISE files per selected object at:
  - `data/wise/{canonical_id}.csv`
- Local blazar catalog:
  - `known_blazars.csv` with at least one ID column among `canonical_id, source_id, name, alias, aliases`

## Optional Inputs
- `data/benchmark/known_clagn.csv` (auto-detected if present)
- Gaia metrics CSV (e.g. `data/real_clagn/cache/crossmatch/gaia.csv`)
- `coordinates.csv` if `results.csv` lacks RA/Dec
- `data/real_clagn/benchmark_master.csv` (used as fallback for coordinates/class if present)

## Exact Commands
Basic run:
```bash
python make_cards.py \
  --results data/results.csv \
  --topn 7 \
  --outdir . \
  --score-thresh 0.7 \
  --wise-dir data/wise \
  --known-blazars known_blazars.csv \
  --policy-config configs/pipeline_policy.yaml \
  --strict-mode
```

With benchmark fallback coordinates/class:
```bash
python make_cards.py \
  --results data/results.csv \
  --topn 7 \
  --outdir . \
  --wise-dir data/wise \
  --known-blazars known_blazars.csv \
  --benchmark-master data/real_clagn/benchmark_master.csv
```

With optional files + PDF export:
```bash
python make_cards.py \
  --results data/results.csv \
  --topn 7 \
  --outdir . \
  --wise-dir data/wise \
  --known-blazars known_blazars.csv \
  --known-clagn data/benchmark/known_clagn.csv \
  --gaia data/real_clagn/cache/crossmatch/gaia.csv \
  --pdf
```

Generate cards and then auto-merge card outputs into benchmark master:
```bash
python make_cards.py \
  --results data/results.csv \
  --topn 7 \
  --outdir . \
  --wise-dir data/wise \
  --known-blazars known_blazars.csv \
  --merge-master \
  --master-csv data/real_clagn/benchmark_master.csv \
  --master-out data/real_clagn/benchmark_master_augmented.csv
```

Run merge step standalone:
```bash
python scripts/merge_master_with_cards.py \
  --master-csv data/real_clagn/benchmark_master.csv \
  --summary-csv summary_table.csv \
  --out data/real_clagn/benchmark_master_augmented.csv
```

Run strict stage-wise evaluation:
```bash
python scripts/evaluate_real_clagn.py \
  --data_dir data/real_clagn \
  --split dev \
  --scores_csv benchmark_scores.csv \
  --policy-config configs/pipeline_policy.yaml \
  --stage-report summary_table.csv \
  --output_dir reports \
  --strict-mode
```

## Output Structure
- `summary_table.csv`
- `benchmark_scores_augmented.csv`
- `cards/`
  - `01_<canonical_id>.md`
  - optional `01_<canonical_id>.pdf`
- `figures/`
  - `<canonical_id>_wise.png`
- `reports/`
  - `benchmark_report.md`
  - `run_manifest.json`
  - `eval_stage_a.csv`
  - `eval_after_blazar_veto.csv`
  - `eval_after_all_gates.csv`
  - `eval_summary.md`

## Common Failure Messages (and fixes)
- **Missing WISE file**: add `data/wise/{canonical_id}.csv` for every selected top-N object, or lower `--topn`, or use `--skip-missing-wise`.
- **Missing coordinates**: add `ra/dec` to `results.csv`, provide `coordinates.csv`, or pass `--benchmark-master data/real_clagn/benchmark_master.csv`.
- **Missing blazar catalog**: provide `--known-blazars known_blazars.csv` or pass `--allow-missing-blazar-catalog` (not recommended).
- **Unsupported WISE schema**: inspect the error message listing columns, then map your file to common IRSA-style time/mag/flag columns.

## Notes
- WISE is primary. No web calls are required.
- The pipeline is deterministic (sorted selection, fixed seasonal bins, fixed plotting settings).
- `score` is treated as a ranking statistic. The card labels are rule-based summaries, not calibrated probabilities.
- Stage columns exported for audit: `stage_a_positive`, `gate_blazar_veto`, `gate_wise_qc_pass`, `gate_data_sufficient`, `final_label`, `decision_trace`.
