# Claude Handoff: CLAGN Pipeline Status (Up-to-date)

## 1) Project Purpose
This repository runs a WISE-first CLAGN candidate pipeline with strict stage gating, catalog vetoes, QA checks, candidate cards, and imbalanced evaluation.

Primary benchmark data lives in:
- `data/real_clagn/benchmark_master.csv`
- `data/real_clagn/benchmark_dev.csv`
- `data/real_clagn/benchmark_test.csv`
- `data/real_clagn/benchmark_scores.csv`

## 2) What Was Added/Changed Recently
The following major hardening updates were implemented.

- Policy-driven pipeline behavior via `configs/pipeline_policy.yaml`.
- Stage outputs and decision trace fields exported to score tables.
- Stage-wise evaluator in `scripts/evaluate_real_clagn.py`.
- New Stage-B discriminators in `make_cards.py`:
  - Gate A: state-change vs jitter ratio.
  - Gate B: transient spike veto.
- Dev false-positive audit export.
- Dev-only tuning grid search export.
- Acceptance checks for recall/blazar-FPR/balanced-accuracy.

## 3) Pipeline Logic (Current)

### Stage A (recovery)
- `stage_a_positive = (status == OK) AND (score finite) AND (score >= t_recall)`
- Default `t_recall` from policy: `0.3493918746`.

### Stage B (purity gates)
Final positive requires all of:
- `gate_blazar_veto == True`
- `gate_wise_qc_pass == True`
- `gate_data_sufficient == True`
- `gate_state_change_ratio_pass == True`
- `gate_transient_spike_pass == True`

### New Gate A (state change ratio)
Computed from seasonal W1/W2 medians and intra-season scatter:
- `delta_season = max(season_median) - min(season_median)`
- `sigma_intra = median(scatter_mag)` (fallback robust MAD if needed)
- `R_band = delta_season / (sigma_intra + eps)`
- `state_change_ratio_value = max(R_band)`
- Pass if `state_change_ratio_value >= r_min`.

Policy defaults:
- `r_min = 2.0`
- `eps = 0.001`

### New Gate B (transient spike veto)
Feature form used for tuning/eval:
- `transient_max_nonpeak_ratio`
- `transient_max_adjacency_ratio`
- `transient_min_band_seasons`

Flagged transient if:
- `max_adjacency_ratio < adjacency_frac`
- `max_nonpeak_ratio < peak_frac`
- `min_band_seasons >= min_seasons`

Gate pass is inverse of transient flag.

Policy defaults:
- `peak_frac = 0.6`
- `adjacency_frac = 0.4`
- `min_seasons = 4`

## 4) Policy File (Current)
`configs/pipeline_policy.yaml` now includes:
- `t_recall`
- `qa`:
  - `min_qual_frame`
  - `cc_flag_clean_only`
  - `moon_lev_reject_min`
  - `min_points_per_season`
  - `min_good_points_per_season`
  - `min_clean_points_total`
  - `min_season_bins_total`
- `gates`:
  - `hard_blazar_veto`
  - `require_wise_qc_pass`
  - `require_data_sufficient`
  - `state_change_ratio {enabled, r_min, eps}`
  - `transient_spike {enabled, peak_frac, adjacency_frac, min_seasons}`
- `tuning` grids and objective.
- `acceptance_targets`:
  - `recall_clagn_min = 0.80`
  - `blazar_fpr_max = 0.10`
  - `balanced_accuracy_min = 0.90`

## 5) Output Contract (Current)
`benchmark_scores_augmented.csv` and summary outputs include at least:
- `stage_a_positive`
- `gate_blazar_veto`
- `gate_wise_qc_pass`
- `gate_data_sufficient`
- `gate_state_change_ratio_pass`
- `gate_transient_spike_pass`
- `state_change_ratio_value`
- `transient_spike_flag`
- `transient_max_nonpeak_ratio`
- `transient_max_adjacency_ratio`
- `transient_min_band_seasons`
- `fp_mode_hint`
- `final_label`
- `decision_trace`

## 6) Evaluation Artifacts (Current)
Produced by `scripts/evaluate_real_clagn.py`:
- `reports/eval_stage_a.csv`
- `reports/eval_after_blazar_veto.csv`
- `reports/eval_after_all_gates.csv`
- `reports/eval_after_tuned_gates.csv`
- `reports/dev_false_positive_audit.csv`
- `reports/dev_tuning_grid.csv`
- `reports/dev_best_policy.json`
- `reports/class_contamination_*.csv`
- `reports/eval_summary.md`
- `reports/metrics_summary.json`

## 7) Latest Metrics Snapshot (from current repo run)
Dev (`reports/eval_*`):
- Stage A: `TP=33 FP=52 FN=3 TN=79`, recall `0.9167`, balanced accuracy `0.7599`.
- After all gates: `TP=2 FP=1 FN=34 TN=130`, recall `0.0556`, balanced accuracy `0.5240`.
- Tuned best policy currently same as above (no gain).

Test (`reports/test_eval/eval_after_all_gates.csv`):
- `TP=0 FP=0 FN=10 TN=31`, recall `0.0000`, balanced accuracy `0.5000`.

Best dev policy selected by current grid (`reports/dev_best_policy.json`):
- `r_min=1.5, peak_frac=0.5, adjacency_frac=0.3, min_clean_points_total=10, min_season_bins_total=4`
- Metrics still poor due low recall.

## 8) Important Current Constraint
`make_cards.py` processing loop is top-N over `status=OK` rows and requires local/fetched WISE per candidate. In current run, only 7 cards were generated, so many rows lacked rich gate features unless fully fetched and processed.

This is the key reason current tuned/eval outputs can collapse recall if feature columns are missing and conservative-failed.

## 9) Full Command Block to Run Entire Benchmark Path
Use this block to run full-row style processing and stage evaluation:

```bash
python make_cards.py \
  --results data/real_clagn/benchmark_scores.csv \
  --topn 100000 \
  --outdir outputs/full_rows \
  --score-thresh 0.0 \
  --benchmark-master data/real_clagn/benchmark_master.csv \
  --known-blazars known_blazars.csv \
  --policy-config configs/pipeline_policy.yaml \
  --strict-mode \
  --overwrite \
  --fetch-wise-if-missing

python scripts/evaluate_real_clagn.py \
  --data_dir data/real_clagn \
  --split dev \
  --scores_csv data/real_clagn/benchmark_scores.csv \
  --stage-report outputs/full_rows/benchmark_scores_augmented.csv \
  --policy-config configs/pipeline_policy.yaml \
  --output_dir reports/full_rows_dev \
  --strict-mode

python scripts/evaluate_real_clagn.py \
  --data_dir data/real_clagn \
  --split test \
  --scores_csv data/real_clagn/benchmark_scores.csv \
  --stage-report outputs/full_rows/benchmark_scores_augmented.csv \
  --policy-config configs/pipeline_policy.yaml \
  --output_dir reports/full_rows_test \
  --strict-mode
```

## 10) Critical Next Steps
- Ensure gate features are computed for all evaluation rows (not only a small processed subset).
- Keep tuning strictly on dev split.
- Freeze selected policy.
- Run one-shot test.
- Re-check acceptance targets:
  - recall >= 0.80
  - blazar FPR <= 0.10
  - balanced accuracy >= 0.90

## 11) Files Most Relevant for Debugging
- `make_cards.py`
- `scripts/evaluate_real_clagn.py`
- `wise_processing.py`
- `configs/pipeline_policy.yaml`
- `outputs/*/benchmark_scores_augmented.csv`
- `reports/*/eval_summary.md`
- `reports/dev_tuning_grid.csv`

## 12) Notes for Claude (Actionable)
- Treat current dev/test metrics as interim and likely feature-coverage-limited.
- Do not claim performance improvements without verifying feature completeness across rows.
- Prioritize failure analysis on false negatives after Stage B.
- Keep all threshold/gate fitting strictly dev-only.
- Preserve deterministic sorting and strict schema checks.
