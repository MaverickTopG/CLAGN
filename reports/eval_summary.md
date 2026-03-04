# Evaluation Summary

- split: `dev`
- rows_joined: `223`
- rows_used_for_metrics: `167`
- imbalance_warning: `Accuracy is not primary KPI for this benchmark.`
- baseline_all_negative_accuracy: `0.7844`
- baseline_random_accuracy: `0.5000`

## Stage Metrics

- Stage A balanced_accuracy: `0.7599`
- After blazar veto balanced_accuracy: `0.7713`
- After all gates balanced_accuracy: `0.5240`
- After tuned gates balanced_accuracy: `0.5240`

## FP Gap Math

- positives: `36`
- negatives: `131`
- current recall used for target gap: `0.0556`
- required_tnr_for_balanced_accuracy_0.90: `1.0000`
- max_fp_allowed_for_target_tnr: `0`
- current_fp: `1`

## Calibration

- Brier score: `0.164982`
- Reliability bins file: `reliability_bins.csv`

## Acceptance Targets

- recall_clagn >= 0.80: FAIL (`0.0556`)
- blazar_fpr <= 0.10: PASS (`0.0000`)
- balanced_accuracy >= 0.90: FAIL (`0.5240`)
