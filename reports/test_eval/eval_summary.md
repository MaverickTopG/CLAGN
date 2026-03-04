# Evaluation Summary

- split: `test`
- rows_joined: `56`
- rows_used_for_metrics: `41`
- imbalance_warning: `Accuracy is not primary KPI for this benchmark.`
- baseline_all_negative_accuracy: `0.7561`
- baseline_random_accuracy: `0.5000`

## Stage Metrics

- Stage A balanced_accuracy: `0.6903`
- After blazar veto balanced_accuracy: `0.6903`
- After all gates balanced_accuracy: `0.5000`
- After tuned gates balanced_accuracy: `N/A`

## FP Gap Math

- positives: `10`
- negatives: `31`
- current recall used for target gap: `0.0000`
- required_tnr_for_balanced_accuracy_0.90: `1.8000`
- max_fp_allowed_for_target_tnr: `0`
- current_fp: `0`

## Calibration

- Brier score: `0.180015`
- Reliability bins file: `reliability_bins.csv`

## Acceptance Targets

- recall_clagn >= 0.80: FAIL (`0.0000`)
- blazar_fpr <= 0.10: PASS (`0.0000`)
- balanced_accuracy >= 0.90: FAIL (`0.5000`)
