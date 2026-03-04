# Evaluation Summary

- split: `test`
- rows_joined: `778`
- rows_used_for_metrics: `41`
- imbalance_warning: `Accuracy is not primary KPI for this benchmark.`
- baseline_all_negative_accuracy: `0.7561`
- baseline_random_accuracy: `0.5000`

## Stage Metrics

- Stage A balanced_accuracy: `0.6903`
- After blazar veto balanced_accuracy: `0.7710`
- After all gates balanced_accuracy: `0.7710`
- After tuned gates balanced_accuracy: `N/A`

## FP Gap Math

- positives: `10`
- negatives: `31`
- current recall used for target gap: `0.8000`
- required_tnr_for_balanced_accuracy_0.90: `1.0000`
- max_fp_allowed_for_target_tnr: `0`
- current_fp: `8`

## Calibration

- Brier score: `0.180015`
- AUC-ROC: `0.7226`
- Reliability bins file: `reliability_bins.csv`
- Precision@K file: `precision_at_k.csv`

## Additional Metrics

- overall_accuracy (after tuned gates): `0.7561`

## Acceptance Targets

- recall_clagn >= 0.80: PASS (`0.8000`)
- blazar_fpr <= 0.10: PASS (`0.0000`)
- balanced_accuracy >= 0.90: FAIL (`0.7710`)
