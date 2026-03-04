# Evaluation Summary

- split: `dev`
- rows_joined: `223`
- rows_used_for_metrics: `167`
- imbalance_warning: `Accuracy is not primary KPI for this benchmark.`
- baseline_all_negative_accuracy: `0.7844`
- baseline_random_accuracy: `0.5000`

## Stage Metrics

- Stage A balanced_accuracy: `0.7599`
- After blazar veto balanced_accuracy: `0.8591`
- After all gates balanced_accuracy: `0.7747`
- After tuned gates balanced_accuracy: `0.8667`

## FP Gap Math

- positives: `36`
- negatives: `131`
- current recall used for target gap: `0.9167`
- required_tnr_for_balanced_accuracy_0.90: `0.8833`
- max_fp_allowed_for_target_tnr: `15`
- current_fp: `24`

## Calibration

- Brier score: `0.164982`
- AUC-ROC: `0.8011`
- Reliability bins file: `reliability_bins.csv`
- Precision@K file: `precision_at_k.csv`

## Additional Metrics

- overall_accuracy (after tuned gates): `0.8383`

## Acceptance Targets

- recall_clagn >= 0.80: PASS (`0.9167`)
- blazar_fpr <= 0.10: PASS (`0.0000`)
- balanced_accuracy >= 0.90: FAIL (`0.8667`)
