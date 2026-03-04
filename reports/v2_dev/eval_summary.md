# Evaluation Summary

- split: `dev`
- rows_joined: `3105`
- rows_used_for_metrics: `167`
- imbalance_warning: `Accuracy is not primary KPI for this benchmark.`
- baseline_all_negative_accuracy: `0.7844`
- baseline_random_accuracy: `0.5000`

## Stage Metrics

- Stage A balanced_accuracy: `0.7737`
- After blazar veto balanced_accuracy: `0.8730`
- After all gates balanced_accuracy: `0.8730`
- After tuned gates balanced_accuracy: `0.8806`

## FP Gap Math

- positives: `36`
- negatives: `131`
- current recall used for target gap: `0.9444`
- required_tnr_for_balanced_accuracy_0.90: `0.8556`
- max_fp_allowed_for_target_tnr: `18`
- current_fp: `24`

## Calibration

- Brier score: `0.164982`
- AUC-ROC: `0.8011`
- Reliability bins file: `reliability_bins.csv`
- Precision@K file: `precision_at_k.csv`

## Additional Metrics

- overall_accuracy (after tuned gates): `0.8443`

## Acceptance Targets

- recall_clagn >= 0.80: PASS (`0.9444`)
- blazar_fpr <= 0.10: PASS (`0.0000`)
- balanced_accuracy >= 0.90: FAIL (`0.8806`)
