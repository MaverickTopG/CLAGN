# Baseline Snapshot (Pre-Round2 Tuning)

## Dev After-All-Gates

- metrics: `{'tp': 34, 'fp': 26, 'fn': 2, 'tn': 105, 'precision': 0.566667, 'recall': 0.944444, 'balanced_accuracy': 0.872986, 'mcc': 0.639354, 'overall_accuracy': 0.832335}`
- fp_by_class: `{'star': 14, 'sn': 7, 'normal_agn': 5}`

## Test After-All-Gates

- metrics: `{'tp': 8, 'fp': 8, 'fn': 2, 'tn': 23, 'precision': 0.5, 'recall': 0.8, 'balanced_accuracy': 0.770968, 'mcc': 0.477088, 'overall_accuracy': 0.756098}`
- fp_by_class: `{'star': 3, 'normal_agn': 3, 'sn': 2}`

## Source Files

- `reports/v2_dev/eval_after_all_gates.csv`
- `reports/v2_test/eval_after_all_gates.csv`
- `reports/v2_dev/dev_false_positive_audit.csv`
- `reports/v2_test/dev_false_positive_audit.csv`