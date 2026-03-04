# Round2 Test Comparison (Frozen Policy, One-Shot)

## Overall Metrics (After All Gates)

| metric | baseline | tuned_once | delta |
|---|---:|---:|---:|
| tp | 8.000000 | 8.000000 | +0.000000 |
| fp | 8.000000 | 8.000000 | +0.000000 |
| fn | 2.000000 | 2.000000 | +0.000000 |
| tn | 23.000000 | 23.000000 | +0.000000 |
| precision | 0.500000 | 0.500000 | +0.000000 |
| recall | 0.800000 | 0.800000 | +0.000000 |
| balanced_accuracy | 0.770968 | 0.770968 | +0.000000 |
| mcc | 0.477088 | 0.477088 | +0.000000 |
| overall_accuracy | 0.756098 | 0.756098 | +0.000000 |

## Class FP Comparison (After All Gates)

| class | fp_neg_base | fp_neg_tuned | delta | fpr_base | fpr_tuned | delta_fpr |
|---|---:|---:|---:|---:|---:|---:|
| blazar | 0 | 0 | +0 | 0.000000 | 0.000000 | +0.000000 |
| clagn | 0 | 0 | +0 | nan | nan | +nan |
| normal_agn | 3 | 3 | +0 | 0.375000 | 0.375000 | +0.000000 |
| sn | 2 | 2 | +0 | 0.400000 | 0.400000 | +0.000000 |
| star | 3 | 3 | +0 | 0.230769 | 0.230769 | +0.000000 |

## Conclusion
- No measurable test change versus baseline in this pass.