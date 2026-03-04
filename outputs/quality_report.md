# Quality Report

## Dataset Summary

- Total rows: 279
- Duplicate `canonical_id` count: 0
- Missing score count: 72

## Counts by Status

| value | count |
| --- | --- |
| OK | 207 |
| TOO_FEW_POINTS | 70 |
| NO_DATA | 2 |

## Counts by Band Coverage

| value | count |
| --- | --- |
| W1,W2 | 241 |
| none | 31 |
| W1_only | 7 |

## ID Integrity

- Rows where `source_id_normalized != canonical_id`: 0
- Rows where raw `source_id != canonical_id`: 31
- Ambiguous ID count (e.g., PKS/PG/MR/HS/B2): 5

### ID Mismatch Breakdown

| id_mismatch_reason | count |
| --- | --- |
| <none> | 248 |
| sdss_double_j | 31 |

### Mismatch Preview (first 20)

| source_id | source_id_normalized | canonical_id | id_mismatch_reason |
| --- | --- | --- | --- |
| SDSSJJ005813.68+001409.09 | SDSSJ005813.68+001409.09 | SDSSJ005813.68+001409.09 | sdss_double_j |
| SDSSJJ012256.19-000252.68 | SDSSJ012256.19-000252.68 | SDSSJ012256.19-000252.68 | sdss_double_j |
| SDSSJJ015653.16-004623.28 | SDSSJ015653.16-004623.28 | SDSSJ015653.16-004623.28 | sdss_double_j |
| SDSSJJ015726.00-000602.15 | SDSSJ015726.00-000602.15 | SDSSJ015726.00-000602.15 | sdss_double_j |
| SDSSJJ020222.00+010557.11 | SDSSJ020222.00+010557.11 | SDSSJ020222.00+010557.11 | sdss_double_j |
| SDSSJJ022930.91-000845.37 | SDSSJ022930.91-000845.37 | SDSSJ022930.91-000845.37 | sdss_double_j |
| SDSSJJ025410.08+034911.9 | SDSSJ025410.08+034911.9 | SDSSJ025410.08+034911.9 | sdss_double_j |
| SDSSJJ080138.67+423354.1 | SDSSJ080138.67+423354.1 | SDSSJ080138.67+423354.1 | sdss_double_j |
| SDSSJJ081319.34+460849.5 | SDSSJ081319.34+460849.5 | SDSSJ081319.34+460849.5 | sdss_double_j |
| SDSSJJ081425.89+294115.6 | SDSSJ081425.89+294115.6 | SDSSJ081425.89+294115.6 | sdss_double_j |
| SDSSJJ082323.88+422048.2 | SDSSJ082323.88+422048.2 | SDSSJ082323.88+422048.2 | sdss_double_j |
| SDSSJJ082942.66+415436.8 | SDSSJ082942.66+415436.8 | SDSSJ082942.66+415436.8 | sdss_double_j |
| SDSSJJ090258.39+252915.15 | SDSSJ090258.39+252915.15 | SDSSJ090258.39+252915.15 | sdss_double_j |
| SDSSJJ091407.16+243633.08 | SDSSJ091407.16+243633.08 | SDSSJ091407.16+243633.08 | sdss_double_j |
| SDSSJJ093812.27+074340.0 | SDSSJ093812.27+074340.0 | SDSSJ093812.27+074340.0 | sdss_double_j |
| SDSSJJ094443.08+580953.2 | SDSSJ094443.08+580953.2 | SDSSJ094443.08+580953.2 | sdss_double_j |
| SDSSJJ100256.22+475027.7 | SDSSJ100256.22+475027.7 | SDSSJ100256.22+475027.7 | sdss_double_j |
| SDSSJJ101322.43+214021.74 | SDSSJ101322.43+214021.74 | SDSSJ101322.43+214021.74 | sdss_double_j |
| SDSSJJ105058.42+241351.18 | SDSSJ105058.42+241351.18 | SDSSJ105058.42+241351.18 | sdss_double_j |
| SDSSJJ105325.40+302419.34 | SDSSJ105325.40+302419.34 | SDSSJ105325.40+302419.34 | sdss_double_j |

## Baseline Consistency Check

- Rows with provided `baseline_years`: 279
- Rows with finite baseline diff: 279
- Rows with `abs(baseline_years - baseline_years_recomputed) > 0.01`: 0
- Max baseline diff: 1.7763568394002505e-15
- Downstream analysis uses `baseline_years_recomputed` as the trusted value.

## Coverage Metrics

- coverage_rate = OK / total = 74.2%
- strict_coverage_rate = strict_quality / total = 67.7%

## Rejection Bucket Histogram

| rejection_bucket | count |
| --- | --- |
| ok | 207 |
| few_epochs | 38 |
| too_few_points_other | 29 |
| short_baseline | 3 |
| no_data | 2 |

## Correlation Summary

| pair | available | rho_spearman | pvalue | n | note |
| --- | --- | --- | --- | --- | --- |
| score_vs_n_points | True | 0.3485 | 2.658e-07 | 207 |  |
| score_vs_baseline_years | True | 0.2116 | 2.209e-03 | 207 |  |
| score_vs_delta_mag | True | 0.0903 | 1.956e-01 | 207 |  |
| score_vs_abs_delta_mag | True | 0.5684 | 4.150e-19 | 207 |  |

## Warnings

- None
