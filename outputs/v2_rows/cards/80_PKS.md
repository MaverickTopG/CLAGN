# Candidate Card: PKS (Rank 80)

## Basic Metadata

- `source_id`: `PKS`
- `canonical_id`: `PKS`
- `score`: `0.456678`
- `status`: `OK`
- `final_label`: `Known blazar`
- `stage_a_positive`: `True`
- `gate_blazar_veto`: `False`
- `gate_wise_qc_pass`: `True`
- `gate_data_sufficient`: `True`
- `decision_trace`: `stage_a_positive=pass;gate_blazar_veto=fail;gate_wise_qc_pass=pass;gate_data_sufficient=pass;gate_state_change_ratio_pass=pass;gate_transient_spike_pass=pass`

## Sampling / Variability Summary

- `n_points` (results table): `181.0`
- `baseline_days`: `5272.62`
- `baseline_years`: `14.436`
- `band_coverage`: `W1,W2`
- `delta_mag`: `-0.069`
- `delta_mag_method`: `seasonal_median_flux`

## Coordinates

- `ra`: `196.387600`
- `dec`: `-10.555400`
- `coordinate_source`: `benchmark_master`

## WISE Light Curve

![WISE light curve](../figures/PKS_wise.png)

## WISE QA / Rejection Accounting

| Metric | Value |
| --- | --- |
| Raw points total | 628 |
| Raw points W1 | 314 |
| Raw points W2 | 314 |
| Kept points total | 627 |
| Kept points W1 | 314 |
| Kept points W2 | 313 |
| Fraction rejected | 0.00159236 |
| Reject: missing time | 0 |
| Reject: missing mag | 0 |
| Reject: invalid band | 0 |
| Reject: cc_flags | 1 |
| Reject: qual_frame | 0 |
| Reject: moon_lev | 0 |

### QA Reject Reason Counts

| reject_reason | count |
| --- | --- |
| reject_cc_flags | 1 |
| reject_invalid_band | 0 |
| reject_missing_mag | 0 |
| reject_missing_time | 0 |
| reject_moon_lev | 0 |
| reject_qual_frame | 0 |

## Flag Summary

### `cc_flags`

| value | count | fraction |
| --- | --- | --- |
| 0000 | 626 | 0.996815 |
| 0h00 | 2 | 0.00318471 |

### `qual_frame`

| value | count | fraction |
| --- | --- | --- |
| <NA> | 628 | 1 |

### `moon_lev`

| value | count | fraction |
| --- | --- | --- |
| 00 | 506 | 0.805732 |
| 0000 | 64 | 0.101911 |
| 11 | 52 | 0.0828025 |
| 01 | 6 | 0.00955414 |

### `qi_fact`

| value | count | fraction |
| --- | --- | --- |
| 1.0 | 540 | 0.859873 |
| 0.5 | 82 | 0.130573 |
| 0.0 | 6 | 0.00955414 |

### `saa_sep`

| value | count | fraction |
| --- | --- | --- |
| 87.0 | 4 | 0.00636943 |
| -2.0 | 4 | 0.00636943 |
| 91.0 | 4 | 0.00636943 |
| 29.0 | 4 | 0.00636943 |
| 89.98200225830078 | 2 | 0.00318471 |
| 36.74599838256836 | 2 | 0.00318471 |
| 16.229999542236328 | 2 | 0.00318471 |
| 78.46199798583984 | 2 | 0.00318471 |

### `ph_qual`

| value | count | fraction |
| --- | --- | --- |
| AA | 562 | 0.894904 |
| <NA> | 64 | 0.101911 |
| AB | 2 | 0.00318471 |

## Coverage Summary

- Seasonal bins (W1): `16`
- Seasonal bins (W2): `16`
- Pre-gap coverage: `True`
- Post-gap coverage: `True`
- Both-sides coverage: `True`

## Systematics Verdict

- Verdict: `PASS`
- Summary: QA-clean points and seasonal coverage support the observed variability signal.
- QA-dominated signal check: Not obviously dominated by flagged/low-quality points

Reasons:
- Seasonal trend supported by sufficient QA-clean W1/W2 points

## Catalog Checks

- Known blazar match: `YES` via `source_id` token `PKS`
- Known CLAGN match: `NO`
- Gaia metrics: not provided / no match

## Final Label

**Known blazar**

- Rank 80 with score=0.4567
- Sampling: n_points=181.0, baseline_years=14.4356384623, band_coverage=W1,W2
- QA rejection fraction=0.2% (main causes summarized in card QA section)
- Systematics verdict=PASS: QA-clean points and seasonal coverage support the observed variability signal.
- Known blazar catalog match=YES
- Dataset metadata class=blazar

## Reproducibility

- `run_timestamp_utc`: `2026-03-02T06:47:01.885248+00:00`
- `results_file`: `data/real_clagn/benchmark_scores_v2.csv`
- `wise_file`: `data/wise/PKS.csv`
- `score_threshold`: `0.0`
- `t_recall`: `0.3493918746`
