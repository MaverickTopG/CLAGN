# Candidate Card: IIZw 1 (Rank 72)

## Basic Metadata

- `source_id`: `IIZw 1`
- `canonical_id`: `IIZw 1`
- `source_id_normalized`: `IIZW 1`
- `canonical_id_normalized`: `IIZW 1`
- `score`: `0.486633`
- `status`: `OK`
- `final_label`: `Known blazar`
- `stage_a_positive`: `True`
- `gate_blazar_veto`: `False`
- `gate_wise_qc_pass`: `True`
- `gate_data_sufficient`: `True`
- `decision_trace`: `stage_a_positive=pass;gate_blazar_veto=fail;gate_wise_qc_pass=pass;gate_data_sufficient=pass;gate_state_change_ratio_pass=pass;gate_transient_spike_pass=pass`

## Sampling / Variability Summary

- `n_points` (results table): `202.0`
- `baseline_days`: `5299.31`
- `baseline_years`: `14.509`
- `band_coverage`: `W1,W2`
- `delta_mag`: `-0.096`
- `delta_mag_method`: `seasonal_median_flux`

## Coordinates

- `ra`: `20.499300`
- `dec`: `-1.040000`
- `coordinate_source`: `benchmark_master`

## WISE Light Curve

![WISE light curve](../figures/IIZw_1_wise.png)

## WISE QA / Rejection Accounting

| Metric | Value |
| --- | --- |
| Raw points total | 626 |
| Raw points W1 | 313 |
| Raw points W2 | 313 |
| Kept points total | 626 |
| Kept points W1 | 313 |
| Kept points W2 | 313 |
| Fraction rejected | 0 |
| Reject: missing time | 0 |
| Reject: missing mag | 0 |
| Reject: invalid band | 0 |
| Reject: cc_flags | 0 |
| Reject: qual_frame | 0 |
| Reject: moon_lev | 0 |

### QA Reject Reason Counts

| reject_reason | count |
| --- | --- |
| reject_cc_flags | 0 |
| reject_invalid_band | 0 |
| reject_missing_mag | 0 |
| reject_missing_time | 0 |
| reject_moon_lev | 0 |
| reject_qual_frame | 0 |

## Flag Summary

### `cc_flags`

| value | count | fraction |
| --- | --- | --- |
| 0000 | 626 | 1 |

### `qual_frame`

| value | count | fraction |
| --- | --- | --- |
| <NA> | 626 | 1 |

### `moon_lev`

| value | count | fraction |
| --- | --- | --- |
| 00 | 528 | 0.84345 |
| 0000 | 54 | 0.086262 |
| 11 | 44 | 0.0702875 |

### `qi_fact`

| value | count | fraction |
| --- | --- | --- |
| 1.0 | 464 | 0.741214 |
| 0.5 | 138 | 0.220447 |
| 0.0 | 24 | 0.0383387 |

### `saa_sep`

| value | count | fraction |
| --- | --- | --- |
| 11.0 | 4 | 0.00638978 |
| 40.03300094604492 | 4 | 0.00638978 |
| 116.0 | 4 | 0.00638978 |
| 90.48300170898438 | 2 | 0.00319489 |
| 88.02999877929688 | 2 | 0.00319489 |
| 45.415000915527344 | 2 | 0.00319489 |
| 12.960000038146973 | 2 | 0.00319489 |
| 18.097999572753906 | 2 | 0.00319489 |

### `ph_qual`

| value | count | fraction |
| --- | --- | --- |
| AA | 572 | 0.913738 |
| <NA> | 54 | 0.086262 |

## Coverage Summary

- Seasonal bins (W1): `17`
- Seasonal bins (W2): `17`
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

- Known blazar match: `YES` via `source_id` token `IIZW 1`
- Known CLAGN match: `NO`
- Gaia metrics: not provided / no match

## Final Label

**Known blazar**

- Rank 72 with score=0.4866
- Sampling: n_points=202.0, baseline_years=14.5087282769, band_coverage=W1,W2
- QA rejection fraction=0.0% (main causes summarized in card QA section)
- Systematics verdict=PASS: QA-clean points and seasonal coverage support the observed variability signal.
- Known blazar catalog match=YES
- Dataset metadata class=blazar

## Reproducibility

- `run_timestamp_utc`: `2026-03-02T06:47:01.885248+00:00`
- `results_file`: `data/real_clagn/benchmark_scores_v2.csv`
- `wise_file`: `data/wise/IIZw 1.csv`
- `score_threshold`: `0.0`
- `t_recall`: `0.3493918746`
