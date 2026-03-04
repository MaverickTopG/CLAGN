# Candidate Card: IIIZw 2 (Rank 54)

## Basic Metadata

- `source_id`: `IIIZw 2`
- `canonical_id`: `IIIZw 2`
- `source_id_normalized`: `IIIZW 2`
- `canonical_id_normalized`: `IIIZW 2`
- `score`: `0.533221`
- `status`: `OK`
- `final_label`: `Known blazar`
- `stage_a_positive`: `True`
- `gate_blazar_veto`: `False`
- `gate_wise_qc_pass`: `True`
- `gate_data_sufficient`: `False`
- `decision_trace`: `stage_a_positive=pass;gate_blazar_veto=fail;gate_wise_qc_pass=pass;gate_data_sufficient=fail;gate_state_change_ratio_pass=pass;gate_transient_spike_pass=pass`

## Sampling / Variability Summary

- `n_points` (results table): `180`
- `baseline_days`: `5116.75`
- `baseline_years`: `14.009`
- `band_coverage`: `W1,W2`
- `delta_mag`: `-0.299`
- `delta_mag_method`: `seasonal_median_flux`

## Coordinates

- `ra`: `2.629200`
- `dec`: `10.974900`
- `coordinate_source`: `benchmark_master`

## WISE Light Curve

![WISE light curve](../figures/IIIZw_2_wise.png)

## WISE QA / Rejection Accounting

| Metric | Value |
| --- | --- |
| Raw points total | 642 |
| Raw points W1 | 321 |
| Raw points W2 | 321 |
| Kept points total | 641 |
| Kept points W1 | 320 |
| Kept points W2 | 321 |
| Fraction rejected | 0.00155763 |
| Reject: missing time | 0 |
| Reject: missing mag | 1 |
| Reject: invalid band | 0 |
| Reject: cc_flags | 0 |
| Reject: qual_frame | 0 |
| Reject: moon_lev | 0 |

### QA Reject Reason Counts

| reject_reason | count |
| --- | --- |
| reject_cc_flags | 0 |
| reject_invalid_band | 0 |
| reject_missing_mag | 1 |
| reject_missing_time | 0 |
| reject_moon_lev | 0 |
| reject_qual_frame | 0 |

## Flag Summary

### `cc_flags`

| value | count | fraction |
| --- | --- | --- |
| 0000 | 642 | 1 |

### `qual_frame`

| value | count | fraction |
| --- | --- | --- |
| <NA> | 642 | 1 |

### `moon_lev`

| value | count | fraction |
| --- | --- | --- |
| 00 | 524 | 0.816199 |
| 11 | 58 | 0.0903427 |
| 0000 | 48 | 0.0747664 |
| 01 | 12 | 0.0186916 |

### `qi_fact`

| value | count | fraction |
| --- | --- | --- |
| 1.0 | 436 | 0.679128 |
| 0.5 | 188 | 0.292835 |
| 0.0 | 18 | 0.0280374 |

### `saa_sep`

| value | count | fraction |
| --- | --- | --- |
| 18.0 | 4 | 0.00623053 |
| 36.0 | 4 | 0.00623053 |
| 79.0 | 2 | 0.00311526 |
| 50.512001037597656 | 2 | 0.00311526 |
| 35.98400115966797 | 2 | 0.00311526 |
| 24.347000122070312 | 2 | 0.00311526 |
| 20.488000869750977 | 2 | 0.00311526 |
| 38.24100112915039 | 2 | 0.00311526 |

### `ph_qual`

| value | count | fraction |
| --- | --- | --- |
| AA | 590 | 0.919003 |
| <NA> | 48 | 0.0747664 |
| XA | 2 | 0.00311526 |
| AB | 2 | 0.00311526 |

## Coverage Summary

- Seasonal bins (W1): `24`
- Seasonal bins (W2): `24`
- Pre-gap coverage: `False`
- Post-gap coverage: `True`
- Both-sides coverage: `False`

## Systematics Verdict

- Verdict: `CAUTION`
- Summary: Variability signal is present but data quality/coverage limitations weaken confidence.
- QA-dominated signal check: Not obviously dominated by flagged/low-quality points

Reasons:
- No clean coverage on both sides of WISE hibernation gap

## Catalog Checks

- Known blazar match: `YES` via `source_id` token `IIIZW 2`
- Known CLAGN match: `NO`
- Gaia metrics: not provided / no match

## Final Label

**Known blazar**

- Rank 54 with score=0.5332
- Sampling: n_points=180, baseline_years=14.008909744996584, band_coverage=W1,W2
- QA rejection fraction=0.2% (main causes summarized in card QA section)
- Systematics verdict=CAUTION: Variability signal is present but data quality/coverage limitations weaken confidence.
- Known blazar catalog match=YES
- Dataset metadata class=blazar

## Reproducibility

- `run_timestamp_utc`: `2026-02-28T17:09:43.673413+00:00`
- `results_file`: `data/real_clagn/benchmark_scores.csv`
- `wise_file`: `data/wise/IIIZw 2.csv`
- `score_threshold`: `0.0`
- `t_recall`: `0.3493918746`
