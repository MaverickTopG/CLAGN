# Candidate Card Benchmark Report

## Dataset Summary

- `rows_total`: `279`
- `rows_ok`: `208`
- `topn_requested`: `279`
- `topn_generated`: `7`
- `score_threshold`: `0.3493918746`
- `detections_above_threshold_ok`: `106`

## Status Counts

| status | count |
| --- | --- |
| NO_DATA | 2 |
| OK | 208 |
| TOO_FEW_POINTS | 69 |

## Class Counts

| class | count |
| --- | --- |
| blazar | 34 |
| clagn | 50 |
| normal_agn | 65 |
| sn | 65 |
| star | 65 |

## Selected Candidates

| rank | canonical_id | score | final_label |
| --- | --- | --- | --- |
| 1 | 4645750160709248 | 0.842179 | Rejected |
| 2 | 3C 279 | 0.825 | Known blazar |
| 3 | BL Lac | 0.795333 | Known blazar |
| 4 | SDSSJ141324.27+530527.0 | 0.761227 | Needs more data |
| 5 | SDSSJ082323.88+422048.2 | 0.757734 | Known AGN |
| 6 | TON 1542 | 0.754087 | Known blazar |
| 7 | SDSSJ153612.80+034245.7 | 0.731372 | Known AGN |

## Top CLAGN Candidates (Rule-Based Final Label)

(none in current top-N selection)

## Blazar Match Summary

- `n_selected_blazar_matches`: `3`
- `n_selected_non_blazar`: `4`

## QA / Systematics Summary (Selected)

- `n_selected`: `7`
- `n_systematics_pass`: `4`
- `n_systematics_caution`: `3`
- `n_systematics_fail`: `0`
- `mean_fraction_rejected_total`: `0.12703676349068896`

## Known CLAGN Recovery

- `n_known_clagn_total`: `60`
- `n_known_clagn_matched_to_results`: `13`
- `n_detected_above_threshold`: `12`
- `n_missed_below_threshold_or_non_ok`: `1`
- `completeness`: `0.9230769230769231`
- `contamination_proxy`: `0.8867924528301887`

| threshold | n_known_clagn_matched | n_detected_above_threshold | n_missed | completeness |
| --- | --- | --- | --- | --- |
| 0.349392 | 13 | 12 | 1 | 0.923077 |
| 0.7 | 13 | 1 | 12 | 0.0769231 |

## Threshold Metrics

| threshold | n_detected_ok | tp | fp | fn | completeness | contamination | contamination_note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.349392 | 106 | 41 | 65 | 9 | 0.82 | 0.613208 | exact benchmark truth |
| 0.7 | 13 | 7 | 6 | 43 | 0.14 | 0.461538 | exact benchmark truth |

## Inputs / Reproducibility

- `results`: `data/real_clagn/benchmark_scores.csv`
- `wise_dir`: `data/wise`
- `known_blazars`: `known_blazars.csv`
- `known_clagn`: `data/benchmark/known_clagn.csv`
- `gaia`: `not_provided`
- `benchmark_master`: `data/real_clagn/benchmark_master.csv`
- `coordinates`: `coordinates.csv`
- `seed`: `0`
- `policy_config`: `configs/pipeline_policy.yaml`
- `strict_mode`: `True`

See `reports/run_manifest.json` for exact configuration and file hashes.
