# Candidate Card Benchmark Report

## Dataset Summary

- `rows_total`: `279`
- `rows_ok`: `208`
- `topn_requested`: `1`
- `topn_generated`: `1`
- `score_threshold`: `0.7`
- `detections_above_threshold_ok`: `13`

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

## Top CLAGN Candidates (Rule-Based Final Label)

(none in current top-N selection)

## Blazar Match Summary

- `n_selected_blazar_matches`: `0`
- `n_selected_non_blazar`: `1`

## QA / Systematics Summary (Selected)

- `n_selected`: `1`
- `n_systematics_pass`: `1`
- `n_systematics_caution`: `0`
- `n_systematics_fail`: `0`
- `mean_fraction_rejected_total`: `0.024834437086092714`

## Known CLAGN Recovery

- `n_known_clagn_total`: `60`
- `n_known_clagn_matched_to_results`: `13`
- `n_detected_above_threshold`: `1`
- `n_missed_below_threshold_or_non_ok`: `12`
- `completeness`: `0.07692307692307693`
- `contamination_proxy`: `0.9230769230769231`

| threshold | n_known_clagn_matched | n_detected_above_threshold | n_missed | completeness |
| --- | --- | --- | --- | --- |
| 0.7 | 13 | 1 | 12 | 0.0769231 |

## Threshold Metrics

| threshold | n_detected_ok | tp | fp | fn | completeness | contamination | contamination_note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.7 | 13 | 7 | 6 | 43 | 0.14 | 0.461538 | exact benchmark truth |

## Inputs / Reproducibility

- `results`: `data/results.csv`
- `wise_dir`: `data/wise`
- `known_blazars`: `known_blazars.csv`
- `known_clagn`: `data/benchmark/known_clagn.csv`
- `gaia`: `not_provided`
- `benchmark_master`: `data/real_clagn/benchmark_master.csv`
- `coordinates`: `coordinates.csv`
- `seed`: `0`

See `reports/run_manifest.json` for exact configuration and file hashes.
