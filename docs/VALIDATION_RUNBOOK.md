# Validation Runbook

## Quickstart (Offline)

```bash
python reproduce_all.py --data_dir tests/data --benchmark_dir tests/data/benchmark --output_dir results_test --offline
```

## Full Offline Validation

1. Build and freeze benchmark:
```bash
python scripts/build_benchmark_datasets.py --benchmark_dir data/benchmark --data_dir data
python scripts/check_label_leakage.py --benchmark data/benchmark/benchmark_master.csv
```

2. Evaluate benchmark:
```bash
python scripts/evaluate_benchmark.py --benchmark_dir data/benchmark --split dev --output_dir results/dev
python scripts/evaluate_benchmark.py --benchmark_dir data/benchmark --split test --output_dir results --threshold <from dev>
```

3. Injection–recovery:
```bash
python scripts/run_injection_recovery.py --data_dir data --benchmark_dir data/benchmark --output_dir results
python scripts/plot_injection_recovery.py --input_csv results/injection_recovery/grid_results.csv --output_dir results
```

4. Crossmatch coverage:
```bash
python scripts/report_crossmatch_coverage.py --data_dir data --output_dir results --offline
```

5. Claims lint:
```bash
python scripts/claims_lint.py --output_dir results --benchmark_dir data/benchmark
```

## Required Inputs (Offline)

- `data/benchmark/sources/*.csv` or legacy benchmark CSVs.
- Cached host light curves in `data/hosts/*.csv`.
- Catalog caches referenced by `data/catalogs/catalogs.json`.

## Outputs

- Metrics/figures under `results/` with metadata sidecars in `results/metadata/`.
- `results/RUN_MANIFEST.json` for end-to-end runs.
