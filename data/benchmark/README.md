# Benchmark Datasets (Validation Hardening)

This directory contains the **frozen benchmark** used for referee‑proof validation.

## Required Schema
The canonical schema is defined in:

```
data/benchmark/schema/benchmark_schema.json
```

Every row must contain:

- `source_id`
- `ra`, `dec`
- `label` (positive / contaminant type / control)
- `label_source` (citation)
- `label_method` (spectroscopic / catalog / etc.)
- `independent_of_pipeline` (must be `True`)
- `notes`

## How to Populate
You can provide sources in **either** format:

1. **Schema‑compliant CSVs** under:
   - `data/benchmark/sources/*.csv`

2. **Legacy benchmark files** (auto‑converted):
   - `data/benchmark/known_clagn.csv`
   - `data/benchmark/known_contaminants.csv`
   - `data/benchmark/normal_agn_control.csv`

## Build and Freeze

```
python scripts/build_benchmark_datasets.py \
  --benchmark_dir data/benchmark \
  --seed 42
```

Outputs:
- `benchmark_master.csv`
- `benchmark_train_dev.csv`
- `benchmark_test.csv`
- `TRAIN_DEV_TEST_SPLIT.csv`
- `SPLIT_HASH.txt`
- `BENCHMARK_FREEZE.json`

If counts or schema fail, the script exits non‑zero with instructions.

## Integrity Rules
- No label leakage (labels must be independent of pipeline scores/features).
- All entries must have citations (`label_source`).
- All entries must be `independent_of_pipeline=True`.
