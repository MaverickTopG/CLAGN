# Real CLAGN Validation Data

This folder holds **spectroscopically anchored** real-data validation inputs.

## Required Inputs

### Spectroscopic CLAGN sources
- `data/real_clagn/raw/literature_sources.json`
  - JSON list of objects, each with:
    - `source_id`, `ra`, `dec`, `redshift`
    - `spectral_state_1`, `spectral_state_2`
    - `epoch_1`, `epoch_2`
    - `spectroscopic_reference` (DOI or ADS bibcode)
    - `label_method` = `spectroscopic`
    - `independent_of_pipeline` = `true`

### Contaminants
- `data/real_clagn/raw/contaminants_sources.json`
  - JSON list of contaminants with:
    - `source_id`, `ra`, `dec`
    - `contaminant_type` (e.g., SN, blazar, star, normal_agn)
    - `citation`
    - `classification_method`
    - `independent_of_pipeline` = `true`

### Host light curves for injection–recovery
- `data/real_clagn/hosts/*.csv`
  - Must contain `mjd` and either `w1_flux_mjy` + `w1_flux_err_mjy`
    or `mag` + `mag_err`.

### Catalog caches for coverage accounting
- `data/real_clagn/catalogs/catalogs.json`
  - A list of catalog entries with `name`, `cache_path`, and optional `queried_count`.

## Generated Outputs
- `data/real_clagn/processed/clagn_spectroscopic.csv`
- `data/real_clagn/processed/contaminants_master.csv`
- `data/real_clagn/benchmark_master.csv`
- `data/real_clagn/BENCHMARK_FREEZE.json`
- `data/real_clagn/benchmark_dev.csv`
- `data/real_clagn/benchmark_test.csv`
- `data/real_clagn/SPLIT_HASH.json`
