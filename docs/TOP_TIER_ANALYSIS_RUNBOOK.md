# Top-Tier Analysis Runbook

## Purpose

This analysis layer audits an existing CLAGN candidate results table and produces reproducible quality reports, bias diagnostics, and case-mining outputs.

## Run

```bash
python -m analysis.run_all --input data/results.csv --outdir outputs/
```

## Key Outputs

- `outputs/cleaned_results.csv`: typed and enriched table with derived columns
- `outputs/quality_report.md`: status/coverage summary + correlations
- `outputs/bias_audit.md`: proxy rejection-bias analysis
- `outputs/top_candidates.csv`: case-mining tables for manual follow-up
- `outputs/*.png`: reproducible figures
- `outputs/run_manifest.json`: run metadata and deterministic settings

## Interpretation Notes

- `score` is a ranking statistic, not a calibrated probability.
- `strict_quality` is a conservative data-quality flag for follow-up prioritization.
- Bias audit here is proxy-only and does not establish causality.
