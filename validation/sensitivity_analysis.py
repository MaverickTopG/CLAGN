"""
Sensitivity analysis sweep over threshold parameters.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np

import clagn.config as cfg
from clagn.pipeline import CLAGNPipeline
from validation.precision_recall import run_full_benchmark

THRESHOLDS_TO_TEST = {
    'MIN_BASELINE_YEARS':       [5.0, 7.0, 10.0],
    'MIN_REDSHIFT':             [0.002, 0.005, 0.01],
    'SIGMA_CLIP_SIGMA':         [3.5, 4.0, 5.0],
    'MIN_DELTA_MAG':            [0.2, 0.3, 0.4, 0.5],
    'MIN_EPOCHS_PER_SEASON':    [2, 3, 5],
    'MIN_SEASONS_REQUIRED':     [3, 4, 5],
    'GAIA_SEARCH_RADIUS_ARCSEC':[1.0, 1.5, 2.0],
    'MAX_PARALLAX_SIG':         [2.0, 3.0, 5.0],
    'operating_threshold':      [0.3, 0.4, 0.5, 0.6],
}


def run_sensitivity_sweep(pipeline: CLAGNPipeline, benchmark_dir: str, param_name: str, values):
    results = []
    orig = getattr(cfg, param_name, None)

    for v in values:
        if param_name == 'operating_threshold':
            pipeline.operating_threshold = float(v)
        elif param_name == 'MIN_DELTA_MAG':
            setattr(cfg, 'MIN_DELTA_MAG', v)
            setattr(cfg, 'CLAGN_MIN_DELTA_MAG', v)
            setattr(cfg, 'MIN_MAG_CHANGE_W1', v)
        else:
            setattr(cfg, param_name, v)

        scores, pr = run_full_benchmark(pipeline, benchmark_dir)
        results.append({
            'param': param_name,
            'value': v,
            'n_candidates': scores.get('n_candidates', 0),
            'precision': pr['operating_point']['precision'],
            'recall': pr['operating_point']['recall'],
            'auc_pr': pr['auc_pr'],
        })

    # Restore
    if param_name == 'MIN_DELTA_MAG':
        if orig is not None:
            setattr(cfg, 'MIN_DELTA_MAG', orig)
            setattr(cfg, 'CLAGN_MIN_DELTA_MAG', orig)
            setattr(cfg, 'MIN_MAG_CHANGE_W1', orig)
    elif param_name != 'operating_threshold':
        if orig is not None:
            setattr(cfg, param_name, orig)
    return pd.DataFrame(results)


def main(results_dir: str = './results/', benchmark_dir: str = 'data/benchmark/'):
    pipeline = CLAGNPipeline(results_dir=results_dir)
    out_dir = Path(results_dir) / 'validation' / 'sensitivity'
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []

    for param, values in THRESHOLDS_TO_TEST.items():
        df = run_sensitivity_sweep(pipeline, benchmark_dir, param, values)
        df.to_csv(out_dir / f"sweep_{param}.csv", index=False)
        if len(df) > 0:
            auc_range = float(df['auc_pr'].max() - df['auc_pr'].min())
            summary.append({'param': param, 'auc_range': auc_range})

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out_dir / 'sensitivity_summary.csv', index=False)

    # Fail if any AUC range > 0.1
    fragile = summary_df[summary_df['auc_range'] > 0.1]
    report = {
        'pass': fragile.empty,
        'fragile_params': fragile.to_dict(orient='records'),
    }
    with open(out_dir / 'sensitivity_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    if not report['pass']:
        raise RuntimeError(f"Sensitivity analysis failed: {fragile}")

    return report


if __name__ == '__main__':
    main()
