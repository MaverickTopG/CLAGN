"""
Score real benchmark sources using the pipeline and save benchmark_scores.csv.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clagn.pipeline import CLAGNPipeline
from clagn.utils.id_canonicalization import add_canonical_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/real_clagn')
    parser.add_argument('--results_dir', default='results_real')
    parser.add_argument('--max_rows', type=int, default=None)
    parser.add_argument('--proxy', action='store_true', help='Use deterministic proxy scoring (no network)')
    parser.add_argument('--disable_gaia', action='store_true', help='Skip Gaia queries (uses null Gaia features)')
    args = parser.parse_args()

    base = Path(args.data_dir)
    bench_path = base / 'benchmark_master.csv'
    if not bench_path.exists():
        print(f"Missing benchmark_master.csv: {bench_path}")
        sys.exit(1)

    df = pd.read_csv(bench_path)
    if 'canonical_id' not in df.columns:
        df = add_canonical_id(df)
    if args.max_rows:
        df = df.head(args.max_rows)

    if args.disable_gaia:
        os.environ['CLAGN_DISABLE_GAIA'] = '1'

    if args.proxy:
        # Deterministic, label-independent proxy score for infrastructure testing
        ra = df['ra'].astype(float).to_numpy()
        dec = df['dec'].astype(float).to_numpy()
        score = (np.sin(np.deg2rad(ra)) + np.cos(np.deg2rad(dec)) + 2.0) / 4.0
        score_df = pd.DataFrame({
            'source_id': df['source_id'].astype(str),
            'canonical_id': df['canonical_id'].astype(str),
            'score': score,
            'status': 'OK',
            'n_points': 0,
            'baseline_days': np.nan,
            'band_coverage': 'proxy',
        })
    else:
        pipeline = CLAGNPipeline(results_dir=args.results_dir)
        scored = pipeline.run(df)
        score_df = scored.rename(columns={'name': 'source_id', 'composite_score_v2': 'score'}).copy()
        score_df = add_canonical_id(score_df)
        for col, default in [
            ('status', 'OK'),
            ('n_points', 0),
            ('baseline_days', np.nan),
            ('band_coverage', 'none'),
        ]:
            if col not in score_df.columns:
                score_df[col] = default
        keep_cols = ['source_id', 'canonical_id', 'score', 'status', 'n_points', 'baseline_days', 'band_coverage']
        extra_cols = [c for c in ['baseline_years', 'delta_mag', 'delta_mag_method', 'rejection_reason'] if c in score_df.columns]
        score_df = score_df[keep_cols + extra_cols]
    out_path = base / 'benchmark_scores.csv'
    score_df.to_csv(out_path, index=False)

    print(f"Saved scores to {out_path}")


if __name__ == '__main__':
    main()
