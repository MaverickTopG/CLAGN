"""
Label leakage checks for benchmark datasets.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

FORBIDDEN_SUBSTRINGS = [
    'score', 'delta_mag', 'drw', 'changepoint', 'sf_', 'nonstation',
    'composite', 'w1_flux', 'w2_flux', 'variability', 'tau', 'sigma'
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', default='data/benchmark/benchmark_master.csv')
    args = parser.parse_args()

    path = Path(args.benchmark)
    if not path.exists():
        print(f"Missing benchmark file: {path}")
        sys.exit(1)

    df = pd.read_csv(path)

    # independent_of_pipeline must be True
    if 'independent_of_pipeline' not in df.columns:
        print("independent_of_pipeline column missing")
        sys.exit(1)

    if not df['independent_of_pipeline'].astype(bool).all():
        print("Label leakage: independent_of_pipeline is False for some rows")
        sys.exit(1)

    # Forbidden columns
    bad_cols = []
    for col in df.columns:
        for sub in FORBIDDEN_SUBSTRINGS:
            if sub.lower() in col.lower():
                bad_cols.append(col)
                break

    if bad_cols:
        print("Label leakage: forbidden columns present:")
        for c in sorted(set(bad_cols)):
            print(f"- {c}")
        sys.exit(1)

    print("Label leakage check PASS")


if __name__ == '__main__':
    main()
