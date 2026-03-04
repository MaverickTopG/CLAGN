"""
Verify Mrk 1018 is rank 1 with negative delta_mag.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def main():
    # Prefer candidate_table.csv if present, else top_candidates.csv
    cand_path = Path('results/candidates/candidate_table.csv')
    if not cand_path.exists():
        cand_path = Path('results/top_candidates.csv')

    if not cand_path.exists():
        print("VERIFY: FAIL — no candidates table found")
        sys.exit(1)

    df = pd.read_csv(cand_path)
    if df.empty:
        print("VERIFY: FAIL — candidates table empty")
        sys.exit(1)

    # Identify Mrk 1018
    mask = df['source_id'].astype(str).str.replace(' ', '').str.contains('Mrk1018', case=False)
    if not mask.any():
        print("VERIFY: FAIL — Mrk 1018 not found in candidates")
        sys.exit(1)

    row = df[mask].iloc[0]
    rank = int(row.get('rank', row.get('Rank', 999)))

    # Prefer seasonal delta_mag column
    if 'delta_mag_w1_seasonal_pogson' in df.columns:
        dm = float(row.get('delta_mag_w1_seasonal_pogson'))
    elif 'delta_mag_w1' in df.columns:
        dm = float(row.get('delta_mag_w1'))
    elif 'w1_delta_mag' in df.columns:
        dm = float(row.get('w1_delta_mag'))
    else:
        dm = float(row.get('delta_mag', 0.0))

    if rank != 1:
        print(f"VERIFY: FAIL — Mrk 1018 rank={rank} (expected 1)")
        sys.exit(1)
    if dm >= 0:
        print(f"VERIFY: FAIL — Mrk 1018 delta_mag={dm:.3f} (expected negative)")
        sys.exit(1)

    print(f"VERIFY: PASS — Mrk 1018 rank=1, delta_mag={dm:.3f}")


if __name__ == '__main__':
    main()
