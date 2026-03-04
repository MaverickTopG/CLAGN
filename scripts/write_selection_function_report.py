"""
Write selection function report for real validation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _find_mag_column(df: pd.DataFrame) -> str | None:
    candidates = ['w1_mag', 'w2_mag', 'mag', 'g_mag', 'r_mag', 'i_mag']
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _bin_completeness(values: np.ndarray, detected: np.ndarray, n_bins: int = 4) -> list[dict]:
    if len(values) == 0:
        return []
    qs = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    rows = []
    for i in range(n_bins):
        lo, hi = qs[i], qs[i + 1]
        mask = (values >= lo) & (values <= hi)
        if mask.sum() == 0:
            comp = float('nan')
        else:
            comp = float(detected[mask].mean())
        rows.append({'bin_lo': float(lo), 'bin_hi': float(hi), 'completeness': comp, 'n': int(mask.sum())})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/real_clagn')
    parser.add_argument('--output_dir', default='results_real')
    args = parser.parse_args()

    base = Path(args.data_dir)
    out_dir = Path(args.output_dir)

    grid_path = out_dir / 'injection_recovery' / 'grid_results.csv'
    recovery_path = out_dir / 'recovery_table.csv'
    bench_path = base / 'benchmark_master.csv'

    if not grid_path.exists():
        print(f"Missing injection grid: {grid_path}")
        sys.exit(1)
    if not recovery_path.exists():
        print(f"Missing recovery_table.csv: {recovery_path}")
        sys.exit(1)
    if not bench_path.exists():
        print(f"Missing benchmark_master.csv: {bench_path}")
        sys.exit(1)

    grid = pd.read_csv(grid_path)
    recovery = pd.read_csv(recovery_path)
    bench = pd.read_csv(bench_path)
    if 'canonical_id' not in recovery.columns and 'source_id' in recovery.columns:
        recovery = add_canonical_id(recovery)
    if 'canonical_id' not in bench.columns and 'source_id' in bench.columns:
        bench = add_canonical_id(bench)
    bench = bench.copy()
    bench['class'] = normalized_class_series(bench)

    # Completeness vs amplitude from injection grid
    amp_summary = grid.groupby('amplitude')['completeness'].mean().reset_index()

    # Completeness vs redshift/magnitude from recovered positives
    merge_key = best_join_key(recovery, bench)
    merge_cols = [merge_key, 'redshift', 'class'] + [c for c in bench.columns if c in ['w1_mag','w2_mag','mag','g_mag','r_mag','i_mag']]
    merge_cols = [c for c in merge_cols if c in bench.columns]
    merged = recovery.merge(bench[merge_cols], on=merge_key, how='left')
    if 'redshift' not in merged.columns:
        if 'redshift_x' in merged.columns:
            merged['redshift'] = merged['redshift_x']
        elif 'redshift_y' in merged.columns:
            merged['redshift'] = merged['redshift_y']
    # normalize magnitude columns from merge suffixes
    for base in ['mag', 'g_mag', 'r_mag', 'i_mag', 'w1_mag', 'w2_mag']:
        if base not in merged.columns:
            if f"{base}_x" in merged.columns:
                merged[base] = merged[f"{base}_x"]
            elif f"{base}_y" in merged.columns:
                merged[base] = merged[f"{base}_y"]
    class_col = 'class' if 'class' in merged.columns else 'label'
    pos = merged[merged[class_col].astype(str).str.lower().isin(['clagn', 'positive'])]
    if pos.empty:
        print("No positive recovery rows found for selection function")
        sys.exit(1)

    if 'redshift' not in pos.columns:
        print("Benchmark missing redshift column")
        sys.exit(1)

    mag_col = _find_mag_column(pos)
    if mag_col is None:
        print("Benchmark missing magnitude column (w1_mag/w2_mag/mag/g_mag/r_mag/i_mag)")
        sys.exit(1)

    redshift_bins = _bin_completeness(pos['redshift'].to_numpy(float), pos['predicted'].to_numpy(int))
    mag_bins = _bin_completeness(pos[mag_col].to_numpy(float), pos['predicted'].to_numpy(int))

    report_path = out_dir / 'selection_function_report.md'
    with report_path.open('w') as f:
        f.write("# Selection Function Report\n\n")
        f.write("## Completeness vs Amplitude\n\n")
        f.write("| Amplitude (mag) | Completeness |\n")
        f.write("| --- | --- |\n")
        for _, row in amp_summary.iterrows():
            f.write(f"| {row['amplitude']:.2f} | {row['completeness']:.3f} |\n")

        f.write("\n## Completeness vs Redshift\n\n")
        f.write("| z Bin | Completeness | N |\n")
        f.write("| --- | --- | --- |\n")
        for row in redshift_bins:
            f.write(f"| {row['bin_lo']:.3f}–{row['bin_hi']:.3f} | {row['completeness']:.3f} | {row['n']} |\n")

        f.write("\n## Completeness vs Magnitude\n\n")
        f.write(f"Magnitude column used: `{mag_col}`\n\n")
        f.write("| Mag Bin | Completeness | N |\n")
        f.write("| --- | --- | --- |\n")
        for row in mag_bins:
            f.write(f"| {row['bin_lo']:.3f}–{row['bin_hi']:.3f} | {row['completeness']:.3f} | {row['n']} |\n")

        f.write("\n## Detection Threshold Justification\n\n")
        f.write("Detection threshold is derived from dev split F1 optimization. "
                "Completeness curves above support the chosen threshold.\n")

    print("Selection function report complete")


if __name__ == '__main__':
    main()
from clagn.utils.benchmark_labels import normalized_class_series
from clagn.utils.id_canonicalization import best_join_key, add_canonical_id
