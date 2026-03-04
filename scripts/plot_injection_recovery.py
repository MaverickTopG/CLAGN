"""
Plot injection-recovery results from saved grid results.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clagn.utils.validation_metadata import build_metadata, write_metadata_sidecar


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', default='results/injection_recovery/grid_results.csv')
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--shape', default='step')
    parser.add_argument('--benchmark_dir', default='data/benchmark')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"Missing grid results: {input_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    fig_dir = output_dir / 'figures'
    metadata_dir = output_dir / 'metadata'
    fig_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if df.empty:
        print("Grid results empty")
        sys.exit(1)

    shape_df = df[df['shape'] == args.shape]
    if shape_df.empty:
        shape_df = df

    meta = build_metadata(Path(args.benchmark_dir))

    if args.dry_run:
        print("Dry run: plot skipped")
        return

    try:
        import matplotlib.pyplot as plt

        pivot = shape_df.pivot_table(index='timescale_yr', columns='amplitude', values='completeness')
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(pivot.values, origin='lower', aspect='auto')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.2f}" for v in pivot.index])
        ax.set_xlabel('Amplitude (mag)')
        ax.set_ylabel('Timescale (yr)')
        ax.set_title('Completeness Heatmap')
        fig.colorbar(im, ax=ax, label='Completeness')
        fig.tight_layout()
        heatmap_path = fig_dir / 'completeness_heatmap.png'
        fig.savefig(heatmap_path, dpi=200)
        plt.close(fig)
        write_metadata_sidecar(heatmap_path, metadata_dir, meta)
    except Exception:
        print("Failed to generate heatmap")
        sys.exit(1)

    print("Injection-recovery plot complete")


if __name__ == '__main__':
    main()
