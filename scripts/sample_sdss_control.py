#!/usr/bin/env python3
"""
Sample SDSS DR16Q quasars as normal AGN control sources for benchmark expansion.

Fetches quasars via VizieR VII/289 (Lyke+2020), stratified by redshift bins:
  [0.0-0.3, 0.3-0.6, 0.6-1.0, 1.0-2.0, 2.0+]
  target_per_bin = [120, 140, 120, 100, 35]  (sum = 515)

Deduplicates against an optional existing benchmark master.

Usage:
    python scripts/sample_sdss_control.py --n_new 515 --seed 42

    # With custom output and dedup:
    python scripts/sample_sdss_control.py \\
        --n_new 515 \\
        --seed 42 \\
        --output data/benchmark/normal_agn_expanded.csv \\
        --benchmark-master data/real_clagn/benchmark_master.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rebuild_benchmark_splits import (
    generate_agn_expansion,
    _build_skycoord,
)


def _load_existing_sc(benchmark_master: Path | None):
    if benchmark_master is None or not benchmark_master.exists():
        return None
    try:
        df = pd.read_csv(benchmark_master)
        return _build_skycoord(df)
    except Exception as exc:
        print(f"WARNING: Could not load {benchmark_master}: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n_new",
        type=int,
        default=515,
        help="Target number of new normal AGN sources (default: 515)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified sampling (default: 42)",
    )
    parser.add_argument(
        "--output",
        default="data/benchmark/normal_agn_expanded.csv",
        help="Output CSV path (default: data/benchmark/normal_agn_expanded.csv)",
    )
    parser.add_argument(
        "--benchmark-master",
        default=None,
        help="Existing benchmark master CSV to dedup against (optional)",
    )
    parser.add_argument(
        "--dedup-radius",
        type=float,
        default=5.0,
        help="Coordinate dedup radius in arcsec (default: 5.0)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    print(f"SDSS control AGN sampling: n_new={args.n_new}, seed={args.seed}")
    print("  Redshift bins: [0.0-0.3, 0.3-0.6, 0.6-1.0, 1.0-2.0, 2.0+]")
    print("  Target per bin: [120, 140, 120, 100, 35]")

    existing_sc = _load_existing_sc(
        Path(args.benchmark_master) if args.benchmark_master else None
    )
    if existing_sc is not None:
        print(f"  Deduplicating against {len(existing_sc)} existing benchmark sources")
    else:
        print("  No existing benchmark master provided — skipping dedup")

    df = generate_agn_expansion(
        output_path=output_path,
        existing_sc=existing_sc,
        target_n=args.n_new,
        seed=args.seed,
        dedup_radius=args.dedup_radius,
    )

    print(f"\nNormal AGN sampling complete: {len(df)} sources written to {output_path}")
    if not df.empty and "z_bin" in df.columns:
        print(f"  Redshift bin distribution:\n{df['z_bin'].value_counts().sort_index().to_string()}")


if __name__ == "__main__":
    main()
