#!/usr/bin/env python3
"""
Expand star contaminator catalog for benchmark construction.

Generates a CSV of variable star sources from Gaia DR3 varisum (I/358/varisum),
deduplicating against an optional existing benchmark master.

Usage:
    python scripts/expand_star_catalog.py \\
        --n_target 1200 \\
        --output data/expansion/star_expansion.csv \\
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
    generate_star_expansion,
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
        "--n_target",
        type=int,
        default=65,
        help="Target number of star sources (default: 65)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling (default: 42)",
    )
    parser.add_argument(
        "--output",
        default="data/expansion/star_expansion.csv",
        help="Output CSV path (default: data/expansion/star_expansion.csv)",
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
    existing_sc = _load_existing_sc(
        Path(args.benchmark_master) if args.benchmark_master else None
    )
    if existing_sc is not None:
        print(f"Deduplicating against {len(existing_sc)} existing benchmark sources")
    else:
        print("No existing benchmark master provided — skipping dedup")

    df = generate_star_expansion(
        output_path=output_path,
        existing_sc=existing_sc,
        target_n=args.n_target,
        seed=args.seed,
        dedup_radius=args.dedup_radius,
    )
    print(f"\nStar expansion complete: {len(df)} sources written to {output_path}")


if __name__ == "__main__":
    main()
