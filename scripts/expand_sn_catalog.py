#!/usr/bin/env python3
"""
Expand SN contaminator catalog for benchmark construction.

Generates a CSV of SN (supernova) sources from hardcoded TYPE_IIn and ASASSN
lists, deduplicating against an optional existing benchmark master.

Usage:
    python scripts/expand_sn_catalog.py \\
        --sources asassn,tns,asiago \\
        --n_target 138 \\
        --output data/benchmark/sn_expanded.csv

    # With dedup against existing benchmark:
    python scripts/expand_sn_catalog.py \\
        --output data/benchmark/sn_expanded.csv \\
        --benchmark-master data/real_clagn/benchmark_master.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rebuild_benchmark_splits import (
    generate_sn_expansion,
    TYPE_IIn_SN,
    ASASSN_SN_IN_AGN,
    _build_skycoord,
)


def _load_existing_sc(benchmark_master: Path | None) -> "SkyCoord | None":
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
        "--sources",
        default="asassn,tns,asiago",
        help=(
            "Comma-separated source tags to include (asassn, tns, asiago). "
            "Currently all map to the hardcoded TYPE_IIn_SN + ASASSN_SN_IN_AGN lists."
        ),
    )
    parser.add_argument(
        "--n_target",
        type=int,
        default=138,
        help="Target number of SN sources (default: 138)",
    )
    parser.add_argument(
        "--output",
        default="data/benchmark/sn_expanded.csv",
        help="Output CSV path (default: data/benchmark/sn_expanded.csv)",
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
    benchmark_master = Path(args.benchmark_master) if args.benchmark_master else None

    sources = [s.strip().lower() for s in args.sources.split(",") if s.strip()]
    print(f"SN catalog expansion: sources={sources}, n_target={args.n_target}")
    print(f"  TYPE_IIn_SN list: {len(TYPE_IIn_SN)} entries")
    print(f"  ASASSN_SN_IN_AGN list: {len(ASASSN_SN_IN_AGN)} entries")

    existing_sc = _load_existing_sc(benchmark_master)
    if existing_sc is not None:
        print(f"  Deduplicating against {len(existing_sc)} existing benchmark sources")
    else:
        print("  No existing benchmark master provided — skipping dedup")

    df = generate_sn_expansion(
        output_path=output_path,
        existing_sc=existing_sc,
        target_n=args.n_target,
        dedup_radius=args.dedup_radius,
    )

    print(f"\nSN expansion complete: {len(df)} sources written to {output_path}")
    if not df.empty:
        print(f"  class distribution:\n{df['class'].value_counts().to_string()}")
        if "sn_type" in df.columns:
            print(f"  sn_type distribution:\n{df['sn_type'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
