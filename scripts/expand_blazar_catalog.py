#!/usr/bin/env python3
"""
Expand blazar contaminator catalog for benchmark construction.

Generates a CSV of blazar sources from hardcoded HIGH_PRIORITY_BZB / HIGH_PRIORITY_BZQ
lists plus optional BZCAT5 VizieR fill. Stratified by subtype (BZB, BZQ, BZU).

Usage:
    python scripts/expand_blazar_catalog.py \\
        --bzcat_path data/raw/ROMABZCAT5.fits \\
        --n_target 115 \\
        --stratify BZB:50,BZQ:50,BZU:15

    # Minimal (uses VizieR for fill):
    python scripts/expand_blazar_catalog.py --n_target 115
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
    generate_blazar_expansion,
    HIGH_PRIORITY_BZB,
    HIGH_PRIORITY_BZQ,
    _build_skycoord,
)


def _parse_stratify(stratify_str: str) -> dict[str, int]:
    """
    Parse '--stratify BZB:50,BZQ:50,BZU:15' into {'BZB': 50, 'BZQ': 50, 'BZU': 15}.
    """
    result: dict[str, int] = {}
    for part in stratify_str.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid stratify token '{part}'; expected format TYPE:N")
        key, val = part.split(":", 1)
        result[key.strip().upper()] = int(val.strip())
    return result


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
        "--bzcat_path",
        default=None,
        help=(
            "Path to ROMABZCAT5.fits (optional). If provided and the file exists, "
            "it is noted but the script still uses VizieR for fill since astropy FITS "
            "parsing is handled internally. Pass the path for record-keeping."
        ),
    )
    parser.add_argument(
        "--n_target",
        type=int,
        default=115,
        help="Total target blazar count (default: 115; overridden by --stratify totals)",
    )
    parser.add_argument(
        "--stratify",
        default="BZB:50,BZQ:50,BZU:15",
        help=(
            "Per-subtype target counts, format 'BZB:N,BZQ:N,BZU:N' "
            "(default: BZB:50,BZQ:50,BZU:15)"
        ),
    )
    parser.add_argument(
        "--output",
        default="data/benchmark/blazars_verified.csv",
        help="Output CSV path (default: data/benchmark/blazars_verified.csv)",
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

    # Parse stratify
    try:
        strat = _parse_stratify(args.stratify)
    except ValueError as exc:
        parser.error(str(exc))
        return

    target_bzb = strat.get("BZB", 50)
    target_bzq = strat.get("BZQ", 50)
    target_bzu = strat.get("BZU", 15)

    total = target_bzb + target_bzq + target_bzu
    print(f"Blazar catalog expansion: BZB={target_bzb}, BZQ={target_bzq}, BZU={target_bzu} (total={total})")
    print(f"  HIGH_PRIORITY_BZB: {len(HIGH_PRIORITY_BZB)} hardcoded entries")
    print(f"  HIGH_PRIORITY_BZQ: {len(HIGH_PRIORITY_BZQ)} hardcoded entries")

    if args.bzcat_path:
        bzcat = Path(args.bzcat_path)
        if bzcat.exists():
            print(f"  BZCAT path provided: {bzcat} (exists — VizieR fill will also run if needed)")
        else:
            print(f"  BZCAT path provided: {bzcat} (NOT found — falling back to VizieR)")

    existing_sc = _load_existing_sc(
        Path(args.benchmark_master) if args.benchmark_master else None
    )
    if existing_sc is not None:
        print(f"  Deduplicating against {len(existing_sc)} existing benchmark sources")
    else:
        print("  No existing benchmark master provided — skipping dedup")

    df = generate_blazar_expansion(
        output_path=output_path,
        existing_sc=existing_sc,
        target_bzb=target_bzb,
        target_bzq=target_bzq,
        target_bzu=target_bzu,
        dedup_radius=args.dedup_radius,
    )

    print(f"\nBlazar expansion complete: {len(df)} sources written to {output_path}")
    if not df.empty and "bztype" in df.columns:
        print(f"  bztype distribution:\n{df['bztype'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
