#!/usr/bin/env python3
"""
Merge expansion CSVs with an existing benchmark master to produce benchmark_master_v2.csv.

Loads:
  - Existing master (preserves split assignments verbatim)
  - CLAGN expansion CSV (from expand_clagn_catalog.py)
  - SN expansion CSV   (from expand_sn_catalog.py)
  - Blazar expansion CSV (from expand_blazar_catalog.py)
  - Normal AGN expansion CSV (from sample_sdss_control.py)

Deduplicates new sources against existing (coordinate radius), assigns dev/test
splits via StratifiedShuffleSplit, conforms to the 19-column master schema,
and writes the merged master v2 CSV.

Usage:
    python scripts/merge_benchmark.py \\
        --existing data/real_clagn/benchmark_master.csv \\
        --clagn_new data/benchmark/known_clagn_expanded.csv \\
        --sn_new data/benchmark/sn_expanded.csv \\
        --blazars_new data/benchmark/blazars_verified.csv \\
        --agn_new data/benchmark/normal_agn_expanded.csv \\
        --output data/benchmark/benchmark_master_v2.csv \\
        --dedup_radius_arcsec 5.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rebuild_benchmark_splits import (
    _build_skycoord,
    _dedup_new_against_existing,
    _dedup_within,
    _conform_to_master_schema,
    _assign_new_splits,
    _MASTER_COLS,
    load_clagn_expansion,
)
from clagn.utils.benchmark_labels import ensure_class_ytrue
from clagn.utils.id_canonicalization import add_canonical_id


def _load_expansion_csv(path: Path | None, label: str) -> pd.DataFrame:
    """Load an expansion CSV, returning empty DataFrame if path is None or missing."""
    if path is None:
        print(f"  {label}: not provided — skipping")
        return pd.DataFrame()
    if not path.exists():
        print(f"  {label}: {path} not found — skipping")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"  {label}: {len(df)} rows loaded from {path}")
    return df


def _load_clagn_csv(path: Path | None) -> pd.DataFrame:
    """
    Load CLAGN expansion CSV. Supports both:
      - clagn_coverage_report.csv format (from expand_clagn_catalog.py, has 'coverage_pass' column)
      - generic CSV format (just needs source_id, ra, dec columns)
    """
    if path is None:
        print("  CLAGN: not provided — skipping")
        return pd.DataFrame()
    if not path.exists():
        print(f"  CLAGN: {path} not found — skipping")
        return pd.DataFrame()

    df = pd.read_csv(path)
    print(f"  CLAGN raw: {len(df)} rows loaded from {path}")

    # If it has the coverage_pass column it's the catalog report format
    if "coverage_pass" in df.columns:
        df = load_clagn_expansion(path)
        print(f"  CLAGN after coverage filter: {len(df)} rows")
    else:
        # Generic format — assume all rows are valid CLAGN
        df["class"] = "clagn"
        df["y_true"] = 1
        if "source_id" not in df.columns and "name" in df.columns:
            df = df.rename(columns={"name": "source_id"})

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--existing",
        required=True,
        help="Existing benchmark master CSV (data/real_clagn/benchmark_master.csv or data/benchmark/benchmark_master.csv)",
    )
    parser.add_argument(
        "--clagn_new",
        default=None,
        help="CLAGN expansion CSV (from expand_clagn_catalog.py)",
    )
    parser.add_argument(
        "--sn_new",
        default=None,
        help="SN expansion CSV (from expand_sn_catalog.py)",
    )
    parser.add_argument(
        "--blazars_new",
        default=None,
        help="Blazar expansion CSV (from expand_blazar_catalog.py)",
    )
    parser.add_argument(
        "--agn_new",
        default=None,
        help="Normal AGN expansion CSV (from sample_sdss_control.py)",
    )
    parser.add_argument(
        "--output",
        default="data/real_clagn/benchmark_master_v2.csv",
        help="Output path for merged master v2 CSV (default: data/real_clagn/benchmark_master_v2.csv)",
    )
    parser.add_argument(
        "--dedup_radius_arcsec",
        type=float,
        default=5.0,
        help="Coordinate dedup radius in arcsec (default: 5.0)",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.20,
        help="Fraction of new sources assigned to test split (default: 0.20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split assignment (default: 42)",
    )
    args = parser.parse_args()

    existing_path = Path(args.existing)
    output_path = Path(args.output)
    radius = args.dedup_radius_arcsec

    # --- Load existing master ---
    if not existing_path.exists():
        print(f"ERROR: Existing master not found: {existing_path}", file=sys.stderr)
        sys.exit(1)

    existing = pd.read_csv(existing_path)
    existing = ensure_class_ytrue(existing)
    if "canonical_id" not in existing.columns:
        existing = add_canonical_id(existing)
    print(f"Existing benchmark: {len(existing)} sources")
    print(existing["class"].value_counts().to_string())

    existing_sc = _build_skycoord(existing)

    # --- Load expansion CSVs ---
    print("\nLoading expansion CSVs...")
    new_clagn = _load_clagn_csv(Path(args.clagn_new) if args.clagn_new else None)
    new_sn = _load_expansion_csv(Path(args.sn_new) if args.sn_new else None, "SN")
    new_blazar = _load_expansion_csv(Path(args.blazars_new) if args.blazars_new else None, "Blazar")
    new_agn = _load_expansion_csv(Path(args.agn_new) if args.agn_new else None, "Normal AGN")

    # --- Ensure class / y_true on all parts ---
    class_defaults = {
        "clagn": ("clagn", 1),
        "sn": ("sn", 0),
        "blazar": ("blazar", 0),
        "normal_agn": ("normal_agn", 0),
    }
    parts_raw = [
        ("clagn", new_clagn),
        ("sn", new_sn),
        ("blazar", new_blazar),
        ("normal_agn", new_agn),
    ]
    all_new_parts: list[pd.DataFrame] = []
    for label, df_part in parts_raw:
        if df_part is None or df_part.empty:
            continue
        df_part = df_part.copy()
        cls, ytrue = class_defaults[label]
        if "class" not in df_part.columns:
            df_part["class"] = cls
        if "y_true" not in df_part.columns:
            df_part["y_true"] = ytrue
        all_new_parts.append(df_part)

    if not all_new_parts:
        print("\nWARNING: No expansion CSVs provided. Output will equal existing master.")
        master_v2 = existing[_MASTER_COLS].copy()
    else:
        new_all = pd.concat(all_new_parts, ignore_index=True)
        print(f"\nTotal new before dedup: {len(new_all)}")

        # Dedup against existing
        new_all = _dedup_new_against_existing(new_all, existing_sc, radius)
        print(f"After dedup vs existing: {len(new_all)}")

        # Assign dev/test splits
        new_all = _assign_new_splits(new_all, test_frac=args.test_frac, seed=args.seed)

        # Conform to master schema
        new_all = _conform_to_master_schema(new_all)
        if "canonical_id" not in new_all.columns or new_all["canonical_id"].isna().all():
            new_all = add_canonical_id(new_all)

        # Assemble v2
        existing_conformant = _conform_to_master_schema(existing)
        master_v2 = pd.concat(
            [existing_conformant[_MASTER_COLS], new_all[_MASTER_COLS]],
            ignore_index=True,
        )

    # Dedup within v2 on coordinates
    master_v2 = _dedup_within(master_v2, radius)

    # Ensure canonical_id
    if "canonical_id" not in master_v2.columns:
        master_v2 = add_canonical_id(master_v2)

    # Fix NaN canonical_ids
    mask_nan = master_v2["canonical_id"].isna() | (master_v2["canonical_id"].astype(str) == "nan")
    if mask_nan.any():
        master_v2.loc[mask_nan, "canonical_id"] = (
            master_v2.loc[mask_nan, "source_id"]
            .map(lambda x: str(x).strip() if x is not None else "")
        )

    # Remove duplicate canonical_ids
    dupes = master_v2["canonical_id"].duplicated()
    if dupes.any():
        print(f"  WARNING: {dupes.sum()} duplicate canonical_ids; keeping first occurrence")
        master_v2 = master_v2[~dupes].reset_index(drop=True)

    # --- Write output ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    master_v2.to_csv(output_path, index=False, float_format="%.10f")
    print(f"\nWrote {output_path} ({len(master_v2)} sources)")

    # Write dev/test splits alongside master
    out_dir = output_path.parent
    dev_v2 = master_v2[master_v2["split"].astype(str).str.lower() == "dev"].reset_index(drop=True)
    test_v2 = master_v2[master_v2["split"].astype(str).str.lower() == "test"].reset_index(drop=True)
    stem = output_path.stem  # e.g. "benchmark_master_v2"

    dev_path = out_dir / stem.replace("master", "dev") + ".csv" if "master" in stem else out_dir / f"{stem}_dev.csv"
    test_path = out_dir / stem.replace("master", "test") + ".csv" if "master" in stem else out_dir / f"{stem}_test.csv"

    # Simpler approach
    base = output_path.stem  # benchmark_master_v2
    dev_path = out_dir / (base.replace("master", "dev") + ".csv")
    test_path = out_dir / (base.replace("master", "test") + ".csv")

    dev_v2.to_csv(dev_path, index=False, float_format="%.10f")
    test_v2.to_csv(test_path, index=False, float_format="%.10f")
    print(f"Wrote {dev_path} ({len(dev_v2)} sources)")
    print(f"Wrote {test_path} ({len(test_v2)} sources)")

    # Write split hash JSON
    master_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    split_hash = {
        "version": "v2",
        "seed": args.seed,
        "test_frac": args.test_frac,
        "dedup_radius_arcsec": radius,
        "n_total": int(len(master_v2)),
        "n_dev": int(len(dev_v2)),
        "n_test": int(len(test_v2)),
        "class_counts": master_v2["class"].value_counts().to_dict(),
        "dev_class_counts": dev_v2["class"].value_counts().to_dict(),
        "test_class_counts": test_v2["class"].value_counts().to_dict(),
        "master_sha256": master_hash,
    }
    hash_path = out_dir / "SPLIT_HASH_v2.json"
    hash_path.write_text(json.dumps(split_hash, indent=2), encoding="utf-8")
    print(f"Wrote {hash_path}")

    # --- Summary ---
    print("\n=== Class x Split counts ===")
    ct = master_v2.groupby(["class", "split"]).size().unstack(fill_value=0)
    print(ct.to_string())
    print("\nCLAGN per split:")
    for sp in ["dev", "test"]:
        n_c = int(
            ((master_v2["class"].str.lower() == "clagn") &
             (master_v2["split"].str.lower() == sp)).sum()
        )
        print(f"  {sp}: {n_c} CLAGN  (each = {100/max(n_c, 1):.1f}% recall swing)")


if __name__ == "__main__":
    main()
