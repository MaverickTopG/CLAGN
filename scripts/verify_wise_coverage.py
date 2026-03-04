#!/usr/bin/env python3
"""
Verify WISE NEOWISE coverage for sources in an arbitrary input CSV.

Reads a CSV with at minimum columns [source_id, ra, dec], queries IRSA NEOWISE
TAP for each source, and appends coverage columns:
  - n_clean_epochs
  - n_seasons_w1
  - n_seasons_w2
  - n_seasons_total
  - coverage_pass  (True if n_clean_epochs >= min_epochs AND n_seasons_total >= min_seasons)

Writes all rows to the output CSV with the coverage columns added.

Usage:
    python scripts/verify_wise_coverage.py \\
        --input data/blazar_candidates.csv \\
        --min_epochs 10 \\
        --min_seasons 3 \\
        --output data/benchmark/blazars_verified.csv

    # Resume from a partial run (skips rows that already have coverage_pass set):
    python scripts/verify_wise_coverage.py \\
        --input data/blazar_candidates.csv \\
        --output data/benchmark/blazars_verified.csv \\
        --resume
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.expand_clagn_catalog import _irsa_wise_coverage


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV with columns [source_id, ra, dec]",
    )
    parser.add_argument(
        "--min_epochs",
        type=int,
        default=10,
        help="Minimum number of clean NEOWISE epochs to pass (default: 10)",
    )
    parser.add_argument(
        "--min_seasons",
        type=int,
        default=3,
        help="Minimum number of NEOWISE seasons (W1+W2 combined) to pass (default: 3)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output CSV path. Defaults to same as input with '_coverage' suffix. "
            "If --resume is set, this file is read first to skip already-processed rows."
        ),
    )
    parser.add_argument(
        "--wise-radius",
        type=float,
        default=3.0,
        help="IRSA WISE query radius in arcsec (default: 3.0)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.4,
        help="Seconds to sleep between IRSA API calls (default: 0.4)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows that already have a non-null coverage_pass value in the output file",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(input_path)
    required_cols = {"source_id", "ra", "dec"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"ERROR: Input CSV missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else (
        input_path.parent / (input_path.stem + "_coverage.csv")
    )

    # Coverage columns to add
    cov_cols = ["n_clean_epochs", "n_seasons_w1", "n_seasons_w2", "n_seasons_total", "coverage_pass"]
    for col in cov_cols:
        if col not in df.columns:
            df[col] = None

    # Resume: load previously processed rows
    already_done: set[str] = set()
    if args.resume and output_path.exists():
        try:
            prev = pd.read_csv(output_path)
            if "source_id" in prev.columns and "coverage_pass" in prev.columns:
                done_mask = prev["coverage_pass"].notna()
                already_done = set(prev.loc[done_mask, "source_id"].astype(str).tolist())
                # Merge previously computed coverage into df
                prev_done = prev[done_mask].set_index("source_id")[cov_cols]
                for col in cov_cols:
                    df_idx = df.set_index("source_id")
                    overlap = df_idx.index.intersection(prev_done.index)
                    if len(overlap) > 0:
                        df.loc[df["source_id"].astype(str).isin(overlap), col] = (
                            df["source_id"].astype(str).map(
                                lambda sid: prev_done.loc[sid, col] if sid in prev_done.index else None
                            )
                        )
                print(f"Resume: {len(already_done)} rows already processed from {output_path}")
        except Exception as exc:
            print(f"WARNING: Could not load resume file {output_path}: {exc}")

    total = len(df)
    n_pass = 0
    n_fail = 0
    n_skip = 0

    print(f"\nVerifying WISE coverage for {total} sources...")
    print(f"  min_epochs={args.min_epochs}, min_seasons={args.min_seasons}, "
          f"wise_radius={args.wise_radius}\"")

    for i, (idx, row) in enumerate(df.iterrows(), 1):
        sid = str(row["source_id"])

        if sid in already_done:
            n_skip += 1
            if i % 50 == 0:
                print(f"[{i}/{total}] {sid}: SKIPPED (resume)")
            continue

        try:
            ra = float(row["ra"])
            dec = float(row["dec"])
        except (ValueError, TypeError):
            print(f"[{i}/{total}] {sid}: SKIP — invalid ra/dec")
            df.at[idx, "coverage_pass"] = False
            df.at[idx, "n_clean_epochs"] = 0
            df.at[idx, "n_seasons_w1"] = 0
            df.at[idx, "n_seasons_w2"] = 0
            df.at[idx, "n_seasons_total"] = 0
            n_fail += 1
            continue

        cov = _irsa_wise_coverage(ra, dec, radius_arcsec=args.wise_radius)
        time.sleep(args.rate_limit)

        # Override coverage_pass with custom thresholds
        n_clean = int(cov.get("n_clean", 0))
        n_total = int(cov.get("n_seasons_total", 0))
        passes = (n_clean >= args.min_epochs) and (n_total >= args.min_seasons)

        df.at[idx, "n_clean_epochs"] = n_clean
        df.at[idx, "n_seasons_w1"] = int(cov.get("n_seasons_w1", 0))
        df.at[idx, "n_seasons_w2"] = int(cov.get("n_seasons_w2", 0))
        df.at[idx, "n_seasons_total"] = n_total
        df.at[idx, "coverage_pass"] = passes

        if passes:
            n_pass += 1
        else:
            n_fail += 1

        status = "PASS" if passes else f"FAIL(n={n_clean},s={n_total})"
        if i % 10 == 0 or i <= 5:
            print(f"[{i}/{total}] {sid}: {status}")

        # Checkpoint every 25 rows
        if i % 25 == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False, float_format="%.6f")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format="%.6f")

    print(f"\n=== WISE Coverage Summary ===")
    print(f"  Total sources:    {total}")
    print(f"  Skipped (resume): {n_skip}")
    processed = total - n_skip
    print(f"  Newly processed:  {processed}")
    print(f"  PASS:             {n_pass} / {processed} ({100 * n_pass / max(processed, 1):.1f}%)")
    print(f"  FAIL:             {n_fail} / {processed}")
    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()
