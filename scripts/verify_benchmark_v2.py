#!/usr/bin/env python3
"""
Verify integrity of benchmark_master_v2.csv after expansion.

Runs 6 assertion checks from the CLAGN_BENCHMARK_EXPANSION_1000.md Part 7:

  1. Minimum CLAGN count per split: n_clagn_dev >= 15, n_clagn_test >= 15
  2. Data leakage check: warn if any test source appears in CLAGN literature list
  3. WISE coverage rate: print summary if augmented CSV is available
  4. Transition type balance: turn_on >= 30, turn_off >= 30 (if transition_type present)
  5. Redshift range: min(z) < 0.05, max(z) > 0.30, >=20 sources at intermediate z
  6. Policy threshold unchanged: t_recall == 0.349392 (from pipeline_policy.yaml)

Exits with code 1 if any hard assertion fails. Warnings do not cause failure.

Usage:
    python scripts/verify_benchmark_v2.py \\
        --master data/real_clagn/benchmark_master_v2.csv \\
        --policy configs/pipeline_policy.yaml

    # Also check augmented CSV for WISE coverage:
    python scripts/verify_benchmark_v2.py \\
        --master data/real_clagn/benchmark_master_v2.csv \\
        --policy configs/pipeline_policy.yaml \\
        --augmented outputs/full_rows/benchmark_scores_augmented.csv
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

# Known CLAGN literature source IDs (from expand_clagn_catalog.py)
# Used for leakage check against test split
try:
    from scripts.expand_clagn_catalog import LITERATURE_SOURCES
    _LITERATURE_NAMES: set[str] = {src[0] for src in LITERATURE_SOURCES}
except ImportError:
    _LITERATURE_NAMES = set()

# Expected t_recall (hard constraint — must never change)
_EXPECTED_T_RECALL = 0.3493918746
_T_RECALL_TOL = 1e-9

# Redshift column candidates
_Z_COLS = ["redshift", "z", "zsp", "z_spec"]


def _find_z_col(df: pd.DataFrame) -> str | None:
    for col in _Z_COLS:
        if col in df.columns:
            return col
    return None


def _sep(label: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  CHECK: {label}")
    print("─" * 60)


PASS = "  [PASS]"
FAIL = "  [FAIL]"
WARN = "  [WARN]"
INFO = "  [INFO]"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--master",
        required=True,
        help="Path to benchmark_master_v2.csv",
    )
    parser.add_argument(
        "--policy",
        default="configs/pipeline_policy.yaml",
        help="Path to pipeline_policy.yaml (default: configs/pipeline_policy.yaml)",
    )
    parser.add_argument(
        "--augmented",
        default=None,
        help="(Optional) Path to benchmark_scores_augmented.csv for WISE coverage check",
    )
    parser.add_argument(
        "--min-clagn-per-split",
        type=int,
        default=15,
        help="Minimum CLAGN sources required in each split (default: 15)",
    )
    parser.add_argument(
        "--min-turn-on",
        type=int,
        default=30,
        help="Minimum turn_on CLAGN required (default: 30)",
    )
    parser.add_argument(
        "--min-turn-off",
        type=int,
        default=30,
        help="Minimum turn_off CLAGN required (default: 30)",
    )
    args = parser.parse_args()

    master_path = Path(args.master)
    policy_path = Path(args.policy)

    failures: list[str] = []
    warnings: list[str] = []

    # Load master
    if not master_path.exists():
        print(f"ERROR: Master file not found: {master_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(master_path)
    print(f"Loaded {len(df)} sources from {master_path}")

    if "class" in df.columns:
        print(f"Class distribution:\n{df['class'].value_counts().to_string()}")
    if "split" in df.columns:
        print(f"Split distribution:\n{df['split'].value_counts().to_string()}")

    # ─────────────────────────────────────────────────────────────────────
    # CHECK 1: Minimum CLAGN count per split
    # ─────────────────────────────────────────────────────────────────────
    _sep("1. Minimum CLAGN count per split")
    min_req = args.min_clagn_per_split

    if "class" not in df.columns or "split" not in df.columns:
        msg = "Missing 'class' or 'split' column — cannot check CLAGN counts"
        print(f"{FAIL} {msg}")
        failures.append(msg)
    else:
        is_clagn = df["class"].str.lower() == "clagn"
        for sp in ["dev", "test"]:
            n_c = int((is_clagn & (df["split"].str.lower() == sp)).sum())
            if n_c >= min_req:
                print(f"{PASS} {sp}: {n_c} CLAGN >= {min_req}")
            else:
                msg = f"{sp}: only {n_c} CLAGN < {min_req} required"
                print(f"{FAIL} {msg}")
                failures.append(msg)

    # ─────────────────────────────────────────────────────────────────────
    # CHECK 2: Data leakage — test sources in CLAGN literature list
    # ─────────────────────────────────────────────────────────────────────
    _sep("2. Data leakage: test sources in CLAGN literature list")

    if not _LITERATURE_NAMES:
        print(f"{WARN} Could not import LITERATURE_SOURCES — leakage check skipped")
        warnings.append("Leakage check skipped (import failed)")
    elif "split" not in df.columns or "source_id" not in df.columns:
        print(f"{WARN} Missing 'split' or 'source_id' column — leakage check skipped")
        warnings.append("Leakage check skipped (missing columns)")
    else:
        test_ids = set(df.loc[df["split"].str.lower() == "test", "source_id"].astype(str))
        leaked = test_ids & _LITERATURE_NAMES
        if leaked:
            print(f"{WARN} {len(leaked)} test source(s) appear in CLAGN literature list:")
            for name in sorted(leaked):
                print(f"       - {name}")
            warnings.append(
                f"Leakage: {len(leaked)} test source(s) in literature list: {sorted(leaked)}"
            )
        else:
            print(f"{PASS} No test sources found in CLAGN literature list")

    # ─────────────────────────────────────────────────────────────────────
    # CHECK 3: WISE coverage rate (requires augmented CSV)
    # ─────────────────────────────────────────────────────────────────────
    _sep("3. WISE coverage rate")

    augmented_path = Path(args.augmented) if args.augmented else None
    if augmented_path is None:
        print(f"{INFO} --augmented not provided; skipping WISE coverage check")
    elif not augmented_path.exists():
        print(f"{WARN} Augmented CSV not found: {augmented_path}")
        warnings.append(f"WISE coverage check skipped: {augmented_path} not found")
    else:
        aug = pd.read_csv(augmented_path)
        if "gate_wise_qc_pass" in aug.columns:
            n_total = len(aug)
            n_pass = int(aug["gate_wise_qc_pass"].astype(str).str.lower().isin({"true", "1", "yes"}).sum())
            rate = 100 * n_pass / max(n_total, 1)
            print(f"{INFO} WISE QC pass rate: {n_pass}/{n_total} ({rate:.1f}%)")
            if rate >= 50.0:
                print(f"{PASS} Coverage rate {rate:.1f}% >= 50% threshold")
            else:
                print(f"{WARN} Coverage rate {rate:.1f}% < 50% — check WISE data availability")
                warnings.append(f"WISE QC pass rate low: {rate:.1f}%")
        elif "n_clean_epochs" in aug.columns:
            n_total = len(aug)
            n_pass = int((pd.to_numeric(aug["n_clean_epochs"], errors="coerce") >= 10).sum())
            rate = 100 * n_pass / max(n_total, 1)
            print(f"{INFO} n_clean_epochs >= 10: {n_pass}/{n_total} ({rate:.1f}%)")
            if rate >= 50.0:
                print(f"{PASS} Coverage rate {rate:.1f}% >= 50%")
            else:
                print(f"{WARN} Coverage rate {rate:.1f}% < 50%")
                warnings.append(f"Low WISE epoch coverage: {rate:.1f}%")
        else:
            print(f"{WARN} No coverage columns found in augmented CSV (tried gate_wise_qc_pass, n_clean_epochs)")
            warnings.append("No WISE coverage columns in augmented CSV")

    # ─────────────────────────────────────────────────────────────────────
    # CHECK 4: Transition type balance (turn_on >= 30, turn_off >= 30)
    # ─────────────────────────────────────────────────────────────────────
    _sep("4. Transition type balance")

    if "transition_type" not in df.columns:
        print(f"{INFO} No 'transition_type' column found — check skipped")
    else:
        is_clagn = df["class"].str.lower() == "clagn" if "class" in df.columns else pd.Series(True, index=df.index)
        clagn_df = df[is_clagn]
        n_on = int((clagn_df["transition_type"].str.lower() == "turn_on").sum())
        n_off = int((clagn_df["transition_type"].str.lower() == "turn_off").sum())
        min_on = args.min_turn_on
        min_off = args.min_turn_off

        for count, label, req in [(n_on, "turn_on", min_on), (n_off, "turn_off", min_off)]:
            if count >= req:
                print(f"{PASS} {label}: {count} >= {req}")
            else:
                msg = f"{label}: only {count} < {req} required"
                print(f"{FAIL} {msg}")
                failures.append(msg)

    # ─────────────────────────────────────────────────────────────────────
    # CHECK 5: Redshift range
    # ─────────────────────────────────────────────────────────────────────
    _sep("5. Redshift range")

    z_col = _find_z_col(df)
    if z_col is None:
        print(f"{WARN} No redshift column found (tried: {_Z_COLS}) — check skipped")
        warnings.append("Redshift range check skipped: no redshift column")
    else:
        z = pd.to_numeric(df[z_col], errors="coerce").dropna()
        if z.empty:
            print(f"{WARN} All redshift values are NaN — check skipped")
            warnings.append("Redshift range check skipped: all NaN")
        else:
            z_min = float(z.min())
            z_max = float(z.max())
            n_intermediate = int(((z >= 0.05) & (z <= 0.30)).sum())

            if z_min < 0.05:
                print(f"{PASS} min(z) = {z_min:.4f} < 0.05")
            else:
                msg = f"min(z) = {z_min:.4f} >= 0.05 (expected nearby sources)"
                print(f"{FAIL} {msg}")
                failures.append(msg)

            if z_max > 0.30:
                print(f"{PASS} max(z) = {z_max:.4f} > 0.30")
            else:
                msg = f"max(z) = {z_max:.4f} <= 0.30 (expected high-z sources)"
                print(f"{FAIL} {msg}")
                failures.append(msg)

            if n_intermediate >= 20:
                print(f"{PASS} Intermediate-z (0.05-0.30): {n_intermediate} >= 20")
            else:
                msg = f"Intermediate-z (0.05-0.30): only {n_intermediate} < 20 required"
                print(f"{FAIL} {msg}")
                failures.append(msg)

    # ─────────────────────────────────────────────────────────────────────
    # CHECK 6: Policy threshold unchanged
    # ─────────────────────────────────────────────────────────────────────
    _sep("6. Policy threshold t_recall unchanged")

    if not policy_path.exists():
        msg = f"Policy file not found: {policy_path}"
        print(f"{FAIL} {msg}")
        failures.append(msg)
    else:
        try:
            import yaml
            with policy_path.open() as fh:
                policy = yaml.safe_load(fh) or {}
            actual_t = float(policy.get("t_recall", -1))
            if abs(actual_t - _EXPECTED_T_RECALL) < _T_RECALL_TOL:
                print(f"{PASS} t_recall = {actual_t} == {_EXPECTED_T_RECALL}")
            else:
                msg = f"t_recall TAMPERED: {actual_t} != {_EXPECTED_T_RECALL}"
                print(f"{FAIL} {msg}")
                failures.append(msg)
        except ImportError:
            print(f"{WARN} PyYAML not installed — reading policy as text")
            text = policy_path.read_text()
            import re
            m = re.search(r"t_recall\s*:\s*([0-9.]+)", text)
            if m:
                actual_t = float(m.group(1))
                if abs(actual_t - _EXPECTED_T_RECALL) < _T_RECALL_TOL:
                    print(f"{PASS} t_recall = {actual_t} == {_EXPECTED_T_RECALL}")
                else:
                    msg = f"t_recall TAMPERED: {actual_t} != {_EXPECTED_T_RECALL}"
                    print(f"{FAIL} {msg}")
                    failures.append(msg)
            else:
                msg = "Could not parse t_recall from policy file"
                print(f"{FAIL} {msg}")
                failures.append(msg)
        except Exception as exc:
            msg = f"Policy read error: {exc}"
            print(f"{FAIL} {msg}")
            failures.append(msg)

    # ─────────────────────────────────────────────────────────────────────
    # Final summary
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("  VERIFICATION SUMMARY")
    print("═" * 60)

    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")

    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        print(f"\n[RESULT] FAILED — {len(failures)} assertion(s) failed")
        sys.exit(1)
    else:
        print(f"\n[RESULT] ALL CHECKS PASSED")
        if warnings:
            print(f"         ({len(warnings)} warning(s) — review above)")
        sys.exit(0)


if __name__ == "__main__":
    main()
