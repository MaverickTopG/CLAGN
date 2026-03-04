#!/usr/bin/env python3
"""
Strict validation checks for CLAGN benchmark v4 outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord


EVAL_REQUIRED = [
    "source_id",
    "ra",
    "dec",
    "redshift",
    "class",
    "y_true",
    "label_source",
    "label_method",
    "confidence",
    "canonical_id",
    "split",
]
DISC_REQUIRED = [
    "source_id",
    "ra",
    "dec",
    "redshift",
    "label",
    "reference",
    "evidence_tier",
    "canonical_id",
]
EVAL_CLASSES = {"clagn", "normal_agn", "blazar", "sn", "star"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify benchmark v4 outputs")
    p.add_argument("--output-dir", default="data/benchmark")
    p.add_argument("--dedup-radius-arcsec", type=float, default=5.0)
    p.add_argument("--strict-discovery-target", type=int, default=150000)
    return p.parse_args()


def _must_exist(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")


def _check_required_cols(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{label}: missing required columns: {missing}")


def _check_no_nulls(df: pd.DataFrame, cols: list[str], label: str) -> None:
    for c in cols:
        if df[c].isna().any():
            n = int(df[c].isna().sum())
            raise SystemExit(f"{label}: column `{c}` has {n} null values")


def _check_eval_dedup(df: pd.DataFrame, radius_arcsec: float) -> None:
    if df.empty:
        raise SystemExit("eval master is empty")
    if df["canonical_id"].duplicated().any():
        n = int(df["canonical_id"].duplicated().sum())
        raise SystemExit(f"eval master has duplicate canonical_id rows: {n}")
    coords = SkyCoord(ra=df["ra"].to_numpy(dtype=float) * u.deg, dec=df["dec"].to_numpy(dtype=float) * u.deg)
    i, j, _, _ = coords.search_around_sky(coords, radius_arcsec * u.arcsec)
    near_pairs = int(np.sum(i < j))
    if near_pairs > 0:
        raise SystemExit(f"eval master has {near_pairs} coordinate pairs within {radius_arcsec}\"")


def _check_splits(master: pd.DataFrame, train: pd.DataFrame, dev: pd.DataFrame, test: pd.DataFrame) -> None:
    ms = set(master["canonical_id"].astype(str))
    tr = set(train["canonical_id"].astype(str))
    dv = set(dev["canonical_id"].astype(str))
    ts = set(test["canonical_id"].astype(str))

    if tr & dv or tr & ts or dv & ts:
        raise SystemExit("split overlap detected among train/dev/test")
    if ms != (tr | dv | ts):
        missing = ms - (tr | dv | ts)
        extra = (tr | dv | ts) - ms
        raise SystemExit(
            f"split union mismatch: missing={len(missing)} extra={len(extra)}"
        )

    for split_name, sdf in [("train", train), ("dev", dev), ("test", test)]:
        classes = set(sdf["class"].astype(str).str.lower().unique().tolist())
        absent = sorted(EVAL_CLASSES - classes)
        if absent:
            raise SystemExit(f"{split_name} split missing classes: {absent}")


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    files = {
        "eval_master": out / "benchmark_eval_master_v4.csv",
        "eval_train": out / "benchmark_eval_train_v4.csv",
        "eval_dev": out / "benchmark_eval_dev_v4.csv",
        "eval_test": out / "benchmark_eval_test_v4.csv",
        "discovery": out / "benchmark_discovery_master_v4.csv",
        "audit": out / "v4_provenance_audit.csv",
        "split_hash": out / "v4_split_hash.json",
        "report": out / "v4_build_report.json",
    }
    for p in files.values():
        _must_exist(p)

    eval_master = pd.read_csv(files["eval_master"], low_memory=False)
    eval_train = pd.read_csv(files["eval_train"], low_memory=False)
    eval_dev = pd.read_csv(files["eval_dev"], low_memory=False)
    eval_test = pd.read_csv(files["eval_test"], low_memory=False)
    discovery = pd.read_csv(files["discovery"], low_memory=False)

    _check_required_cols(eval_master, EVAL_REQUIRED, "eval_master")
    _check_no_nulls(eval_master, EVAL_REQUIRED, "eval_master")
    _check_eval_dedup(eval_master, radius_arcsec=args.dedup_radius_arcsec)

    _check_required_cols(discovery, DISC_REQUIRED, "discovery")
    _check_no_nulls(discovery, DISC_REQUIRED, "discovery")
    if discovery["canonical_id"].duplicated().any():
        n = int(discovery["canonical_id"].duplicated().sum())
        raise SystemExit(f"discovery has duplicate canonical_id rows: {n}")

    eval_ids = set(eval_master["canonical_id"].astype(str))
    disc_ids = set(discovery["canonical_id"].astype(str))
    overlap = len(eval_ids & disc_ids)
    if overlap:
        raise SystemExit(f"eval/discovery overlap detected: {overlap} canonical_ids")

    _check_splits(eval_master, eval_train, eval_dev, eval_test)

    if len(discovery) < args.strict_discovery_target:
        raise SystemExit(
            f"discovery rows {len(discovery)} < strict target {args.strict_discovery_target}"
        )

    split_hash = json.loads(files["split_hash"].read_text(encoding="utf-8"))
    if int(split_hash["master"]["rows"]) != len(eval_master):
        raise SystemExit("split_hash master row count mismatch")

    print("PASS: v4 dataset verification")
    print(f"  eval rows: {len(eval_master)}")
    print(f"  eval class counts: {eval_master['class'].value_counts().to_dict()}")
    print(f"  discovery rows: {len(discovery)}")
    print(f"  discovery label counts: {discovery['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
