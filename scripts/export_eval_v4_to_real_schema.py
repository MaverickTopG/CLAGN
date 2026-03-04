#!/usr/bin/env python3
"""
Export benchmark_eval_*_v4.csv into the real benchmark schema contract.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REAL_COLUMNS = [
    "source_id",
    "ra",
    "dec",
    "redshift",
    "spectral_state_1",
    "spectral_state_2",
    "epoch_1",
    "epoch_2",
    "class",
    "label_source",
    "label_method",
    "independent_of_pipeline",
    "confidence",
    "g_mag",
    "notes",
    "mag",
    "y_true",
    "canonical_id",
    "split",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export eval v4 to real schema")
    p.add_argument("--input-dir", default="data/benchmark")
    p.add_argument("--output-dir", default="data/real_clagn")
    p.add_argument("--prefix", default="benchmark")
    return p.parse_args()


def _to_real(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "source_id": df.get("source_id", ""),
            "ra": df.get("ra", 0.0),
            "dec": df.get("dec", 0.0),
            "redshift": df.get("redshift", 0.0),
            "spectral_state_1": "",
            "spectral_state_2": "",
            "epoch_1": "",
            "epoch_2": "",
            "class": df.get("class", ""),
            "label_source": df.get("label_source", ""),
            "label_method": df.get("label_method", "literature"),
            "independent_of_pipeline": True,
            "confidence": df.get("confidence", "high"),
            "g_mag": "",
            "notes": df.get("reference", ""),
            "mag": "",
            "y_true": df.get("y_true", 0),
            "canonical_id": df.get("canonical_id", ""),
            "split": df.get("split", ""),
        }
    )
    return out[REAL_COLUMNS]


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_master = pd.read_csv(in_dir / "benchmark_eval_master_v4.csv", low_memory=False)
    real_master = _to_real(eval_master)

    out_master = out_dir / f"{args.prefix}_master_v4.csv"
    out_dev = out_dir / f"{args.prefix}_dev_v4.csv"
    out_test = out_dir / f"{args.prefix}_test_v4.csv"

    real_master.to_csv(out_master, index=False)
    real_master[real_master["split"] == "dev"].to_csv(out_dev, index=False)
    real_master[real_master["split"] == "test"].to_csv(out_test, index=False)

    print(f"Wrote: {out_master}")
    print(f"Wrote: {out_dev}")
    print(f"Wrote: {out_test}")


if __name__ == "__main__":
    main()
