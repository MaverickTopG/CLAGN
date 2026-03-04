"""
Freeze the real CLAGN benchmark with a cryptographic hash and summary counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clagn.utils.benchmark_labels import ensure_class_ytrue
from clagn.utils.validation_metadata import git_commit


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to benchmark_master.csv")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Missing input file: {in_path}")
        sys.exit(1)

    df = pd.read_csv(in_path)
    df = ensure_class_ytrue(df)
    # Ensure the on-disk file is normalized before hashing/freeze.
    df.to_csv(in_path, index=False)

    out_path = in_path.parent / "BENCHMARK_FREEZE.json"
    h = sha256_file(in_path)
    class_counts = df["class"].astype(str).value_counts().to_dict()
    payload = {
        "sha256": h,
        "benchmark_hash": h,  # compatibility with existing metadata readers
        "row_count_total": int(len(df)),
        "row_count_per_class": class_counts,
        "y_true_positive_count": int(df["y_true"].sum()),
        "git_commit": git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote freeze file: {out_path}")


if __name__ == "__main__":
    main()
