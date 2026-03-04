from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from catalog_match import normalize_identifier
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from catalog_match import normalize_identifier

MERGE_COLUMNS = [
    "score",
    "status",
    "rank",
    "n_points_results",
    "baseline_days_results",
    "baseline_years_results",
    "band_coverage_results",
    "delta_mag_results",
    "raw_points_total",
    "kept_points_total",
    "fraction_rejected_total",
    "kept_points_w1",
    "kept_points_w2",
    "season_bins_w1",
    "season_bins_w2",
    "has_pre_gap",
    "has_post_gap",
    "has_both_sides",
    "systematics_verdict",
    "blazar_match",
    "known_clagn_match",
    "final_label",
    "coordinate_source",
    "is_top_clagn_candidate",
    "card_path",
    "figure_path",
    "source_id_normalized",
    "canonical_id_normalized",
]

NUMERIC_MERGE_COLUMNS = {
    "score",
    "rank",
    "n_points_results",
    "baseline_days_results",
    "baseline_years_results",
    "delta_mag_results",
    "raw_points_total",
    "kept_points_total",
    "fraction_rejected_total",
    "kept_points_w1",
    "kept_points_w2",
    "season_bins_w1",
    "season_bins_w2",
}


def _read_csv_str(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=True,
        na_values=["", " ", "NA", "NaN", "null", "None"],
    )


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "canonical_id" in out.columns:
        out["_cid_norm"] = out["canonical_id"].map(normalize_identifier).str.upper()
    else:
        out["_cid_norm"] = np.nan
    if "source_id" in out.columns:
        out["_sid_norm"] = out["source_id"].map(normalize_identifier).str.upper()
    else:
        out["_sid_norm"] = np.nan
    return out


def _prepare_summary(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    summary = _normalize_ids(summary_df)
    warnings: list[str] = []

    for col in NUMERIC_MERGE_COLUMNS:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")

    for col in ["score", "rank", "canonical_id"]:
        if col not in summary.columns:
            if col == "canonical_id":
                summary[col] = ""
            else:
                summary[col] = np.nan

    summary = summary.sort_values(
        by=["score", "rank", "canonical_id"],
        ascending=[False, True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)

    cid_dup = int(summary["_cid_norm"].dropna().duplicated().sum())
    sid_dup = int(summary["_sid_norm"].dropna().duplicated().sum())
    if cid_dup:
        warnings.append(
            f"Found {cid_dup} duplicate canonical_id keys in summary; used deterministic tie-break (score desc, rank asc, canonical_id asc)."
        )
    if sid_dup:
        warnings.append(
            f"Found {sid_dup} duplicate source_id keys in summary; used deterministic tie-break (score desc, rank asc, canonical_id asc)."
        )

    return summary, warnings


def _build_rename_map(master_cols: set[str], merge_cols: list[str]) -> dict[str, str]:
    rename_map: dict[str, str] = {}
    for col in merge_cols:
        if col in master_cols:
            rename_map[col] = f"cards_{col}"
        else:
            rename_map[col] = col
    return rename_map


def merge_master_with_summary(master_csv: str, summary_csv: str, out_csv: str) -> dict[str, Any]:
    master_path = Path(master_csv)
    summary_path = Path(summary_csv)
    out_path = Path(out_csv)

    if not master_path.exists():
        raise FileNotFoundError(f"Master CSV not found: {master_csv}")
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Summary CSV not found: {summary_csv}. Run make_cards.py first or pass --summary-csv."
        )

    master = _read_csv_str(str(master_path))
    summary_raw = _read_csv_str(str(summary_path))

    if "canonical_id" not in master.columns and "source_id" not in master.columns:
        raise ValueError(
            f"Master CSV must contain canonical_id or source_id. Found columns: {list(master.columns)}"
        )
    if "canonical_id" not in summary_raw.columns and "source_id" not in summary_raw.columns:
        raise ValueError(
            f"Summary CSV must contain canonical_id or source_id. Found columns: {list(summary_raw.columns)}"
        )

    summary, warnings = _prepare_summary(summary_raw)
    merge_cols = [c for c in MERGE_COLUMNS if c in summary.columns]
    rename_map = _build_rename_map(set(master.columns), merge_cols)

    master = _normalize_ids(master)
    master["_row_order"] = np.arange(len(master), dtype=int)

    summary_cid = summary.dropna(subset=["_cid_norm"]).drop_duplicates(subset=["_cid_norm"], keep="first")
    summary_sid = summary.dropna(subset=["_sid_norm"]).drop_duplicates(subset=["_sid_norm"], keep="first")

    cid_payload = summary_cid[["_cid_norm"] + merge_cols].rename(columns={c: f"__cid__{c}" for c in merge_cols})
    cid_payload["__cid_match"] = True
    merged = master.merge(cid_payload, on="_cid_norm", how="left")
    merged["merge_match_key"] = np.where(merged["__cid_match"].eq(True), "canonical_id", "")

    sid_payload = summary_sid[["_sid_norm"] + merge_cols].rename(columns={c: f"__sid__{c}" for c in merge_cols})
    sid_payload["__sid_match"] = True
    merged = merged.merge(sid_payload, on="_sid_norm", how="left")

    for src_col in merge_cols:
        out_col = rename_map[src_col]
        cid_col = f"__cid__{src_col}"
        sid_col = f"__sid__{src_col}"
        merged[out_col] = merged[cid_col]
        merged[out_col] = merged[out_col].where(merged[out_col].notna(), merged[sid_col])

    sid_only_mask = merged["merge_match_key"].eq("") & merged["__sid_match"].eq(True)
    merged.loc[sid_only_mask, "merge_match_key"] = "source_id"
    merged["merge_matched"] = merged["merge_match_key"].ne("")
    merged["merge_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    merged["merge_summary_sha256"] = _sha256(summary_path)

    temp_cols = [c for c in merged.columns if c.startswith("__cid__") or c.startswith("__sid__")] + [
        "__cid_match",
        "__sid_match",
        "_cid_norm",
        "_sid_norm",
    ]
    merged = merged.drop(columns=[c for c in temp_cols if c in merged.columns], errors="ignore")
    merged = merged.sort_values("_row_order").drop(columns=["_row_order"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    stats = {
        "master_csv": str(master_path),
        "summary_csv": str(summary_path),
        "out_csv": str(out_path),
        "rows_master": int(len(master)),
        "rows_out": int(len(merged)),
        "rows_summary": int(len(summary_raw)),
        "matched_rows": int(merged["merge_matched"].sum()),
        "unmatched_rows": int((~merged["merge_matched"]).sum()),
        "match_key_counts": {
            "canonical_id": int((merged["merge_match_key"] == "canonical_id").sum()),
            "source_id": int((merged["merge_match_key"] == "source_id").sum()),
            "unmatched": int((merged["merge_match_key"] == "").sum()),
        },
        "merged_columns": [rename_map[c] for c in merge_cols],
        "warnings": warnings,
    }
    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge candidate-card summary_table outputs into benchmark master CSV.")
    p.add_argument("--master-csv", default="data/real_clagn/benchmark_master.csv")
    p.add_argument("--summary-csv", default="summary_table.csv")
    p.add_argument("--out", default="data/real_clagn/benchmark_master_augmented.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stats = merge_master_with_summary(args.master_csv, args.summary_csv, args.out)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
