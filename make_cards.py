from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback as _tb
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from catalog_match import (
    load_gaia_metrics,
    load_known_blazars,
    load_known_clagn,
    lookup_gaia_metrics,
    match_blazar,
    match_known_clagn,
    normalize_identifier,
)
from reporting import render_benchmark_report, render_candidate_card, sanitize_filename, write_text, write_text_atomic
from wise_processing import (
    WiseQAConfig,
    apply_wise_qa_filters,
    build_systematics_verdict,
    compute_coverage_metrics,
    compute_seasonal_medians,
    load_wise_table,
    normalize_wise_schema,
    plot_wise_lightcurve,
)
from scripts.merge_master_with_cards import merge_master_with_summary

REQUIRED_RESULTS_COLS = {
    "source_id",
    "canonical_id",
    "score",
    "status",
    "n_points",
    "baseline_days",
    "band_coverage",
}
OPTIONAL_RESULTS_COLS = {
    "baseline_years",
    "delta_mag",
    "delta_mag_method",
    "rejection_reason",
    "ra",
    "dec",
    "class",
    "y_true",
}
NUMERIC_COLS = ["score", "n_points", "baseline_days", "baseline_years", "delta_mag", "ra", "dec", "y_true"]


@dataclass
class Config:
    results: str
    topn: int
    outdir: str
    score_thresh: float = 0.7
    wise_dir: str = "data/wise"
    gaia: str | None = None
    known_blazars: str | None = None
    allow_missing_blazar_catalog: bool = False
    known_clagn: str | None = None
    coordinates: str | None = "coordinates.csv"
    benchmark_master: str | None = "data/real_clagn/benchmark_master.csv"
    pdf: bool = False
    seed: int = 0
    overwrite: bool = False
    skip_missing_wise: bool = False
    fetch_wise_if_missing: bool = False
    merge_master: bool = False
    master_csv: str = "data/real_clagn/benchmark_master.csv"
    summary_csv: str | None = None
    master_out: str = "data/real_clagn/benchmark_master_augmented.csv"
    policy_config: str = "configs/pipeline_policy.yaml"
    t_recall: float | None = None
    strict_mode: bool = False


def query_irsa_wise_timeseries(ra: float, dec: float) -> pd.DataFrame:
    raise NotImplementedError(
        "No web calls are implemented. Provide local WISE files in data/wise/{canonical_id}.csv."
    )


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Generate CLAGN candidate cards from results.csv")
    p.add_argument("--results", required=True)
    p.add_argument("--topn", required=True, type=int)
    p.add_argument("--outdir", required=True)
    p.add_argument("--score-thresh", type=float, default=0.7)
    p.add_argument("--wise-dir", default="data/wise")
    p.add_argument("--gaia", default=None)
    p.add_argument("--known-blazars", default=None)
    p.add_argument("--allow-missing-blazar-catalog", action="store_true")
    p.add_argument("--known-clagn", default=None)
    p.add_argument("--coordinates", default="coordinates.csv")
    p.add_argument("--benchmark-master", default="data/real_clagn/benchmark_master.csv")
    p.add_argument("--pdf", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-missing-wise", action="store_true")
    p.add_argument("--fetch-wise-if-missing", action="store_true", help="Query IRSA via local clagn.ingestion.wise helpers and cache per-object CSVs")
    p.add_argument("--merge-master", action="store_true", help="Merge summary_table outputs into benchmark master CSV")
    p.add_argument("--master-csv", default="data/real_clagn/benchmark_master.csv", help="Master benchmark CSV for merge step")
    p.add_argument("--summary-csv", default=None, help="Summary CSV path for merge step (defaults to <outdir>/summary_table.csv)")
    p.add_argument("--master-out", default="data/real_clagn/benchmark_master_augmented.csv", help="Output augmented master CSV path")
    p.add_argument("--policy-config", default="configs/pipeline_policy.yaml", help="Policy YAML controlling thresholds and hard gates")
    p.add_argument("--t-recall", type=float, default=None, help="Stage-A threshold override (default from policy)")
    p.add_argument("--strict-mode", action="store_true", help="Fail-fast on schema/data integrity violations")
    args = p.parse_args()
    return Config(**vars(args))


def _read_csv_str(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=True,
        na_values=["", " ", "NA", "NaN", "null", "None"],
    )


def _load_policy_config(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Policy config not found: {path}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Policy config must be a mapping: {path}")
    return data


def _strict_results_checks(df: pd.DataFrame, strict_mode: bool) -> None:
    required = {"source_id", "canonical_id", "score", "status"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"results.csv missing required columns for strict checks: {missing}")

    scores = pd.to_numeric(df["score"], errors="coerce")
    status_ok = df["status"].astype(str).str.upper().eq("OK")
    if strict_mode and (status_ok & ~np.isfinite(scores)).any():
        n_bad = int((status_ok & ~np.isfinite(scores)).sum())
        raise ValueError(f"Strict mode: found {n_bad} status=OK rows with non-finite scores")

    cid_norm = df["canonical_id"].map(normalize_identifier).str.upper()
    sid_norm = df["source_id"].map(normalize_identifier).str.upper()
    dup_cid = cid_norm[(cid_norm.notna()) & (cid_norm != "") & cid_norm.duplicated(keep=False)]
    dup_sid = sid_norm[(sid_norm.notna()) & (sid_norm != "") & sid_norm.duplicated(keep=False)]
    if strict_mode and (not dup_cid.empty or not dup_sid.empty):
        raise ValueError(
            f"Strict mode: duplicate key values found (canonical_id duplicates={dup_cid.nunique()}, "
            f"source_id duplicates={dup_sid.nunique()})"
        )

    if {"ra", "dec"}.issubset(df.columns):
        ra = pd.to_numeric(df["ra"], errors="coerce")
        dec = pd.to_numeric(df["dec"], errors="coerce")
        bad_coords = ((ra.notna() & ((ra < 0) | (ra > 360))) | (dec.notna() & ((dec < -90) | (dec > 90))))
        if strict_mode and bad_coords.any():
            raise ValueError(f"Strict mode: found {int(bad_coords.sum())} rows with invalid RA/Dec ranges")

    if "split" in df.columns:
        split_vals = df["split"].dropna().astype(str).str.strip().str.lower()
        bad_split = ~split_vals.isin({"dev", "test"})
        if strict_mode and bad_split.any():
            bad = sorted(split_vals[bad_split].unique().tolist())
            raise ValueError(f"Strict mode: invalid split values found: {bad}")


def _coerce_results(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    missing = sorted(REQUIRED_RESULTS_COLS - set(out.columns))
    if missing:
        raise ValueError(f"results.csv missing required columns: {missing}")
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype(str).replace("nan", np.nan)
            out[c] = out[c].str.strip()
    out["status"] = out["status"].fillna("").astype(str).str.upper().str.strip()
    for col in NUMERIC_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "baseline_years" not in out.columns:
        out["baseline_years"] = pd.to_numeric(out["baseline_days"], errors="coerce") / 365.25
    return out


def _load_optional_coordinates(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    df = _read_csv_str(str(p))
    cols = {c.lower(): c for c in df.columns}
    if "ra" not in cols or "dec" not in cols:
        raise ValueError("coordinates.csv must contain ra and dec columns")
    if "canonical_id" not in cols and "source_id" not in cols:
        raise ValueError("coordinates.csv must contain canonical_id or source_id (plus ra, dec)")
    for c in [cols["ra"], cols["dec"]]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _load_benchmark_master(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    df = _read_csv_str(str(p))
    for col in ["ra", "dec", "y_true"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _enrich_results(results: pd.DataFrame, coordinates: pd.DataFrame | None, benchmark: pd.DataFrame | None) -> pd.DataFrame:
    df = results.copy()
    df["_cid_norm"] = df["canonical_id"].map(normalize_identifier).str.upper()
    df["_sid_norm"] = df["source_id"].map(normalize_identifier).str.upper()
    df["canonical_id_normalized"] = df["_cid_norm"]
    df["source_id_normalized"] = df["_sid_norm"]
    if "ra" not in df.columns:
        df["ra"] = np.nan
    if "dec" not in df.columns:
        df["dec"] = np.nan
    ra0 = pd.to_numeric(df["ra"], errors="coerce")
    dec0 = pd.to_numeric(df["dec"], errors="coerce")
    has_results_coords = np.isfinite(ra0) & np.isfinite(dec0)
    df["coordinate_source"] = np.where(has_results_coords, "results", "missing")

    def merge_from(source: pd.DataFrame, source_name: str) -> pd.DataFrame:
        src = source.copy()
        src_cols = set(src.columns)
        if "canonical_id" in src_cols:
            src["_cid_norm"] = src["canonical_id"].map(normalize_identifier).str.upper()
        if "source_id" in src_cols:
            src["_sid_norm"] = src["source_id"].map(normalize_identifier).str.upper()
        keep_cols = [c for c in ["ra", "dec", "class", "y_true", "split", "source_id", "canonical_id"] if c in src.columns]
        src = src[[c for c in ["_cid_norm", "_sid_norm"] + keep_cols if c in src.columns]].copy()

        df_local = df
        fill_cols = [c for c in ["ra", "dec", "class", "y_true", "split"] if c in src.columns]

        # canonical join first
        if "_cid_norm" in src.columns and fill_cols:
            src_cid = src.dropna(subset=["_cid_norm"]).drop_duplicates(subset=["_cid_norm"], keep="first")
            src_cid = src_cid[["_cid_norm"] + fill_cols].rename(columns={c: f"__{source_name}_cid__{c}" for c in fill_cols})
            merged = df_local.merge(src_cid, on="_cid_norm", how="left")
            before_missing_coords = ~(
                np.isfinite(pd.to_numeric(merged["ra"], errors="coerce")) &
                np.isfinite(pd.to_numeric(merged["dec"], errors="coerce"))
            )
            for col in fill_cols:
                tmp = f"__{source_name}_cid__{col}"
                if col not in merged.columns:
                    merged[col] = np.nan
                merged[col] = merged[col].where(merged[col].notna(), merged[tmp])
                merged = merged.drop(columns=[tmp], errors="ignore")
            after_has_coords = (
                np.isfinite(pd.to_numeric(merged["ra"], errors="coerce")) &
                np.isfinite(pd.to_numeric(merged["dec"], errors="coerce"))
            )
            if "coordinate_source" in merged.columns:
                fill_mask = before_missing_coords & after_has_coords & merged["coordinate_source"].eq("missing")
                merged.loc[fill_mask, "coordinate_source"] = source_name
            df_local = merged

        # source_id join fallback
        if "_sid_norm" in src.columns and fill_cols:
            src_sid = src.dropna(subset=["_sid_norm"]).drop_duplicates(subset=["_sid_norm"], keep="first")
            src_sid = src_sid[["_sid_norm"] + fill_cols].rename(columns={c: f"__{source_name}_sid__{c}" for c in fill_cols})
            merged = df_local.merge(src_sid, on="_sid_norm", how="left")
            before_missing_coords = ~(
                np.isfinite(pd.to_numeric(merged["ra"], errors="coerce")) &
                np.isfinite(pd.to_numeric(merged["dec"], errors="coerce"))
            )
            for col in fill_cols:
                tmp = f"__{source_name}_sid__{col}"
                if col not in merged.columns:
                    merged[col] = np.nan
                merged[col] = merged[col].where(merged[col].notna(), merged[tmp])
                merged = merged.drop(columns=[tmp], errors="ignore")
            after_has_coords = (
                np.isfinite(pd.to_numeric(merged["ra"], errors="coerce")) &
                np.isfinite(pd.to_numeric(merged["dec"], errors="coerce"))
            )
            if "coordinate_source" in merged.columns:
                fill_mask = before_missing_coords & after_has_coords & merged["coordinate_source"].eq("missing")
                merged.loc[fill_mask, "coordinate_source"] = source_name
            df_local = merged
        return df_local

    if coordinates is not None:
        df = merge_from(coordinates, "coordinates")
    if benchmark is not None:
        df = merge_from(benchmark, "benchmark_master")
    return df


def _file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _resolve_wise_file(wise_dir: Path, candidate: pd.Series) -> Path | None:
    attempts = []
    for key in [candidate.get("canonical_id"), sanitize_filename(candidate.get("canonical_id")), candidate.get("source_id"), sanitize_filename(candidate.get("source_id"))]:
        if key is None or (isinstance(key, float) and np.isnan(key)):
            continue
        p = wise_dir / f"{key}.csv"
        attempts.append(p)
        if p.exists():
            return p
    return None


def _fetch_and_cache_wise_raw(candidate: pd.Series, wise_dir: Path) -> Path:
    """Fetch raw-ish AllWISE/NEOWISE tables using local CLAGN ingestion helpers and cache to CSV."""
    try:
        from clagn.ingestion import wise as wise_mod  # local project module
    except Exception as e:
        raise RuntimeError(
            "Unable to import clagn.ingestion.wise for live WISE fetch. "
            "Install project deps or provide local data/wise/*.csv files."
        ) from e

    ra = pd.to_numeric(pd.Series([candidate.get("ra")]), errors="coerce").iloc[0]
    dec = pd.to_numeric(pd.Series([candidate.get("dec")]), errors="coerce").iloc[0]
    if not (np.isfinite(ra) and np.isfinite(dec)):
        raise RuntimeError(f"Cannot fetch WISE for {candidate.get('canonical_id')}: missing RA/Dec")

    frames = []
    errors = []
    # These are private helpers, but they return the raw IRSA tables we need for QA accounting.
    for label, fn_name in [("allwise", "_query_allwise_mept"), ("neowise", "_query_neowise_r")]:
        fn = getattr(wise_mod, fn_name, None)
        if fn is None:
            errors.append(f"{fn_name} unavailable")
            continue
        try:
            df = fn(float(ra), float(dec))
            if df is not None and len(df) > 0:
                part = pd.DataFrame(df).copy()
                part["source_table"] = label
                frames.append(part)
        except Exception as e:
            errors.append(f"{label}: {e}")

    if not frames:
        raise RuntimeError(
            f"IRSA WISE fetch produced no rows for {candidate.get('canonical_id')} "
            f"(ra={ra}, dec={dec}). Errors: {'; '.join(errors) if errors else 'none'}"
        )

    raw = pd.concat(frames, ignore_index=True, sort=False)
    wise_dir.mkdir(parents=True, exist_ok=True)
    out = wise_dir / f"{candidate.get('canonical_id')}.csv"
    raw.to_csv(out, index=False, float_format="%.6f")
    return out


def _main_rejection_causes(qa: dict, topk: int = 3) -> str:
    keys = [
        "reject_missing_time",
        "reject_missing_mag",
        "reject_invalid_band",
        "reject_cc_flags",
        "reject_qual_frame",
        "reject_moon_lev",
    ]
    pairs = [(k, int(qa.get(k, 0) or 0)) for k in keys]
    pairs = [p for p in pairs if p[1] > 0]
    if not pairs:
        return "none"
    pairs.sort(key=lambda x: (-x[1], x[0]))
    return ", ".join(f"{k.replace('reject_', '')}={v}" for k, v in pairs[:topk])


def _compute_state_change_ratio(seasonal: pd.DataFrame, min_seasons: int, eps: float) -> tuple[float, dict[str, float]]:
    if seasonal is None or seasonal.empty:
        return np.nan, {}
    per_band: dict[str, float] = {}
    for band in ["W1", "W2"]:
        s = seasonal[seasonal["band"] == band].copy()
        if len(s) < min_seasons:
            continue
        med = pd.to_numeric(s["median_mag"], errors="coerce").to_numpy(dtype=float)
        scat = pd.to_numeric(s["scatter_mag"], errors="coerce").to_numpy(dtype=float)
        med = med[np.isfinite(med)]
        scat = scat[np.isfinite(scat) & (scat > 0)]
        if med.size < min_seasons:
            continue
        delta = float(np.nanmax(med) - np.nanmin(med))
        sigma = float(np.nanmedian(scat)) if scat.size else np.nan
        if not np.isfinite(sigma) or sigma <= 0:
            # Robust fallback to preserve determinism if scatter is unavailable.
            sigma = float(np.nanmedian(np.abs(med - np.nanmedian(med)))) if med.size else np.nan
        if not np.isfinite(delta) or not np.isfinite(sigma):
            continue
        per_band[band] = float(delta / (sigma + eps))
    if not per_band:
        return np.nan, {}
    return float(max(per_band.values())), per_band


def _compute_transient_spike_features(seasonal: pd.DataFrame, min_seasons: int) -> dict[str, float]:
    features = {
        "transient_max_nonpeak_ratio": np.nan,
        "transient_max_adjacency_ratio": np.nan,
        "transient_min_band_seasons": 0,
    }
    if seasonal is None or seasonal.empty:
        return features
    max_nonpeak_ratio: list[float] = []
    max_adj_ratio: list[float] = []
    season_counts: list[int] = []
    for band in ["W1", "W2"]:
        s = seasonal[seasonal["band"] == band].copy().sort_values("season_center_mjd")
        if len(s) < min_seasons:
            continue
        med = pd.to_numeric(s["median_mag"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(med).sum() < min_seasons:
            continue
        idx = np.where(np.isfinite(med))[0]
        vals = med[idx]
        if vals.size < min_seasons:
            continue

        # Peak season against baseline excluding that season.
        best_peak = None
        best_delta = -np.inf
        for j in range(vals.size):
            baseline = np.nanmedian(np.delete(vals, j)) if vals.size > 1 else vals[j]
            delta = abs(vals[j] - baseline)
            if delta > best_delta:
                best_delta = delta
                best_peak = j
        if best_peak is None or not np.isfinite(best_delta) or best_delta <= 0:
            continue

        baseline = np.nanmedian(np.delete(vals, best_peak)) if vals.size > 1 else vals[best_peak]
        peak_delta = abs(vals[best_peak] - baseline)
        adj_ratio = 0.0
        for k in [best_peak - 1, best_peak + 1]:
            if 0 <= k < vals.size:
                adj_ratio = max(adj_ratio, float(abs(vals[k] - baseline) / peak_delta))
        nonpeak_ratio = 0.0
        for j in range(vals.size):
            if j == best_peak:
                continue
            nonpeak_ratio = max(nonpeak_ratio, float(abs(vals[j] - baseline) / peak_delta))
        max_nonpeak_ratio.append(nonpeak_ratio)
        max_adj_ratio.append(adj_ratio)
        season_counts.append(int(vals.size))
    if not max_nonpeak_ratio:
        return features
    features["transient_max_nonpeak_ratio"] = float(max(max_nonpeak_ratio))
    features["transient_max_adjacency_ratio"] = float(max(max_adj_ratio))
    features["transient_min_band_seasons"] = int(min(season_counts))
    return features


def _compute_transient_spike_flag(
    transient_max_nonpeak_ratio: float,
    transient_max_adjacency_ratio: float,
    transient_min_band_seasons: int,
    peak_frac: float,
    adjacency_frac: float,
    min_seasons: int,
) -> bool:
    if transient_min_band_seasons < min_seasons:
        return True
    if not np.isfinite(transient_max_nonpeak_ratio) or not np.isfinite(transient_max_adjacency_ratio):
        return True
    return bool((transient_max_adjacency_ratio < adjacency_frac) and (transient_max_nonpeak_ratio < peak_frac))


def _fp_mode_hint(
    klass: str,
    fraction_rejected_total: float,
    kept_points_total: int,
    season_bins_total: int,
    state_change_ratio_value: float,
    transient_spike_flag: bool,
    r_min: float,
    min_clean_points_total: int,
    min_season_bins_total: int,
) -> str:
    k = (klass or "").strip().lower()
    if (
        (np.isfinite(fraction_rejected_total) and fraction_rejected_total > 0.5)
        or kept_points_total < min_clean_points_total
        or season_bins_total < min_season_bins_total
    ):
        return "artifact_like"
    if transient_spike_flag:
        return "sn_like"
    if np.isfinite(state_change_ratio_value) and state_change_ratio_value < r_min:
        return "star_like"
    if k == "normal_agn":
        return "normal_agn_like"
    return "unknown"


def _final_label(
    candidate: pd.Series,
    systematics: dict,
    blazar_match: dict,
    known_clagn_match: dict,
    stage_a_positive: bool,
    gate_blazar_veto: bool,
    gate_wise_qc_pass: bool,
    gate_data_sufficient: bool,
    gate_state_change_ratio_pass: bool,
    gate_transient_spike_pass: bool,
) -> tuple[str, list[str], str]:
    bullets: list[str] = []
    bullets.append(f"Rank {int(candidate['rank'])} with score={candidate.get('score', np.nan):.4f}" if pd.notna(candidate.get("score")) else f"Rank {int(candidate['rank'])} with missing score")
    bullets.append(
        f"Sampling: n_points={candidate.get('n_points')}, baseline_years={candidate.get('baseline_years') if pd.notna(candidate.get('baseline_years')) else 'NA'}, band_coverage={candidate.get('band_coverage')}"
    )
    bullets.append(
        f"QA rejection fraction={systematics.get('fraction_rejected_total', np.nan):.1%} (main causes summarized in card QA section)" if pd.notna(systematics.get('fraction_rejected_total')) else "QA rejection fraction unavailable"
    )
    bullets.append(f"Systematics verdict={systematics.get('systematics_verdict')}: {systematics.get('systematics_summary')}")
    bullets.append("Known blazar catalog match=YES" if blazar_match.get("matched") else "Known blazar catalog match=NO")
    if pd.notna(candidate.get("class")):
        bullets.append(f"Dataset metadata class={candidate.get('class')}")
    if known_clagn_match.get("matched"):
        bullets.append("Known CLAGN file match=YES")

    klass = str(candidate.get("class", "")).strip().lower()
    trace = [
        f"stage_a_positive={'pass' if stage_a_positive else 'fail'}",
        f"gate_blazar_veto={'pass' if gate_blazar_veto else 'fail'}",
        f"gate_wise_qc_pass={'pass' if gate_wise_qc_pass else 'fail'}",
        f"gate_data_sufficient={'pass' if gate_data_sufficient else 'fail'}",
        f"gate_state_change_ratio_pass={'pass' if gate_state_change_ratio_pass else 'fail'}",
        f"gate_transient_spike_pass={'pass' if gate_transient_spike_pass else 'fail'}",
    ]
    decision_trace = ";".join(trace)

    if not stage_a_positive:
        return "Rejected", bullets, decision_trace
    if not gate_blazar_veto:
        return "Known blazar", bullets, decision_trace
    if (not gate_wise_qc_pass) or (not gate_data_sufficient):
        return "Needs more data", bullets, decision_trace
    if (not gate_state_change_ratio_pass) or (not gate_transient_spike_pass):
        return "Needs more data", bullets, decision_trace
    if klass in {"normal_agn", "clagn"}:
        return "Known AGN", bullets, decision_trace
    if klass in {"star", "sn"}:
        bullets.append(f"Rejected because dataset metadata class={klass} is a known contaminant class")
        return "Rejected", bullets, decision_trace
    return "Candidate CLAGN", bullets, decision_trace


def _ensure_output_paths(cfg: Config) -> dict[str, Path]:
    root = Path(cfg.outdir)
    cards = root / "cards"
    figs = root / "figures"
    reports = root / "reports"
    for p in [root, cards, figs, reports]:
        p.mkdir(parents=True, exist_ok=True)
    return {"root": root, "cards": cards, "figs": figs, "reports": reports}


def _convert_md_to_pdf(md_path: Path, pdf_path: Path) -> tuple[bool, str]:
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        return False, "pandoc not installed"
    try:
        subprocess.run([pandoc, str(md_path), "-o", str(pdf_path)], check=True, capture_output=True)
        return True, "ok"
    except subprocess.CalledProcessError as e:
        return False, f"pandoc failed: {e.stderr.decode(errors='ignore')[:300]}"


def _load_optional_inputs(cfg: Config) -> dict[str, Any]:
    known_blazars = None
    if cfg.known_blazars:
        known_blazars = load_known_blazars(cfg.known_blazars)
    elif not cfg.allow_missing_blazar_catalog:
        raise FileNotFoundError(
            "Missing required known_blazars.csv. Provide --known-blazars <path> or pass --allow-missing-blazar-catalog."
        )

    known_clagn_path = cfg.known_clagn
    if known_clagn_path is None:
        default = Path("data/benchmark/known_clagn.csv")
        if default.exists():
            known_clagn_path = str(default)
    known_clagn = load_known_clagn(known_clagn_path) if known_clagn_path and Path(known_clagn_path).exists() else None

    gaia = load_gaia_metrics(cfg.gaia) if cfg.gaia and Path(cfg.gaia).exists() else None
    coordinates = _load_optional_coordinates(cfg.coordinates)
    benchmark = _load_benchmark_master(cfg.benchmark_master)
    return {
        "known_blazars": known_blazars,
        "known_clagn": known_clagn,
        "gaia": gaia,
        "coordinates": coordinates,
        "benchmark": benchmark,
        "known_clagn_path": known_clagn_path,
    }


def _load_results(cfg: Config, benchmark: pd.DataFrame | None, coordinates: pd.DataFrame | None) -> pd.DataFrame:
    if not Path(cfg.results).exists():
        raise FileNotFoundError(f"results.csv not found: {cfg.results}")
    results = _coerce_results(_read_csv_str(cfg.results))
    enriched = _enrich_results(results, coordinates, benchmark)

    if not {"ra", "dec"}.issubset(set(enriched.columns)):
        enriched["ra"] = np.nan
        enriched["dec"] = np.nan
    missing_coords = ~(np.isfinite(pd.to_numeric(enriched["ra"], errors="coerce")) & np.isfinite(pd.to_numeric(enriched["dec"], errors="coerce")))
    # Only error later for selected candidates if needed; keeps global report possible.
    enriched["has_coords"] = ~missing_coords
    return enriched


def _select_topn(df: pd.DataFrame, topn: int) -> pd.DataFrame:
    ok = df[(df["status"] == "OK") & pd.to_numeric(df["score"], errors="coerce").notna()].copy()
    if ok.empty:
        raise ValueError("No rows with status=='OK' and finite score found in results.csv")
    ok["score"] = pd.to_numeric(ok["score"], errors="coerce")
    ok["_results_index"] = ok.index
    ok = ok.sort_values(["score", "canonical_id", "source_id"], ascending=[False, True, True]).reset_index(drop=True)
    top = ok.head(topn).copy()
    top["rank"] = np.arange(1, len(top) + 1)
    return top


def _compute_threshold_metrics(df: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    score = pd.to_numeric(df.get("score"), errors="coerce")
    ok = df.get("status", "").astype(str).str.upper().eq("OK")
    has_truth = "y_true" in df.columns and pd.to_numeric(df["y_true"], errors="coerce").notna().any()
    y_true = pd.to_numeric(df["y_true"], errors="coerce") if has_truth else None
    for thr in sorted(set(thresholds)):
        det = ok & score.notna() & (score >= thr)
        row = {
            "threshold": float(thr),
            "n_detected_ok": int(det.sum()),
        }
        if has_truth:
            pos = y_true == 1
            neg = y_true == 0
            tp = int((det & pos).sum())
            fp = int((det & neg).sum())
            fn = int((~det & pos).sum())
            row.update(
                {
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "completeness": float(tp / (tp + fn)) if (tp + fn) else np.nan,
                    "contamination": float(fp / (tp + fp)) if (tp + fp) else np.nan,
                    "contamination_note": "exact benchmark truth",
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _known_clagn_recovery(df: pd.DataFrame, known_clagn_df: pd.DataFrame | None, thresholds: list[float]) -> dict | None:
    if known_clagn_df is None:
        return None
    # Match known CLAGN rows to results by identifier first, then coordinate (2 arcsec).
    results = df.copy()
    results["_cid_norm"] = results["canonical_id"].map(normalize_identifier).str.upper()
    results["_sid_norm"] = results["source_id"].map(normalize_identifier).str.upper()
    known = known_clagn_df.copy()

    matched_result_idx = set()
    matched_known_idx = set()
    # identifier matches from a name-like column if present
    name_cols = [c for c in known.columns if c.lower() in {"name", "canonical_id", "source_id", "alias", "aliases"}]
    for kidx, row in known.iterrows():
        tokens = []
        for col in name_cols:
            val = row.get(col)
            if pd.isna(val):
                continue
            tokens.extend([v.strip() for v in str(val).replace("|", ",").replace(";", ",").split(",") if v.strip()])
        norms = {normalize_identifier(t).upper() for t in tokens if normalize_identifier(t)}
        hit = pd.DataFrame()
        for tok in norms:
            h = results[(results["_cid_norm"] == tok) | (results["_sid_norm"] == tok)]
            if not h.empty:
                hit = h
                break
        if hit.empty:
            continue
        matched_known_idx.add(int(kidx))
        matched_result_idx.add(int(hit.index[0]))

    # coordinate fallback if available
    if {c.lower() for c in known.columns} >= {"ra", "dec"}:
        ra_col = next(c for c in known.columns if c.lower() == "ra")
        dec_col = next(c for c in known.columns if c.lower() == "dec")
        rra = pd.to_numeric(results.get("ra"), errors="coerce")
        rdec = pd.to_numeric(results.get("dec"), errors="coerce")
        kra = pd.to_numeric(known[ra_col], errors="coerce")
        kdec = pd.to_numeric(known[dec_col], errors="coerce")
        for kidx in known.index:
            if int(kidx) in matched_known_idx:
                continue
            if not (np.isfinite(kra.loc[kidx]) and np.isfinite(kdec.loc[kidx])):
                continue
            # cheap small-N search using planar approx with cos(dec) correction then exact-ish threshold
            dra = (rra - float(kra.loc[kidx])) * np.cos(np.deg2rad(float(kdec.loc[kidx])))
            ddec = rdec - float(kdec.loc[kidx])
            sep_arcsec = np.sqrt(dra**2 + ddec**2) * 3600.0
            sep_arcsec = pd.to_numeric(sep_arcsec, errors="coerce")
            h = results[sep_arcsec <= 2.0]
            if not h.empty:
                matched_known_idx.add(int(kidx))
                matched_result_idx.add(int(h.index[0]))

    matched_results = results.loc[sorted(matched_result_idx)].copy() if matched_result_idx else results.iloc[0:0].copy()
    score = pd.to_numeric(matched_results.get("score"), errors="coerce") if not matched_results.empty else pd.Series(dtype=float)
    ok = matched_results.get("status", pd.Series(dtype=object)).astype(str).str.upper().eq("OK") if not matched_results.empty else pd.Series(dtype=bool)

    thr_rows = []
    for thr in sorted(set(thresholds)):
        detected = (score >= thr) & ok if not matched_results.empty else pd.Series(dtype=bool)
        n_det = int(detected.sum()) if not matched_results.empty else 0
        n_match = int(len(matched_results))
        thr_rows.append(
            {
                "threshold": float(thr),
                "n_known_clagn_matched": n_match,
                "n_detected_above_threshold": n_det,
                "n_missed": int(n_match - n_det),
                "completeness": float(n_det / n_match) if n_match else np.nan,
            }
        )

    # contamination proxy at primary threshold if no truth columns.
    primary_thr = thresholds[0]
    all_score = pd.to_numeric(df.get("score"), errors="coerce")
    all_ok = df.get("status", "").astype(str).str.upper().eq("OK")
    detections = df[all_ok & all_score.notna() & (all_score >= primary_thr)]
    det_known_hits = 0
    if not detections.empty and matched_result_idx:
        det_known_hits = int(sum(idx in matched_result_idx for idx in detections.index))
    contamination_proxy = float((len(detections) - det_known_hits) / len(detections)) if len(detections) else np.nan

    return {
        "n_known_clagn_total": int(len(known_clagn_df)),
        "n_known_clagn_matched_to_results": int(len(matched_results)),
        "n_detected_above_threshold": int(thr_rows[0]["n_detected_above_threshold"]) if thr_rows else 0,
        "n_missed_below_threshold_or_non_ok": int(thr_rows[0]["n_missed"]) if thr_rows else 0,
        "completeness": thr_rows[0]["completeness"] if thr_rows else np.nan,
        "contamination_proxy": contamination_proxy,
        "threshold_table": pd.DataFrame(thr_rows),
    }


def _write_summary_table(summary_rows: list[dict[str, Any]], out_path: Path) -> pd.DataFrame:
    df = pd.DataFrame(summary_rows)
    if not df.empty:
        df = df.sort_values(["rank", "canonical_id", "source_id"]).reset_index(drop=True)
    df.to_csv(out_path, index=False, float_format="%.6f")
    return df


def run_pipeline(config: Config) -> int:
    np.random.seed(config.seed)
    paths = _ensure_output_paths(config)
    policy = _load_policy_config(config.policy_config)
    strict_mode = bool(config.strict_mode or policy.get("strict_mode_default", False))
    t_recall = float(config.t_recall if config.t_recall is not None else policy.get("t_recall", 0.3493918746))

    optional = _load_optional_inputs(config)
    results = _load_results(config, optional["benchmark"], optional["coordinates"])
    _strict_results_checks(results, strict_mode)

    score_num = pd.to_numeric(results["score"], errors="coerce")
    status_ok = results["status"].astype(str).str.upper().eq("OK")
    results["stage_a_positive"] = status_ok & score_num.notna() & (score_num >= t_recall)
    results["gate_blazar_veto"] = True
    results["gate_wise_qc_pass"] = pd.NA
    results["gate_data_sufficient"] = pd.NA
    results["gate_state_change_ratio_pass"] = pd.NA
    results["gate_transient_spike_pass"] = pd.NA
    results["state_change_ratio_value"] = np.nan
    results["transient_spike_flag"] = pd.NA
    results["transient_max_nonpeak_ratio"] = np.nan
    results["transient_max_adjacency_ratio"] = np.nan
    results["transient_min_band_seasons"] = pd.NA
    results["kept_points_total"] = pd.NA
    results["season_bins_w1"] = pd.NA
    results["season_bins_w2"] = pd.NA
    results["fraction_rejected_total"] = np.nan
    results["fp_mode_hint"] = "unknown"
    results["final_label"] = "Rejected"
    results["decision_trace"] = "stage_a_only"
    selected = _select_topn(results, config.topn)

    wise_dir = Path(config.wise_dir)
    if not wise_dir.exists() and config.fetch_wise_if_missing:
        wise_dir.mkdir(parents=True, exist_ok=True)
    if not wise_dir.exists():
        raise FileNotFoundError(
            f"WISE directory not found: {wise_dir}\n"
            "Place per-object WISE files at data/wise/{canonical_id}.csv, pass --wise-dir, "
            "or rerun with --fetch-wise-if-missing."
        )

    run_ts = datetime.now(timezone.utc).isoformat()
    qa_policy = policy.get("qa", {})
    qa_cfg_obj = WiseQAConfig(
        min_qual_frame=float(qa_policy.get("min_qual_frame", 1.0)),
        cc_flag_clean_only=bool(qa_policy.get("cc_flag_clean_only", True)),
        moon_lev_reject_min=int(qa_policy.get("moon_lev_reject_min", 5)),
        min_points_per_season=int(qa_policy.get("min_points_per_season", 2)),
    )
    qa_cfg = asdict(qa_cfg_obj)
    min_clean_points_total = int(qa_policy.get("min_clean_points_total", 10))
    min_season_bins_total = int(qa_policy.get("min_season_bins_total", 4))
    min_good_points_per_season = int(qa_policy.get("min_good_points_per_season", qa_cfg_obj.min_points_per_season))
    gate_state_cfg = policy.get("gates", {}).get("state_change_ratio", {})
    gate_transient_cfg = policy.get("gates", {}).get("transient_spike", {})
    state_gate_enabled = bool(gate_state_cfg.get("enabled", True))
    transient_gate_enabled = bool(gate_transient_cfg.get("enabled", True))
    r_min = float(gate_state_cfg.get("r_min", 2.0))
    r_eps = float(gate_state_cfg.get("eps", 1e-3))
    transient_peak_frac = float(gate_transient_cfg.get("peak_frac", 0.6))
    transient_adj_frac = float(gate_transient_cfg.get("adjacency_frac", 0.4))
    transient_min_seasons = int(gate_transient_cfg.get("min_seasons", 4))
    summary_rows: list[dict[str, Any]] = []
    card_index_rows: list[dict[str, Any]] = []

    missing_wise: list[str] = []
    selected_cards = []
    skipped_count = 0
    failure_count = 0
    git_commit = _git_hash()

    for _, cand in selected.iterrows():
        source_id = str(cand.get("source_id", "unknown"))
        canonical_id = str(cand.get("canonical_id", "unknown"))
        try:
            if not bool(cand.get("has_coords", False)):
                raise RuntimeError(
                    f"Missing coordinates for selected candidate {canonical_id}. "
                    "Provide ra/dec in results.csv, coordinates.csv, or --benchmark-master."
                )

            wise_path = _resolve_wise_file(wise_dir, cand)
            if wise_path is None:
                if config.fetch_wise_if_missing:
                    wise_path = _fetch_and_cache_wise_raw(cand, wise_dir)
                else:
                    missing_wise.append(canonical_id)
                    if config.skip_missing_wise:
                        continue
                    raise FileNotFoundError(
                        f"Missing WISE file for candidate {canonical_id}.\n"
                        f"Expected e.g. {wise_dir / (canonical_id + '.csv')}\n"
                        "Fix: populate local WISE tables or rerun with --fetch-wise-if-missing."
                    )

            raw_table = load_wise_table(str(wise_path))
            wise_tables = sorted(raw_table["source_table"].dropna().unique().tolist()) if "source_table" in raw_table.columns else []
            norm = normalize_wise_schema(raw_table)
            kept, qa_accounting, rejected = apply_wise_qa_filters(norm, qa_cfg)
            norm_with_qa = norm.copy()
            # align QA columns by row order/index (same length/order)
            for col in ["qa_keep", "qa_reject_step", "qa_reject_reason"]:
                if col in kept.columns or col in rejected.columns:
                    # easier from concatenating original qa state present in apply return 'kept/rejected' only; rebuild from copies if not in norm
                    pass
            # Re-run accounting frame source for plotting overlays.
            qa_frame = pd.concat([kept, rejected], axis=0).sort_index().reset_index(drop=True)
            seasonal = compute_seasonal_medians(kept)
            coverage = compute_coverage_metrics(qa_frame, kept)
            systematics = build_systematics_verdict(qa_frame, kept, seasonal)

            safe_id = sanitize_filename(canonical_id)
            fig_path = paths["figs"] / f"{safe_id}_wise.png"
            plot_wise_lightcurve(
                qa_frame,
                kept,
                seasonal,
                str(fig_path),
                title=f"{canonical_id} | score={cand.get('score'):.4f} | status={cand.get('status')}",
            )

            blazar_match = match_blazar(cand, optional["known_blazars"]) if optional["known_blazars"] is not None else {"matched": False}
            known_clagn_match = match_known_clagn(cand, optional["known_clagn"]) if optional["known_clagn"] is not None else {"matched": False}
            gaia_metrics = lookup_gaia_metrics(cand, optional["gaia"]) if optional["gaia"] is not None else None

            stage_a_positive = bool(cand.get("stage_a_positive", False))
            gate_blazar_veto = not bool(blazar_match.get("matched", False))
            gate_wise_qc_pass = (
                int(qa_accounting.get("kept_points_total", 0) or 0) >= min_clean_points_total
                and str(systematics.get("systematics_verdict", "")).upper() != "FAIL"
            )
            season_bins_total = int(systematics.get("season_bins_w1", 0) or 0) + int(systematics.get("season_bins_w2", 0) or 0)
            good_season_bins_total = 0
            if not seasonal.empty and "n_points" in seasonal.columns:
                good_season_bins_total = int((pd.to_numeric(seasonal["n_points"], errors="coerce").fillna(0) >= min_good_points_per_season).sum())
            gate_data_sufficient = (
                int(qa_accounting.get("kept_points_total", 0) or 0) >= 20
                and good_season_bins_total >= 5
                and float(cand.get("baseline_days", 0) or 0) >= 1000.0
            )
            state_change_ratio_value, _ = _compute_state_change_ratio(
                seasonal,
                min_seasons=transient_min_seasons,
                eps=r_eps,
            )
            gate_state_change_ratio_pass = (state_change_ratio_value >= r_min) if np.isfinite(state_change_ratio_value) else False
            if not state_gate_enabled:
                gate_state_change_ratio_pass = True

            transient_features = _compute_transient_spike_features(
                seasonal,
                min_seasons=transient_min_seasons,
            )
            transient_spike_flag = _compute_transient_spike_flag(
                transient_max_nonpeak_ratio=float(transient_features["transient_max_nonpeak_ratio"]),
                transient_max_adjacency_ratio=float(transient_features["transient_max_adjacency_ratio"]),
                transient_min_band_seasons=int(transient_features["transient_min_band_seasons"]),
                peak_frac=transient_peak_frac,
                adjacency_frac=transient_adj_frac,
                min_seasons=transient_min_seasons,
            )
            gate_transient_spike_pass = not transient_spike_flag
            if not transient_gate_enabled:
                gate_transient_spike_pass = True

            fp_mode_hint = _fp_mode_hint(
                klass=str(cand.get("class", "")).strip().lower(),
                fraction_rejected_total=float(qa_accounting.get("fraction_rejected_total")) if pd.notna(qa_accounting.get("fraction_rejected_total")) else np.nan,
                kept_points_total=int(qa_accounting.get("kept_points_total", 0) or 0),
                season_bins_total=good_season_bins_total,
                state_change_ratio_value=float(state_change_ratio_value) if np.isfinite(state_change_ratio_value) else np.nan,
                transient_spike_flag=bool(transient_spike_flag),
                r_min=r_min,
                min_clean_points_total=min_clean_points_total,
                min_season_bins_total=min_season_bins_total,
            )

            final_label, bullets, decision_trace = _final_label(
                cand,
                systematics,
                blazar_match,
                known_clagn_match,
                stage_a_positive=stage_a_positive,
                gate_blazar_veto=gate_blazar_veto,
                gate_wise_qc_pass=gate_wise_qc_pass,
                gate_data_sufficient=gate_data_sufficient,
                gate_state_change_ratio_pass=gate_state_change_ratio_pass,
                gate_transient_spike_pass=gate_transient_spike_pass,
            )
            res_idx = int(cand.get("_results_index"))
            results.loc[res_idx, "gate_blazar_veto"] = gate_blazar_veto
            results.loc[res_idx, "gate_wise_qc_pass"] = gate_wise_qc_pass
            results.loc[res_idx, "gate_data_sufficient"] = gate_data_sufficient
            results.loc[res_idx, "gate_state_change_ratio_pass"] = gate_state_change_ratio_pass
            results.loc[res_idx, "gate_transient_spike_pass"] = gate_transient_spike_pass
            results.loc[res_idx, "state_change_ratio_value"] = state_change_ratio_value
            results.loc[res_idx, "transient_spike_flag"] = bool(transient_spike_flag)
            results.loc[res_idx, "transient_max_nonpeak_ratio"] = transient_features["transient_max_nonpeak_ratio"]
            results.loc[res_idx, "transient_max_adjacency_ratio"] = transient_features["transient_max_adjacency_ratio"]
            results.loc[res_idx, "transient_min_band_seasons"] = transient_features["transient_min_band_seasons"]
            results.loc[res_idx, "kept_points_total"] = qa_accounting.get("kept_points_total")
            results.loc[res_idx, "season_bins_w1"] = systematics.get("season_bins_w1")
            results.loc[res_idx, "season_bins_w2"] = systematics.get("season_bins_w2")
            results.loc[res_idx, "fraction_rejected_total"] = qa_accounting.get("fraction_rejected_total")
            results.loc[res_idx, "fp_mode_hint"] = fp_mode_hint
            results.loc[res_idx, "final_label"] = final_label
            results.loc[res_idx, "decision_trace"] = decision_trace
            rank = int(cand["rank"])
            md_path = paths["cards"] / f"{rank:02d}_{safe_id}.md"
            if md_path.exists() and not config.overwrite:
                print(f"INFO: Skipping {canonical_id} — card already exists (use --overwrite to replace)", file=sys.stderr)
                skipped_count += 1
                continue

            card_ctx = {
                "candidate": cand.to_dict(),
                "qa_accounting": qa_accounting,
                "coverage": coverage,
                "systematics": systematics,
                "seasonal": seasonal,
                "blazar_match": blazar_match,
                "known_clagn_match": known_clagn_match,
                "gaia_metrics": gaia_metrics,
                "figure_relpath": os.path.relpath(fig_path, md_path.parent),
                "final_label": final_label,
                "decision_trace": decision_trace,
                "justification_bullets": bullets,
                "run_timestamp_utc": run_ts,
                "results_file": config.results,
                "wise_file": str(wise_path),
                "score_threshold": config.score_thresh,
                "t_recall": t_recall,
                "stage_a_positive": stage_a_positive,
                "gate_blazar_veto": gate_blazar_veto,
                "gate_wise_qc_pass": gate_wise_qc_pass,
                "gate_data_sufficient": gate_data_sufficient,
                "gate_state_change_ratio_pass": gate_state_change_ratio_pass,
                "gate_transient_spike_pass": gate_transient_spike_pass,
                "state_change_ratio_value": state_change_ratio_value,
                "transient_spike_flag": bool(transient_spike_flag),
                "transient_max_nonpeak_ratio": transient_features["transient_max_nonpeak_ratio"],
                "transient_max_adjacency_ratio": transient_features["transient_max_adjacency_ratio"],
                "transient_min_band_seasons": transient_features["transient_min_band_seasons"],
                "fp_mode_hint": fp_mode_hint,
                "score_version": "1.0",
                "git_commit": git_commit,
                "wise_source_tables": wise_tables,
            }
            write_text_atomic(md_path, render_candidate_card(card_ctx))

            pdf_path = None
            pdf_status = None
            if config.pdf:
                pdf_path = paths["cards"] / f"{rank:02d}_{safe_id}.pdf"
                ok_pdf, msg = _convert_md_to_pdf(md_path, pdf_path)
                pdf_status = msg
                if not ok_pdf:
                    pdf_path = None

            summary_row = {
                "rank": rank,
                "source_id": cand.get("source_id"),
                "canonical_id": cand.get("canonical_id"),
                "source_id_normalized": cand.get("source_id_normalized"),
                "canonical_id_normalized": cand.get("canonical_id_normalized"),
                "score": float(cand.get("score")) if pd.notna(cand.get("score")) else np.nan,
                "status": cand.get("status"),
                "n_points_results": cand.get("n_points"),
                "baseline_days_results": cand.get("baseline_days"),
                "baseline_years_results": cand.get("baseline_years"),
                "band_coverage_results": cand.get("band_coverage"),
                "delta_mag_results": cand.get("delta_mag"),
                "raw_points_total": qa_accounting.get("raw_points_total"),
                "kept_points_total": qa_accounting.get("kept_points_total"),
                "fraction_rejected_total": qa_accounting.get("fraction_rejected_total"),
                "kept_points_w1": qa_accounting.get("kept_points_w1"),
                "kept_points_w2": qa_accounting.get("kept_points_w2"),
                "season_bins_w1": systematics.get("season_bins_w1"),
                "season_bins_w2": systematics.get("season_bins_w2"),
                "has_pre_gap": systematics.get("has_pre_gap"),
                "has_post_gap": systematics.get("has_post_gap"),
                "has_both_sides": systematics.get("has_both_sides"),
                "systematics_verdict": systematics.get("systematics_verdict"),
                "blazar_match": bool(blazar_match.get("matched", False)),
                "known_clagn_match": bool(known_clagn_match.get("matched", False)),
                "stage_a_positive": stage_a_positive,
                "gate_blazar_veto": gate_blazar_veto,
                "gate_wise_qc_pass": gate_wise_qc_pass,
                "gate_data_sufficient": gate_data_sufficient,
                "gate_state_change_ratio_pass": gate_state_change_ratio_pass,
                "gate_transient_spike_pass": gate_transient_spike_pass,
                "state_change_ratio_value": state_change_ratio_value,
                "transient_spike_flag": bool(transient_spike_flag),
                "transient_max_nonpeak_ratio": transient_features["transient_max_nonpeak_ratio"],
                "transient_max_adjacency_ratio": transient_features["transient_max_adjacency_ratio"],
                "transient_min_band_seasons": transient_features["transient_min_band_seasons"],
                "fp_mode_hint": fp_mode_hint,
                "final_label": final_label,
                "decision_trace": decision_trace,
                "coordinate_source": cand.get("coordinate_source"),
                "is_top_clagn_candidate": final_label == "Candidate CLAGN",
                "card_path": str(md_path),
                "figure_path": str(fig_path),
            }
            if pdf_path is not None:
                summary_row["pdf_path"] = str(pdf_path)
            if pdf_status is not None:
                summary_row["pdf_status"] = pdf_status

            summary_rows.append(summary_row)
            selected_cards.append(card_ctx)
            card_index_rows.append({"rank": rank, "canonical_id": cand.get("canonical_id"), "score": cand.get("score"), "final_label": final_label})

        except Exception as exc:
            fail_card = {
                "source_id": source_id,
                "canonical_id": canonical_id,
                "error": str(exc),
                "traceback": _tb.format_exc(),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            fail_path = paths["cards"] / f"{sanitize_filename(source_id)}_FAILED.json"
            try:
                write_text_atomic(fail_path, json.dumps(fail_card, indent=2))
            except Exception:
                pass
            print(f"ERROR: Candidate {canonical_id} failed: {exc}", file=sys.stderr)
            failure_count += 1

    summary_df = _write_summary_table(summary_rows, paths["root"] / "summary_table.csv")
    results_aug_path = paths["root"] / "benchmark_scores_augmented.csv"
    results.sort_values(["score", "canonical_id", "source_id"], ascending=[False, True, True]).to_csv(results_aug_path, index=False, float_format="%.6f")

    status_counts = results["status"].fillna("<NA>").astype(str).value_counts(dropna=False).to_dict()
    class_counts = results["class"].fillna("<NA>").astype(str).value_counts(dropna=False).to_dict() if "class" in results.columns else None
    score_num = pd.to_numeric(results["score"], errors="coerce")
    detections_thr = int(((results["status"] == "OK") & score_num.notna() & (score_num >= config.score_thresh)).sum())
    thresholds = [config.score_thresh, 0.7]
    threshold_metrics = _compute_threshold_metrics(results, thresholds)
    known_clagn_recovery = _known_clagn_recovery(results, optional["known_clagn"], thresholds) if optional["known_clagn"] is not None else None

    qa_sys_summary = {
        "n_selected": int(len(summary_df)),
        "n_systematics_pass": int((summary_df.get("systematics_verdict") == "PASS").sum()) if not summary_df.empty else 0,
        "n_systematics_caution": int((summary_df.get("systematics_verdict") == "CAUTION").sum()) if not summary_df.empty else 0,
        "n_systematics_fail": int((summary_df.get("systematics_verdict") == "FAIL").sum()) if not summary_df.empty else 0,
        "mean_fraction_rejected_total": float(pd.to_numeric(summary_df.get("fraction_rejected_total"), errors="coerce").mean()) if not summary_df.empty else np.nan,
    }
    blazar_summary = {
        "n_selected_blazar_matches": int(summary_df.get("blazar_match", pd.Series(dtype=bool)).fillna(False).sum()) if not summary_df.empty else 0,
        "n_selected_non_blazar": int((~summary_df.get("blazar_match", pd.Series(dtype=bool)).fillna(False)).sum()) if not summary_df.empty else 0,
    }

    selected_report_table = pd.DataFrame(card_index_rows).sort_values(["rank", "canonical_id"]) if card_index_rows else pd.DataFrame(columns=["rank", "canonical_id", "score", "final_label"])
    top_clagn_table = pd.DataFrame(columns=["rank", "canonical_id", "score", "systematics_verdict"])
    if not summary_df.empty:
        top_clagn_table = (
            summary_df.loc[summary_df["final_label"] == "Candidate CLAGN", ["rank", "canonical_id", "score", "systematics_verdict"]]
            .sort_values(["rank", "canonical_id"])
            .reset_index(drop=True)
        )
        top_clagn_table.to_csv(paths["reports"] / "top_clagn_candidates.csv", index=False, float_format="%.6f")

    report_ctx = {
        "dataset_summary": {
            "rows_total": int(len(results)),
            "rows_ok": int((results["status"] == "OK").sum()),
            "topn_requested": int(config.topn),
            "topn_generated": int(len(summary_df)),
            "score_threshold": float(config.score_thresh),
            "t_recall": float(t_recall),
            "detections_above_threshold_ok": detections_thr,
        },
        "status_counts": status_counts,
        "class_counts": class_counts,
        "selected_table": selected_report_table,
        "top_clagn_candidates_table": top_clagn_table,
        "blazar_summary": blazar_summary,
        "qa_systematics_summary": qa_sys_summary,
        "known_clagn_recovery": known_clagn_recovery,
        "threshold_metrics": threshold_metrics,
        "inputs": {
            "results": config.results,
            "wise_dir": config.wise_dir,
            "known_blazars": config.known_blazars if config.known_blazars else ("MISSING_ALLOWED" if config.allow_missing_blazar_catalog else "REQUIRED_MISSING"),
            "known_clagn": optional.get("known_clagn_path") if optional.get("known_clagn") is not None else "not_provided",
            "gaia": config.gaia or "not_provided",
            "benchmark_master": config.benchmark_master or "not_provided",
            "coordinates": config.coordinates or "not_provided",
            "seed": config.seed,
            "policy_config": config.policy_config,
            "strict_mode": strict_mode,
        },
    }
    write_text(paths["reports"] / "benchmark_report.md", render_benchmark_report(report_ctx))

    merge_stats = None
    if config.merge_master:
        summary_for_merge = config.summary_csv or str(paths["root"] / "summary_table.csv")
        merge_stats = merge_master_with_summary(config.master_csv, summary_for_merge, config.master_out)

    manifest = {
        "timestamp_utc": run_ts,
        "score_version": "1.0",
        "git_commit": git_commit,
        "config": asdict(config),
        "policy": policy,
        "inputs": {
            "results": config.results,
            "results_sha256": _file_sha256(config.results),
            "known_blazars": config.known_blazars,
            "known_blazars_sha256": _file_sha256(config.known_blazars) if config.known_blazars and Path(config.known_blazars).exists() else None,
            "known_clagn": optional.get("known_clagn_path") if optional.get("known_clagn") is not None else None,
            "known_clagn_sha256": _file_sha256(optional["known_clagn_path"]) if optional.get("known_clagn") is not None and optional.get("known_clagn_path") and Path(optional["known_clagn_path"]).exists() else None,
        },
        "selected_count": int(len(summary_df)),
        "skipped_count": skipped_count,
        "failure_count": failure_count,
        "results_augmented_csv": str(results_aug_path),
        "missing_wise_candidates": missing_wise,
        "qa_config": qa_cfg,
        "python": sys.version,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "master_merge": merge_stats,
    }
    write_text_atomic(paths["reports"] / "run_manifest.json", json.dumps(manifest, indent=2))

    print(
        f"Generated {len(summary_df)} cards | topn={config.topn} | "
        f"threshold={config.score_thresh:.3f} | t_recall={t_recall:.6f} | outdir={config.outdir} | "
        f"blazar_matches={blazar_summary['n_selected_blazar_matches']} | "
        f"skipped={skipped_count} | failed={failure_count}"
    )
    if merge_stats is not None:
        print(
            f"Merged master rows={merge_stats.get('rows_master')} -> "
            f"{merge_stats.get('out_csv')} | matched={merge_stats.get('matched_rows')} "
            f"unmatched={merge_stats.get('unmatched_rows')}"
        )
    return 0


def main() -> None:
    try:
        cfg = parse_args()
        raise SystemExit(run_pipeline(cfg))
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
