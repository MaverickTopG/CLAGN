from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MJD0 = 51544.5  # J2000 reference for deterministic season bins
SEASON_DAYS = 182.625  # fixed 6-month bins
WISE_GAP_START_MJD = 55340.0
WISE_GAP_END_MJD = 56650.0

W1_COLOR = "#1f77b4"
W2_COLOR = "#d62728"


@dataclass(frozen=True)
class WiseQAConfig:
    min_qual_frame: float = 1.0
    cc_flag_clean_only: bool = True
    moon_lev_reject_min: int = 5
    min_points_per_season: int = 2


def load_wise_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"WISE file not found: {p}")
    try:
        df = pd.read_csv(
            p,
            dtype=str,
            keep_default_na=True,
            na_values=["", " ", "NA", "NaN", "null", "None"],
        )
    except Exception as e:
        raise RuntimeError(f"Failed to read WISE CSV {p}: {e}") from e
    if df.empty:
        raise ValueError(f"WISE file is empty: {p}")
    return df


def _find_first(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _normalize_str_col(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype="object")
    return df[col].astype(str).replace("nan", "").str.strip()


def normalize_wise_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw IRSA-like WISE tables into a long schema.

    Returns columns at least: mjd, band, mag, mag_err, cc_flags, qual_frame, moon_lev,
    source_table, raw_row_index. Extra columns are carried where possible.
    """
    raw = df.copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    raw["raw_row_index"] = np.arange(len(raw), dtype=int)

    # Common shared columns.
    time_col = _find_first(raw, ["mjd", "mjd_obs", "mean_mjd", "w1mjdmean", "w2mjdmean"])
    band_col = _find_first(raw, ["band", "filter", "wise_band"])
    mag_col = _find_first(raw, ["mag", "mpro", "w1mpro_ep", "w1mpro", "w2mpro_ep", "w2mpro"])
    magerr_col = _find_first(raw, ["mag_err", "sigmag", "sigmpro", "w1sigmpro", "w2sigmpro"])

    # Long format path (already has band + mag + time).
    if band_col and time_col and mag_col and band_col.lower() not in {"w1mpro_ep", "w2mpro_ep"}:
        out = pd.DataFrame(
            {
                "mjd": _coerce_numeric(raw[time_col]),
                "band": raw[band_col].astype(str).str.upper().str.strip(),
                "mag": _coerce_numeric(raw[mag_col]),
                "mag_err": _coerce_numeric(raw[magerr_col]) if magerr_col else np.nan,
                "raw_row_index": raw["raw_row_index"],
            }
        )
        for extra in ["cc_flags", "qual_frame", "moon_lev", "moon_masked", "qi_fact", "saa_sep", "ph_qual"]:
            col = _find_first(raw, [extra])
            # Normalize moon_masked into moon_lev field.
            target = "moon_lev" if extra == "moon_masked" else extra
            if target in out.columns and col:
                out[target] = out[target].where(out[target].notna(), raw[col])
            else:
                out[target] = raw[col] if col else out.get(target, pd.NA)
        table_col = _find_first(raw, ["source_table", "table", "catalog", "dataset"])
        out["source_table"] = raw[table_col].astype(str).str.lower().str.strip() if table_col else "unknown"
        return out

    # Wide format path: separate W1/W2 columns; expand into long rows.
    # Prefer epoch mags if present.
    col_mjd = _find_first(raw, ["mjd", "mjd_obs", "mean_mjd"])
    w1_mjd = _find_first(raw, ["w1mjdmean"])
    w2_mjd = _find_first(raw, ["w2mjdmean"])
    # Keep all candidate columns; selection may be row-wise based on source_table.
    w1_mag_ep = _find_first(raw, ["w1mpro_ep"])
    w2_mag_ep = _find_first(raw, ["w2mpro_ep"])
    w1_mag_neo = _find_first(raw, ["w1mpro"])
    w2_mag_neo = _find_first(raw, ["w2mpro"])
    w1_err_ep = _find_first(raw, ["w1sigmpro_ep"])
    w2_err_ep = _find_first(raw, ["w2sigmpro_ep"])
    w1_err_neo = _find_first(raw, ["w1sigmpro"])
    w2_err_neo = _find_first(raw, ["w2sigmpro"])

    if not ((w1_mag_ep or w2_mag_ep or w1_mag_neo or w2_mag_neo) and (col_mjd or w1_mjd or w2_mjd)):
        cols = ", ".join(raw.columns)
        raise ValueError(
            "Unsupported WISE schema. Could not detect time/magnitude columns. "
            f"Found columns: {cols}"
        )

    rows: list[dict[str, Any]] = []
    shared_flag_cols = {}
    for extra in ["cc_flags", "qual_frame", "moon_lev", "moon_masked", "qi_fact", "saa_sep", "ph_qual"]:
        found = _find_first(raw, [extra])
        shared_flag_cols[extra] = found
    table_col = _find_first(raw, ["source_table", "table", "catalog", "dataset"])

    for _, r in raw.iterrows():
        source_table = str(r[table_col]).strip().lower() if table_col else "unknown"
        is_allwise = "allwise" in source_table

        def _pick(ep_col: str | None, neo_col: str | None) -> Any:
            # Row-wise selection fixes mixed AllWISE/NEOWISE cached files.
            if is_allwise:
                if ep_col is not None and pd.notna(r[ep_col]):
                    return r[ep_col]
                if neo_col is not None:
                    return r[neo_col]
                return pd.NA
            if neo_col is not None and pd.notna(r[neo_col]):
                return r[neo_col]
            if ep_col is not None:
                return r[ep_col]
            return pd.NA

        base = {
            "raw_row_index": int(r["raw_row_index"]),
            "cc_flags": r[shared_flag_cols["cc_flags"]] if shared_flag_cols["cc_flags"] else pd.NA,
            "qual_frame": r[shared_flag_cols["qual_frame"]] if shared_flag_cols["qual_frame"] else pd.NA,
            "moon_lev": (
                r[shared_flag_cols["moon_lev"]]
                if shared_flag_cols["moon_lev"]
                else (r[shared_flag_cols["moon_masked"]] if shared_flag_cols["moon_masked"] else pd.NA)
            ),
            "qi_fact": r[shared_flag_cols["qi_fact"]] if shared_flag_cols["qi_fact"] else pd.NA,
            "saa_sep": r[shared_flag_cols["saa_sep"]] if shared_flag_cols["saa_sep"] else pd.NA,
            "ph_qual": r[shared_flag_cols["ph_qual"]] if shared_flag_cols["ph_qual"] else pd.NA,
            "source_table": source_table,
        }
        w1_mag_val = _pick(w1_mag_ep, w1_mag_neo)
        w2_mag_val = _pick(w2_mag_ep, w2_mag_neo)
        w1_err_val = _pick(w1_err_ep, w1_err_neo)
        w2_err_val = _pick(w2_err_ep, w2_err_neo)

        if w1_mag_ep or w1_mag_neo:
            rows.append({
                **base,
                "mjd": r[w1_mjd] if w1_mjd else r[col_mjd] if col_mjd else pd.NA,
                "band": "W1",
                "mag": w1_mag_val,
                "mag_err": w1_err_val,
            })
        if w2_mag_ep or w2_mag_neo:
            rows.append({
                **base,
                "mjd": r[w2_mjd] if w2_mjd else r[col_mjd] if col_mjd else pd.NA,
                "band": "W2",
                "mag": w2_mag_val,
                "mag_err": w2_err_val,
            })

    out = pd.DataFrame(rows)
    out["mjd"] = _coerce_numeric(out["mjd"])
    out["mag"] = _coerce_numeric(out["mag"])
    out["mag_err"] = _coerce_numeric(out["mag_err"])
    out["band"] = out["band"].astype(str).str.upper().str.strip()
    out["qual_frame"] = _coerce_numeric(out["qual_frame"])
    return out


def _cc_flag_bad(cc_flags: Any, band: str) -> bool:
    # If cc_flags are unavailable, do not reject solely on missing metadata.
    if pd.isna(cc_flags):
        return False
    s = str(cc_flags).strip()
    if not s:
        return False
    idx = 0 if band == "W1" else 1
    if len(s) <= idx:
        return False
    c = s[idx]
    return c != "0"


def _moon_flag_bad(moon_lev: Any, band: str, reject_min: int) -> bool:
    if pd.isna(moon_lev):
        return False
    s = str(moon_lev).strip()
    if not s or s.lower() == "nan":
        return False
    idx = 0 if band == "W1" else 1
    c = s[idx] if len(s) > idx else s[0]
    if c.isdigit():
        return int(c) >= reject_min
    return False


def apply_wise_qa_filters(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    cfg = WiseQAConfig(**{k: v for k, v in config.items() if hasattr(WiseQAConfig, k)})
    work = df.copy()
    work["mjd"] = _coerce_numeric(work["mjd"])
    work["mag"] = _coerce_numeric(work["mag"])
    if "mag_err" in work.columns:
        work["mag_err"] = _coerce_numeric(work["mag_err"])
    if "qual_frame" in work.columns:
        work["qual_frame"] = _coerce_numeric(work["qual_frame"])
    work["band"] = work["band"].astype(str).str.upper().str.strip()
    for col in ["cc_flags", "moon_lev", "qi_fact", "saa_sep", "ph_qual", "source_table"]:
        if col not in work.columns:
            work[col] = pd.NA

    n = len(work)
    keep = pd.Series(True, index=work.index)
    step_counts: dict[str, int] = {}
    reject_step = pd.Series(pd.NA, index=work.index, dtype="object")

    def apply_step(name: str, mask_bad: pd.Series) -> None:
        nonlocal keep, reject_step
        mask_bad = mask_bad.fillna(False)
        newly = keep & mask_bad
        step_counts[name] = int(newly.sum())
        reject_step.loc[newly] = name
        keep = keep & ~mask_bad

    apply_step("reject_missing_time", work["mjd"].isna())
    apply_step("reject_missing_mag", work["mag"].isna())
    apply_step("reject_invalid_band", ~work["band"].isin(["W1", "W2"]))

    cc_available = "cc_flags" in work.columns and work["cc_flags"].notna().any()
    if cfg.cc_flag_clean_only and cc_available:
        apply_step(
            "reject_cc_flags",
            work.apply(lambda r: _cc_flag_bad(r.get("cc_flags"), str(r.get("band"))), axis=1),
        )
    else:
        step_counts["reject_cc_flags"] = 0

    if "qual_frame" in work.columns:
        apply_step("reject_qual_frame", work["qual_frame"].notna() & (work["qual_frame"] < cfg.min_qual_frame))
    else:
        step_counts["reject_qual_frame"] = 0

    apply_step(
        "reject_moon_lev",
        work.apply(lambda r: _moon_flag_bad(r.get("moon_lev"), str(r.get("band")), cfg.moon_lev_reject_min), axis=1),
    )

    work["qa_keep"] = keep
    work["qa_reject_step"] = reject_step
    work["qa_reject_reason"] = reject_step.fillna("")

    kept = work[work["qa_keep"]].copy()
    rejected = work[~work["qa_keep"]].copy()

    def _band_count(frame: pd.DataFrame, band: str) -> int:
        if frame.empty:
            return 0
        return int((frame["band"] == band).sum())

    flag_summary: dict[str, Any] = {
        "available_flag_columns": [c for c in ["cc_flags", "qual_frame", "moon_lev", "qi_fact", "saa_sep", "ph_qual"] if c in work.columns],
        "flag_columns_with_values": [c for c in ["cc_flags", "qual_frame", "moon_lev", "qi_fact", "saa_sep", "ph_qual"] if c in work.columns and work[c].notna().any()],
        "cc_flags": _value_freq_summary(work["cc_flags"]),
        "qual_frame": _value_freq_summary(work["qual_frame"]),
        "moon_lev": _value_freq_summary(work["moon_lev"]),
    }
    for extra in ["qi_fact", "saa_sep", "ph_qual"]:
        if extra in work.columns:
            flag_summary[extra] = _value_freq_summary(work[extra])

    accounting = {
        "raw_points_total": int(n),
        "raw_points_w1": _band_count(work, "W1"),
        "raw_points_w2": _band_count(work, "W2"),
        "kept_points_total": int(len(kept)),
        "kept_points_w1": _band_count(kept, "W1"),
        "kept_points_w2": _band_count(kept, "W2"),
        "fraction_rejected_total": float((n - len(kept)) / n) if n else np.nan,
        "reject_missing_time": step_counts.get("reject_missing_time", 0),
        "reject_missing_mag": step_counts.get("reject_missing_mag", 0),
        "reject_invalid_band": step_counts.get("reject_invalid_band", 0),
        "reject_cc_flags": step_counts.get("reject_cc_flags", 0),
        "reject_qual_frame": step_counts.get("reject_qual_frame", 0),
        "reject_moon_lev": step_counts.get("reject_moon_lev", 0),
        "reject_other": 0,
        "reject_reason_counts": {k: int(v) for k, v in step_counts.items()},
        "flag_summary": flag_summary,
    }
    return kept, accounting, rejected


def _value_freq_summary(series: pd.Series, topn: int = 10) -> list[dict[str, Any]]:
    if series is None:
        return []
    s = series.copy()
    s = s.astype(str).replace("nan", "<NA>").fillna("<NA>").str.strip()
    vc = s.value_counts(dropna=False)
    total = int(vc.sum())
    out = []
    for val, cnt in vc.head(topn).items():
        out.append({"value": str(val), "count": int(cnt), "fraction": float(cnt / total) if total else np.nan})
    return out


def assign_season_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["mjd"] = _coerce_numeric(out["mjd"])
    out["season_bin"] = np.floor((out["mjd"] - MJD0) / SEASON_DAYS).astype("Int64")
    return out


def _mad(values: np.ndarray) -> float:
    if values.size == 0:
        return np.nan
    med = np.nanmedian(values)
    return float(np.nanmedian(np.abs(values - med)))


def compute_seasonal_medians(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["band", "season_bin", "season_center_mjd", "median_mag", "scatter_mag", "n_points", "scatter_method"])
    work = assign_season_bins(df)
    rows: list[dict[str, Any]] = []
    for (band, sbin), g in work.groupby(["band", "season_bin"], dropna=True):
        mags = pd.to_numeric(g["mag"], errors="coerce").to_numpy(dtype=float)
        mjds = pd.to_numeric(g["mjd"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(mags) & np.isfinite(mjds)
        if finite.sum() == 0:
            continue
        mags = mags[finite]
        mjds = mjds[finite]
        if mags.size >= 3:
            scatter = _mad(mags)
            method = "MAD"
        elif mags.size >= 2:
            scatter = float(np.nanstd(mags, ddof=1))
            method = "STD"
        else:
            scatter = np.nan
            method = "NA"
        rows.append(
            {
                "band": band,
                "season_bin": int(sbin),
                "season_center_mjd": float(np.nanmedian(mjds)),
                "median_mag": float(np.nanmedian(mags)),
                "scatter_mag": scatter,
                "n_points": int(mags.size),
                "scatter_method": method,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["band", "season_center_mjd", "season_bin"]).reset_index(drop=True)


def compute_coverage_metrics(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> dict:
    def side_flags(frame: pd.DataFrame, band: str) -> tuple[bool, bool]:
        if frame.empty:
            return False, False
        g = frame[frame["band"] == band]
        if g.empty:
            return False, False
        mjd = pd.to_numeric(g["mjd"], errors="coerce")
        pre = bool((mjd < WISE_GAP_START_MJD).fillna(False).any())
        post = bool((mjd > WISE_GAP_END_MJD).fillna(False).any())
        return pre, post

    pre_w1, post_w1 = side_flags(df_clean, "W1")
    pre_w2, post_w2 = side_flags(df_clean, "W2")
    return {
        "has_pre_gap_w1": pre_w1,
        "has_post_gap_w1": post_w1,
        "has_pre_gap_w2": pre_w2,
        "has_post_gap_w2": post_w2,
        "has_pre_gap": bool(pre_w1 or pre_w2),
        "has_post_gap": bool(post_w1 or post_w2),
        "has_both_sides": bool((pre_w1 or pre_w2) and (post_w1 or post_w2)),
        "wise_gap_start_mjd": WISE_GAP_START_MJD,
        "wise_gap_end_mjd": WISE_GAP_END_MJD,
        "raw_rows": int(len(df_raw)),
        "clean_rows": int(len(df_clean)),
    }


def build_systematics_verdict(df_raw: pd.DataFrame, df_clean: pd.DataFrame, seasonal: pd.DataFrame) -> dict:
    raw_n = int(len(df_raw))
    clean_n = int(len(df_clean))
    frac_rej = float((raw_n - clean_n) / raw_n) if raw_n else np.nan
    coverage = compute_coverage_metrics(df_raw, df_clean)

    season_bins_w1 = int((seasonal["band"] == "W1").sum()) if not seasonal.empty else 0
    season_bins_w2 = int((seasonal["band"] == "W2").sum()) if not seasonal.empty else 0
    reasons: list[str] = []

    verdict = "PASS"
    if clean_n < 10 or (season_bins_w1 + season_bins_w2) < 3:
        verdict = "FAIL"
        reasons.append("Too few QA-clean points / seasonal bins for robust variability assessment")
    if raw_n > 0 and frac_rej > 0.5:
        if verdict != "FAIL":
            verdict = "CAUTION"
        reasons.append(f"High QA rejection fraction ({frac_rej:.1%})")
    if not coverage["has_both_sides"]:
        if verdict == "PASS":
            verdict = "CAUTION"
        reasons.append("No clean coverage on both sides of WISE hibernation gap")
    if season_bins_w1 == 0 or season_bins_w2 == 0:
        if verdict == "PASS":
            verdict = "CAUTION"
        reasons.append("Only one band has usable seasonal support")

    if not reasons:
        reasons.append("Seasonal trend supported by sufficient QA-clean W1/W2 points")

    summary = {
        "PASS": "QA-clean points and seasonal coverage support the observed variability signal.",
        "CAUTION": "Variability signal is present but data quality/coverage limitations weaken confidence.",
        "FAIL": "Data quality and/or sampling is insufficient; signal may be dominated by systematics.",
    }[verdict]

    return {
        "systematics_verdict": verdict,
        "systematics_summary": summary,
        "justifications": reasons,
        "fraction_rejected_total": frac_rej,
        "season_bins_w1": season_bins_w1,
        "season_bins_w2": season_bins_w2,
        **coverage,
    }


def plot_wise_lightcurve(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    seasonal: pd.DataFrame,
    out_png: str,
    title: str,
) -> None:
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)

    def _plot_points(frame: pd.DataFrame, band: str, kept: bool) -> None:
        if frame.empty:
            return
        g = frame[frame["band"] == band].copy()
        if g.empty:
            return
        g["mjd"] = pd.to_numeric(g["mjd"], errors="coerce")
        g["mag"] = pd.to_numeric(g["mag"], errors="coerce")
        g = g[np.isfinite(g["mjd"]) & np.isfinite(g["mag"])]
        if g.empty:
            return
        g = g.sort_values("mjd")
        color = W1_COLOR if band == "W1" else W2_COLOR
        if kept:
            ax.scatter(g["mjd"], g["mag"], s=10, alpha=0.55, color=color, label=f"{band} kept ({len(g)})")
        else:
            ax.scatter(g["mjd"], g["mag"], s=12, alpha=0.25, color=color, marker="x", label=f"{band} rejected ({len(g)})")

    rejected = df_raw.copy()
    if "qa_keep" in df_raw.columns:
        rejected = df_raw[~df_raw["qa_keep"].fillna(False)]

    _plot_points(df_clean, "W1", kept=True)
    _plot_points(df_clean, "W2", kept=True)
    _plot_points(rejected, "W1", kept=False)
    _plot_points(rejected, "W2", kept=False)

    if not seasonal.empty:
        for band, color in [("W1", W1_COLOR), ("W2", W2_COLOR)]:
            s = seasonal[seasonal["band"] == band].copy()
            if s.empty:
                continue
            s = s.sort_values("season_center_mjd")
            yerr = pd.to_numeric(s["scatter_mag"], errors="coerce").to_numpy(dtype=float)
            yerr = np.where(np.isfinite(yerr), np.maximum(yerr, 0.0), np.nan)
            ax.errorbar(
                s["season_center_mjd"],
                s["median_mag"],
                yerr=yerr,
                fmt="o-",
                color=color,
                linewidth=1.5,
                markersize=4,
                capsize=2,
                label=f"{band} seasonal medians ({len(s)})",
            )

    ax.axvspan(WISE_GAP_START_MJD, WISE_GAP_END_MJD, color="gray", alpha=0.15, label="WISE hibernation gap")
    ax.set_xlabel("MJD")
    ax.set_ylabel("Magnitude")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.invert_yaxis()
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        uniq = []
        seen = set()
        for h, l in zip(handles, labels):
            if l in seen:
                continue
            seen.add(l)
            uniq.append((h, l))
        ax.legend([h for h, _ in uniq], [l for _, l in uniq], fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
