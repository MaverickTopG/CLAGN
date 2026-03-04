from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


def sanitize_filename(text: str) -> str:
    s = str(text).strip()
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
    s = re.sub(r"\s+", "_", s)
    s = s[:120] if len(s) > 120 else s
    return s or "unknown"


def write_text_atomic(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    """Write *text* to *path* atomically via a temp-file-then-rename."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".tmp_")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
        os.replace(tmp, p)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_text(path: str | Path, text: str) -> None:
    write_text_atomic(path, text)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "(none)"
    work = df.copy()
    if max_rows is not None:
        work = work.head(max_rows)
    cols = [str(c) for c in work.columns]
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in work.iterrows():
        vals = []
        for c in work.columns:
            v = row[c]
            if pd.isna(v):
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:.6g}")
            else:
                vals.append(str(v).replace("\n", " "))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _fmt(v: Any, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "NA"
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, (int,)):
        return str(v)
    try:
        fv = float(v)
        return f"{fv:.{digits}f}"
    except Exception:
        return str(v)


def _flag_summary_to_table(flag_summary: dict[str, Any], field: str, topn: int = 8) -> pd.DataFrame:
    rows = flag_summary.get(field, []) if isinstance(flag_summary, dict) else []
    if not rows:
        return pd.DataFrame(columns=["value", "count", "fraction"])
    return pd.DataFrame(rows).head(topn)


def render_candidate_card(card_ctx: dict) -> str:
    cand = card_ctx["candidate"]
    qa = card_ctx["qa_accounting"]
    sysv = card_ctx["systematics"]
    cov = card_ctx["coverage"]
    seasonal = card_ctx.get("seasonal", pd.DataFrame())
    blazar = card_ctx.get("blazar_match", {})
    known_clagn = card_ctx.get("known_clagn_match", {})
    gaia = card_ctx.get("gaia_metrics")
    fig_rel = card_ctx["figure_relpath"]
    final_label = card_ctx["final_label"]
    just = card_ctx["justification_bullets"]

    seasonal_counts = {
        "W1": int((seasonal["band"] == "W1").sum()) if isinstance(seasonal, pd.DataFrame) and not seasonal.empty else 0,
        "W2": int((seasonal["band"] == "W2").sum()) if isinstance(seasonal, pd.DataFrame) and not seasonal.empty else 0,
    }

    qa_table = pd.DataFrame(
        [
            ["Raw points total", qa.get("raw_points_total")],
            ["Raw points W1", qa.get("raw_points_w1")],
            ["Raw points W2", qa.get("raw_points_w2")],
            ["Kept points total", qa.get("kept_points_total")],
            ["Kept points W1", qa.get("kept_points_w1")],
            ["Kept points W2", qa.get("kept_points_w2")],
            ["Fraction rejected", qa.get("fraction_rejected_total")],
            ["Reject: missing time", qa.get("reject_missing_time")],
            ["Reject: missing mag", qa.get("reject_missing_mag")],
            ["Reject: invalid band", qa.get("reject_invalid_band")],
            ["Reject: cc_flags", qa.get("reject_cc_flags")],
            ["Reject: qual_frame", qa.get("reject_qual_frame")],
            ["Reject: moon_lev", qa.get("reject_moon_lev")],
        ],
        columns=["Metric", "Value"],
    )

    lines = []
    lines.append(f"# Candidate Card: {cand.get('canonical_id')} (Rank {cand.get('rank')})")
    lines.append("")
    lines.append("## Basic Metadata")
    lines.append("")
    lines.append(f"- `source_id`: `{cand.get('source_id')}`")
    lines.append(f"- `canonical_id`: `{cand.get('canonical_id')}`")
    if cand.get("source_id_normalized") and str(cand.get("source_id_normalized")) != str(cand.get("source_id")):
        lines.append(f"- `source_id_normalized`: `{cand.get('source_id_normalized')}`")
    if cand.get("canonical_id_normalized") and str(cand.get("canonical_id_normalized")) != str(cand.get("canonical_id")):
        lines.append(f"- `canonical_id_normalized`: `{cand.get('canonical_id_normalized')}`")
    lines.append(f"- `score`: `{_fmt(cand.get('score'), 6)}`")
    lines.append(f"- `status`: `{cand.get('status')}`")
    lines.append(f"- `final_label`: `{final_label}`")
    lines.append(f"- `stage_a_positive`: `{card_ctx.get('stage_a_positive')}`")
    lines.append(f"- `gate_blazar_veto`: `{card_ctx.get('gate_blazar_veto')}`")
    lines.append(f"- `gate_wise_qc_pass`: `{card_ctx.get('gate_wise_qc_pass')}`")
    lines.append(f"- `gate_data_sufficient`: `{card_ctx.get('gate_data_sufficient')}`")
    lines.append(f"- `decision_trace`: `{card_ctx.get('decision_trace', '')}`")
    lines.append("")
    lines.append("## Sampling / Variability Summary")
    lines.append("")
    lines.append(f"- `n_points` (results table): `{cand.get('n_points')}`")
    lines.append(f"- `baseline_days`: `{_fmt(cand.get('baseline_days'), 2)}`")
    lines.append(f"- `baseline_years`: `{_fmt(cand.get('baseline_years'), 3)}`")
    lines.append(f"- `band_coverage`: `{cand.get('band_coverage')}`")
    lines.append(f"- `delta_mag`: `{_fmt(cand.get('delta_mag'), 3)}`")
    lines.append(f"- `delta_mag_method`: `{cand.get('delta_mag_method')}`")
    lines.append("")
    lines.append("## Coordinates")
    lines.append("")
    lines.append(f"- `ra`: `{_fmt(cand.get('ra'), 6)}`")
    lines.append(f"- `dec`: `{_fmt(cand.get('dec'), 6)}`")
    lines.append(f"- `coordinate_source`: `{cand.get('coordinate_source', 'unknown')}`")
    lines.append("")
    lines.append("## WISE Light Curve")
    lines.append("")
    lines.append(f"![WISE light curve]({fig_rel})")
    lines.append("")
    lines.append("## WISE QA / Rejection Accounting")
    lines.append("")
    lines.append(markdown_table(qa_table))
    lines.append("")
    reason_counts = qa.get("reject_reason_counts", {})
    if reason_counts:
        rc_df = pd.DataFrame(sorted(reason_counts.items()), columns=["reject_reason", "count"])
        lines.append("### QA Reject Reason Counts")
        lines.append("")
        lines.append(markdown_table(rc_df))
        lines.append("")
    lines.append("## Flag Summary")
    lines.append("")
    flag_summary = qa.get("flag_summary", {})
    for fld in ["cc_flags", "qual_frame", "moon_lev", "qi_fact", "saa_sep", "ph_qual"]:
        tbl = _flag_summary_to_table(flag_summary, fld)
        if tbl.empty:
            continue
        lines.append(f"### `{fld}`")
        lines.append("")
        lines.append(markdown_table(tbl))
        lines.append("")

    lines.append("## Coverage Summary")
    lines.append("")
    lines.append(f"- Seasonal bins (W1): `{seasonal_counts['W1']}`")
    lines.append(f"- Seasonal bins (W2): `{seasonal_counts['W2']}`")
    lines.append(f"- Pre-gap coverage: `{cov.get('has_pre_gap')}`")
    lines.append(f"- Post-gap coverage: `{cov.get('has_post_gap')}`")
    lines.append(f"- Both-sides coverage: `{cov.get('has_both_sides')}`")
    lines.append("")

    lines.append("## Systematics Verdict")
    lines.append("")
    lines.append(f"- Verdict: `{sysv.get('systematics_verdict')}`")
    lines.append(f"- Summary: {sysv.get('systematics_summary')}")
    lines.append("- QA-dominated signal check: " + ("Potentially dominated by flagged/low-quality points" if sysv.get("systematics_verdict") == "FAIL" else "Not obviously dominated by flagged/low-quality points"))
    lines.append("")
    lines.append("Reasons:")
    for r in sysv.get("justifications", []):
        lines.append(f"- {r}")
    lines.append("")

    lines.append("## Catalog Checks")
    lines.append("")
    if blazar.get("matched"):
        lines.append(f"- Known blazar match: `YES` via `{blazar.get('match_field_catalog')}` token `{blazar.get('matched_token')}`")
    else:
        lines.append("- Known blazar match: `NO`")
    if known_clagn.get("matched"):
        lines.append(f"- Known CLAGN match: `YES` (method={known_clagn.get('method', 'identifier')})")
    else:
        lines.append("- Known CLAGN match: `NO`")
    if gaia:
        lines.append("- Gaia metrics: available")
        preview = {k: v for k, v in gaia.items() if k not in {"source_id", "canonical_id"}}
        if preview:
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(preview, indent=2, default=str)[:2000])
            lines.append("```")
    else:
        lines.append("- Gaia metrics: not provided / no match")
    lines.append("")

    lines.append("## Final Label")
    lines.append("")
    lines.append(f"**{final_label}**")
    lines.append("")
    for b in just:
        lines.append(f"- {b}")
    lines.append("")

    lines.append("## Reproducibility")
    lines.append("")
    lines.append(f"- `run_timestamp_utc`: `{card_ctx.get('run_timestamp_utc')}`")
    lines.append(f"- `results_file`: `{card_ctx.get('results_file')}`")
    lines.append(f"- `wise_file`: `{card_ctx.get('wise_file')}`")
    lines.append(f"- `score_threshold`: `{card_ctx.get('score_threshold')}`")
    lines.append(f"- `t_recall`: `{card_ctx.get('t_recall')}`")
    lines.append("")
    return "\n".join(lines).replace("```\n```", "")


def render_benchmark_report(report_ctx: dict) -> str:
    lines = []
    ds = report_ctx["dataset_summary"]
    lines.append("# Candidate Card Benchmark Report")
    lines.append("")
    lines.append("## Dataset Summary")
    lines.append("")
    for k in [
        "rows_total",
        "rows_ok",
        "topn_requested",
        "topn_generated",
        "score_threshold",
        "detections_above_threshold_ok",
    ]:
        if k in ds:
            lines.append(f"- `{k}`: `{ds[k]}`")
    lines.append("")

    if report_ctx.get("status_counts") is not None:
        lines.append("## Status Counts")
        lines.append("")
        sc = pd.DataFrame(sorted(report_ctx["status_counts"].items()), columns=["status", "count"])
        lines.append(markdown_table(sc))
        lines.append("")

    if report_ctx.get("class_counts"):
        lines.append("## Class Counts")
        lines.append("")
        cc = pd.DataFrame(sorted(report_ctx["class_counts"].items()), columns=["class", "count"])
        lines.append(markdown_table(cc))
        lines.append("")

    lines.append("## Selected Candidates")
    lines.append("")
    lines.append(markdown_table(report_ctx.get("selected_table", pd.DataFrame())))
    lines.append("")

    tcl = report_ctx.get("top_clagn_candidates_table")
    if isinstance(tcl, pd.DataFrame):
        lines.append("## Top CLAGN Candidates (Rule-Based Final Label)")
        lines.append("")
        if tcl.empty:
            lines.append("(none in current top-N selection)")
        else:
            lines.append(markdown_table(tcl))
        lines.append("")

    lines.append("## Blazar Match Summary")
    lines.append("")
    bsum = report_ctx.get("blazar_summary", {})
    for k, v in bsum.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")

    lines.append("## QA / Systematics Summary (Selected)")
    lines.append("")
    qsum = report_ctx.get("qa_systematics_summary", {})
    for k, v in qsum.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")

    kcr = report_ctx.get("known_clagn_recovery")
    if kcr:
        lines.append("## Known CLAGN Recovery")
        lines.append("")
        for k, v in kcr.items():
            if k == "threshold_table":
                continue
            lines.append(f"- `{k}`: `{v}`")
        if isinstance(kcr.get("threshold_table"), pd.DataFrame):
            lines.append("")
            lines.append(markdown_table(kcr["threshold_table"]))
        lines.append("")

    tmetrics = report_ctx.get("threshold_metrics")
    if isinstance(tmetrics, pd.DataFrame) and not tmetrics.empty:
        lines.append("## Threshold Metrics")
        lines.append("")
        lines.append(markdown_table(tmetrics))
        lines.append("")

    lines.append("## Inputs / Reproducibility")
    lines.append("")
    for k, v in report_ctx.get("inputs", {}).items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("See `reports/run_manifest.json` for exact configuration and file hashes.")
    lines.append("")
    return "\n".join(lines)
