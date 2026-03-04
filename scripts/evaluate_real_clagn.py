"""
Evaluate real CLAGN benchmark with stage-wise metrics and strict imbalance-aware reporting.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clagn.utils.benchmark_labels import ensure_y_true, normalized_class_series
from clagn.utils.id_canonicalization import add_canonical_id, best_join_key


def _load_policy_config(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Policy config not found: {path}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("Policy config must be a mapping")
    return data


def _load_scores(scores_path: Path) -> pd.DataFrame:
    df = pd.read_csv(scores_path)
    required = {"source_id", "score"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"scores_csv missing required columns: {missing}")
    if "canonical_id" not in df.columns:
        df = add_canonical_id(df)
    if "status" not in df.columns:
        df["status"] = "OK"
    return df


def _strict_schema_checks(df: pd.DataFrame) -> None:
    required = {"source_id", "canonical_id", "score", "status", "y_true", "class"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns after merge: {missing}")
    if df["canonical_id"].duplicated().any():
        raise ValueError("Duplicate canonical_id values found after merge")
    score = pd.to_numeric(df["score"], errors="coerce")
    bad = df["status"].astype(str).str.upper().eq("OK") & ~np.isfinite(score)
    if bad.any():
        raise ValueError(f"Found {int(bad.sum())} status=OK rows with non-finite scores")
    valid_split = {"dev", "test"}
    if "split" in df.columns:
        vals = df["split"].dropna().astype(str).str.lower()
        invalid = sorted(vals[~vals.isin(valid_split)].unique().tolist())
        if invalid:
            raise ValueError(f"Invalid split values found: {invalid}")


def _confusion_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    bal = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
    total = tp + fp + fn + tn
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(prec),
        "recall": float(rec),
        "balanced_accuracy": float(bal),
        "mcc": float(mcc),
        "overall_accuracy": float((tp + tn) / total) if total > 0 else 0.0,
    }


def _class_contamination(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows = []
    for klass, g in df.groupby("class", dropna=False):
        y = (g["y_true"].astype(int) == 0).astype(int)
        pred_pos = g[pred_col].astype(bool)
        denom = int(y.sum())
        fp = int(((y == 1) & pred_pos).sum())
        rows.append(
            {
                "class": str(klass),
                "n_rows": int(len(g)),
                "n_negatives": denom,
                "predicted_positive_on_negative": fp,
                "fpr_within_class": float(fp / denom) if denom else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["class"]).reset_index(drop=True)


def _recall_at_k(df: pd.DataFrame, ks: list[int]) -> pd.DataFrame:
    rows = []
    ranked = df.sort_values(["score", "canonical_id"], ascending=[False, True]).reset_index(drop=True)
    total_pos = int((ranked["y_true"] == 1).sum())
    for k in ks:
        top = ranked.head(k)
        tp = int((top["y_true"] == 1).sum())
        rows.append(
            {
                "k": int(k),
                "tp_in_top_k": tp,
                "total_known_clagn": total_pos,
                "recall_at_k": float(tp / total_pos) if total_pos else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _precision_at_k(df: pd.DataFrame, ks: list[int]) -> pd.DataFrame:
    rows = []
    ranked = df.sort_values(["score", "canonical_id"], ascending=[False, True]).reset_index(drop=True)
    for k in ks:
        top = ranked.head(k)
        tp = int((top["y_true"] == 1).sum())
        rows.append({"k": k, "tp_in_top_k": tp, "precision_at_k": float(tp / k) if k > 0 else np.nan})
    return pd.DataFrame(rows)


def _reliability_bins(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    work = df.copy()
    work["score"] = pd.to_numeric(work["score"], errors="coerce")
    work = work[np.isfinite(work["score"])]
    if work.empty:
        return pd.DataFrame(columns=["bin", "n", "avg_score", "empirical_positive_rate"])
    work["bin"] = pd.cut(work["score"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True, labels=False)
    rows = []
    for b, g in work.groupby("bin", dropna=False):
        rows.append(
            {
                "bin": int(b),
                "n": int(len(g)),
                "avg_score": float(g["score"].mean()),
                "empirical_positive_rate": float(g["y_true"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("bin").reset_index(drop=True)


def _synthetic_shift(df: pd.DataFrame, shifts: list[float], rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    pos = df[df["y_true"] == 1]
    neg = df[df["y_true"] == 0]
    if pos.empty or neg.empty:
        return pd.DataFrame(columns=["target_positive_rate", "sample_size", "actual_positive_rate", "brier"])
    n = min(len(df), 5000)
    for target in shifts:
        n_pos = max(1, int(round(n * float(target))))
        n_neg = max(1, n - n_pos)
        sp = pos.sample(n=min(n_pos, len(pos)), replace=n_pos > len(pos), random_state=int(rng.integers(0, 1_000_000)))
        sn = neg.sample(n=min(n_neg, len(neg)), replace=n_neg > len(neg), random_state=int(rng.integers(0, 1_000_000)))
        mix = pd.concat([sp, sn], ignore_index=True)
        brier = brier_score_loss(mix["y_true"].astype(int), pd.to_numeric(mix["score"], errors="coerce"))
        rows.append(
            {
                "target_positive_rate": float(target),
                "sample_size": int(len(mix)),
                "actual_positive_rate": float(mix["y_true"].mean()),
                "brier": float(brier),
            }
        )
    return pd.DataFrame(rows)


def _fp_refinement_tables(eval_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return three additive FP refinement tables:
    1) by-class FP counts
    2) by-class quantiles for key gate features
    3) by-class gate-pass matrix over FP rows
    """
    fp_mask = (eval_df["y_true"].astype(int) == 0) & eval_df["pred_after_all_gates"].astype(bool)
    fps = eval_df.loc[fp_mask].copy()
    if fps.empty:
        return (
            pd.DataFrame(columns=["class", "fp_count"]),
            pd.DataFrame(columns=["class", "feature", "q25", "q50", "q75", "min", "max"]),
            pd.DataFrame(columns=["class", "gate", "pass_rate"]),
        )

    counts = (
        fps["class"]
        .astype(str)
        .value_counts()
        .rename_axis("class")
        .reset_index(name="fp_count")
        .sort_values(["fp_count", "class"], ascending=[False, True])
    )

    quant_rows: list[dict[str, Any]] = []
    for cls, g in fps.groupby("class", dropna=False):
        for feat in ["state_change_ratio_value", "transient_max_nonpeak_ratio", "transient_max_adjacency_ratio"]:
            if feat not in g.columns:
                continue
            s = pd.to_numeric(g[feat], errors="coerce").dropna()
            if s.empty:
                continue
            quant_rows.append(
                {
                    "class": str(cls),
                    "feature": feat,
                    "q25": float(s.quantile(0.25)),
                    "q50": float(s.quantile(0.50)),
                    "q75": float(s.quantile(0.75)),
                    "min": float(s.min()),
                    "max": float(s.max()),
                }
            )
    quants = pd.DataFrame(quant_rows)
    if not quants.empty:
        quants = quants.sort_values(["class", "feature"]).reset_index(drop=True)

    gate_cols = [
        c for c in [
            "gate_blazar_veto",
            "gate_wise_qc_pass",
            "gate_data_sufficient",
            "gate_state_change_ratio_pass",
            "gate_transient_spike_pass",
        ]
        if c in fps.columns
    ]
    gate_rows: list[dict[str, Any]] = []
    for cls, g in fps.groupby("class", dropna=False):
        for gc in gate_cols:
            pass_rate = pd.to_numeric(g[gc].astype(int), errors="coerce").mean()
            gate_rows.append({"class": str(cls), "gate": gc, "pass_rate": float(pass_rate)})
    gate_matrix = pd.DataFrame(gate_rows)
    if not gate_matrix.empty:
        gate_matrix = gate_matrix.sort_values(["class", "gate"]).reset_index(drop=True)

    return counts, quants, gate_matrix


def _blazar_fpr(eval_df: pd.DataFrame, pred_col: str) -> float:
    blazar = eval_df[eval_df["class"].astype(str).str.lower() == "blazar"]
    if blazar.empty:
        return np.nan
    denom = int((blazar["y_true"] == 0).sum())
    if denom == 0:
        return np.nan
    fp = int(((blazar["y_true"] == 0) & eval_df.loc[blazar.index, pred_col].astype(bool)).sum())
    return float(fp / denom)


def _class_fpr(eval_df: pd.DataFrame, pred_col: str, class_name: str) -> float:
    subset = eval_df[eval_df["class"].astype(str).str.lower() == str(class_name).lower()]
    if subset.empty:
        return np.nan
    denom = int((subset["y_true"] == 0).sum())
    if denom == 0:
        return np.nan
    fp = int(((subset["y_true"] == 0) & subset[pred_col].astype(bool)).sum())
    return float(fp / denom)


def _confusion_frame(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return pd.DataFrame([{"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}])


_DEDUP_DONE: set[int] = set()


def _dedup_eval_df(df: pd.DataFrame, key: str = "canonical_id") -> pd.DataFrame:
    """Drop duplicates on *key*, with an id-based guard to avoid double-dedup."""
    tag = id(df)
    if tag in _DEDUP_DONE:
        return df
    out = df.drop_duplicates(subset=[key], keep="first")
    _DEDUP_DONE.add(tag)
    return out


def _apply_tuned_prediction(
    eval_df: pd.DataFrame,
    r_min: float,
    peak_frac: float,
    adjacency_frac: float,
    min_clean_points_total: int,
    min_season_bins_total: int,
) -> pd.Series:
    out = eval_df["stage_a_positive"].astype(bool).copy()
    out &= eval_df["gate_blazar_veto"].astype(bool)

    # Re-evaluate QA/data-sufficiency from raw feature columns
    # so threshold tuning is not blocked by pre-computed booleans.
    kept = pd.to_numeric(eval_df["kept_points_total"], errors="coerce").fillna(-1) \
        if "kept_points_total" in eval_df.columns else pd.Series(-1, index=eval_df.index)
    season_bins = pd.to_numeric(eval_df["season_bins_total"], errors="coerce").fillna(-1) \
        if "season_bins_total" in eval_df.columns else pd.Series(-1, index=eval_df.index)
    out &= kept >= min_clean_points_total
    out &= season_bins >= min_season_bins_total

    # Gate A: state-change ratio.
    state_ratio = pd.to_numeric(eval_df["state_change_ratio_value"], errors="coerce") if "state_change_ratio_value" in eval_df.columns else pd.Series(np.nan, index=eval_df.index)
    out &= state_ratio >= r_min

    # Gate B: transient spike veto with thresholded feature ratios.
    max_nonpeak = pd.to_numeric(eval_df["transient_max_nonpeak_ratio"], errors="coerce") if "transient_max_nonpeak_ratio" in eval_df.columns else pd.Series(np.nan, index=eval_df.index)
    max_adj = pd.to_numeric(eval_df["transient_max_adjacency_ratio"], errors="coerce") if "transient_max_adjacency_ratio" in eval_df.columns else pd.Series(np.nan, index=eval_df.index)
    min_band_seasons = pd.to_numeric(eval_df["transient_min_band_seasons"], errors="coerce").fillna(0) if "transient_min_band_seasons" in eval_df.columns else pd.Series(0, index=eval_df.index)
    transient_flag = (max_nonpeak < peak_frac) & (max_adj < adjacency_frac) & (min_band_seasons >= min_season_bins_total)
    transient_flag = transient_flag.fillna(False)
    out &= ~transient_flag.astype(bool)
    return out


def _run_dev_tuning(
    eval_df: pd.DataFrame,
    policy: dict[str, Any],
    recall_target: float,
    blazar_target: float,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    tune = policy.get("tuning", {})
    r_min_grid = [float(v) for v in tune.get("r_min_grid", [1.5, 2.0, 2.5, 3.0])]
    peak_frac_grid = [float(v) for v in tune.get("peak_frac_grid", [0.5, 0.6, 0.7])]
    adjacency_frac_grid = [float(v) for v in tune.get("adjacency_frac_grid", [0.3, 0.4, 0.5])]
    min_clean_grid = [int(v) for v in tune.get("min_clean_points_total_grid", [10, 12, 14])]
    min_season_grid = [int(v) for v in tune.get("min_season_bins_total_grid", [4, 5, 6])]
    use_cv = bool(tune.get("cross_validate", False))
    n_splits = int(tune.get("cv_n_splits", 5))
    class_obj_cfg = tune.get("class_fp_objective", {}) if isinstance(tune.get("class_fp_objective", {}), dict) else {}
    class_obj_enabled = bool(class_obj_cfg.get("enabled", False))
    class_weights_cfg = class_obj_cfg.get("weights", {}) if isinstance(class_obj_cfg.get("weights", {}), dict) else {}
    w_normal_agn = float(class_weights_cfg.get("normal_agn", 0.0))
    w_star = float(class_weights_cfg.get("star", 0.0))
    w_sn = float(class_weights_cfg.get("sn", 0.0))
    class_caps_cfg = tune.get("class_fp_caps", {}) if isinstance(tune.get("class_fp_caps", {}), dict) else {}
    cap_normal_agn = class_caps_cfg.get("normal_agn", None)
    cap_star = class_caps_cfg.get("star", None)

    needed = {
        "state_change_ratio_value",
        "transient_max_nonpeak_ratio",
        "transient_max_adjacency_ratio",
        "transient_min_band_seasons",
        "kept_points_total",
        "season_bins_total",
    }
    if not needed.issubset(eval_df.columns):
        return pd.DataFrame(), None

    y_true = eval_df["y_true"].astype(int).to_numpy()

    # Build CV folds once (only when cross_validate: true in policy).
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    if use_cv:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        folds = list(skf.split(np.zeros(len(eval_df)), y_true))

    rows: list[dict[str, Any]] = []
    for r_min, peak_frac, adjacency_frac, min_clean_points_total, min_season_bins_total in itertools.product(
        r_min_grid, peak_frac_grid, adjacency_frac_grid, min_clean_grid, min_season_grid
    ):
        # Cross-validated balanced accuracy (mean across folds).
        cv_balanced_accuracy = np.nan
        if use_cv and folds:
            fold_scores: list[float] = []
            for train_idx, val_idx in folds:
                val_df = eval_df.iloc[val_idx]
                pred_val = _apply_tuned_prediction(
                    eval_df=val_df,
                    r_min=r_min,
                    peak_frac=peak_frac,
                    adjacency_frac=adjacency_frac,
                    min_clean_points_total=min_clean_points_total,
                    min_season_bins_total=min_season_bins_total,
                )
                fold_stats = _confusion_stats(y_true[val_idx], pred_val.astype(int).to_numpy())
                fold_scores.append(fold_stats["balanced_accuracy"])
            cv_balanced_accuracy = float(np.mean(fold_scores))

        # Full-dataset stats (always computed for reporting).
        pred = _apply_tuned_prediction(
            eval_df=eval_df,
            r_min=r_min,
            peak_frac=peak_frac,
            adjacency_frac=adjacency_frac,
            min_clean_points_total=min_clean_points_total,
            min_season_bins_total=min_season_bins_total,
        )
        stats = _confusion_stats(y_true, pred.astype(int).to_numpy())
        eval_pred = eval_df.assign(_pred=pred)
        blazar_fpr = _blazar_fpr(eval_pred, "_pred")
        fpr_normal_agn = _class_fpr(eval_pred, "_pred", "normal_agn")
        fpr_star = _class_fpr(eval_pred, "_pred", "star")
        fpr_sn = _class_fpr(eval_pred, "_pred", "sn")
        weighted_class_fp = np.nan
        if class_obj_enabled:
            weighted_class_fp = (
                (w_normal_agn * (fpr_normal_agn if np.isfinite(fpr_normal_agn) else 0.0))
                + (w_star * (fpr_star if np.isfinite(fpr_star) else 0.0))
                + (w_sn * (fpr_sn if np.isfinite(fpr_sn) else 0.0))
            )
        meets_caps = True
        if cap_normal_agn is not None and np.isfinite(fpr_normal_agn):
            meets_caps = meets_caps and (fpr_normal_agn <= float(cap_normal_agn))
        if cap_star is not None and np.isfinite(fpr_star):
            meets_caps = meets_caps and (fpr_star <= float(cap_star))
        row = {
            "r_min": r_min,
            "peak_frac": peak_frac,
            "adjacency_frac": adjacency_frac,
            "min_clean_points_total": int(min_clean_points_total),
            "min_season_bins_total": int(min_season_bins_total),
            **stats,
            "cv_balanced_accuracy": cv_balanced_accuracy,
            "blazar_fpr": blazar_fpr,
            "fpr_normal_agn": fpr_normal_agn,
            "fpr_star": fpr_star,
            "fpr_sn": fpr_sn,
            "weighted_class_fp": weighted_class_fp,
            "meets_recall": bool(stats["recall"] >= recall_target),
            "meets_blazar": bool(np.isfinite(blazar_fpr) and blazar_fpr <= blazar_target),
            "meets_class_caps": bool(meets_caps),
        }
        rows.append(row)
    grid = pd.DataFrame(rows)
    if grid.empty:
        return grid, None

    # Select best by CV score when available, otherwise by full-dataset balanced accuracy.
    sort_primary = "cv_balanced_accuracy" if (use_cv and grid["cv_balanced_accuracy"].notna().any()) else "balanced_accuracy"
    grid["weighted_class_fp_sort"] = pd.to_numeric(grid["weighted_class_fp"], errors="coerce").fillna(np.inf)
    eligible = grid[(grid["meets_recall"]) & (grid["meets_blazar"]) & (grid["meets_class_caps"])].copy()
    if eligible.empty:
        best = grid.sort_values(
            [sort_primary, "recall", "weighted_class_fp_sort", "blazar_fpr", "tn", "precision"],
            ascending=[False, False, True, True, False, False],
        ).iloc[0]
    else:
        best = eligible.sort_values(
            [sort_primary, "weighted_class_fp_sort", "tn", "precision"],
            ascending=[False, True, False, False],
        ).iloc[0]
    best_policy = {
        "r_min": float(best["r_min"]),
        "peak_frac": float(best["peak_frac"]),
        "adjacency_frac": float(best["adjacency_frac"]),
        "min_clean_points_total": int(best["min_clean_points_total"]),
        "min_season_bins_total": int(best["min_season_bins_total"]),
        "cross_validated": use_cv,
        "cv_n_splits": n_splits if use_cv else None,
        "metrics": {
            "balanced_accuracy": float(best["balanced_accuracy"]),
            "cv_balanced_accuracy": float(best["cv_balanced_accuracy"]) if np.isfinite(best["cv_balanced_accuracy"]) else None,
            "recall": float(best["recall"]),
            "precision": float(best["precision"]),
            "blazar_fpr": float(best["blazar_fpr"]) if np.isfinite(best["blazar_fpr"]) else np.nan,
            "fpr_normal_agn": float(best["fpr_normal_agn"]) if np.isfinite(best["fpr_normal_agn"]) else np.nan,
            "fpr_star": float(best["fpr_star"]) if np.isfinite(best["fpr_star"]) else np.nan,
            "fpr_sn": float(best["fpr_sn"]) if np.isfinite(best["fpr_sn"]) else np.nan,
            "weighted_class_fp": float(best["weighted_class_fp"]) if np.isfinite(best["weighted_class_fp"]) else None,
            "tp": int(best["tp"]),
            "fp": int(best["fp"]),
            "fn": int(best["fn"]),
            "tn": int(best["tn"]),
        },
        "class_fp_objective": {
            "enabled": class_obj_enabled,
            "weights": {
                "normal_agn": w_normal_agn,
                "star": w_star,
                "sn": w_sn,
            },
            "caps": {
                "normal_agn": float(cap_normal_agn) if cap_normal_agn is not None else None,
                "star": float(cap_star) if cap_star is not None else None,
            },
        },
    }
    return grid, best_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/real_clagn")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--scores_csv", default=None)
    parser.add_argument("--policy-config", default="configs/pipeline_policy.yaml")
    parser.add_argument("--stage-report", default=None, help="CSV with gate columns from make_cards summary/augmented output")
    parser.add_argument("--output_dir", default="reports")
    parser.add_argument("--strict-mode", action="store_true")
    parser.add_argument("--benchmark-file", default=None, help="Override default split CSV path (e.g. for v2 dev/test sets)")
    args = parser.parse_args()

    policy = _load_policy_config(args.policy_config)
    strict_mode = bool(args.strict_mode or policy.get("strict_mode_default", False))
    t_recall = float(policy.get("t_recall", 0.3493918746))
    recall_ks = [int(v) for v in policy.get("evaluation", {}).get("recall_at_k", [10, 20, 50])]
    rel_bins = int(policy.get("evaluation", {}).get("reliability_bins", 10))
    shifts = [float(v) for v in policy.get("evaluation", {}).get("synthetic_prevalence_shift", [0.05, 0.1, 0.2, 0.5])]
    rng = np.random.default_rng(int(policy.get("random_seed", 42)))

    base = Path(args.data_dir)
    if args.benchmark_file:
        split_path = Path(args.benchmark_file)
    else:
        split_path = base / ("benchmark_dev.csv" if args.split == "dev" else "benchmark_test.csv")
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    scores_path = Path(args.scores_csv) if args.scores_csv else (base / "benchmark_scores.csv")
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing scores file: {scores_path}")

    split_df = pd.read_csv(split_path)
    split_df["y_true"] = ensure_y_true(split_df)
    split_df["class"] = normalized_class_series(split_df)
    if "canonical_id" not in split_df.columns:
        split_df = add_canonical_id(split_df)
    scores_df = _load_scores(scores_path)
    join_key = best_join_key(split_df, scores_df)
    merged = split_df.merge(scores_df, on=join_key, how="inner", suffixes=("", "_score"))
    if merged.empty:
        raise ValueError("No overlap between split and scores")

    merged["status"] = merged["status"].astype(str).str.upper()
    merged["score"] = pd.to_numeric(merged["score"], errors="coerce")
    metric_mask = merged["status"].eq("OK") & np.isfinite(merged["score"])
    eval_df = merged.loc[metric_mask].copy()
    if eval_df.empty:
        raise ValueError("No rows available for evaluation after status/score filtering")

    if strict_mode:
        _strict_schema_checks(eval_df)
        if args.split == "test" and "split" in eval_df.columns:
            if not eval_df["split"].astype(str).str.lower().eq("test").all():
                raise ValueError("Strict mode: split leakage detected in test run")

    # Stage columns
    eval_df["stage_a_positive"] = eval_df["score"] >= t_recall
    eval_df["gate_blazar_veto"] = ~eval_df["class"].astype(str).str.lower().eq("blazar")
    eval_df["gate_wise_qc_pass"] = True
    eval_df["gate_data_sufficient"] = True

    if args.stage_report:
        stage_path = Path(args.stage_report)
        if not stage_path.exists():
            raise FileNotFoundError(f"stage-report file not found: {stage_path}")
        stage_df = pd.read_csv(stage_path)
        if "canonical_id" not in stage_df.columns and "source_id" not in stage_df.columns:
            raise ValueError("stage-report must contain canonical_id or source_id")
        if "canonical_id" not in stage_df.columns:
            stage_df = add_canonical_id(stage_df)
        stage_df = _dedup_eval_df(stage_df)
        keep_cols = [
            c
            for c in [
                "canonical_id",
                "gate_blazar_veto",
                "gate_wise_qc_pass",
                "gate_data_sufficient",
                "gate_state_change_ratio_pass",
                "gate_transient_spike_pass",
                "state_change_ratio_value",
                "transient_spike_flag",
                "transient_max_nonpeak_ratio",
                "transient_max_adjacency_ratio",
                "transient_min_band_seasons",
                "kept_points_total",
                "season_bins_w1",
                "season_bins_w2",
                "fraction_rejected_total",
                "fp_mode_hint",
                "final_label",
                "decision_trace",
            ]
            if c in stage_df.columns
        ]
        eval_df = eval_df.merge(stage_df[keep_cols], on="canonical_id", how="left", suffixes=("", "_stage"))
        for c in ["gate_blazar_veto", "gate_wise_qc_pass", "gate_data_sufficient"]:
            if f"{c}_stage" in eval_df.columns:
                eval_df[c] = eval_df[f"{c}_stage"].where(eval_df[f"{c}_stage"].notna(), eval_df[c])
                eval_df.drop(columns=[f"{c}_stage"], inplace=True)
        passthrough_cols = [
            "gate_state_change_ratio_pass",
            "gate_transient_spike_pass",
            "state_change_ratio_value",
            "transient_spike_flag",
            "transient_max_nonpeak_ratio",
            "transient_max_adjacency_ratio",
            "transient_min_band_seasons",
            "kept_points_total",
            "season_bins_w1",
            "season_bins_w2",
            "fraction_rejected_total",
            "fp_mode_hint",
            "final_label",
            "decision_trace",
        ]
        for c in passthrough_cols:
            stage_c = f"{c}_stage"
            if stage_c in eval_df.columns:
                if c in eval_df.columns:
                    eval_df[c] = eval_df[stage_c].where(eval_df[stage_c].notna(), eval_df[c])
                else:
                    eval_df[c] = eval_df[stage_c]
                eval_df.drop(columns=[stage_c], inplace=True)

    if "season_bins_total" not in eval_df.columns:
        w1 = pd.to_numeric(eval_df["season_bins_w1"], errors="coerce").fillna(0) if "season_bins_w1" in eval_df.columns else pd.Series(0, index=eval_df.index)
        w2 = pd.to_numeric(eval_df["season_bins_w2"], errors="coerce").fillna(0) if "season_bins_w2" in eval_df.columns else pd.Series(0, index=eval_df.index)
        eval_df["season_bins_total"] = w1 + w2

    eval_df["gate_blazar_veto"] = eval_df["gate_blazar_veto"].astype(bool)
    eval_df["gate_wise_qc_pass"] = eval_df["gate_wise_qc_pass"].astype(bool)
    eval_df["gate_data_sufficient"] = eval_df["gate_data_sufficient"].astype(bool)
    if "gate_state_change_ratio_pass" not in eval_df.columns:
        eval_df["gate_state_change_ratio_pass"] = True
    if "gate_transient_spike_pass" not in eval_df.columns:
        eval_df["gate_transient_spike_pass"] = True
    eval_df["gate_state_change_ratio_pass"] = eval_df["gate_state_change_ratio_pass"].astype("boolean").fillna(False).astype(bool)
    eval_df["gate_transient_spike_pass"] = eval_df["gate_transient_spike_pass"].astype("boolean").fillna(False).astype(bool)

    eval_df["pred_stage_a"] = eval_df["stage_a_positive"]
    eval_df["pred_after_blazar_veto"] = eval_df["stage_a_positive"] & eval_df["gate_blazar_veto"]
    eval_df["pred_after_all_gates"] = (
        eval_df["pred_after_blazar_veto"]
        & eval_df["gate_wise_qc_pass"]
        & eval_df["gate_data_sufficient"]
        & eval_df["gate_state_change_ratio_pass"]
        & eval_df["gate_transient_spike_pass"]
    )

    y_true = eval_df["y_true"].astype(int).to_numpy()
    stage_metrics = {}
    for col in ["pred_stage_a", "pred_after_blazar_veto", "pred_after_all_gates"]:
        stage_metrics[col] = _confusion_stats(y_true, eval_df[col].astype(int).to_numpy())

    recall_at_k = _recall_at_k(eval_df[["canonical_id", "score", "y_true"]], recall_ks)
    precision_at_k = _precision_at_k(eval_df[["canonical_id", "score", "y_true"]], recall_ks)
    try:
        auc_roc = float(roc_auc_score(
            eval_df["y_true"].astype(int),
            pd.to_numeric(eval_df["score"], errors="coerce").fillna(0.0)
        ))
    except Exception:
        auc_roc = np.nan
    class_stage_a = _class_contamination(eval_df, "pred_stage_a")
    class_after_blazar = _class_contamination(eval_df, "pred_after_blazar_veto")
    class_after_all = _class_contamination(eval_df, "pred_after_all_gates")

    # Dev false-positive audit after all gates.
    fp_mask = (eval_df["y_true"].astype(int) == 0) & eval_df["pred_after_all_gates"].astype(bool)
    fp_cols = [
        "source_id",
        "canonical_id",
        "class",
        "score",
        "kept_points_total",
        "season_bins_total",
        "season_bins_w1",
        "season_bins_w2",
        "fraction_rejected_total",
        "state_change_ratio_value",
        "transient_spike_flag",
        "transient_max_nonpeak_ratio",
        "transient_max_adjacency_ratio",
        "transient_min_band_seasons",
        "fp_mode_hint",
        "final_label",
        "decision_trace",
    ]
    fp_cols = [c for c in fp_cols if c in eval_df.columns]
    fp_audit = eval_df.loc[fp_mask, fp_cols].copy().sort_values(
        ["score", "canonical_id"], ascending=[False, True]
    )
    fp_counts_by_class, fp_feature_quants, fp_gate_matrix = _fp_refinement_tables(eval_df)

    reliability = _reliability_bins(eval_df[["score", "y_true"]], rel_bins)
    brier = float(brier_score_loss(eval_df["y_true"].astype(int), eval_df["score"].astype(float)))
    shift_df = _synthetic_shift(eval_df[["score", "y_true"]], shifts, rng)

    acceptance = policy.get("acceptance_targets", {})
    recall_target = float(acceptance.get("recall_clagn_min", 0.80))
    blazar_target = float(acceptance.get("blazar_fpr_max", 0.10))
    bal_target = float(acceptance.get("balanced_accuracy_min", 0.90))

    tuning_cfg = policy.get("tuning", {})
    run_tuning = (
        args.split == str(tuning_cfg.get("only_split", "dev")).lower()
        and bool(tuning_cfg.get("enabled", True))
    )
    tuning_grid = pd.DataFrame()
    best_policy: dict[str, Any] | None = None
    tuned_metrics: dict[str, Any] | None = None
    tuned_class_contamination = pd.DataFrame()
    if run_tuning:
        tuning_grid, best_policy = _run_dev_tuning(
            eval_df=eval_df,
            policy=policy,
            recall_target=recall_target,
            blazar_target=blazar_target,
        )
        if best_policy is not None:
            tuned_pred = _apply_tuned_prediction(
                eval_df=eval_df,
                r_min=float(best_policy["r_min"]),
                peak_frac=float(best_policy["peak_frac"]),
                adjacency_frac=float(best_policy["adjacency_frac"]),
                min_clean_points_total=int(best_policy["min_clean_points_total"]),
                min_season_bins_total=int(best_policy["min_season_bins_total"]),
            )
            eval_df["pred_after_tuned_gates"] = tuned_pred.astype(bool)
            tuned_metrics = _confusion_stats(y_true, tuned_pred.astype(int).to_numpy())
            tuned_class_contamination = _class_contamination(eval_df, "pred_after_tuned_gates")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_stage_a_path = out_dir / "eval_stage_a.csv"
    eval_after_blazar_path = out_dir / "eval_after_blazar_veto.csv"
    eval_after_all_path = out_dir / "eval_after_all_gates.csv"
    eval_stage_a = pd.DataFrame([stage_metrics["pred_stage_a"]])
    eval_after_blazar = pd.DataFrame([stage_metrics["pred_after_blazar_veto"]])
    eval_after_all = pd.DataFrame([stage_metrics["pred_after_all_gates"]])
    eval_stage_a.to_csv(eval_stage_a_path, index=False, float_format="%.6f")
    eval_after_blazar.to_csv(eval_after_blazar_path, index=False, float_format="%.6f")
    eval_after_all.to_csv(eval_after_all_path, index=False, float_format="%.6f")
    class_stage_a.to_csv(out_dir / "class_contamination_stage_a.csv", index=False, float_format="%.6f")
    class_after_blazar.to_csv(out_dir / "class_contamination_after_blazar_veto.csv", index=False, float_format="%.6f")
    class_after_all.to_csv(out_dir / "class_contamination_after_all_gates.csv", index=False, float_format="%.6f")
    fp_audit.to_csv(out_dir / "dev_false_positive_audit.csv", index=False, float_format="%.6f")
    fp_counts_by_class.to_csv(out_dir / "fp_audit_by_class.csv", index=False, float_format="%.6f")
    fp_feature_quants.to_csv(out_dir / "fp_audit_feature_quantiles.csv", index=False, float_format="%.6f")
    fp_gate_matrix.to_csv(out_dir / "fp_audit_gate_pass_matrix.csv", index=False, float_format="%.6f")
    recall_at_k.to_csv(out_dir / "recall_at_k.csv", index=False, float_format="%.6f")
    precision_at_k.to_csv(out_dir / "precision_at_k.csv", index=False, float_format="%.6f")
    reliability.to_csv(out_dir / "reliability_bins.csv", index=False, float_format="%.6f")
    shift_df.to_csv(out_dir / "synthetic_prevalence_shift.csv", index=False, float_format="%.6f")
    if not tuning_grid.empty:
        tuning_grid.sort_values(
            ["balanced_accuracy", "weighted_class_fp_sort", "recall", "blazar_fpr", "tn", "precision"],
            ascending=[False, True, False, True, False, False],
        ).to_csv(out_dir / "dev_tuning_grid.csv", index=False, float_format="%.6f")
    if best_policy is not None:
        (out_dir / "dev_best_policy.json").write_text(json.dumps(best_policy, indent=2), encoding="utf-8")
    if tuned_metrics is not None:
        pd.DataFrame([tuned_metrics]).to_csv(out_dir / "eval_after_tuned_gates.csv", index=False, float_format="%.6f")
    if not tuned_class_contamination.empty:
        tuned_class_contamination.to_csv(out_dir / "class_contamination_after_tuned_gates.csv", index=False, float_format="%.6f")

    blazar_row = class_after_all[class_after_all["class"].str.lower() == "blazar"]
    blazar_fpr = float(blazar_row["fpr_within_class"].iloc[0]) if not blazar_row.empty else np.nan
    final_metrics = stage_metrics["pred_after_all_gates"]
    report_metrics = tuned_metrics if tuned_metrics is not None else final_metrics
    report_blazar_fpr = (
        _blazar_fpr(eval_df, "pred_after_tuned_gates")
        if tuned_metrics is not None
        else blazar_fpr
    )

    pass_recall = report_metrics["recall"] >= recall_target
    pass_blazar = np.isfinite(report_blazar_fpr) and report_blazar_fpr <= blazar_target
    pass_bal = report_metrics["balanced_accuracy"] >= bal_target

    n_pos = int((eval_df["y_true"] == 1).sum())
    n_neg = int((eval_df["y_true"] == 0).sum())
    target_tnr = min(1.0, max(0.0, (2.0 * bal_target) - report_metrics["recall"]))
    target_fp_max = int(np.floor(n_neg * max(0.0, 1.0 - target_tnr)))

    summary_lines = [
        "# Evaluation Summary",
        "",
        f"- split: `{args.split}`",
        f"- rows_joined: `{len(merged)}`",
        f"- rows_used_for_metrics: `{len(eval_df)}`",
        f"- imbalance_warning: `Accuracy is not primary KPI for this benchmark.`",
        f"- baseline_all_negative_accuracy: `{float((eval_df['y_true'] == 0).mean()):.4f}`",
        f"- baseline_random_accuracy: `{0.5000:.4f}`",
        "",
        "## Stage Metrics",
        "",
        f"- Stage A balanced_accuracy: `{stage_metrics['pred_stage_a']['balanced_accuracy']:.4f}`",
        f"- After blazar veto balanced_accuracy: `{stage_metrics['pred_after_blazar_veto']['balanced_accuracy']:.4f}`",
        f"- After all gates balanced_accuracy: `{stage_metrics['pred_after_all_gates']['balanced_accuracy']:.4f}`",
        f"- After tuned gates balanced_accuracy: `{tuned_metrics['balanced_accuracy']:.4f}`" if tuned_metrics is not None else "- After tuned gates balanced_accuracy: `N/A`",
        "",
        "## FP Gap Math",
        "",
        f"- positives: `{n_pos}`",
        f"- negatives: `{n_neg}`",
        f"- current recall used for target gap: `{report_metrics['recall']:.4f}`",
        f"- required_tnr_for_balanced_accuracy_{bal_target:.2f}: `{target_tnr:.4f}`",
        f"- max_fp_allowed_for_target_tnr: `{target_fp_max}`",
        f"- current_fp: `{report_metrics['fp']}`",
        "",
        "## Calibration",
        "",
        f"- Brier score: `{brier:.6f}`",
        f"- AUC-ROC: `{auc_roc:.4f}`" if np.isfinite(auc_roc) else "- AUC-ROC: `N/A`",
        f"- Reliability bins file: `reliability_bins.csv`",
        f"- Precision@K file: `precision_at_k.csv`",
        "",
        "## Additional Metrics",
        "",
        f"- overall_accuracy (after tuned gates): `{report_metrics.get('overall_accuracy', 0.0):.4f}`",
        "",
        "## Acceptance Targets",
        "",
        f"- recall_clagn >= {recall_target:.2f}: {'PASS' if pass_recall else 'FAIL'} "
        f"(`{report_metrics['recall']:.4f}`)",
        f"- blazar_fpr <= {blazar_target:.2f}: {'PASS' if pass_blazar else 'FAIL'} "
        f"(`{report_blazar_fpr:.4f}`)",
        f"- balanced_accuracy >= {bal_target:.2f}: {'PASS' if pass_bal else 'FAIL'} "
        f"(`{report_metrics['balanced_accuracy']:.4f}`)",
        "",
    ]
    (out_dir / "eval_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    p20_row = precision_at_k[precision_at_k["k"] == 20]
    precision_at_20 = float(p20_row["precision_at_k"].iloc[0]) if not p20_row.empty else np.nan
    payload = {
        "split": args.split,
        "t_recall": t_recall,
        "join_key": join_key,
        "stage_metrics": stage_metrics,
        "tuned_metrics": tuned_metrics,
        "brier_score": brier,
        "auc_roc": auc_roc if np.isfinite(auc_roc) else None,
        "precision_at_20": precision_at_20 if np.isfinite(precision_at_20) else None,
        "blazar_fpr_after_all_gates": blazar_fpr,
        "blazar_fpr_after_tuned_gates": report_blazar_fpr,
        "best_policy": best_policy,
        "acceptance": {
            "recall": bool(pass_recall),
            "blazar_fpr": bool(pass_blazar),
            "balanced_accuracy": bool(pass_bal),
        },
    }
    (out_dir / "metrics_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Evaluation complete")
    print(f"Stage-A metrics: {stage_metrics['pred_stage_a']}")
    print(f"After-all-gates metrics: {stage_metrics['pred_after_all_gates']}")


if __name__ == "__main__":
    main()
