"""
Evaluate benchmark split with PR/ROC/CM metrics and bootstrap CIs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clagn.utils.validation_metadata import build_metadata, write_metadata_sidecar, write_json_with_metadata
from clagn.utils.benchmark_labels import ensure_y_true, normalized_class_series
from clagn.utils.id_canonicalization import best_join_key, add_canonical_id


def _bootstrap_ci(values: list[float]) -> tuple[float, float]:
    if len(values) == 0:
        return (np.nan, np.nan)
    lo = np.percentile(values, 2.5)
    hi = np.percentile(values, 97.5)
    return float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark_dir', default='data/benchmark')
    parser.add_argument('--scores_csv', default=None)
    parser.add_argument('--split', choices=['dev', 'test'], default='dev')
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--n_bootstrap', type=int, default=200)
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / 'metrics'
    figures_dir = output_dir / 'figures'
    metadata_dir = output_dir / 'metadata'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    split_path = benchmark_dir / ('benchmark_train_dev.csv' if args.split == 'dev' else 'benchmark_test.csv')
    if not split_path.exists():
        print(f"Missing split file: {split_path}")
        sys.exit(1)

    split_df = pd.read_csv(split_path)

    scores_path = Path(args.scores_csv) if args.scores_csv else benchmark_dir / 'scores' / 'benchmark_scores.csv'
    if not scores_path.exists():
        print(f"Missing scores file: {scores_path}. Provide --scores_csv.")
        sys.exit(1)

    scores_df = pd.read_csv(scores_path)
    if 'source_id' not in scores_df.columns or 'score' not in scores_df.columns:
        print("scores_csv must have columns: source_id, score")
        sys.exit(1)
    if 'canonical_id' not in scores_df.columns:
        scores_df = add_canonical_id(scores_df)
    if 'canonical_id' not in split_df.columns and 'source_id' in split_df.columns:
        split_df = add_canonical_id(split_df)
    if 'status' not in scores_df.columns:
        scores_df['status'] = 'OK'

    join_key = best_join_key(split_df, scores_df)
    merged = split_df.merge(scores_df, on=join_key, how='inner')
    if merged.empty:
        print("No overlap between benchmark split and scores")
        sys.exit(1)

    merged['y_true'] = ensure_y_true(merged)
    merged['class'] = normalized_class_series(merged)
    metric_mask = (merged['status'].astype(str) == 'OK') & np.isfinite(pd.to_numeric(merged['score'], errors='coerce'))
    if metric_mask.sum() == 0:
        print("No status=OK rows with finite scores available for metrics")
        sys.exit(1)
    y_true = merged.loc[metric_mask, 'y_true'].astype(int).values
    y_score = pd.to_numeric(merged.loc[metric_mask, 'score'], errors='coerce').astype(float).values

    # Curves
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)

    auc_pr = auc(recall, precision)
    auc_roc = auc(fpr, tpr)

    # Threshold selection
    if args.threshold is None:
        if args.split != 'dev':
            # Must not auto-tune on test
            thresh_path = metrics_dir / 'operating_threshold.json'
            if thresh_path.exists():
                args.threshold = json.loads(thresh_path.read_text()).get('threshold', 0.5)
            else:
                print("Threshold required for test split (no operating_threshold.json found)")
                sys.exit(1)
        else:
            f1 = 2 * precision * recall / (precision + recall + 1e-12)
            best_idx = int(np.argmax(f1))
            if len(pr_thresholds) == 0:
                args.threshold = 0.5
            elif best_idx >= len(pr_thresholds):
                args.threshold = float(pr_thresholds[-1])
            else:
                args.threshold = float(pr_thresholds[best_idx])

            # Save operating threshold
            op_path = metrics_dir / 'operating_threshold.json'
            meta = build_metadata(benchmark_dir, extra={'split': args.split})
            write_json_with_metadata(op_path, {'threshold': args.threshold}, metadata_dir, meta)

    # Confusion matrix
    y_pred = (y_score >= args.threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision_at = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_at = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_at = (2 * precision_at * recall_at / (precision_at + recall_at)
             if (precision_at + recall_at) > 0 else 0.0)

    # Bootstrap CIs
    rng = np.random.default_rng(seed=42)
    prec_samples = []
    rec_samples = []
    f1_samples = []
    auc_pr_samples = []
    auc_roc_samples = []

    n = len(y_true)
    target = args.n_bootstrap
    attempts = 0
    max_attempts = max(target * 5, 50)
    while len(auc_pr_samples) < target and attempts < max_attempts:
        attempts += 1
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        # Skip resamples that lack positives or negatives to avoid undefined metrics
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        pr_p, pr_r, _ = precision_recall_curve(yt, ys)
        roc_f, roc_t, _ = roc_curve(yt, ys)
        auc_pr_samples.append(auc(pr_r, pr_p))
        auc_roc_samples.append(auc(roc_f, roc_t))
        yp = (ys >= args.threshold).astype(int)
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        p = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0.0
        r = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0
        f1_b = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        prec_samples.append(p)
        rec_samples.append(r)
        f1_samples.append(f1_b)

    # Outputs
    pr_df = pd.DataFrame({
        'threshold': list(pr_thresholds) + [np.nan],
        'precision': precision,
        'recall': recall,
    })
    roc_df = pd.DataFrame({
        'threshold': roc_thresholds,
        'fpr': fpr,
        'tpr': tpr,
    })
    cm_df = pd.DataFrame({
        'metric': ['tp', 'fp', 'fn', 'tn'],
        'value': [tp, fp, fn, tn],
    })

    meta = build_metadata(benchmark_dir, extra={'split': args.split})

    if not args.dry_run:
        pr_path = metrics_dir / 'pr_curve.csv'
        roc_path = metrics_dir / 'roc_curve.csv'
        cm_path = metrics_dir / 'confusion_matrix.csv'
        pr_df.to_csv(pr_path, index=False)
        roc_df.to_csv(roc_path, index=False)
        cm_df.to_csv(cm_path, index=False)
        write_metadata_sidecar(pr_path, metadata_dir, meta)
        write_metadata_sidecar(roc_path, metadata_dir, meta)
        write_metadata_sidecar(cm_path, metadata_dir, meta)

        # Plots
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(recall, precision)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('PR Curve')
        fig.tight_layout()
        pr_fig = figures_dir / 'pr_curve.png'
        fig.savefig(pr_fig, dpi=200)
        plt.close(fig)
        write_metadata_sidecar(pr_fig, metadata_dir, meta)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr)
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title('ROC Curve')
        fig.tight_layout()
        roc_fig = figures_dir / 'roc_curve.png'
        fig.savefig(roc_fig, dpi=200)
        plt.close(fig)
        write_metadata_sidecar(roc_fig, metadata_dir, meta)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow([[tp, fp], [fn, tn]], cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred+','Pred-'])
        ax.set_yticklabels(['True+','True-'])
        ax.set_title('Confusion Matrix')
        fig.tight_layout()
        cm_fig = figures_dir / 'confusion_matrix.png'
        fig.savefig(cm_fig, dpi=200)
        plt.close(fig)
        write_metadata_sidecar(cm_fig, metadata_dir, meta)

    summary = {
        'split': args.split,
        'threshold': args.threshold,
        'precision': precision_at,
        'recall': recall_at,
        'f1': f1_at,
        'auc_pr': auc_pr,
        'auc_roc': auc_roc,
        'ci_precision': _bootstrap_ci(prec_samples),
        'ci_recall': _bootstrap_ci(rec_samples),
        'ci_f1': _bootstrap_ci(f1_samples),
        'ci_auc_pr': _bootstrap_ci(auc_pr_samples),
        'ci_auc_roc': _bootstrap_ci(auc_roc_samples),
    }

    if not args.dry_run:
        write_json_with_metadata(metrics_dir / 'metrics_summary.json', summary, metadata_dir, meta)

    print("Benchmark evaluation complete")


if __name__ == '__main__':
    main()
