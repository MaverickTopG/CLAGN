#!/usr/bin/env python3
"""
Prediction wrapper for the two-layer classifier.
Runs Isolation Forest (Layer 1) + LightGBM (Layer 2) on a given split
and outputs predictions + metrics JSON.

Usage:
  python scripts/predict_v2.py --split test --output results/FINAL_TEST_RESULTS.json
  python scripts/predict_v2.py --split dev  --output results/dev_predictions.json
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, recall_score,
    precision_score, confusion_matrix, roc_auc_score,
    classification_report, matthews_corrcoef,
)

FEATURES_CSV = "data/features/all_features_with_scores.csv"
BENCHMARK_DIR = "data/benchmark"


def parse_args():
    p = argparse.ArgumentParser(description="Two-layer classifier prediction")
    p.add_argument("--split", required=True, choices=["train", "dev", "test"],
                   help="Which benchmark split to evaluate")
    p.add_argument("--output", required=True,
                   help="Output JSON path for results")
    p.add_argument("--threshold", type=float, default=None,
                   help="Override LightGBM threshold (default: from "
                        "models/threshold_v2.json)")
    return p.parse_args()


def main():
    args = parse_args()

    # Load models
    iso = joblib.load("models/isolation_forest.pkl")
    scaler = joblib.load("models/scaler.pkl")
    clf = joblib.load("models/lgbm_classifier.pkl")
    iso_thresh = joblib.load("models/iso_threshold.pkl")
    feat_cols = joblib.load("models/feature_columns.pkl")

    # Load threshold
    if args.threshold is not None:
        lgbm_thresh = args.threshold
    elif os.path.exists("models/threshold_v2.json"):
        with open("models/threshold_v2.json") as f:
            lgbm_thresh = json.load(f)['threshold_v2']
    else:
        lgbm_thresh = 0.5
    print(f"Using LightGBM threshold: {lgbm_thresh:.4f}")

    # Load data
    split_csv = os.path.join(
        BENCHMARK_DIR, f"benchmark_v3_{args.split}.csv")
    feat = pd.read_csv(FEATURES_CSV)
    split = pd.read_csv(split_csv)[['source_id', 'label']]
    feat['source_id'] = feat['source_id'].astype(str).str.strip()
    split['source_id'] = split['source_id'].astype(str).str.strip()

    df = split.merge(feat, on='source_id', how='left',
                     suffixes=('', '_feat'))
    if 'label_feat' in df.columns:
        df = df.drop(columns=['label_feat'])
    df['y'] = (df['label'] == 'CLAGN').astype(int)

    avail = [f for f in feat_cols if f in df.columns]
    X = df[avail].copy()
    medians = X.median()
    X = X.fillna(medians)
    y = df['y'].values

    print(f"Split '{args.split}': {len(df):,} sources | "
          f"{y.sum()} CLAGN | {len(avail)} features")

    # Layer 1: Isolation Forest
    X_scaled = scaler.transform(X)
    iso_scores = iso.score_samples(X_scaled)
    l2_mask = iso_scores > iso_thresh
    print(f"Layer 1 pass: {l2_mask.sum():,} / {len(df):,} "
          f"({l2_mask.mean():.1%})")

    # Layer 2: LightGBM
    y_pred = np.zeros(len(y), dtype=int)
    proba = np.zeros(len(y))
    if l2_mask.sum() > 0:
        proba_l2 = clf.predict_proba(
            X[l2_mask].reset_index(drop=True))[:, 1]
        proba[l2_mask] = proba_l2
        y_pred[l2_mask] = (proba_l2 >= lgbm_thresh).astype(int)

    # Metrics
    cm = confusion_matrix(y, y_pred)
    try:
        auc = float(roc_auc_score(y, proba))
    except ValueError:
        auc = 0.0

    blazar_mask = df['label'] == 'blazar'
    blazar_fpr = (float(y_pred[blazar_mask].mean())
                  if blazar_mask.sum() > 0 else 0.0)

    metrics = {
        'split': args.split,
        'n_sources': len(df),
        'n_clagn': int(y.sum()),
        'n_features': len(avail),
        'lgbm_threshold': lgbm_thresh,
        'overall_accuracy': round(float(accuracy_score(y, y_pred)), 4),
        'balanced_accuracy': round(float(
            balanced_accuracy_score(y, y_pred)), 4),
        'clagn_recall': round(float(
            recall_score(y, y_pred, zero_division=0)), 4),
        'clagn_precision': round(float(
            precision_score(y, y_pred, zero_division=0)), 4),
        'roc_auc': round(auc, 4),
        'mcc': round(float(matthews_corrcoef(y, y_pred)), 4),
        'blazar_fpr': round(blazar_fpr, 4),
        'confusion_matrix': cm.tolist(),
    }

    print(f"\n=== {args.split.upper()} RESULTS ===")
    print(classification_report(
        y, y_pred, target_names=['non-CLAGN', 'CLAGN']))
    print(f"Confusion matrix:\n{cm}")
    for k, v in metrics.items():
        if k not in ('confusion_matrix', 'split'):
            print(f"  {k:25s}: {v}")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved {args.output}")

    # Save per-source predictions
    df['y_pred'] = y_pred
    df['proba'] = proba
    df['iso_score'] = iso_scores
    df['l2_pass'] = l2_mask
    pred_csv = args.output.replace('.json', '_predictions.csv')
    df[['source_id', 'label', 'y', 'y_pred', 'proba',
        'iso_score', 'l2_pass']].to_csv(pred_csv, index=False)
    print(f"Saved {pred_csv}")


if __name__ == "__main__":
    main()
