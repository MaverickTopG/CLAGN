#!/usr/bin/env python3
"""
Find optimal Layer 2 probability threshold on dev set.
Maximizes balanced accuracy subject to recall >= 0.80 and blazar FPR = 0.
Threshold is frozen after this step and never adjusted again.

Precondition: dev balanced accuracy >= 0.82 from classifier_v2.py.

Usage:
  python scripts/calibrate_threshold_v2.py
"""
import numpy as np
import pandas as pd
import json
import joblib
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score

# ── Load models ───────────────────────────────────────────────────
iso        = joblib.load("models/isolation_forest.pkl")
scaler     = joblib.load("models/scaler.pkl")
clf        = joblib.load("models/lgbm_classifier.pkl")
iso_thresh = joblib.load("models/iso_threshold.pkl")
feat_cols  = joblib.load("models/feature_columns.pkl")

# ── Load dev data (same logic as classifier_v2.py) ────────────────

FEATURES_CSV = "data/features/all_features_with_scores.csv"
DEV_CSV      = "data/benchmark/benchmark_v3_dev.csv"

feat = pd.read_csv(FEATURES_CSV)
split = pd.read_csv(DEV_CSV)[['source_id', 'label']]
feat['source_id'] = feat['source_id'].astype(str).str.strip()
split['source_id'] = split['source_id'].astype(str).str.strip()

df_dev = split.merge(feat, on='source_id', how='left',
                     suffixes=('', '_feat'))
if 'label_feat' in df_dev.columns:
    df_dev = df_dev.drop(columns=['label_feat'])
df_dev['y'] = (df_dev['label'] == 'CLAGN').astype(int)

avail = [f for f in feat_cols if f in df_dev.columns]
X_dev = df_dev[avail].copy()
medians = X_dev.median()
X_dev = X_dev.fillna(medians)
y_dev = df_dev['y'].values

print(f"Dev set: {len(df_dev):,} sources | {y_dev.sum()} CLAGN | "
      f"{len(avail)} features")

# ── Compute Layer 1 scores ────────────────────────────────────────
dev_scores = iso.score_samples(scaler.transform(X_dev))
l2_mask = dev_scores > iso_thresh
print(f"Layer 1 pass rate: {l2_mask.mean():.1%} "
      f"({l2_mask.sum()} sources)")

proba_dev_l2 = clf.predict_proba(
    X_dev[l2_mask].reset_index(drop=True))[:, 1]

# Blazar mask for FPR calculation
blazar_mask = df_dev['label'] == 'blazar'
print(f"Blazars in dev: {blazar_mask.sum()}")

# ── Sweep thresholds ──────────────────────────────────────────────
thresholds = np.arange(0.05, 0.95, 0.005)
records = []

for t in thresholds:
    y_pred_full = np.zeros(len(y_dev), dtype=int)
    y_pred_full[l2_mask] = (proba_dev_l2 >= t).astype(int)
    bal_acc = balanced_accuracy_score(y_dev, y_pred_full)
    recall = recall_score(y_dev, y_pred_full, zero_division=0)
    precision = precision_score(y_dev, y_pred_full, zero_division=0)
    blazar_fpr = (y_pred_full[blazar_mask].mean()
                  if blazar_mask.sum() > 0 else 0.0)
    records.append({
        'threshold': round(float(t), 4),
        'bal_acc': round(float(bal_acc), 4),
        'recall': round(float(recall), 4),
        'precision': round(float(precision), 4),
        'blazar_fpr': round(float(blazar_fpr), 4),
    })

df_t = pd.DataFrame(records)

# ── Select best threshold ─────────────────────────────────────────
# Constraints: recall >= 0.80, blazar_fpr == 0
valid = df_t[(df_t['recall'] >= 0.80) & (df_t['blazar_fpr'] == 0.0)]
if len(valid) == 0:
    print("WARNING: No threshold satisfies recall>=0.80 AND blazar_fpr=0")
    print("Relaxing to recall>=0.75...")
    valid = df_t[df_t['recall'] >= 0.75]
if len(valid) == 0:
    print("WARNING: No threshold satisfies recall>=0.75 either.")
    print("Selecting threshold that maximizes balanced accuracy overall.")
    valid = df_t

best_row = valid.loc[valid['bal_acc'].idxmax()]
best_t = float(best_row['threshold'])

print(f"\nOptimal threshold: {best_t:.4f}")
print(f"  Balanced accuracy: {best_row['bal_acc']:.4f}")
print(f"  Recall:            {best_row['recall']:.4f}")
print(f"  Precision:         {best_row['precision']:.4f}")
print(f"  Blazar FPR:        {best_row['blazar_fpr']:.4f}")

config = {
    'threshold_v2': best_t,
    'dev_balanced_acc': float(best_row['bal_acc']),
    'dev_recall': float(best_row['recall']),
    'dev_precision': float(best_row['precision']),
    'dev_blazar_fpr': float(best_row['blazar_fpr']),
    'frozen': True,
    'note': 'This threshold was set on dev set ONCE and must not be changed.',
}
with open("models/threshold_v2.json", 'w') as f:
    json.dump(config, f, indent=2)
print("\nThreshold frozen. DO NOT modify models/threshold_v2.json")

df_t.to_csv("results/threshold_sweep_dev.csv", index=False)
print(f"Saved results/threshold_sweep_dev.csv ({len(df_t)} rows)")
