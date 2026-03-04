#!/usr/bin/env python3
"""
Train classifier with each feature group added incrementally.
Produces the ablation table for the paper.
Shows contribution of each novel feature group.

Usage:
  python scripts/ablation_study.py 2>&1 | tee logs/ablation.log
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, recall_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

FEATURES_CSV = "data/features/all_features_with_scores.csv"
TRAIN_CSV    = "data/benchmark/benchmark_v3_train.csv"
DEV_CSV      = "data/benchmark/benchmark_v3_dev.csv"

# Load best params from Optuna trials CSV (not JSON — bug fix)
trials_df = pd.read_csv("results/optuna_trials.csv")
best_row = trials_df.loc[trials_df['value'].idxmax()]
# Extract only LightGBM hyperparams (drop 'value' column)
best_params = best_row.drop('value').to_dict()
# Convert integer params
for k in ['num_leaves', 'max_depth', 'min_child_samples']:
    if k in best_params:
        best_params[k] = int(best_params[k])
print(f"Best Optuna params: {best_params}")

# Load saved feature column list
feat_cols_loaded = joblib.load("models/feature_columns.pkl")

# ── Cumulative feature group definitions (bug fix: fully specified) ──

FEATURE_GROUPS_ORDERED = [
    ('color_asymmetry', [
        'color_slope_early', 'color_slope_late', 'color_asymmetry',
        'color_range', 'color_flux_corr', 'dust_lag_proxy',
    ]),
    ('wavelets', [
        'wavelet_d1', 'wavelet_d2', 'wavelet_d3', 'wavelet_approx_power',
        'wavelet_power_ratio_3_1', 'wavelet_power_ratio_3_2',
        'wavelet_entropy', 'wavelet_hurst',
    ]),
    ('structural_break', [
        'break_magnitude', 'break_significance', 'pre_break_slope',
        'post_break_slope', 'slope_change', 'break_fractional_position',
        'break_duration_yr', 'n_detected_breaks',
    ]),
    ('coherence', [
        'mean_season_coherence', 'coherence_trend', 'min_season_coherence',
        'max_season_coherence', 'coherence_drop_amplitude',
        'coherence_recovery', 'n_coherent_seasons',
    ]),
    ('flux_ratio', [
        'flux_ratio_mean', 'flux_ratio_std', 'flux_ratio_trend',
        'flux_ratio_skewness', 'flux_ratio_peak_to_trough',
        'flux_ratio_percentile_90_10', 'flux_ratio_late_minus_early',
    ]),
    ('drw_residuals', [
        'drw_sigma', 'drw_tau', 'drw_residual_rms',
        'drw_residual_skewness', 'drw_residual_autocorr',
        'drw_fit_quality', 'drw_residual_trend', 'drw_chi2_per_dof',
    ]),
    ('redshift_corr', [
        'z_corrected_amplitude', 'z_corrected_color_slope',
        'luminosity_change_proxy', 'z_corrected_drw_sigma',
        'intrinsic_variability_proxy',
    ]),
]

# Metadata features always included
METADATA_FEATS = ['n_epochs', 'n_seasons', 'baseline_days', 'z']


def load_split(feat_csv, split_csv, feature_cols, label='split'):
    """Load features merged with split labels for a specific feature subset."""
    feat = pd.read_csv(feat_csv)
    split = pd.read_csv(split_csv)[['source_id', 'label']]
    feat['source_id'] = feat['source_id'].astype(str).str.strip()
    split['source_id'] = split['source_id'].astype(str).str.strip()
    df = split.merge(feat, on='source_id', how='left',
                     suffixes=('', '_feat'))
    if 'label_feat' in df.columns:
        df = df.drop(columns=['label_feat'])
    df['y'] = (df['label'] == 'CLAGN').astype(int)
    avail = [f for f in feature_cols if f in df.columns]
    X = df[avail].copy()
    medians = X.median()
    X = X.fillna(medians)
    y = df['y'].values
    return X, y, avail


def run_ablation():
    results = []

    # Baseline: metadata + composite_score only
    base_feats = METADATA_FEATS + ['composite_score']
    cumulative = list(base_feats)

    print("\n=== ABLATION STUDY ===\n")

    # Baseline run
    X_train, y_train, avail = load_split(
        FEATURES_CSV, TRAIN_CSV, cumulative, 'train')
    X_dev, y_dev, _ = load_split(
        FEATURES_CSV, DEV_CSV, cumulative, 'dev')

    clf = lgb.LGBMClassifier(
        **{**best_params, 'n_estimators': 500, 'verbosity': -1,
           'objective': 'binary', 'random_state': 42})
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_dev)[:, 1]
    pred = (proba >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y_dev, proba)
    except ValueError:
        auc = 0.5
    results.append({
        'feature_set': 'baseline_only',
        'n_features': len(avail),
        'balanced_acc': round(balanced_accuracy_score(y_dev, pred), 4),
        'recall': round(recall_score(y_dev, pred, zero_division=0), 4),
        'roc_auc': round(auc, 4),
    })
    print(f"{'baseline_only':30s}: bal_acc={results[-1]['balanced_acc']:.4f} "
          f"recall={results[-1]['recall']:.4f} auc={results[-1]['roc_auc']:.4f}")

    # Incrementally add each feature group
    for group_name, group_feats in FEATURE_GROUPS_ORDERED:
        cumulative = cumulative + group_feats
        X_train, y_train, avail = load_split(
            FEATURES_CSV, TRAIN_CSV, cumulative, 'train')
        X_dev, y_dev, _ = load_split(
            FEATURES_CSV, DEV_CSV, cumulative, 'dev')

        clf = lgb.LGBMClassifier(
            **{**best_params, 'n_estimators': 500, 'verbosity': -1,
               'objective': 'binary', 'random_state': 42})
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_dev)[:, 1]
        pred = (proba >= 0.5).astype(int)
        try:
            auc = roc_auc_score(y_dev, proba)
        except ValueError:
            auc = 0.5
        results.append({
            'feature_set': f"+{group_name}",
            'n_features': len(avail),
            'balanced_acc': round(balanced_accuracy_score(y_dev, pred), 4),
            'recall': round(recall_score(y_dev, pred, zero_division=0), 4),
            'roc_auc': round(auc, 4),
        })
        print(f"+{group_name:29s}: bal_acc={results[-1]['balanced_acc']:.4f} "
              f"recall={results[-1]['recall']:.4f} "
              f"auc={results[-1]['roc_auc']:.4f}")

    # Extra row: novel features only (no composite_score)
    novel_only = METADATA_FEATS[:]
    for _, gf in FEATURE_GROUPS_ORDERED:
        novel_only.extend(gf)
    X_train, y_train, avail = load_split(
        FEATURES_CSV, TRAIN_CSV, novel_only, 'train')
    X_dev, y_dev, _ = load_split(
        FEATURES_CSV, DEV_CSV, novel_only, 'dev')
    clf = lgb.LGBMClassifier(
        **{**best_params, 'n_estimators': 500, 'verbosity': -1,
           'objective': 'binary', 'random_state': 42})
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_dev)[:, 1]
    pred = (proba >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y_dev, proba)
    except ValueError:
        auc = 0.5
    results.append({
        'feature_set': 'novel_only_no_composite',
        'n_features': len(avail),
        'balanced_acc': round(balanced_accuracy_score(y_dev, pred), 4),
        'recall': round(recall_score(y_dev, pred, zero_division=0), 4),
        'roc_auc': round(auc, 4),
    })
    print(f"{'novel_only_no_composite':30s}: "
          f"bal_acc={results[-1]['balanced_acc']:.4f} "
          f"recall={results[-1]['recall']:.4f} "
          f"auc={results[-1]['roc_auc']:.4f}")

    # Save results
    df_abl = pd.DataFrame(results)
    df_abl.to_csv("results/ablation_study.csv", index=False)
    print(f"\nSaved results/ablation_study.csv")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_abl['feature_set'], df_abl['balanced_acc'],
            'o-', color='#2D6BE4', label='Balanced Accuracy',
            linewidth=2, markersize=8)
    ax.plot(df_abl['feature_set'], df_abl['recall'],
            's--', color='#E44B3F', label='CLAGN Recall',
            linewidth=1.5, markersize=6)
    ax.axhline(0.85, color='gray', linestyle=':', linewidth=0.8,
               label='Target (85%)')
    ax.set_xlabel('Feature set', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Ablation Study: Feature Group Contributions', fontsize=12)
    plt.xticks(rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    plt.tight_layout()
    plt.savefig("figures/fig_ablation.pdf", dpi=300, bbox_inches='tight')
    plt.savefig("figures/fig_ablation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved ablation figure and CSV")


if __name__ == "__main__":
    run_ablation()
