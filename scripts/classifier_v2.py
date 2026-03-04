#!/usr/bin/env python3
"""
Two-layer classifier:
  Layer 1: Isolation Forest pre-filter (removes obvious negatives)
  Layer 2: LightGBM with Optuna tuning (50 trials, maximize balanced accuracy)
  SHAP values computed for interpretability (required for paper).

Usage:
  python scripts/classifier_v2.py 2>&1 | tee logs/classifier_v2_training.log
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import shap
import joblib
import os
import json
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score, accuracy_score,
    recall_score, precision_score,
    confusion_matrix, roc_auc_score,
    classification_report, matthews_corrcoef,
)
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

optuna.logging.set_verbosity(optuna.logging.WARNING)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

FEATURES_CSV = "data/features/all_features_with_scores.csv"
TRAIN_CSV    = "data/benchmark/benchmark_v3_train.csv"
DEV_CSV      = "data/benchmark/benchmark_v3_dev.csv"

ALL_FEATURE_COLS = [
    'composite_score',
    # A — Color Asymmetry
    'color_slope_early', 'color_slope_late', 'color_asymmetry',
    'color_range', 'color_flux_corr', 'dust_lag_proxy',
    # B — Wavelet Decomposition
    'wavelet_d1', 'wavelet_d2', 'wavelet_d3', 'wavelet_approx_power',
    'wavelet_power_ratio_3_1', 'wavelet_power_ratio_3_2',
    'wavelet_entropy', 'wavelet_hurst',
    # C — Structural Break
    'break_magnitude', 'break_significance', 'pre_break_slope',
    'post_break_slope', 'slope_change', 'break_fractional_position',
    'break_duration_yr', 'n_detected_breaks',
    # D — Seasonal Coherence
    'mean_season_coherence', 'coherence_trend', 'min_season_coherence',
    'max_season_coherence', 'coherence_drop_amplitude',
    'coherence_recovery', 'n_coherent_seasons',
    # E — Flux Ratio Evolution
    'flux_ratio_mean', 'flux_ratio_std', 'flux_ratio_trend',
    'flux_ratio_skewness', 'flux_ratio_peak_to_trough',
    'flux_ratio_percentile_90_10', 'flux_ratio_late_minus_early',
    # F — DRW Residuals
    'drw_sigma', 'drw_tau', 'drw_residual_rms', 'drw_residual_skewness',
    'drw_residual_autocorr', 'drw_fit_quality',
    'drw_residual_trend', 'drw_chi2_per_dof',
    # G — Redshift-Corrected
    'z_corrected_amplitude', 'z_corrected_color_slope',
    'luminosity_change_proxy', 'z_corrected_drw_sigma',
    'intrinsic_variability_proxy',
    # Metadata features
    'n_epochs', 'n_seasons', 'baseline_days', 'z',
]


def load_split(feat_csv, split_csv, label='split'):
    """Load features merged with split labels. Returns X, y, df, feature_cols."""
    feat = pd.read_csv(feat_csv)
    split = pd.read_csv(split_csv)[['source_id', 'label']]

    # Normalize join keys
    feat['source_id'] = feat['source_id'].astype(str).str.strip()
    split['source_id'] = split['source_id'].astype(str).str.strip()

    df = split.merge(feat, on='source_id', how='left',
                     suffixes=('', '_feat'))

    # Use the label from split CSV (authoritative), drop duplicate if any
    if 'label_feat' in df.columns:
        df = df.drop(columns=['label_feat'])

    df['y'] = (df['label'] == 'CLAGN').astype(int)
    avail = [f for f in ALL_FEATURE_COLS if f in df.columns]
    missing = [f for f in ALL_FEATURE_COLS if f not in df.columns]
    if missing:
        print(f"  [{label}] Missing features ({len(missing)}): {missing[:5]}...")
    X = df[avail].copy()
    # Fill NaN with column medians (computed on this split only)
    medians = X.median()
    X = X.fillna(medians)
    y = df['y'].values
    print(f"  [{label}] {len(df):,} sources | {len(avail)} features | "
          f"{y.sum()} CLAGN ({y.mean():.2%})")
    return X, y, df, avail


def main():
    print("Loading data...")
    X_train, y_train, df_train, feat_cols = load_split(
        FEATURES_CSV, TRAIN_CSV, 'train')
    X_dev, y_dev, df_dev, _ = load_split(
        FEATURES_CSV, DEV_CSV, 'dev')

    # ── Layer 1: Isolation Forest ─────────────────────────────────
    print("\nTraining Isolation Forest (Layer 1)...")
    scaler = StandardScaler()
    Xts = scaler.fit_transform(X_train)
    Xds = scaler.transform(X_dev)

    iso = IsolationForest(
        n_estimators=500,
        contamination=0.12,
        max_samples='auto',
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(Xts)

    train_scores = iso.score_samples(Xts)
    dev_scores = iso.score_samples(Xds)

    # Threshold: all CLAGN pass Layer 1 (set below 5th percentile of CLAGN)
    clagn_scores = train_scores[y_train == 1]
    if len(clagn_scores) == 0:
        iso_threshold = np.percentile(train_scores, 10)
    else:
        iso_threshold = np.percentile(clagn_scores, 5) - 0.01
    print(f"Isolation Forest threshold: {iso_threshold:.4f}")
    print(f"Train pass rate: {(train_scores > iso_threshold).mean():.1%} "
          f"(CLAGN pass: {(train_scores[y_train == 1] > iso_threshold).mean():.1%})")
    print(f"Dev pass rate:   {(dev_scores > iso_threshold).mean():.1%}")

    joblib.dump(iso, "models/isolation_forest.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(iso_threshold, "models/iso_threshold.pkl")

    # ── Layer 2: LightGBM + Optuna ────────────────────────────────
    l2_mask_train = train_scores > iso_threshold
    X_l2 = X_train[l2_mask_train].reset_index(drop=True)
    y_l2 = y_train[l2_mask_train]
    n_neg = (y_l2 == 0).sum()
    n_pos = (y_l2 == 1).sum()
    spw = n_neg / max(n_pos, 1)
    print(f"\nLayer 2 training data: {len(X_l2):,} sources "
          f"({n_pos} CLAGN, base scale_pos_weight={spw:.1f})")

    def objective(trial):
        p = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 15, 255),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float(
                'learning_rate', 0.005, 0.3, log=True),
            'min_child_samples': trial.suggest_int(
                'min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'subsample_freq': 1,
            'colsample_bytree': trial.suggest_float(
                'colsample_bytree', 0.4, 1.0),
            'reg_alpha': trial.suggest_float(
                'reg_alpha', 1e-5, 10.0, log=True),
            'reg_lambda': trial.suggest_float(
                'reg_lambda', 1e-5, 10.0, log=True),
            'scale_pos_weight': trial.suggest_float(
                'scale_pos_weight', spw * 0.5, spw * 3.0),
            'min_split_gain': trial.suggest_float(
                'min_split_gain', 0.0, 1.0),
            'n_estimators': 1000,
            'random_state': 42,
        }
        cv_bal, cv_auc = [], []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for tr_idx, val_idx in skf.split(X_l2, y_l2):
            clf = lgb.LGBMClassifier(**p)
            clf.fit(
                X_l2.iloc[tr_idx], y_l2[tr_idx],
                eval_set=[(X_l2.iloc[val_idx], y_l2[val_idx])],
                callbacks=[lgb.early_stopping(40, verbose=False),
                           lgb.log_evaluation(period=-1)],
            )
            proba = clf.predict_proba(X_l2.iloc[val_idx])[:, 1]
            pred = (proba >= 0.5).astype(int)
            cv_bal.append(balanced_accuracy_score(y_l2[val_idx], pred))
            try:
                cv_auc.append(roc_auc_score(y_l2[val_idx], proba))
            except ValueError:
                cv_auc.append(0.5)
        # Weighted objective: 70% balanced accuracy, 30% AUC
        return 0.7 * np.mean(cv_bal) + 0.3 * np.mean(cv_auc)

    print("\nRunning Optuna (50 trials)...")
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )
    study.optimize(objective, n_trials=50, show_progress_bar=True, n_jobs=1)

    print(f"\nBest trial value: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # Save Optuna study
    pd.DataFrame([{**t.params, 'value': t.value}
                   for t in study.trials
                   if t.value is not None]).to_csv(
        "results/optuna_trials.csv", index=False)

    # Train final model with best params
    best_p = {
        **study.best_params,
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'n_estimators': 2000,
        'random_state': 42,
    }
    print("\nTraining final model on all Layer 2 training data...")
    final_clf = lgb.LGBMClassifier(**best_p)

    # Dev data that passes Layer 1 for early stopping
    l2_mask_dev = dev_scores > iso_threshold
    if l2_mask_dev.sum() > 0:
        eval_X = X_dev[l2_mask_dev].reset_index(drop=True)
        eval_y = y_dev[l2_mask_dev]
        final_clf.fit(
            X_l2, y_l2,
            eval_set=[(eval_X, eval_y)],
            callbacks=[lgb.early_stopping(50, verbose=True),
                       lgb.log_evaluation(period=100)],
        )
    else:
        final_clf.fit(X_l2, y_l2)

    joblib.dump(final_clf, "models/lgbm_classifier.pkl")
    joblib.dump(feat_cols, "models/feature_columns.pkl")

    # ── SHAP values (for paper) ───────────────────────────────────
    print("\nComputing SHAP values...")
    explainer = shap.TreeExplainer(final_clf)
    shap_sample = X_l2.sample(min(2000, len(X_l2)), random_state=42)
    shap_vals = explainer.shap_values(shap_sample)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_vals, shap_sample, show=False,
                      max_display=20, plot_type='bar')
    plt.title("Feature Importance (SHAP)", fontsize=13)
    plt.tight_layout()
    plt.savefig("figures/fig_shap_importance.pdf", dpi=300,
                bbox_inches='tight')
    plt.savefig("figures/fig_shap_importance.png", dpi=150,
                bbox_inches='tight')
    plt.close()
    print("Saved SHAP figure")

    # Save SHAP mean values
    shap_df = pd.DataFrame({
        'feature': X_l2.columns,
        'mean_abs_shap': np.abs(shap_vals).mean(axis=0),
    }).sort_values('mean_abs_shap', ascending=False)
    shap_df.to_csv("results/shap_values.csv", index=False)

    # ── Dev evaluation ────────────────────────────────────────────
    print("\n=== DEV EVALUATION (not the final test) ===")
    y_dev_pred = np.zeros(len(y_dev), dtype=int)
    proba_dev = np.zeros(len(y_dev))

    if l2_mask_dev.sum() > 0:
        proba_l2 = final_clf.predict_proba(
            X_dev[l2_mask_dev].reset_index(drop=True))[:, 1]
        proba_dev[l2_mask_dev] = proba_l2
        y_dev_pred[l2_mask_dev] = (proba_l2 >= 0.5).astype(int)

    print(classification_report(
        y_dev, y_dev_pred, target_names=['non-CLAGN', 'CLAGN']))

    cm = confusion_matrix(y_dev, y_dev_pred)
    try:
        auc = float(roc_auc_score(y_dev, proba_dev))
    except ValueError:
        auc = 0.0

    metrics = {
        'overall_accuracy': round(float(accuracy_score(y_dev, y_dev_pred)), 4),
        'balanced_accuracy': round(float(
            balanced_accuracy_score(y_dev, y_dev_pred)), 4),
        'clagn_recall': round(float(
            recall_score(y_dev, y_dev_pred, zero_division=0)), 4),
        'clagn_precision': round(float(
            precision_score(y_dev, y_dev_pred, zero_division=0)), 4),
        'roc_auc': round(auc, 4),
        'mcc': round(float(matthews_corrcoef(y_dev, y_dev_pred)), 4),
        'confusion_matrix': cm.tolist(),
    }
    print(f"\nKey metrics:")
    for k, v in metrics.items():
        if k != 'confusion_matrix':
            print(f"  {k:25s}: {v}")
    print(f"\nConfusion matrix (TN, FP / FN, TP):")
    print(f"  {cm}")

    with open("results/dev_metrics_v2.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    # Fail-fast check
    if metrics['balanced_accuracy'] < 0.80:
        print(f"\nWARNING: Dev balanced accuracy "
              f"{metrics['balanced_accuracy']:.3f} < 0.80")
        print("Do NOT proceed to Step 6 until this is diagnosed and fixed.")
        print("Check: feature completeness, class balance, WISE coverage rate")
    else:
        print(f"\nDev balanced accuracy "
              f"{metrics['balanced_accuracy']:.3f} >= 0.80")
        print("Proceed to Step 6 (threshold calibration).")

    print("\nDO NOT run evaluate_benchmark.py test mode yet.")


if __name__ == "__main__":
    main()
