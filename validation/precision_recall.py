"""
Precision-recall characterization for the CLAGN pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_curve, auc

from clagn.pipeline import CLAGNPipeline


def compute_precision_recall_curve(pipeline_scores: pd.DataFrame,
                                   benchmark_labels: pd.DataFrame,
                                   score_col: str = 'composite_score_v2',
                                   label_col: str = 'is_clagn') -> dict:
    merged = pipeline_scores.merge(benchmark_labels, on=['ra', 'dec'], how='inner')
    if merged.empty:
        raise ValueError("No overlap between pipeline scores and benchmark labels")

    y_true = merged[label_col].astype(int).values
    y_score = merged[score_col].values

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    auc_pr = auc(recall, precision)

    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = int(np.argmax(f1))
    if len(thresholds) == 0:
        best_threshold = 0.5
    elif best_idx >= len(thresholds):
        best_threshold = float(thresholds[-1])
    else:
        best_threshold = float(thresholds[best_idx])

    return {
        'thresholds': thresholds,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_pr': float(auc_pr),
        'operating_point': {
            'threshold': float(best_threshold),
            'precision': float(precision[best_idx]),
            'recall': float(recall[best_idx]),
            'f1': float(f1[best_idx]),
        }
    }


def run_full_benchmark(pipeline: CLAGNPipeline, benchmark_dir: str = 'data/benchmark/'):
    bench = Path(benchmark_dir)
    known_clagn = pd.read_csv(bench / 'known_clagn.csv')
    contaminants = pd.read_csv(bench / 'known_contaminants.csv')
    control_agn = pd.read_csv(bench / 'normal_agn_control.csv')

    results = {}
    for name, df in [('known_clagn', known_clagn),
                     ('contaminants', contaminants),
                     ('control_agn', control_agn)]:
        results[name] = pipeline.run(df)

    all_sources = pd.concat([
        known_clagn.assign(is_clagn=1),
        contaminants.assign(is_clagn=0),
        control_agn.assign(is_clagn=0),
    ], ignore_index=True)

    all_scores = pd.concat([
        results['known_clagn'],
        results['contaminants'],
        results['control_agn'],
    ], ignore_index=True)

    pr = compute_precision_recall_curve(all_scores, all_sources)

    # Contaminant rejection rates
    for ctype in ['sn_in_host', 'blazar', 'variable_star']:
        ct_subset = contaminants[contaminants['contaminant_type'] == ctype]
        ct_scores = results['contaminants'][
            results['contaminants']['name'].isin(ct_subset['name'])
        ]
        if len(ct_scores) == 0:
            results[f'{ctype}_rejection_rate'] = np.nan
            continue
        reject_rate = (ct_scores['composite_score_v2'] < pr['operating_point']['threshold']).mean()
        results[f'{ctype}_rejection_rate'] = float(reject_rate)

    return results, pr


def _plot_precision_recall(pr: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(pr['recall'], pr['precision'], color='steelblue', lw=2)
    op = pr['operating_point']
    ax.scatter([op['recall']], [op['precision']], color='red', zorder=5)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_score_distributions(scores: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, df, color in [
        ('known_clagn', scores['known_clagn'], 'seagreen'),
        ('contaminants', scores['contaminants'], 'tomato'),
        ('control_agn', scores['control_agn'], 'gray')
    ]:
        if 'composite_score_v2' in df.columns:
            ax.hist(df['composite_score_v2'].values, bins=30, alpha=0.5, label=label, color=color)
    ax.set_xlabel('Composite Score')
    ax.set_ylabel('Count')
    ax.set_title('Score Distributions')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_recovery_vs_delta_mag(scores: dict, known_clagn: pd.DataFrame, threshold: float, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    merged = scores['known_clagn'].merge(known_clagn, on=['ra', 'dec'], how='inner')
    if 'delta_mag_reported' not in merged.columns:
        return
    merged = merged.dropna(subset=['delta_mag_reported'])
    if merged.empty:
        return
    merged['recovered'] = merged['composite_score_v2'] >= threshold
    bins = np.linspace(0, np.nanmax(np.abs(merged['delta_mag_reported'])) + 0.1, 8)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    rec = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (np.abs(merged['delta_mag_reported']) >= lo) & (np.abs(merged['delta_mag_reported']) < hi)
        if mask.sum() == 0:
            rec.append(np.nan)
        else:
            rec.append(float(merged.loc[mask, 'recovered'].mean()))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(bin_centers, rec, 'o-')
    ax.set_xlabel('|delta_mag| (reported)')
    ax.set_ylabel('Recovery rate')
    ax.set_title('Recovery vs delta_mag')
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_recovery_vs_baseline(scores: dict, known_clagn: pd.DataFrame, threshold: float, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    merged = scores['known_clagn'].merge(known_clagn, on=['ra', 'dec'], how='inner')
    if merged.empty:
        return
    merged['recovered'] = merged['composite_score_v2'] >= threshold
    bins = np.linspace(0, np.nanmax(merged['baseline_years']) + 0.1, 8)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    rec = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (merged['baseline_years'] >= lo) & (merged['baseline_years'] < hi)
        if mask.sum() == 0:
            rec.append(np.nan)
        else:
            rec.append(float(merged.loc[mask, 'recovered'].mean()))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(bin_centers, rec, 'o-')
    ax.set_xlabel('Baseline (years)')
    ax.set_ylabel('Recovery rate')
    ax.set_title('Recovery vs baseline')
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main(results_dir: str = './results/', benchmark_dir: str = 'data/benchmark/') -> dict:
    pipeline = CLAGNPipeline(results_dir=results_dir)
    scores, pr = run_full_benchmark(pipeline, benchmark_dir=benchmark_dir)

    out_dir = Path(results_dir) / 'validation'
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_precision_recall(pr, out_dir / 'precision_recall.png')
    _plot_score_distributions(scores, out_dir / 'score_distributions.png')

    # Save score tables for downstream checks
    scores['known_clagn'].to_csv(out_dir / 'benchmark_scores_known_clagn.csv', index=False)
    scores['contaminants'].to_csv(out_dir / 'benchmark_scores_contaminants.csv', index=False)
    scores['control_agn'].to_csv(out_dir / 'benchmark_scores_control.csv', index=False)

    known_clagn = pd.read_csv(Path(benchmark_dir) / 'known_clagn.csv')
    _plot_recovery_vs_delta_mag(scores, known_clagn, pr['operating_point']['threshold'],
                                out_dir / 'recovery_vs_delta_mag.png')
    _plot_recovery_vs_baseline(scores, known_clagn, pr['operating_point']['threshold'],
                               out_dir / 'recovery_vs_baseline.png')

    # Save operating threshold
    op_path = out_dir / 'operating_threshold.json'
    with open(op_path, 'w') as f:
        json.dump(pr['operating_point'], f, indent=2)

    # Control AGN false positive rate
    control_scores = scores['control_agn']
    if len(control_scores) > 0:
        fpr_control = float((control_scores['composite_score_v2'] >= pr['operating_point']['threshold']).mean())
    else:
        fpr_control = np.nan

    summary = {
        'auc_pr': pr['auc_pr'],
        'precision': pr['operating_point']['precision'],
        'recall': pr['operating_point']['recall'],
        'threshold': pr['operating_point']['threshold'],
        'sn_rejection_rate': scores.get('sn_in_host_rejection_rate', np.nan),
        'blazar_rejection_rate': scores.get('blazar_rejection_rate', np.nan),
        'variable_star_rejection_rate': scores.get('variable_star_rejection_rate', np.nan),
        'control_false_positive_rate': fpr_control,
    }
    with open(out_dir / 'precision_recall_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return {'scores': scores, 'pr': pr}


if __name__ == '__main__':
    main()
