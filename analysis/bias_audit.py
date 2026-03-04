from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clagn_audit.constants import DEFAULT_BIAS_BINS, DEFAULT_DPI, DEFAULT_FIGSIZE
from clagn_audit.io import read_results_csv, write_csv
from clagn_audit.schema import validate_required_columns, coerce_types, warn_duplicate_canonical_ids
from clagn_audit.cleaning import add_derived_columns
from clagn_audit.markdown import df_to_markdown_table
from clagn_audit.utils import ensure_parent_dir, stable_sort_df


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1.0 + (z**2 / n)
    center = (p + z**2 / (2*n)) / denom
    margin = (z / denom) * math.sqrt((p*(1-p)/n) + (z**2 / (4*n*n)))
    lo = center - margin
    hi = center + margin
    # Numerical guardrails for plotting/error bars.
    lo = max(0.0, lo)
    hi = min(1.0, hi)
    return (lo, hi)



def _qcut_summary(df: pd.DataFrame, value_col: str, n_bins: int, label: str) -> pd.DataFrame:
    sub = df.copy()
    sub[value_col] = pd.to_numeric(sub[value_col], errors='coerce')
    sub = sub[np.isfinite(sub[value_col])].copy()
    if sub.empty:
        return pd.DataFrame(columns=['audit', 'bin', 'n_total', 'n_rejected', 'rejection_rate', 'ci_low', 'ci_high'])
    try:
        bins = pd.qcut(sub[value_col], q=n_bins, duplicates='drop')
    except ValueError:
        bins = pd.cut(sub[value_col], bins=min(n_bins, max(1, sub[value_col].nunique())), duplicates='drop')
    sub['_bin'] = bins.astype(str)
    rows = []
    for b, grp in sub.groupby('_bin', dropna=False):
        n_total = int(len(grp))
        n_rej = int((~grp['is_ok']).sum())
        rate = n_rej / n_total if n_total else math.nan
        lo, hi = wilson_ci(n_rej, n_total)
        rows.append({'audit': label, 'bin': str(b), 'n_total': n_total, 'n_rejected': n_rej, 'rejection_rate': rate, 'ci_low': lo, 'ci_high': hi})
    out = pd.DataFrame(rows)
    return out.sort_values('bin', kind='mergesort').reset_index(drop=True)



def _category_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cat, grp in df.groupby(df['band_coverage'].astype(str), dropna=False):
        n_total = int(len(grp))
        n_rej = int((~grp['is_ok']).sum())
        rate = n_rej / n_total if n_total else math.nan
        lo, hi = wilson_ci(n_rej, n_total)
        rows.append({'audit': 'band_coverage', 'bin': str(cat), 'n_total': n_total, 'n_rejected': n_rej, 'rejection_rate': rate, 'ci_low': lo, 'ci_high': hi})
    return pd.DataFrame(rows).sort_values('bin', kind='mergesort').reset_index(drop=True)



def _plot_rate(summary: pd.DataFrame, out_path: Path, title: str, x_label: str) -> None:
    ensure_parent_dir(out_path)
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    if summary.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
    else:
        x = np.arange(len(summary))
        y = summary['rejection_rate'].to_numpy(float)
        ci_low = np.clip(summary['ci_low'].to_numpy(float), 0.0, 1.0)
        ci_high = np.clip(summary['ci_high'].to_numpy(float), 0.0, 1.0)
        y_clipped = np.clip(y, 0.0, 1.0)
        # Matplotlib errorbar requires non-negative uncertainties.
        yerr_low = np.maximum(0.0, y_clipped - ci_low)
        yerr_high = np.maximum(0.0, ci_high - y_clipped)
        ax.errorbar(x, y_clipped, yerr=[yerr_low, yerr_high], fmt='o-', capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(summary['bin'].astype(str).tolist(), rotation=25, ha='right')
        ax.set_ylim(0, 1)
    ax.set_ylabel('Rejection Rate')
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)



def run_bias_audit(df: pd.DataFrame, outdir: Path, n_bins: int = DEFAULT_BIAS_BINS) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    baseline_summary = _qcut_summary(df, 'baseline_years_recomputed', n_bins, 'baseline_years')
    npoints_summary = _qcut_summary(df, 'n_points', n_bins, 'n_points')
    band_summary = _category_summary(df)

    summary_df = pd.concat([baseline_summary, npoints_summary, band_summary], ignore_index=True)
    write_csv(summary_df, outdir / 'bias_audit_summary.csv')

    _plot_rate(baseline_summary, outdir / 'fig_reject_rate_vs_baseline_bins.png', 'Rejection Rate vs Baseline Years Bins', 'Baseline-year quantile bins')
    _plot_rate(npoints_summary, outdir / 'fig_reject_rate_vs_npoints_bins.png', 'Rejection Rate vs N Points Bins', 'N-point quantile bins')
    _plot_rate(band_summary, outdir / 'fig_reject_rate_vs_bandcoverage.png', 'Rejection Rate vs Band Coverage', 'Band coverage')

    md_lines = [
        '# Bias Audit (Proxy)',
        '',
        '## Definitions',
        '',
        '- `rejected = ~is_ok`',
        '- Baseline and n_points bins are quantile-based (deterministic `qcut`, `duplicates="drop"`).',
        '- This is a proxy bias audit using available columns only; it does not establish causality.',
        '',
        '## Rejection vs Baseline Bins',
        '',
        df_to_markdown_table(baseline_summary),
        '',
        '## Rejection vs N Points Bins',
        '',
        df_to_markdown_table(npoints_summary),
        '',
        '## Band Coverage Rejection Rates',
        '',
        df_to_markdown_table(band_summary),
        '',
        '## Interpretation Caveats',
        '',
        '- Rejection rates reflect current status logic and data availability, not true astrophysical prevalence.',
        '- Bin-wise differences may be driven by survey cadence, sky position, brightness, or pipeline guardrails.',
        '',
    ]
    (outdir / 'bias_audit.md').write_text('\n'.join(md_lines))

    return {
        'baseline_summary': baseline_summary,
        'npoints_summary': npoints_summary,
        'band_summary': band_summary,
        'summary_df': summary_df,
    }



def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--n_bins_bias', type=int, default=DEFAULT_BIAS_BINS)
    args = parser.parse_args()

    df = read_results_csv(Path(args.input))
    validate_required_columns(df)
    df = coerce_types(df)
    _ = warn_duplicate_canonical_ids(df)
    df = add_derived_columns(stable_sort_df(df, ['canonical_id', 'source_id']))
    run_bias_audit(df, Path(args.outdir), n_bins=args.n_bins_bias)
    print('Bias audit complete')


if __name__ == '__main__':
    _cli()
