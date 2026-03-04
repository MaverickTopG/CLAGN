from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import DEFAULT_DPI, DEFAULT_FIGSIZE
from .utils import ensure_parent_dir



def _save(fig: plt.Figure, path: Path) -> None:
    ensure_parent_dir(path)
    fig.tight_layout()
    fig.savefig(path, dpi=DEFAULT_DPI)
    plt.close(fig)



def plot_score_hist(df: pd.DataFrame, out_path: Path, bins: int = 30) -> None:
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ok = pd.to_numeric(df.loc[df['is_ok'], 'score'], errors='coerce').dropna()
    rej = pd.to_numeric(df.loc[~df['is_ok'], 'score'], errors='coerce').dropna()
    if len(ok):
        ax.hist(ok, bins=bins, alpha=0.6, label=f'OK (N={len(ok)})')
    if len(rej):
        ax.hist(rej, bins=bins, alpha=0.6, label=f'Not OK (N={len(rej)})')
    if not len(ok) and not len(rej):
        ax.text(0.5, 0.5, 'No finite scores', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Score')
    ax.set_ylabel('Count')
    ax.set_title('Score Distribution: OK vs Rejected')
    ax.grid(True, alpha=0.25)
    if len(ok) or len(rej):
        ax.legend()
    _save(fig, out_path)



def _scatter(df: pd.DataFrame, x_col: str, y_col: str, out_path: Path, xlabel: str, ylabel: str, title: str) -> None:
    work = df.copy()
    work[x_col] = pd.to_numeric(work[x_col], errors='coerce')
    work[y_col] = pd.to_numeric(work[y_col], errors='coerce')
    work = work[np.isfinite(work[x_col]) & np.isfinite(work[y_col])].copy()
    # deterministic overplot: rejected first, then OK
    work['is_ok_sort'] = work['is_ok'].astype(int)
    work = work.sort_values(['is_ok_sort', 'canonical_id', 'source_id'], kind='mergesort')

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    if work.empty:
        ax.text(0.5, 0.5, f'No finite data for {title}', ha='center', va='center', transform=ax.transAxes)
    else:
        rej = work[~work['is_ok']]
        ok = work[work['is_ok']]
        if not rej.empty:
            ax.scatter(rej[x_col], rej[y_col], s=14, alpha=0.6, label=f'Not OK (N={len(rej)})', color='tab:gray')
        if not ok.empty:
            ax.scatter(ok[x_col], ok[y_col], s=14, alpha=0.7, label=f'OK (N={len(ok)})', color='tab:blue')
        ax.legend(loc='best')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    _save(fig, out_path)



def plot_score_vs_baseline(df: pd.DataFrame, out_path: Path) -> None:
    _scatter(df, 'baseline_years_recomputed', 'score', out_path, 'Baseline Years (recomputed)', 'Score', 'Score vs Baseline Years')



def plot_score_vs_npoints(df: pd.DataFrame, out_path: Path) -> None:
    _scatter(df, 'n_points', 'score', out_path, 'N Points', 'Score', 'Score vs N Points')



def plot_score_vs_deltamag(df: pd.DataFrame, out_path: Path) -> None:
    if 'delta_mag' not in df.columns:
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        ax.text(0.5, 0.5, 'delta_mag column missing', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Score vs Delta Mag')
        ax.grid(True, alpha=0.25)
        _save(fig, out_path)
        return
    _scatter(df, 'delta_mag', 'score', out_path, 'delta_mag', 'Score', 'Score vs delta_mag')
