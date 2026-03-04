from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def prepare_ok_vs_rejected_subsets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        'ok': df[df['is_ok']].copy(),
        'rejected': df[~df['is_ok']].copy(),
    }



def safe_spearman(x: pd.Series, y: pd.Series) -> dict[str, Any]:
    x_num = pd.to_numeric(x, errors='coerce')
    y_num = pd.to_numeric(y, errors='coerce')
    mask = np.isfinite(x_num.to_numpy(float)) & np.isfinite(y_num.to_numpy(float))
    n = int(mask.sum())
    if n < 3:
        return {'available': False, 'rho': None, 'pvalue': None, 'n': n, 'reason': 'fewer than 3 finite pairs'}
    rho, p = spearmanr(x_num[mask], y_num[mask])
    if not np.isfinite(rho) or not np.isfinite(p):
        return {'available': False, 'rho': None, 'pvalue': None, 'n': n, 'reason': 'undefined statistic'}
    return {'available': True, 'rho': float(rho), 'pvalue': float(p), 'n': n}



def compute_internal_correlations(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    ok = df[df['is_ok']].copy()
    results: dict[str, dict[str, Any]] = {}
    if ok.empty:
        return {
            'score_vs_n_points': {'available': False, 'rho': None, 'pvalue': None, 'n': 0, 'reason': 'no OK rows'},
            'score_vs_baseline_years': {'available': False, 'rho': None, 'pvalue': None, 'n': 0, 'reason': 'no OK rows'},
            'score_vs_delta_mag': {'available': False, 'rho': None, 'pvalue': None, 'n': 0, 'reason': 'no OK rows'},
            'score_vs_abs_delta_mag': {'available': False, 'rho': None, 'pvalue': None, 'n': 0, 'reason': 'no OK rows'},
        }

    results['score_vs_n_points'] = safe_spearman(ok['score'], ok['n_points'])
    results['score_vs_baseline_years'] = safe_spearman(ok['score'], ok['baseline_years_recomputed'])

    if 'delta_mag' in ok.columns:
        results['score_vs_delta_mag'] = safe_spearman(ok['score'], ok['delta_mag'])
        abs_dm = pd.to_numeric(ok['delta_mag'], errors='coerce').abs()
        results['score_vs_abs_delta_mag'] = safe_spearman(ok['score'], abs_dm)
    else:
        for k in ['score_vs_delta_mag', 'score_vs_abs_delta_mag']:
            results[k] = {'available': False, 'rho': None, 'pvalue': None, 'n': 0, 'reason': 'delta_mag column missing'}
    return results
