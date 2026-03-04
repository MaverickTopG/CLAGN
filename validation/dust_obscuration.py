"""
Dust obscuration check via W1-W2 color vs W1 flux correlation.
"""
from __future__ import annotations

import numpy as np


def color_flux_correlation(w1_seasonal: dict, w2_seasonal: dict):
    """
    Compute correlation between W1 seasonal flux and W1-W2 color.

    Returns dict with correlation coefficient and dust_obscuration_flag.
    """
    seasons = sorted(set(w1_seasonal.keys()) & set(w2_seasonal.keys()))
    if len(seasons) < 3:
        return {'color_flux_correlation': np.nan, 'dust_obscuration_flag': False}

    w1v = np.array([w1_seasonal[s] for s in seasons])
    color = np.array([w1_seasonal[s] - w2_seasonal[s] for s in seasons])

    if len(color) < 3:
        return {'color_flux_correlation': np.nan, 'dust_obscuration_flag': False}

    r = float(np.corrcoef(w1v, color)[0, 1])

    return {
        'color_flux_correlation': r,
        'dust_obscuration_flag': bool(r > 0.7),
        'color_flux_correlation_method': 'seasonal_median_pearson'
    }
