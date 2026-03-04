"""
Supernova rejection heuristics and transient catalog crossmatch.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from clagn.models.variability import cluster_epochs_into_seasons
from clagn.utils.crossmatch import angular_separation
from clagn.config import WISE_SEASON_ANCHOR_MJD

TRANSIENT_CACHE_DIR = Path('data/cache/transients')


def _seasonal_medians(times, fluxes):
    seasons = cluster_epochs_into_seasons(times)
    medians = {}
    for s in np.unique(seasons):
        medians[int(s)] = float(np.nanmedian(fluxes[seasons == s]))
    return medians


def check_sn_morphology(times_mjd, flux_mjy, season_medians=None):
    """
    Flag light curves that match SN-like morphology.
    """
    times_mjd = np.asarray(times_mjd, dtype=float)
    flux_mjy = np.asarray(flux_mjy, dtype=float)
    if season_medians is None:
        season_medians = _seasonal_medians(times_mjd, flux_mjy)

    seasons = sorted(season_medians.keys())
    if len(seasons) < 3:
        return {
            'single_season_dominance_flag': False,
            'transition_recurrence_check': False,
            'sn_like_morphology': False,
            'sn_morphology_note': 'insufficient_seasons'
        }

    med = np.array([season_medians[s] for s in seasons])
    baseline = np.nanmedian([med[0], med[-1]])
    if not np.isfinite(baseline) or baseline <= 0:
        return {
            'single_season_dominance_flag': False,
            'transition_recurrence_check': False,
            'sn_like_morphology': False,
            'sn_morphology_note': 'invalid_baseline'
        }

    peak_idx = int(np.nanargmax(med))
    peak = med[peak_idx]

    # Elevated if peak > 1.5x baseline
    elevated = med > 1.5 * baseline
    n_elev = int(elevated.sum())

    single_season = n_elev == 1
    returns_to_baseline = True
    if peak_idx > 0:
        returns_to_baseline &= (med[peak_idx - 1] < 1.2 * baseline)
    if peak_idx < len(med) - 1:
        returns_to_baseline &= (med[peak_idx + 1] < 1.2 * baseline)

    sn_like = bool(single_season and returns_to_baseline)

    return {
        'single_season_dominance_flag': single_season,
        'transition_recurrence_check': n_elev > 1,  # recurrent = not SN
        'sn_like_morphology': sn_like,
        'sn_morphology_note': 'single_season_peak' if sn_like else 'not_sn_like'
    }


def color_reversal_timescale_flag(w1_seasonal: dict, w2_seasonal: dict, max_seasons: int = 2):
    """
    Flag if W1-W2 color reverses (rise then fall) within ~1 year (~2 seasons).
    """
    seasons = sorted(set(w1_seasonal.keys()) & set(w2_seasonal.keys()))
    if len(seasons) < 3:
        return False

    color = np.array([w1_seasonal[s] - w2_seasonal[s] for s in seasons])
    # Detect sign change in first derivative within max_seasons steps
    dcolor = np.diff(color)
    for i in range(len(dcolor) - 1):
        if np.sign(dcolor[i]) != 0 and np.sign(dcolor[i + 1]) != 0:
            if np.sign(dcolor[i]) != np.sign(dcolor[i + 1]) and (i + 1) <= max_seasons:
                return True
    return False


def transient_catalog_xmatch(ra, dec, transition_mjd,
                             max_sep_arcsec=2.0, max_dt_days=730.0):
    """
    Crossmatch with cached transient catalogs (TNS/ASAS-SN/ALeRCE).
    Expected cache files: tns.csv, asas_sn.csv, alerce.csv with columns
    ra, dec, discovery_mjd, name.
    """
    matches = []
    if transition_mjd is None or not np.isfinite(transition_mjd):
        return matches

    for fname in ['tns.csv', 'asas_sn.csv', 'alerce.csv']:
        path = TRANSIENT_CACHE_DIR / fname
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if not {'ra', 'dec', 'discovery_mjd'}.issubset(df.columns):
            continue

        seps = angular_separation(ra, dec, df['ra'].values, df['dec'].values)
        dt = np.abs(df['discovery_mjd'].values - transition_mjd)
        mask = (seps <= max_sep_arcsec) & (dt <= max_dt_days)
        for _, row in df[mask].iterrows():
            matches.append({
                'catalog': fname.replace('.csv', ''),
                'name': row.get('name', ''),
                'sep_arcsec': float(angular_separation(ra, dec, row['ra'], row['dec'])),
                'dt_days': float(abs(row['discovery_mjd'] - transition_mjd)),
            })

    return matches
