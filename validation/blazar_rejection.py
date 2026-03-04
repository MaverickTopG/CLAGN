"""
Blazar rejection using radio crossmatch and BZCAT membership.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from clagn.utils.crossmatch import angular_separation
from .radio_crossmatch import crossmatch_radio, _load_catalog_cfg, _vizier_cone_search

CACHE_DIR = Path('data/cache/radio')


def _bzcat_local_match(ra: float, dec: float, max_sep_arcsec: float = 3.0):
    path = CACHE_DIR / 'bzcat.csv'
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if not {'ra', 'dec'}.issubset(df.columns):
        return []
    seps = angular_separation(ra, dec, df['ra'].values, df['dec'].values)
    mask = seps <= max_sep_arcsec
    matches = []
    for _, row in df[mask].iterrows():
        matches.append({
            'name': row.get('name', row.get('bzcat', '')),
            'sep_arcsec': float(angular_separation(ra, dec, row['ra'], row['dec'])),
        })
    return matches


def _bzcat_vizier_match(ra: float, dec: float, max_sep_arcsec: float = 3.0):
    cfg = _load_catalog_cfg()
    entry = cfg.get('bzcat', {})
    catalog = entry.get('catalog')
    if not catalog:
        return []
    cols = entry.get('columns', ['RAJ2000', 'DEJ2000'])
    try:
        matches = _vizier_cone_search(catalog, ra, dec, max_sep_arcsec, cols)
    except Exception:
        return []
    return matches


def is_blazar(source_id: str, ra: float, dec: float, allow_network: bool = True):
    radio = crossmatch_radio(source_id, ra, dec, allow_network=allow_network)
    radio_detected = any(len(v) > 0 for v in radio.values())

    bz_matches = _bzcat_local_match(ra, dec)
    if not bz_matches and allow_network:
        bz_matches = _bzcat_vizier_match(ra, dec)

    is_bzcat = len(bz_matches) > 0

    return {
        'radio_detected': radio_detected,
        'bzcat_match': is_bzcat,
        'bzcat_matches': bz_matches,
        'radio_matches': radio,
        'blazar_flag': bool(is_bzcat or (radio_detected and is_bzcat)),
    }
