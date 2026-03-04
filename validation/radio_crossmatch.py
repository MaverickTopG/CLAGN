"""
Radio catalog crossmatch (FIRST/NVSS) with caching.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from clagn.utils.crossmatch import angular_separation

CACHE_DIR = Path('data/cache/radio')
CATALOG_CFG = CACHE_DIR / 'catalogs.json'


def _load_catalog_cfg():
    if not CATALOG_CFG.exists():
        return {}
    try:
        return json.loads(CATALOG_CFG.read_text())
    except Exception:
        return {}


def _vizier_cone_search(catalog: str, ra: float, dec: float, radius_arcsec: float, columns):
    try:
        from astroquery.vizier import Vizier
    except Exception:
        raise RuntimeError("astroquery.vizier required for radio crossmatch")

    v = Vizier(columns=columns)
    radius = radius_arcsec / 3600.0
    tbls = v.query_region(f"{ra} {dec}", radius=f"{radius}d", catalog=catalog)
    if not tbls:
        return []
    df = tbls[0].to_pandas()
    matches = []
    for _, row in df.iterrows():
        r = float(row[columns[0]]) if columns[0] in row else np.nan
        d = float(row[columns[1]]) if columns[1] in row else np.nan
        sep = float(angular_separation(ra, dec, r, d)) if np.isfinite(r) and np.isfinite(d) else np.nan
        matches.append({'ra': r, 'dec': d, 'sep_arcsec': sep, 'row': row.to_dict()})
    return matches


def crossmatch_radio(source_id: str, ra: float, dec: float, allow_network: bool = True):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{source_id}_radio.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass

    cfg = _load_catalog_cfg()
    results = {}

    for key in ['first', 'nvss']:
        entry = cfg.get(key, {})
        catalog = entry.get('catalog')
        columns = entry.get('columns', ['RAJ2000', 'DEJ2000'])
        radius_arcsec = float(entry.get('radius_arcsec', 5.0))
        if not catalog:
            results[key] = []
            continue
        if not allow_network:
            results[key] = []
            continue
        try:
            results[key] = _vizier_cone_search(catalog, ra, dec, radius_arcsec, columns)
        except Exception:
            results[key] = []

    cache_path.write_text(json.dumps(results, indent=2, default=str))
    return results
