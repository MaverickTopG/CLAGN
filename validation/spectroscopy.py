"""
Spectroscopic archive crossmatch and evidence tier classification.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from clagn.utils.crossmatch import angular_separation

SPECTROSCOPIC_ARCHIVES_TO_CHECK = [
    {'name': 'SDSS',  'method': 'astroquery.sdss', 'radius': 3.0},
    {'name': 'LAMOST', 'method': 'local_cache', 'radius': 3.0},
    {'name': 'DESI', 'method': 'local_cache', 'radius': 1.5},
    {'name': 'NED',  'method': 'astroquery.ned', 'radius': 3.0},
    {'name': '6dFGS', 'method': 'local_cache', 'radius': 3.0},
]

SPECTROSCOPIC_EVIDENCE_TIERS = {
    'tier_1_confirmed': {
        'criteria': [
            'Multi-epoch spectra showing broad line appearance OR disappearance',
            'Seyfert type change documented',
        ],
        'label': 'Spectroscopically confirmed CLAGN',
    },
    'tier_2_strong': {
        'criteria': [
            'Single spectrum showing broad lines consistent with AGN state',
            'Redshift confirmed spectroscopically',
            'No evidence of star/SN in spectrum',
        ],
        'label': 'Spectroscopically supported CLAGN candidate',
    },
    'tier_3_weak': {
        'criteria': [
            'Redshift from archival photometric redshift only',
            'No spectral typing available',
        ],
        'label': 'Photometric candidate (unconfirmed)',
    },
}

CACHE_DIR = Path('data/cache/spectroscopy')


def _query_sdss(ra, dec, radius_arcsec=3.0):
    try:
        from astroquery.sdss import SDSS
    except Exception:
        return []
    try:
        pos = f"{ra} {dec}"
        radius = radius_arcsec / 3600.0
        res = SDSS.query_region(pos, radius=radius, spectro=True)
        if res is None:
            return []
        df = res.to_pandas()
    except Exception:
        return []

    matches = []
    for _, row in df.iterrows():
        matches.append({
            'archive': 'SDSS',
            'ra': float(row.get('ra', np.nan)),
            'dec': float(row.get('dec', np.nan)),
            'z_spectroscopic': float(row.get('z', np.nan)) if 'z' in row else np.nan,
            'broad_lines_present': False,
            'seyfert_type_change': False,
            'meta': row.to_dict(),
        })
    return matches


def _query_ned(ra, dec, radius_arcsec=3.0):
    try:
        from astroquery.ned import Ned
    except Exception:
        return []
    try:
        res = Ned.query_region(f"{ra} {dec}", radius=radius_arcsec/3600.0)
        if res is None or len(res) == 0:
            return []
        df = res.to_pandas()
    except Exception:
        return []

    matches = []
    for _, row in df.iterrows():
        matches.append({
            'archive': 'NED',
            'ra': float(row.get('RA', np.nan)),
            'dec': float(row.get('DEC', np.nan)),
            'z_spectroscopic': float(row.get('Redshift', np.nan)) if 'Redshift' in row else np.nan,
            'broad_lines_present': False,
            'seyfert_type_change': False,
            'meta': row.to_dict(),
        })
    return matches


def _query_local_cache(archive: str, ra: float, dec: float, radius_arcsec: float):
    path = CACHE_DIR / f"{archive.lower()}.csv"
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if not {'ra', 'dec'}.issubset(df.columns):
        return []
    seps = angular_separation(ra, dec, df['ra'].values, df['dec'].values)
    mask = seps <= radius_arcsec
    matches = []
    for _, row in df[mask].iterrows():
        matches.append({
            'archive': archive,
            'ra': float(row.get('ra', np.nan)),
            'dec': float(row.get('dec', np.nan)),
            'z_spectroscopic': float(row.get('redshift', np.nan)) if 'redshift' in row else np.nan,
            'broad_lines_present': bool(row.get('broad_lines_present', False)),
            'seyfert_type_change': bool(row.get('seyfert_type_change', False)),
            'meta': row.to_dict(),
        })
    return matches


def classify_spectroscopic_evidence(candidate, spec_matches):
    if not spec_matches:
        return {'spectroscopic_tier': 'tier_3_weak', 'n_spectra': 0}

    n_spec = len(spec_matches)
    has_broad_lines = any(s.get('broad_lines_present') for s in spec_matches)
    has_type_change = any(s.get('seyfert_type_change') for s in spec_matches)
    z_confirmed = any(np.isfinite(s.get('z_spectroscopic', np.nan)) for s in spec_matches)

    if has_type_change or (n_spec >= 2 and has_broad_lines):
        tier = 'tier_1_confirmed'
    elif has_broad_lines and z_confirmed:
        tier = 'tier_2_strong'
    else:
        tier = 'tier_3_weak'

    return {
        'spectroscopic_tier': tier,
        'n_spectra': n_spec,
        'has_broad_lines': has_broad_lines,
        'has_type_change': has_type_change,
        'z_spectroscopic': z_confirmed,
        'spectroscopic_archives_checked': [s['archive'] for s in spec_matches],
    }


def crossmatch_candidate(ra: float, dec: float):
    matches = []
    for entry in SPECTROSCOPIC_ARCHIVES_TO_CHECK:
        name = entry['name']
        radius = entry['radius']
        method = entry['method']
        if method == 'astroquery.sdss':
            matches += _query_sdss(ra, dec, radius_arcsec=radius)
        elif method == 'astroquery.ned':
            matches += _query_ned(ra, dec, radius_arcsec=radius)
        else:
            matches += _query_local_cache(name, ra, dec, radius_arcsec=radius)
    return matches


def run_spectroscopic_crossmatch(candidates_csv: str = 'results/candidates/candidate_table.csv',
                                 output_dir: str = 'results/validation'):
    df = pd.read_csv(candidates_csv)
    top20 = df.head(20)

    rows = []
    for _, row in top20.iterrows():
        ra = float(row['ra'])
        dec = float(row['dec'])
        spec_matches = crossmatch_candidate(ra, dec)
        tier = classify_spectroscopic_evidence(row, spec_matches)

        rows.append({
            'source_id': row.get('source_id', row.get('name', '')),
            'ra': ra,
            'dec': dec,
            'spectroscopic_tier': tier['spectroscopic_tier'],
            'n_spectra': tier['n_spectra'],
            'has_broad_lines': tier.get('has_broad_lines', False),
            'has_type_change': tier.get('has_type_change', False),
            'z_spectroscopic': tier.get('z_spectroscopic', False),
            'archives_checked': ';'.join(tier.get('spectroscopic_archives_checked', [])),
        })

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'spectroscopy_matches.csv'
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    summary = {
        'n_top20': len(top20),
        'n_tier1_or_2': int(sum(r['spectroscopic_tier'] in ('tier_1_confirmed', 'tier_2_strong') for r in rows)),
    }
    with open(out_dir / 'spectroscopy_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return rows, summary


if __name__ == '__main__':
    run_spectroscopic_crossmatch()
