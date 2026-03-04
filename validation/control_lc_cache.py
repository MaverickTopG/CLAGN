"""
Cache WISE light curves for control AGN for injection-recovery.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from clagn.ingestion.wise import query_wise_lightcurve


def load_control_lcs(benchmark_dir: str = 'data/benchmark/',
                     cache_dir: str = 'data/cache/control_lcs/',
                     n_hosts: int = 50,
                     force: bool = False):
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    cache_file = cache_path / f'control_lcs_n{n_hosts}.pkl'
    if cache_file.exists() and not force:
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    control = pd.read_csv(Path(benchmark_dir) / 'normal_agn_control.csv')
    control = control.head(n_hosts)

    lcs = []
    for _, row in control.iterrows():
        source_id = str(row.get('name', row.get('source_id', 'control')))
        ra = float(row['ra'])
        dec = float(row['dec'])
        z = float(row.get('redshift', row.get('z', 0.1)))

        wise = query_wise_lightcurve(ra, dec, source_id, z=z)
        lc = wise.get('lc')
        if lc is None or len(lc) < 10:
            continue
        lcs.append({
            'source_id': source_id,
            'mjd': lc['mjd'].values,
            'w1_flux_mjy': lc['w1_flux_mjy'].values,
            'w1_flux_err_mjy': lc['w1_flux_err_mjy'].values,
            'z': z,
        })

    with open(cache_file, 'wb') as f:
        pickle.dump(lcs, f, protocol=pickle.HIGHEST_PROTOCOL)

    return lcs
