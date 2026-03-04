"""
Fetch real WISE light curves for control AGN to use as injection hosts.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from astroquery.vizier import Vizier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clagn.ingestion.wise import query_wise_lightcurve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/real_clagn')
    parser.add_argument('--n_hosts', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min_epochs', type=int, default=6)
    args = parser.parse_args()

    base = Path(args.data_dir)
    hosts_dir = base / 'hosts'
    hosts_dir.mkdir(parents=True, exist_ok=True)

    v = Vizier(columns=['*'])
    agn = v.get_catalogs('VII/289/dr16q')[0].to_pandas()
    agn = agn.sample(n=min(len(agn), args.n_hosts * 6), random_state=args.seed)

    candidates = []
    for _, row in agn.iterrows():
        candidates.append({
            'ra': float(row['RAJ2000']),
            'dec': float(row['DEJ2000']),
            'source_id': str(row['SDSS']),
            'z': float(row['z']) if 'z' in row and pd.notna(row['z']) else 0.1,
        })

    saved = 0
    attempted = 0

    def fetch_one(item: dict):
        try:
            res = query_wise_lightcurve(item['ra'], item['dec'], item['source_id'], z=item['z'])
        except Exception:
            return None
        lc = res.get('lc')
        if lc is None or len(lc) < args.min_epochs:
            return None
        return lc

    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(fetch_one, item): item for item in candidates}
        for fut in as_completed(futures):
            attempted += 1
            if saved >= args.n_hosts:
                break
            lc = fut.result()
            if lc is None:
                continue
            out = hosts_dir / f"host_{saved:03d}.csv"
            lc[['mjd', 'w1_flux_mjy', 'w1_flux_err_mjy']].to_csv(out, index=False)
            saved += 1
            print(f"Saved host {saved}/{args.n_hosts} after {attempted} attempts")

    if saved < args.n_hosts:
        print(f"Only saved {saved}/{args.n_hosts} hosts. Try increasing sampling.")
        sys.exit(1)

    print(f"Saved {saved} host light curves to {hosts_dir}")


if __name__ == '__main__':
    main()
