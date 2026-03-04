"""
Gaia confusion matrix validation on known AGN.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from clagn.ingestion.gaia import query_gaia_dr3


def compute_gaia_confusion_matrix(known_agn_list, output_dir: str = './results/'):
    results = []
    for source in known_agn_list:
        gaia_result = query_gaia_dr3(source['ra'], source['dec'], source['name'])
        results.append({
            'name': source['name'],
            'gaia_matched': gaia_result.get('gaia_found', False),
            'reject_as_star': gaia_result.get('star_reject_as_star', gaia_result.get('reject_as_star', False)),
            'rejection_reason': gaia_result.get('star_rejection_reason', gaia_result.get('rejection_reason', 'none')),
            'pm_sig': gaia_result.get('pm_sig', np.nan),
            'parallax_sig': gaia_result.get('parallax_sig', np.nan),
            'ruwe': gaia_result.get('ruwe', np.nan),
        })

    df = pd.DataFrame(results)
    false_negative_rate = float(df['reject_as_star'].mean()) if len(df) else 0.0

    if false_negative_rate > 0.10:
        raise ValueError(
            f"Gaia filter rejects {false_negative_rate:.1%} of known AGN — "
            f"threshold too aggressive. Max allowed: 10%."
        )

    report = {
        'n_known_agn_tested': len(df),
        'false_negative_rate': false_negative_rate,
        'rejection_reasons': df[df['reject_as_star']]['rejection_reason'].value_counts().to_dict(),
        'pass_criterion': false_negative_rate <= 0.10,
    }

    out_dir = Path(output_dir) / 'validation'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'gaia_confusion.json', 'w') as f:
        json.dump(report, f, indent=2)

    return report


def main():
    # Use benchmark known_clagn.csv as known AGN list
    import pandas as pd
    df = pd.read_csv('data/benchmark/known_clagn.csv')
    known = df[['name', 'ra', 'dec']].to_dict(orient='records')
    compute_gaia_confusion_matrix(known)


if __name__ == '__main__':
    main()
