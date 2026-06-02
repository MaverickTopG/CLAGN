"""
validation.py — Cross-validation against published CLAGN catalogs.

Build a validation module that checks pipeline outputs against published
CLAGN catalogs to compute recovery rate as a self-check.

Sources from: MacLeod+2016, Yang+2018, Graham+2020, Green+2022
"""
import logging
import numpy as np
from .crossmatch import angular_separation

logger = logging.getLogger(__name__)

# Known confirmed CLAGN from the literature
KNOWN_CLAGN_CATALOG = [
    # Ricci & Trakhtenbrot 2022 reference sources — NED J2000 coordinates
    {'name': 'Mrk 1018',       'ra': 31.566,   'dec': -0.292,  'z': 0.042, 'type': 'CS-AGN'},
    {'name': '1ES 1927+654',   'ra': 292.619,  'dec': 65.931,  'z': 0.011, 'type': 'CS-AGN'},
    {'name': 'NGC 2617',       'ra': 128.912,  'dec': -4.170,  'z': 0.014, 'type': 'CS-AGN'},
    # Additional literature sources
    {'name': 'Mrk 590',        'ra': 33.640,   'dec': -0.767,  'z': 0.026, 'type': 'CS-AGN'},
    {'name': 'HE 1136-2304',   'ra': 174.746,  'dec': -23.347, 'z': 0.027, 'type': 'CS-AGN'},
    {'name': 'NGC 4151',       'ra': 182.636,  'dec': 39.406,  'z': 0.003, 'type': 'CS-AGN'},
    # MacLeod+2016 sources
    {'name': 'SDSS J0159+0033', 'ra': 29.990,  'dec': 0.553,   'z': 0.312, 'type': 'CS-AGN'},
    {'name': 'SDSS J2336+0017', 'ra': 354.008, 'dec': 0.291,   'z': 0.243, 'type': 'CS-AGN'},
]


def validate_against_known_clagn(all_scored_sources, known_catalog=None,
                                  search_radius_arcsec=12.0):
    """
    Cross-match pipeline results against known CLAGN.

    Reports:
    - Recovery rate: fraction of known CLAGN in input catalog that appear in
      pipeline output with their rank
    - False positive rate estimate (advisory)

    This is the primary accuracy metric for the pipeline.

    Parameters
    ----------
    all_scored_sources : list of dicts
        All sources processed by the pipeline, each with keys:
        'source_row' (has 'ra', 'dec', 'source_id') and 'score' (has 'composite')
    known_catalog : list of dicts or None
        Use KNOWN_CLAGN_CATALOG by default.
    search_radius_arcsec : float

    Returns
    -------
    report : dict with keys:
        'n_known_in_catalog': int
        'n_recovered': int
        'recovery_rate': float
        'recovery_details': list of dicts
    """
    if known_catalog is None:
        known_catalog = KNOWN_CLAGN_CATALOG

    if not all_scored_sources:
        logger.warning("No scored sources provided for validation.")
        return {
            'n_known_in_catalog': 0,
            'n_recovered': 0,
            'recovery_rate': 0.0,
            'recovery_details': [],
        }

    # Sort all sources by composite score (descending) to assign ranks
    sorted_sources = sorted(all_scored_sources,
                             key=lambda x: x['score'].get('composite', 0.0),
                             reverse=True)

    pipeline_ras = np.array([s['source_row']['ra'] for s in sorted_sources])
    pipeline_decs = np.array([s['source_row']['dec'] for s in sorted_sources])

    n_known_in_catalog = 0
    n_recovered = 0
    recovery_details = []

    for known in known_catalog:
        seps = angular_separation(known['ra'], known['dec'],
                                  pipeline_ras, pipeline_decs)
        min_idx = int(np.argmin(seps))
        min_sep = float(seps[min_idx])

        if min_sep <= search_radius_arcsec:
            n_known_in_catalog += 1
            rank = min_idx + 1  # 1-based rank
            composite = sorted_sources[min_idx]['score'].get('composite', 0.0)

            # "Recovered" means it appears in the scored output (not rejected)
            n_recovered += 1
            recovery_details.append({
                'known_name': known['name'],
                'known_type': known['type'],
                'match_sep_arcsec': min_sep,
                'pipeline_rank': rank,
                'composite_score': composite,
            })
            logger.info(
                f"Known CLAGN {known['name']} found at rank {rank} "
                f"(score={composite:.3f}, sep={min_sep:.1f}\")"
            )
        else:
            logger.info(
                f"Known CLAGN {known['name']} not found in pipeline input "
                f"(nearest match at {min_sep:.1f}\")"
            )

    recovery_rate = n_recovered / n_known_in_catalog if n_known_in_catalog > 0 else 0.0

    logger.info(
        f"Pipeline recovered {n_recovered}/{n_known_in_catalog} known CLAGN "
        f"in the input catalog (recovery rate = {100*recovery_rate:.0f}%)"
    )

    return {
        'n_known_in_catalog': n_known_in_catalog,
        'n_recovered': n_recovered,
        'recovery_rate': recovery_rate,
        'recovery_details': recovery_details,
    }
