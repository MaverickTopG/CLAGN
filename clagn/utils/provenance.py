"""
provenance.py — Write JSON provenance sidecar files alongside output CSVs.

Every output CSV produced by the pipeline gets a companion
*_provenance.json file recording all thresholds, constants,
scoring weights, and the git commit used.

FLAW C1 FIX: Without provenance, results are not reproducible.
"""
import json
import datetime
import subprocess
import sys

import numpy as np

from clagn import config


def get_git_hash():
    """Return short git commit hash, or 'unknown' if git is unavailable."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return 'unknown'


def write_provenance_sidecar(output_csv_path, pipeline_config):
    """
    Write a JSON sidecar next to every output CSV.

    Example: top_candidates.csv → top_candidates_provenance.json

    Parameters
    ----------
    output_csv_path : str or Path
        Path to the output CSV file.
    pipeline_config : dict
        Pipeline configuration (e.g. vars(args) from argparse).

    Returns
    -------
    sidecar_path : str
        Path to the written JSON sidecar.
    """
    provenance = {
        'generated_at': datetime.datetime.utcnow().isoformat() + 'Z',
        'git_commit': get_git_hash(),
        'python_version': sys.version,

        # All threshold values that affect results
        'thresholds': {
            'min_baseline_years': pipeline_config.get('min_baseline_years', 10.0),
            'min_epochs': pipeline_config.get('min_epochs', 20),
            'min_delta_mag': pipeline_config.get('min_delta_mag', 0.3),
            'min_changepoint_bic': pipeline_config.get('min_changepoint_bic', 6.0),
            'min_drw_nonstat_sigma': pipeline_config.get('min_drw_nonstat_sigma', 2.0),
            'gaia_max_pm_sig': 3.0,
            'gaia_max_parallax_sig': 3.0,
            'gaia_max_ruwe': 1.4,
            'wise_min_snr': 5.0,
            'wise_season_anchor_mjd': config.WISE_SEASON_ANCHOR_MJD,
        },

        # All physical constants used
        'constants': {
            'wise_w1_zero_point_jy': config.WISE_VEGA_ZERO_POINTS['W1'],
            'wise_w2_zero_point_jy': config.WISE_VEGA_ZERO_POINTS['W2'],
            'wise_hibernation_start_mjd': config.WISE_HIBERNATION_MJD_START,
            'wise_hibernation_end_mjd': config.WISE_HIBERNATION_MJD_END,
        },

        # Scoring weights
        'scoring_weights': {
            'w_amplitude': 3.0,
            'w_temporal': 3.0,
            'w_color': 2.0,
            'w_statistical': 1.5,
            'w_astrometric': 0.5,
            'scoring_version': 'v3_no_double_counting',
        },

        # DRW configuration
        'drw': {
            'backend': 'celerite2_with_numpy_fallback',
            'fit_in_flux_space': True,
            'mean_subtracted_before_fit': True,
            'rest_frame_correction_applied': True,
            'kozlowski2017_reliability_check': True,
        },

        # Delta-mag method
        'delta_mag': {
            'method': 'seasonal_weighted_mean_flux_comparison',
            'season_anchor_mjd': config.WISE_SEASON_ANCHOR_MJD,
            'sign_convention': 'positive_is_brightening',
            'early_seasons': 'first_2',
            'late_seasons': 'last_2',
        },
    }

    sidecar_path = str(output_csv_path).replace('.csv', '_provenance.json')
    with open(sidecar_path, 'w') as f:
        json.dump(provenance, f, indent=2)

    return sidecar_path
