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
from importlib import metadata

import numpy as np

from clagn import config


def get_git_hash():
    """Return git commit hash, or 'unavailable' if git is unavailable."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return 'unavailable'


def get_git_describe():
    """Return git describe --tags, or 'unavailable'."""
    try:
        return subprocess.check_output(
            ['git', 'describe', '--tags', '--always'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return 'unavailable'


def _pkg_version(name: str) -> str:
    try:
        return metadata.version(name)
    except Exception:
        return 'unavailable'


def write_provenance_sidecar(output_csv_path, config_module=config,
                             results_summary=None, pipeline_args=None,
                             operating_threshold=None, wise_tables_queried=None,
                             random_seed=None, irsa_api_version=None):
    """
    Write a JSON sidecar next to every output CSV.

    Example: top_candidates.csv → top_candidates_provenance.json
    """
    results_summary = results_summary or {}
    pipeline_args = pipeline_args or {}

    provenance = {
        'pipeline_version': get_git_describe(),
        'git_hash': get_git_hash(),
        'run_timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
        'config_snapshot': {k: v for k, v in vars(config_module).items() if not k.startswith('_')},
        'python_version': sys.version,
        'key_package_versions': {
            'numpy': _pkg_version('numpy'),
            'scipy': _pkg_version('scipy'),
            'astropy': _pkg_version('astropy'),
            'celerite2': _pkg_version('celerite2'),
            'emcee': _pkg_version('emcee'),
        },
        'random_seed': random_seed,
        'n_sources_input': results_summary.get('n_sources_input'),
        'n_sources_output': results_summary.get('n_sources_output'),
        'operating_threshold': operating_threshold,
        'wise_tables_queried': wise_tables_queried or [],
        'irsa_api_version': irsa_api_version or 'unavailable',
        'results_summary': results_summary,
        'pipeline_args': pipeline_args,
    }

    sidecar_path = str(output_csv_path).replace('.csv', '_provenance.json')
    with open(sidecar_path, 'w') as f:
        json.dump(provenance, f, indent=2, default=str)

    return sidecar_path
