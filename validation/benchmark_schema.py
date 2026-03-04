"""
Benchmark dataset schema and count validation.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

KNOWN_CLAGN_REQUIRED_COLS = [
    'name', 'ra', 'dec', 'redshift', 'transition_type',
    'transition_epoch_mjd', 'delta_mag_reported', 'reference', 'tier'
]

CONTAMINANT_REQUIRED_COLS = [
    'name', 'ra', 'dec', 'redshift', 'contaminant_type', 'reference', 'notes'
]

CONTAMINANT_TYPES_REQUIRED = [
    'sn_in_host', 'blazar', 'variable_star', 'normal_agn'
]

MIN_KNOWN_CLAGN = 30
MIN_CONTAMINANTS_TOTAL = 80
MIN_CONTROL_AGN = 200


def _check_cols(df: pd.DataFrame, required: list[str], label: str) -> list[str]:
    missing = [c for c in required if c not in df.columns]
    if missing:
        return [f"{label} missing columns: {missing}"]
    return []


def validate_benchmark_files(benchmark_dir: str = 'data/benchmark/') -> dict:
    bench = Path(benchmark_dir)
    errors = []

    known_path = bench / 'known_clagn.csv'
    cont_path = bench / 'known_contaminants.csv'
    ctrl_path = bench / 'normal_agn_control.csv'

    if not known_path.exists():
        errors.append(f"Missing {known_path}")
        known = pd.DataFrame()
    else:
        known = pd.read_csv(known_path)
        errors += _check_cols(known, KNOWN_CLAGN_REQUIRED_COLS, 'known_clagn')

    if not cont_path.exists():
        errors.append(f"Missing {cont_path}")
        contaminants = pd.DataFrame()
    else:
        contaminants = pd.read_csv(cont_path)
        errors += _check_cols(contaminants, CONTAMINANT_REQUIRED_COLS, 'known_contaminants')

    if not ctrl_path.exists():
        errors.append(f"Missing {ctrl_path}")
        control = pd.DataFrame()
    else:
        control = pd.read_csv(ctrl_path)

    # Counts
    if len(known) < MIN_KNOWN_CLAGN:
        errors.append(f"known_clagn count {len(known)} < {MIN_KNOWN_CLAGN}")

    if len(contaminants) < MIN_CONTAMINANTS_TOTAL:
        errors.append(f"known_contaminants count {len(contaminants)} < {MIN_CONTAMINANTS_TOTAL}")

    if len(control) < MIN_CONTROL_AGN:
        errors.append(f"normal_agn_control count {len(control)} < {MIN_CONTROL_AGN}")

    # Contaminant type counts
    if not contaminants.empty and 'contaminant_type' in contaminants.columns:
        for ctype in CONTAMINANT_TYPES_REQUIRED:
            n = int((contaminants['contaminant_type'] == ctype).sum())
            if ctype in ('sn_in_host', 'blazar', 'variable_star') and n < 10:
                errors.append(f"contaminant_type {ctype} count {n} < 10")
            if ctype == 'normal_agn' and n < 50:
                errors.append(f"contaminant_type {ctype} count {n} < 50")

    # Reference presence
    if not known.empty and 'reference' in known.columns:
        if known['reference'].isna().any():
            errors.append("known_clagn contains missing references")
    if not contaminants.empty and 'reference' in contaminants.columns:
        if contaminants['reference'].isna().any():
            errors.append("known_contaminants contains missing references")

    return {
        'pass': len(errors) == 0,
        'errors': errors,
        'counts': {
            'known_clagn': len(known),
            'contaminants': len(contaminants),
            'control_agn': len(control),
        },
    }


def main() -> None:
    report = validate_benchmark_files()
    if report['pass']:
        print("BENCHMARK SCHEMA: PASS")
        print(report['counts'])
        sys.exit(0)
    print("BENCHMARK SCHEMA: FAIL")
    for err in report['errors']:
        print(f"- {err}")
    sys.exit(1)


if __name__ == '__main__':
    main()
