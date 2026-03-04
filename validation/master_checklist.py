"""
Master checklist gate for CLAGN validation standard.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from validation.benchmark_schema import validate_benchmark_files
from validation.claims_check import check_paper


RESULTS_DIR = Path('results')
BENCH_DIR = Path('data/benchmark')


def _fail(msg, errors):
    errors.append(msg)


def main():
    errors = []

    # Science Definition
    if not Path('definitions.md').exists():
        _fail('definitions.md missing', errors)
    if not (Path('paper') / 'definitions.md').exists():
        _fail('paper/definitions.md missing', errors)

    # Benchmark dataset
    bench_report = validate_benchmark_files(str(BENCH_DIR))
    if not bench_report['pass']:
        errors += bench_report['errors']

    # Pipeline validation (precision-recall)
    pr_path = RESULTS_DIR / 'validation' / 'precision_recall_summary.json'
    if not pr_path.exists():
        _fail('precision_recall_summary.json missing', errors)
    else:
        pr = json.loads(pr_path.read_text())
        if pr.get('auc_pr', 0) <= 0.7:
            _fail('AUC-PR <= 0.7', errors)
        if pr.get('precision', 0) <= 0.6 or pr.get('recall', 0) <= 0.6:
            _fail('precision/recall below 0.6 at operating threshold', errors)
        if pr.get('sn_rejection_rate', 0) < 0.8:
            _fail('SN rejection rate < 80%', errors)
        if pr.get('blazar_rejection_rate', 0) < 0.7:
            _fail('Blazar rejection rate < 70%', errors)
        if pr.get('control_false_positive_rate', 1.0) >= 0.05:
            _fail('Control AGN false positive rate >= 5%', errors)

    # Injection-recovery
    inj_csv = RESULTS_DIR / 'validation' / 'injection_recovery_results.csv'
    if not inj_csv.exists():
        _fail('injection_recovery_results.csv missing', errors)
    else:
        df = pd.read_csv(inj_csv)
        if df['morphology'].nunique() < 4:
            _fail('Not all 4 morphologies tested', errors)
    for fig in ['injection_recovery_heatmap.png']:
        if not (RESULTS_DIR / 'validation' / fig).exists():
            _fail(f'{fig} missing', errors)
    if not (RESULTS_DIR / 'validation' / 'detection_threshold.json').exists():
        _fail('detection_threshold.json missing', errors)

    # Contaminant war plan (implementation presence)
    for f in ['validation/sn_rejection.py', 'validation/blazar_rejection.py', 'validation/dust_obscuration.py']:
        if not Path(f).exists():
            _fail(f'{f} missing', errors)

    # Gaia validation
    gaia_path = RESULTS_DIR / 'validation' / 'gaia_confusion.json'
    if not gaia_path.exists():
        _fail('gaia_confusion.json missing', errors)
    else:
        gaia = json.loads(gaia_path.read_text())
        if gaia.get('false_negative_rate', 1.0) >= 0.10:
            _fail('Gaia false negative rate >= 10%', errors)

    # Spectroscopic validation
    spec_path = RESULTS_DIR / 'validation' / 'spectroscopy_summary.json'
    if not spec_path.exists():
        _fail('spectroscopy_summary.json missing', errors)
    else:
        spec = json.loads(spec_path.read_text())
        if spec.get('n_tier1_or_2', 0) < 3:
            _fail('Fewer than 3 Tier 1/2 candidates', errors)

    # Reproducibility
    if not Path('reproduce_all.py').exists():
        _fail('reproduce_all.py missing', errors)
    if not Path('environment.yml').exists():
        _fail('environment.yml missing', errors)
    if not Path('requirements.txt').exists():
        _fail('requirements.txt missing', errors)

    # Provenance sidecar
    cand_csv = RESULTS_DIR / 'candidates' / 'candidate_table.csv'
    if cand_csv.exists():
        prov = cand_csv.with_name(cand_csv.stem + '_provenance.json')
        if not prov.exists():
            _fail('candidate_table_provenance.json missing', errors)

    # Sensitivity analysis
    sens_report = RESULTS_DIR / 'validation' / 'sensitivity' / 'sensitivity_report.json'
    if not sens_report.exists():
        _fail('sensitivity_report.json missing', errors)
    else:
        report = json.loads(sens_report.read_text())
        if not report.get('pass', False):
            _fail('Sensitivity analysis failed', errors)

    # Paper text
    if not Path('limitations.md').exists():
        _fail('limitations.md missing', errors)
    paper_tex = Path('paper/clagn_paper.tex')
    if not paper_tex.exists():
        _fail('paper/clagn_paper.tex missing', errors)
    else:
        claims = check_paper(str(paper_tex))
        if not claims['pass']:
            errors += claims['errors']

    if errors:
        print('MASTER CHECKLIST: FAIL')
        for e in errors:
            print(f'- {e}')
        sys.exit(1)

    print('MASTER CHECKLIST: PASS')
    sys.exit(0)


if __name__ == '__main__':
    main()
