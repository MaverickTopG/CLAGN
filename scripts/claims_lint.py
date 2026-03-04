"""
Claims lint gate for manuscript text.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clagn.utils.validation_metadata import build_metadata, write_json_with_metadata

FORBIDDEN_PHRASES = [
    "confirmed clagn",
    "confirm changing-look",
    "confirmed changing-look",
    "confirmed changing look",
    "we discovered clagn",
    "we confirm changing-look behavior",
    "confirmed changing-look behavior",
    "discovered changing-look",
]

METRIC_PATTERNS = [
    r'\\bprecision\\b',
    r'\\brecall\\b',
    r'\\bf1\\b',
    r'\\bf-1\\b',
    r'\\bauc\\b',
    r'\\broc\\b',
    r'pr\\s*curve',
    r'roc\\s*curve',
    r'\\baccuracy\\b',
]

CI_PATTERNS = [
    r'confidence interval',
    r'\\bci\\b',
    r'bootstrap',
]


def _find_scan_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    if Path('manuscript').exists():
        return Path('manuscript')
    return Path('paper')


def _spectroscopy_table_exists(output_dir: Path) -> bool:
    candidates = [
        output_dir / 'validation' / 'spectroscopy_matches.csv',
        output_dir / 'metrics' / 'spectroscopy_table.csv',
        output_dir / 'validation' / 'spectroscopy_table.csv',
    ]
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan_dir', default=None)
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--benchmark_dir', default='data/benchmark')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    scan_dir = _find_scan_dir(args.scan_dir)
    if not scan_dir.exists():
        print(f"Scan directory not found: {scan_dir}")
        sys.exit(1)

    files = list(scan_dir.rglob('*.tex')) + list(scan_dir.rglob('*.md')) + list(scan_dir.rglob('*.txt'))
    if not files:
        print("No manuscript files found to scan")
        sys.exit(1)

    violations = []
    metric_mentions = False
    ci_mentions = False
    for path in files:
        text = path.read_text(errors='ignore').lower()
        for phrase in FORBIDDEN_PHRASES:
            if phrase in text:
                violations.append({'file': str(path), 'phrase': phrase})
        if any(re.search(pat, text) for pat in METRIC_PATTERNS):
            metric_mentions = True
        if any(re.search(pat, text) for pat in CI_PATTERNS):
            ci_mentions = True

    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / 'metrics'
    metadata_dir = output_dir / 'metadata'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    spec_present = _spectroscopy_table_exists(output_dir)

    # Metrics CI enforcement if metrics are reported in text
    ci_missing = False
    if metric_mentions and not ci_mentions:
        ci_missing = True

    # If metrics_summary exists, it must contain CI fields
    metrics_summary = None
    for path in [output_dir / 'metrics_summary.json', output_dir / 'metrics' / 'metrics_summary.json']:
        if path.exists():
            metrics_summary = json.loads(path.read_text())
            break
    if metric_mentions and metrics_summary is not None:
        ci_fields = [k for k in metrics_summary.keys() if k.startswith('ci_')]
        if not ci_fields:
            ci_missing = True

    status = 'pass'
    if violations and not spec_present:
        status = 'fail'
    elif violations and spec_present:
        status = 'pass_with_spectroscopy'
    if ci_missing:
        status = 'fail'

    summary = {
        'status': status,
        'violations': violations,
        'spectroscopy_table_present': spec_present,
        'scanned_files': [str(p) for p in files],
        'metric_mentions': metric_mentions,
        'ci_mentions': ci_mentions,
        'ci_missing': ci_missing,
    }

    meta = build_metadata(Path(args.benchmark_dir))

    if not args.dry_run:
        write_json_with_metadata(metrics_dir / 'claims_lint.json', summary, metadata_dir, meta)

    if status == 'fail':
        if ci_missing:
            print("Claims lint FAILED. Metrics mentioned without confidence intervals.")
        else:
            print("Claims lint FAILED. Forbidden phrases found without spectroscopy table.")
        sys.exit(1)

    print("Claims lint PASS")


if __name__ == '__main__':
    main()
