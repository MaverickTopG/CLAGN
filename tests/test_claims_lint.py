from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_claims_lint_blocks_forbidden(tmp_path):
    scan_dir = tmp_path / 'paper'
    scan_dir.mkdir(parents=True, exist_ok=True)
    (scan_dir / 'main.md').write_text('We confirmed CLAGN in this sample.')

    out_dir = tmp_path / 'results'
    cmd = [sys.executable, 'scripts/claims_lint.py',
           '--scan_dir', str(scan_dir),
           '--output_dir', str(out_dir),
           '--benchmark_dir', 'tests/data/benchmark']
    # The lint command should fail when forbidden certainty wording is present.
    proc = subprocess.run(cmd)
    assert proc.returncode != 0


def test_claims_lint_allows_candidates(tmp_path):
    scan_dir = tmp_path / 'paper'
    scan_dir.mkdir(parents=True, exist_ok=True)
    (scan_dir / 'main.md').write_text('We report changing-state AGN candidates.')

    out_dir = tmp_path / 'results'
    cmd = [sys.executable, 'scripts/claims_lint.py',
           '--scan_dir', str(scan_dir),
           '--output_dir', str(out_dir),
           '--benchmark_dir', 'tests/data/benchmark']
    proc = subprocess.run(cmd)
    assert proc.returncode == 0
    assert (out_dir / 'metrics' / 'claims_lint.json').exists()
