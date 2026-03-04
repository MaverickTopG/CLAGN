from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_metrics_offline(tmp_path):
    bench_dir = Path('tests/data/benchmark')
    scores = bench_dir / 'scores' / 'benchmark_scores.csv'

    # Ensure benchmark split exists
    subprocess.run([
        sys.executable, 'scripts/build_benchmark_datasets.py',
        '--benchmark_dir', str(bench_dir),
        '--data_dir', 'tests/data'
    ], check=True)

    # Dev evaluation
    dev_out = tmp_path / 'dev'
    subprocess.run([
        sys.executable, 'scripts/evaluate_benchmark.py',
        '--benchmark_dir', str(bench_dir),
        '--scores_csv', str(scores),
        '--split', 'dev',
        '--output_dir', str(dev_out),
        '--n_bootstrap', '10'
    ], check=True)

    op = dev_out / 'metrics' / 'operating_threshold.json'
    assert op.exists()
    threshold = json.loads(op.read_text())['threshold']

    # Test evaluation
    test_out = tmp_path / 'test'
    subprocess.run([
        sys.executable, 'scripts/evaluate_benchmark.py',
        '--benchmark_dir', str(bench_dir),
        '--scores_csv', str(scores),
        '--split', 'test',
        '--output_dir', str(test_out),
        '--threshold', str(threshold),
        '--n_bootstrap', '10'
    ], check=True)

    summary_path = test_out / 'metrics' / 'metrics_summary.json'
    assert summary_path.exists()
    assert (test_out / 'figures' / 'pr_curve.png').exists()

    summary = json.loads(summary_path.read_text())
    assert 'git_commit' in summary
    assert 'config_hash' in summary
    assert 'benchmark_hash' in summary
    assert 'created_utc' in summary
