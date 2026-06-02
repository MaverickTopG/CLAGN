from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_benchmark_build_and_freeze():
    bench_dir = Path('tests/data/benchmark')
    cmd = [sys.executable, 'scripts/build_benchmark_datasets.py',
           '--benchmark_dir', str(bench_dir),
           '--data_dir', 'tests/data']
    subprocess.run(cmd, check=True)

    master = bench_dir / 'benchmark_master.csv'
    train_dev = bench_dir / 'benchmark_train_dev.csv'
    test = bench_dir / 'benchmark_test.csv'
    split = bench_dir / 'TRAIN_DEV_TEST_SPLIT.csv'
    split_hash = bench_dir / 'SPLIT_HASH.txt'
    freeze = bench_dir / 'BENCHMARK_FREEZE.json'

    assert master.exists()
    assert train_dev.exists()
    assert test.exists()
    assert split.exists()
    assert split_hash.exists()
    assert freeze.exists()

    data = json.loads(freeze.read_text())
    assert 'benchmark_hash' in data
    assert 'counts' in data and data['counts']
    assert 'timestamp_utc' in data
