"""
Single-command reproduction of validation outputs.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import random

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clagn.utils.validation_metadata import build_metadata, write_json_with_metadata


def _run(cmd: list[str]) -> None:
    cmd = [c for c in cmd if c]
    print("[reproduce] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _read_threshold(path: Path) -> float:
    if not path.exists():
        raise FileNotFoundError(f"Missing operating threshold: {path}")
    data = json.loads(path.read_text())
    if 'threshold' not in data:
        raise ValueError("operating_threshold.json missing threshold")
    return float(data['threshold'])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--benchmark_dir', default='data/benchmark')
    parser.add_argument('--mode', choices=['standard', 'real'], default='standard')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--allow_missing_injection_hosts', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    data_dir = Path(args.data_dir)
    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now(timezone.utc).isoformat()

    if args.dry_run:
        print("Dry run: reproduce_all skipping execution")
        return

    if args.mode == 'real':
        real_dir = Path('data/real_clagn')
        real_out = Path('results_real')
        real_out.mkdir(parents=True, exist_ok=True)

        # Build benchmark + freeze
        _run([sys.executable, 'scripts/build_real_benchmark.py',
              '--data_dir', str(real_dir)])

        # Split benchmark
        _run([sys.executable, 'scripts/split_real_benchmark.py',
              '--data_dir', str(real_dir)])

        # Dev evaluation (threshold selection)
        _run([sys.executable, 'scripts/evaluate_real_clagn.py',
              '--data_dir', str(real_dir),
              '--split', 'dev',
              '--output_dir', str(real_out)])

        # Test evaluation (must be once)
        _run([sys.executable, 'scripts/evaluate_real_clagn.py',
              '--data_dir', str(real_dir),
              '--split', 'test',
              '--output_dir', str(real_out)])

        # Injection-recovery
        _run([sys.executable, 'scripts/run_real_injection_recovery.py',
              '--data_dir', str(real_dir),
              '--output_dir', str(real_out)])

        # Contaminant audit + coverage
        _run([sys.executable, 'scripts/contaminant_audit.py',
              '--data_dir', str(real_dir),
              '--output_dir', str(real_out)])

        # Selection function report
        _run([sys.executable, 'scripts/write_selection_function_report.py',
              '--data_dir', str(real_dir),
              '--output_dir', str(real_out)])

        # Claims lint
        scan_dir = None
        if (real_dir / 'paper').exists():
            scan_dir = real_dir / 'paper'
        elif (real_dir / 'manuscript').exists():
            scan_dir = real_dir / 'manuscript'
        _run([sys.executable, 'scripts/claims_lint.py',
              '--output_dir', str(real_out),
              '--benchmark_dir', str(real_dir),
              '--scan_dir', str(scan_dir) if scan_dir else None])

        required = [
            real_dir / 'BENCHMARK_FREEZE.json',
            real_dir / 'SPLIT_HASH.json',
            real_out / 'metrics_summary.json',
            real_out / 'pr_curve.png',
            real_out / 'roc_curve.png',
            real_out / 'confusion_matrix.png',
            real_out / 'completeness_heatmap.png',
            real_out / 'contaminant_report.json',
            real_out / 'selection_function_report.md',
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            print("Missing required artifacts:\n" + "\n".join(missing))
            sys.exit(1)
    else:
        # Benchmark build + freeze
        _run([sys.executable, 'scripts/build_benchmark_datasets.py',
              '--data_dir', str(args.data_dir),
              '--benchmark_dir', str(args.benchmark_dir)])
        _run([sys.executable, 'scripts/check_label_leakage.py',
              '--benchmark', str(benchmark_dir / 'benchmark_master.csv')])

        # Dev evaluation (threshold selection)
        dev_out = output_dir / 'dev'
        dev_out.mkdir(parents=True, exist_ok=True)
        _run([sys.executable, 'scripts/evaluate_benchmark.py',
              '--benchmark_dir', str(args.benchmark_dir),
              '--split', 'dev',
              '--output_dir', str(dev_out),
              '--offline' if args.offline else None])

        threshold_path = dev_out / 'metrics' / 'operating_threshold.json'
        threshold = _read_threshold(threshold_path)

        # Test evaluation
        _run([sys.executable, 'scripts/evaluate_benchmark.py',
              '--benchmark_dir', str(args.benchmark_dir),
              '--split', 'test',
              '--output_dir', str(output_dir),
              '--threshold', str(threshold),
              '--offline' if args.offline else None])

        # Injection-recovery
        hosts_dir = data_dir / 'hosts'
        if not hosts_dir.exists() or not list(hosts_dir.glob('*.csv')):
            if args.allow_missing_injection_hosts:
                print("[reproduce] injection hosts missing; skipping injection-recovery")
            else:
                print("[reproduce] injection hosts missing. Provide data/hosts CSVs or use --allow_missing_injection_hosts")
                sys.exit(1)
        else:
            _run([sys.executable, 'scripts/run_injection_recovery.py',
                  '--data_dir', str(args.data_dir),
                  '--benchmark_dir', str(args.benchmark_dir),
                  '--output_dir', str(output_dir),
                  '--offline' if args.offline else None])
            grid_path = output_dir / 'injection_recovery' / 'grid_results.csv'
            if grid_path.exists():
                _run([sys.executable, 'scripts/plot_injection_recovery.py',
                      '--input_csv', str(grid_path),
                      '--output_dir', str(output_dir),
                      '--benchmark_dir', str(args.benchmark_dir)])

        # Coverage report
        _run([sys.executable, 'scripts/report_crossmatch_coverage.py',
              '--data_dir', str(args.data_dir),
              '--output_dir', str(output_dir),
              '--benchmark_dir', str(args.benchmark_dir),
              '--offline' if args.offline else None])

        # Claims lint
        scan_dir = None
        if (data_dir / 'paper').exists():
            scan_dir = data_dir / 'paper'
        elif (data_dir / 'manuscript').exists():
            scan_dir = data_dir / 'manuscript'
        _run([sys.executable, 'scripts/claims_lint.py',
              '--output_dir', str(output_dir),
              '--benchmark_dir', str(args.benchmark_dir),
              '--scan_dir', str(scan_dir) if scan_dir else None,
              '--offline' if args.offline else None])

    end_time = datetime.now(timezone.utc).isoformat()

    if args.mode == 'real':
        real_dir = Path('data/real_clagn')
        real_out = Path('results_real')
        split_hash = None
        split_path = real_dir / 'SPLIT_HASH.json'
        if split_path.exists():
            try:
                split_hash = json.loads(split_path.read_text()).get('split_hash')
            except Exception:
                split_hash = None
        meta = build_metadata(real_dir, extra={
            'command': ' '.join(sys.argv),
            'start_utc': start_time,
            'end_utc': end_time,
            'output_dir': str(real_out),
            'benchmark_dir': str(real_dir),
            'split_hash': split_hash,
        })
        write_json_with_metadata(real_out / 'RUN_MANIFEST.json', meta.copy(), real_out / 'metadata', meta)
    else:
        meta = build_metadata(benchmark_dir, extra={
            'command': ' '.join(sys.argv),
            'start_utc': start_time,
            'end_utc': end_time,
            'output_dir': str(output_dir),
            'data_dir': str(data_dir),
            'benchmark_dir': str(benchmark_dir),
        })
        write_json_with_metadata(output_dir / 'RUN_MANIFEST.json', meta.copy(), output_dir / 'metadata', meta)

    print("[reproduce] complete")


if __name__ == '__main__':
    main()
