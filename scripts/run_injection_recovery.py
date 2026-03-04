"""
Offline injection-recovery grid using cached host light curves.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clagn.utils.validation_metadata import build_metadata, write_metadata_sidecar, write_json_with_metadata


def _parse_list(values: str) -> list[float]:
    parts = [p.strip() for p in values.replace(';', ',').split(',') if p.strip()]
    return [float(p) for p in parts]


def _load_host_lightcurves(hosts_dir: Path) -> list[dict]:
    host_lcs: list[dict] = []
    for path in sorted(hosts_dir.glob('*.csv')):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        cols = {c.lower(): c for c in df.columns}
        time_col = cols.get('mjd') or cols.get('time') or cols.get('t')
        mag_col = cols.get('mag') or cols.get('w1_mag') or cols.get('magnitude')
        err_col = cols.get('mag_err') or cols.get('err') or cols.get('magerr')
        if not time_col or not mag_col:
            continue
        time = pd.to_numeric(df[time_col], errors='coerce').values
        mag = pd.to_numeric(df[mag_col], errors='coerce').values
        if err_col:
            err = pd.to_numeric(df[err_col], errors='coerce').values
        else:
            err = np.full_like(mag, 0.05, dtype=float)
        mask = np.isfinite(time) & np.isfinite(mag) & np.isfinite(err)
        if mask.sum() < 6:
            continue
        host_lcs.append({'path': path, 'time': time[mask], 'mag': mag[mask], 'err': err[mask]})
    return host_lcs


def _choose_t0(time: np.ndarray, timescale_days: float, rng: np.random.Generator) -> float | None:
    time = np.sort(time)
    candidates = []
    for t0 in time:
        pre = (time < t0).sum()
        post = (time >= t0 + timescale_days).sum()
        if pre >= 3 and post >= 3:
            candidates.append(t0)
    if not candidates:
        return None
    return float(rng.choice(candidates))


def _inject(time: np.ndarray, mag: np.ndarray, amp: float, timescale_days: float, shape: str, t0: float) -> np.ndarray:
    delta = np.zeros_like(mag, dtype=float)
    if shape == 'step':
        delta[time >= t0] = -amp
    elif shape == 'ramp':
        t1 = t0 + timescale_days
        ramp_mask = (time >= t0) & (time <= t1)
        after_mask = time > t1
        if timescale_days <= 0:
            delta[time >= t0] = -amp
        else:
            delta[ramp_mask] = -amp * (time[ramp_mask] - t0) / timescale_days
            delta[after_mask] = -amp
    else:
        delta[time >= t0] = -amp
    return mag + delta


def _detect(time: np.ndarray, mag: np.ndarray, timescale_days: float, t0: float, threshold: float) -> bool:
    pre_mask = time < t0
    post_mask = time >= (t0 + timescale_days)
    if pre_mask.sum() < 3 or post_mask.sum() < 3:
        return False
    pre_med = np.median(mag[pre_mask])
    post_med = np.median(mag[post_mask])
    delta = pre_med - post_med
    return abs(delta) >= threshold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--hosts_dir', default=None)
    parser.add_argument('--benchmark_dir', default=None)
    parser.add_argument('--n_hosts', type=int, default=50)
    parser.add_argument('--n_realizations', type=int, default=20)
    parser.add_argument('--amplitudes', default='0.2,0.5,1.0,1.5')
    parser.add_argument('--timescales', default='0.5,1.0,2.0,5.0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    hosts_dir = Path(args.hosts_dir) if args.hosts_dir else data_dir / 'hosts'
    inj_dir = output_dir / 'injection_recovery'
    fig_dir = output_dir / 'figures'
    metrics_dir = output_dir / 'metrics'
    metadata_dir = output_dir / 'metadata'
    inj_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    if not hosts_dir.exists():
        print(f"Host light curve directory not found: {hosts_dir}")
        print("Provide host light curves or use --hosts_dir to point to cached CSVs.")
        sys.exit(1)

    host_lcs = _load_host_lightcurves(hosts_dir)
    if not host_lcs:
        print("No valid host light curves found. Expected CSVs with time/mag columns.")
        sys.exit(1)

    if args.n_hosts < len(host_lcs):
        rng = np.random.default_rng(args.seed)
        host_lcs = list(rng.choice(host_lcs, size=args.n_hosts, replace=False))

    amplitudes = _parse_list(args.amplitudes)
    timescales = _parse_list(args.timescales)
    shapes = ['step', 'ramp']

    rng = np.random.default_rng(args.seed)

    results = []

    def run_grid(amp: float, timescale: float, shape: str) -> tuple[int, int, int]:
        n_trials = 0
        n_detected = 0
        n_skipped = 0
        ts_days = timescale * 365.25
        for _ in range(args.n_realizations):
            host = rng.choice(host_lcs)
            t0 = _choose_t0(host['time'], ts_days, rng)
            if t0 is None:
                n_skipped += 1
                continue
            n_trials += 1
            mag = host['mag']
            err = host['err']
            injected = _inject(host['time'], mag, amp, ts_days, shape, t0)
            noisy = injected + rng.normal(0.0, err)
            if _detect(host['time'], noisy, ts_days, t0, args.threshold):
                n_detected += 1
        return n_trials, n_detected, n_skipped

    for amp in amplitudes:
        for ts in timescales:
            for shape in shapes:
                n_trials, n_detected, n_skipped = run_grid(amp, ts, shape)
                completeness = (n_detected / n_trials) if n_trials > 0 else np.nan
                results.append({
                    'amplitude': amp,
                    'timescale_yr': ts,
                    'shape': shape,
                    'n_trials': n_trials,
                    'n_detected': n_detected,
                    'n_skipped': n_skipped,
                    'completeness': completeness,
                })

    # False positive rate using null injections
    fp_trials, fp_detected, fp_skipped = run_grid(0.0, timescales[0], 'step')
    fp_rate = (fp_detected / fp_trials) if fp_trials > 0 else np.nan

    grid_df = pd.DataFrame(results)

    bench_dir = Path(args.benchmark_dir) if args.benchmark_dir else (Path(args.data_dir) / 'benchmark')
    meta = build_metadata(bench_dir)
    meta.update({'n_hosts_used': len(host_lcs), 'n_realizations': args.n_realizations, 'threshold': args.threshold})

    if not args.dry_run:
        grid_path = inj_dir / 'grid_results.csv'
        grid_df.to_csv(grid_path, index=False)
        write_metadata_sidecar(grid_path, metadata_dir, meta)

        # Heatmap for step shape
        try:
            import matplotlib.pyplot as plt

            step_df = grid_df[grid_df['shape'] == 'step']
            if not step_df.empty:
                pivot = step_df.pivot_table(index='timescale_yr', columns='amplitude', values='completeness')
                fig, ax = plt.subplots(figsize=(6, 4))
                im = ax.imshow(pivot.values, origin='lower', aspect='auto')
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns])
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels([f"{v:.2f}" for v in pivot.index])
                ax.set_xlabel('Amplitude (mag)')
                ax.set_ylabel('Timescale (yr)')
                ax.set_title('Completeness Heatmap (Step)')
                fig.colorbar(im, ax=ax, label='Completeness')
                fig.tight_layout()
                heatmap_path = fig_dir / 'completeness_heatmap.png'
                fig.savefig(heatmap_path, dpi=200)
                plt.close(fig)
                write_metadata_sidecar(heatmap_path, metadata_dir, meta)
        except Exception:
            pass

        summary = {
            'n_hosts_available': len(host_lcs),
            'n_hosts_used': len(host_lcs),
            'n_realizations': args.n_realizations,
            'amplitudes': amplitudes,
            'timescales_yr': timescales,
            'threshold': args.threshold,
            'false_positive_rate': fp_rate,
            'false_positive_trials': fp_trials,
            'false_positive_skipped': fp_skipped,
        }
        write_json_with_metadata(metrics_dir / 'injection_summary.json', summary, metadata_dir, meta)

    print("Injection-recovery complete")


if __name__ == '__main__':
    main()
