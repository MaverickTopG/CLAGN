"""
Injection-recovery on empirical WISE light curves.
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

from clagn.pipeline import CLAGNPipeline
from clagn.utils.validation_metadata import build_metadata, write_metadata_sidecar, write_json_with_metadata


def _parse_list(values: str) -> list[float]:
    parts = [p.strip() for p in values.replace(';', ',').split(',') if p.strip()]
    return [float(p) for p in parts]


def _mag_to_flux(mag: np.ndarray) -> np.ndarray:
    return 10 ** (-0.4 * mag)


def _magerr_to_fluxerr(mag: np.ndarray, mag_err: np.ndarray) -> np.ndarray:
    flux = _mag_to_flux(mag)
    return flux * np.log(10) * 0.4 * mag_err


def _load_hosts(hosts_dir: Path) -> list[dict]:
    hosts = []
    for path in sorted(hosts_dir.glob('*.csv')):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        cols = {c.lower(): c for c in df.columns}
        time_col = cols.get('mjd') or cols.get('time') or cols.get('t')
        flux_col = cols.get('w1_flux_mjy') or cols.get('flux') or cols.get('w1_flux')
        ferr_col = cols.get('w1_flux_err_mjy') or cols.get('flux_err') or cols.get('w1_flux_err')
        mag_col = cols.get('mag') or cols.get('w1_mag')
        merr_col = cols.get('mag_err') or cols.get('w1_mag_err')

        if not time_col:
            continue
        time = pd.to_numeric(df[time_col], errors='coerce').values

        if flux_col:
            flux = pd.to_numeric(df[flux_col], errors='coerce').values
            if ferr_col:
                ferr = pd.to_numeric(df[ferr_col], errors='coerce').values
            else:
                ferr = np.full_like(flux, 0.05, dtype=float)
        elif mag_col:
            mag = pd.to_numeric(df[mag_col], errors='coerce').values
            if merr_col:
                merr = pd.to_numeric(df[merr_col], errors='coerce').values
            else:
                merr = np.full_like(mag, 0.05, dtype=float)
            flux = _mag_to_flux(mag)
            ferr = _magerr_to_fluxerr(mag, merr)
        else:
            continue

        mask = np.isfinite(time) & np.isfinite(flux) & np.isfinite(ferr)
        if mask.sum() < 6:
            continue
        hosts.append({'path': path, 'time': time[mask], 'flux': flux[mask], 'ferr': ferr[mask]})
    return hosts


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


def _inject_flux(time: np.ndarray, flux: np.ndarray, amp_mag: float, timescale_days: float, shape: str, t0: float) -> np.ndarray:
    factor = 10 ** (0.4 * amp_mag)
    delta = np.ones_like(flux, dtype=float)
    if shape == 'step':
        delta[time >= t0] = factor
    elif shape == 'ramp':
        t1 = t0 + timescale_days
        ramp = (time >= t0) & (time <= t1)
        after = time > t1
        if timescale_days <= 0:
            delta[time >= t0] = factor
        else:
            frac = (time[ramp] - t0) / timescale_days
            delta[ramp] = 1.0 + frac * (factor - 1.0)
            delta[after] = factor
    else:
        delta[time >= t0] = factor
    return flux * delta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/real_clagn')
    parser.add_argument('--output_dir', default='results_real')
    parser.add_argument('--n_hosts', type=int, default=50)
    parser.add_argument('--n_realizations', type=int, default=20)
    parser.add_argument('--amplitudes', default='0.2,0.5,1.0,1.5')
    parser.add_argument('--timescales', default='0.5,1.0,2.0,5.0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    base = Path(args.data_dir)
    hosts_dir = base / 'hosts'
    if not hosts_dir.exists():
        print(f"Missing hosts directory: {hosts_dir}")
        sys.exit(1)

    hosts = _load_hosts(hosts_dir)
    if len(hosts) < 50:
        print(f"Need at least 50 real hosts, found {len(hosts)}")
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    if args.n_hosts < len(hosts):
        hosts = list(rng.choice(hosts, size=args.n_hosts, replace=False))

    if args.threshold is None:
        op_path = Path(args.output_dir) / 'operating_threshold.json'
        if not op_path.exists():
            print("Missing operating_threshold.json. Run dev evaluation first or provide --threshold")
            sys.exit(1)
        args.threshold = json.loads(op_path.read_text()).get('threshold', 0.5)

    amplitudes = _parse_list(args.amplitudes)
    timescales = _parse_list(args.timescales)
    shapes = ['step', 'ramp']

    pipeline = CLAGNPipeline(results_dir=args.output_dir)

    results = []

    def run_grid(amp: float, timescale: float, shape: str) -> tuple[int, int, int]:
        n_trials = 0
        n_detected = 0
        n_skipped = 0
        ts_days = timescale * 365.25
        for _ in range(args.n_realizations):
            host = rng.choice(hosts)
            t0 = _choose_t0(host['time'], ts_days, rng)
            if t0 is None:
                n_skipped += 1
                continue
            n_trials += 1
            injected = _inject_flux(host['time'], host['flux'], amp, ts_days, shape, t0)
            noisy = injected + rng.normal(0.0, host['ferr'])
            score = pipeline.score_single(host['time'], noisy, host['ferr'])
            if score.get('composite', 0.0) >= args.threshold:
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

    # False positive rate
    fp_trials, fp_detected, fp_skipped = run_grid(0.0, timescales[0], 'step')
    fp_rate = (fp_detected / fp_trials) if fp_trials > 0 else np.nan

    out_dir = Path(args.output_dir)
    inj_dir = out_dir / 'injection_recovery'
    fig_dir = out_dir
    metadata_dir = out_dir / 'metadata'
    inj_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    meta = build_metadata(base)
    meta.update({'n_hosts_used': len(hosts), 'threshold': args.threshold})

    if not args.dry_run:
        grid_df = pd.DataFrame(results)
        grid_path = inj_dir / 'grid_results.csv'
        grid_df.to_csv(grid_path, index=False)
        write_metadata_sidecar(grid_path, metadata_dir, meta)

        # Heatmap for step
        try:
            import matplotlib.pyplot as plt

            step_df = grid_df[grid_df['shape'] == 'step']
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
            print("Failed to generate completeness heatmap")
            sys.exit(1)

        summary = {
            'n_hosts_available': len(hosts),
            'n_hosts_used': len(hosts),
            'n_realizations': args.n_realizations,
            'amplitudes': amplitudes,
            'timescales_yr': timescales,
            'threshold': args.threshold,
            'false_positive_rate': fp_rate,
            'false_positive_trials': fp_trials,
            'false_positive_skipped': fp_skipped,
        }
        write_json_with_metadata(out_dir / 'injection_summary.json', summary, metadata_dir, meta)

    print("Real injection-recovery complete")


if __name__ == '__main__':
    main()
