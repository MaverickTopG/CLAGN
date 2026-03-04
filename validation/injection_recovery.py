"""
Injection-recovery campaign for CLAGN pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from clagn.pipeline import CLAGNPipeline
from .control_lc_cache import load_control_lcs

INJECTION_MORPHOLOGIES = {
    'step': 'instantaneous state change — most CLAGN look like this',
    'ramp': 'linear transition over N days — some slow CLAGN',
    'gaussian': 'temporary excursion — peak + return (SN-like / eclipse)',
    'two_step': 'partial transition followed by second change',
}

INJECTION_GRID = {
    'delta_mag': [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5],
    'transition_epoch_frac': [0.1, 0.3, 0.5, 0.7, 0.9],
    'ramp_duration_days': [30, 90, 180, 365],
    'morphology': list(INJECTION_MORPHOLOGIES.keys()),
}

N_REALIZATIONS = 20
N_HOSTS = 50


def inject_transition(times_mjd, flux_mjy, flux_err_mjy,
                      delta_mag, transition_mjd, morphology='step',
                      ramp_duration=90.0):
    flux_ratio = 10.0 ** (delta_mag / 2.5)

    if morphology == 'step':
        multiplier = np.where(times_mjd >= transition_mjd, flux_ratio, 1.0)
    elif morphology == 'ramp':
        t_end = transition_mjd + ramp_duration
        frac = np.clip((times_mjd - transition_mjd) / ramp_duration, 0, 1)
        multiplier = 1.0 + frac * (flux_ratio - 1.0)
    elif morphology == 'gaussian':
        sigma_days = ramp_duration / 2.355
        frac = np.exp(-0.5 * ((times_mjd - transition_mjd) / sigma_days) ** 2)
        multiplier = 1.0 + frac * (flux_ratio - 1.0)
    elif morphology == 'two_step':
        t2 = transition_mjd + ramp_duration
        m1 = np.where(times_mjd >= transition_mjd, np.sqrt(flux_ratio), 1.0)
        m2 = np.where(times_mjd >= t2, np.sqrt(flux_ratio), 1.0)
        multiplier = m1 * m2
    else:
        raise ValueError(f"Unknown morphology: {morphology}")

    flux_injected = flux_mjy * multiplier
    truth = {
        'delta_mag_injected': delta_mag,
        'transition_mjd': transition_mjd,
        'morphology': morphology,
        'ramp_duration_days': ramp_duration,
        'flux_ratio_injected': float(flux_ratio),
    }
    return flux_injected, truth


def run_injection_recovery(pipeline: CLAGNPipeline,
                           control_agn_lcs,
                           grid=None,
                           n_real=N_REALIZATIONS):
    if grid is None:
        grid = INJECTION_GRID

    results = []

    for morph in grid['morphology']:
        for dm in grid['delta_mag']:
            for epoch_frac in grid['transition_epoch_frac']:
                recovered = []
                dm_recovered = []

                for host_lc in control_agn_lcs[:n_real]:
                    t = host_lc['mjd']
                    f = host_lc['w1_flux_mjy']
                    e = host_lc['w1_flux_err_mjy']

                    t_trans = float(np.min(t) + epoch_frac * (np.max(t) - np.min(t)))
                    f_inj, truth = inject_transition(
                        t, f, e,
                        delta_mag=dm,
                        transition_mjd=t_trans,
                        morphology=morph,
                    )

                    score_result = pipeline.score_single(
                        times=t, flux=f_inj, flux_err=e, z=host_lc.get('z', 0.1)
                    )
                    score = score_result.get('composite', 0.0)
                    recovered.append(score >= pipeline.operating_threshold)
                    dm_recovered.append(score_result.get('delta_mag_w1', 0.0))

                results.append({
                    'morphology': morph,
                    'delta_mag_injected': dm,
                    'transition_epoch_frac': epoch_frac,
                    'recovery_rate': float(np.mean(recovered)),
                    'n_realizations': len(recovered),
                    'mean_dm_recovered': float(np.nanmean(dm_recovered)),
                    'dm_bias': float(np.nanmean(dm_recovered) - dm),
                })

    return pd.DataFrame(results)


def compute_detection_threshold_from_injection(efficiency_df: pd.DataFrame,
                                                target_completeness: float = 0.90):
    step_eff = efficiency_df[efficiency_df['morphology'] == 'step'].copy()
    step_eff = step_eff.groupby('delta_mag_injected')['recovery_rate'].mean()

    above = step_eff[step_eff >= target_completeness]
    if above.empty:
        return np.nan, 'threshold_not_reached'
    min_dm = float(above.index.min())
    return min_dm, f'injection_recovery_completeness={target_completeness}'


def _plot_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    morphs = df['morphology'].unique().tolist()
    n = len(morphs)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, morph in zip(axes, morphs):
        sub = df[df['morphology'] == morph]
        piv = sub.pivot_table(index='delta_mag_injected', columns='transition_epoch_frac',
                              values='recovery_rate', aggfunc='mean')
        im = ax.imshow(piv.values, aspect='auto', origin='lower',
                       extent=[piv.columns.min(), piv.columns.max(),
                               piv.index.min(), piv.index.max()])
        ax.set_title(morph)
        ax.set_xlabel('transition_epoch_frac')
        ax.set_ylabel('delta_mag_injected')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_recovery_curve(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    for morph in df['morphology'].unique():
        sub = df[df['morphology'] == morph]
        grp = sub.groupby('delta_mag_injected')['recovery_rate'].mean()
        ax.plot(grp.index, grp.values, 'o-', label=morph)
    ax.axhline(0.5, color='gray', ls=':')
    ax.axhline(0.8, color='gray', ls='--')
    ax.axhline(0.9, color='gray', ls='-')
    ax.set_xlabel('delta_mag_injected')
    ax.set_ylabel('recovery_rate')
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_amplitude_bias(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    grp = df.groupby('delta_mag_injected')['mean_dm_recovered'].mean()
    ax.plot(grp.index, grp.values, 'o-')
    ax.plot(grp.index, grp.index, 'k--', alpha=0.6)
    ax.set_xlabel('delta_mag_injected')
    ax.set_ylabel('mean_delta_mag_recovered')
    ax.set_title('Amplitude Bias')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main(results_dir: str = './results/', benchmark_dir: str = 'data/benchmark/'):
    pipeline = CLAGNPipeline(results_dir=results_dir)
    control_lcs = load_control_lcs(benchmark_dir=benchmark_dir, n_hosts=N_HOSTS)
    if not control_lcs:
        raise RuntimeError("No control light curves available for injection-recovery")

    df = run_injection_recovery(pipeline, control_lcs, n_real=N_REALIZATIONS)

    out_dir = Path(results_dir) / 'validation'
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / 'injection_recovery_results.csv', index=False)

    _plot_heatmap(df, out_dir / 'injection_recovery_heatmap.png')
    _plot_recovery_curve(df, out_dir / 'injection_recovery_curves.png')
    _plot_amplitude_bias(df, out_dir / 'amplitude_bias.png')

    thresh, reason = compute_detection_threshold_from_injection(df)
    with open(out_dir / 'detection_threshold.json', 'w') as f:
        json.dump({'min_delta_mag': thresh, 'reason': reason}, f, indent=2)

    return df


if __name__ == '__main__':
    main()
