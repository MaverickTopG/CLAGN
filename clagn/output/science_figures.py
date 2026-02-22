"""
science_figures.py — Publication-quality figures for the CLAGN paper.

All figures use matplotlib with the 'Agg' backend (non-interactive, suitable
for headless servers). Each figure function:
    - Wraps all code in try/except (never crashes the pipeline)
    - Saves both .pdf and .png at 300 DPI
    - Always calls plt.close(fig) to prevent memory leaks
    - Shows a placeholder text box if data is missing

Figures:
    1. Discovery: 2×3 grid of light curves with DRW posterior
    2. Population: redshift, luminosity, and color distributions
    3. Physics: black hole mass, Eddington ratio, disk timescales
    4. Dust temperature: W1-W2 color evolution over time
    5. Validation: injection-recovery curves and FAR histogram
    6. Gaia-WISE: optical-IR variability comparison
"""
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------
_STYLE = {
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
}

_COLORS = {
    'allsky': '#2196F3',
    '3band': '#4CAF50',
    'postcryo': '#9C27B0',
    'allwise': '#FF9800',
    'neowise': '#F44336',
    'drw_fit': '#212121',
    'changepoint': '#E91E63',
    'candidate': '#FF5722',
    'all_sources': '#607D8B',
}

_MJD_TO_YEAR_OFFSET = 51544.0   # MJD of J2000.0
_DAYS_PER_YEAR = 365.25


def _mjd_to_year(mjd):
    """Convert MJD to decimal year."""
    return 2000.0 + (np.asarray(mjd) - _MJD_TO_YEAR_OFFSET) / _DAYS_PER_YEAR


def _make_dual_xaxis(ax: plt.Axes) -> plt.Axes:
    """
    Add a secondary x-axis showing calendar year above the MJD axis.

    Returns the twin axes object.
    """
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())

    # Tick positions: every year from MJD range
    mjd_lo, mjd_hi = ax.get_xlim()
    year_lo = _mjd_to_year(mjd_lo)
    year_hi = _mjd_to_year(mjd_hi)
    year_ticks_int = np.arange(int(np.ceil(year_lo)), int(np.floor(year_hi)) + 1)

    if len(year_ticks_int) > 10:
        year_ticks_int = year_ticks_int[::2]  # every 2 years if many

    mjd_ticks = [_MJD_TO_YEAR_OFFSET + (y - 2000.0) * _DAYS_PER_YEAR
                 for y in year_ticks_int]

    ax_top.set_xticks(mjd_ticks)
    ax_top.set_xticklabels([str(int(y)) for y in year_ticks_int], fontsize=9)
    ax_top.set_xlabel('Year', fontsize=10, labelpad=4)

    return ax_top


def _placeholder_text(ax: plt.Axes, message: str) -> None:
    """Render a gray placeholder box with an error/missing data message."""
    ax.set_facecolor('#F5F5F5')
    ax.text(
        0.5, 0.5, f"Data unavailable\n{message}",
        transform=ax.transAxes, ha='center', va='center',
        fontsize=10, color='#757575',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#BDBDBD'),
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _save_figure(fig: plt.Figure, output_path: str) -> None:
    """Save figure as both .pdf and .png at 300 DPI."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for ext in ['pdf', 'png']:
        path = output_path.with_suffix(f'.{ext}')
        try:
            fig.savefig(str(path), dpi=300, bbox_inches='tight')
            logger.debug(f"Saved: {path}")
        except Exception as exc:
            logger.warning(f"Failed to save {path}: {exc}")


def _dataset_color(dataset_label: str) -> str:
    """Return color for a WISE dataset label."""
    label = str(dataset_label).lower()
    for key, color in _COLORS.items():
        if key in label:
            return color
    return '#9E9E9E'


# ---------------------------------------------------------------------------
# Figure 1: Discovery light curves
# ---------------------------------------------------------------------------

def plot_figure1_discovery(candidates: pd.DataFrame, output_path: str) -> None:
    """
    Figure 1: 2×3 grid of the six best CLAGN light curves.

    Each panel shows:
        - Individual WISE single-exposure photometry (color by dataset)
        - DRW GP posterior mean and 1σ band
        - Changepoint location (vertical dashed line)
        - Dual x-axis: MJD (bottom) and calendar year (top)
        - Hibernate gap shaded region

    Parameters
    ----------
    candidates : DataFrame with top CLAGN candidates
    output_path: str, base path (without extension) for output files
    """
    import pickle

    fig = None
    try:
        with plt.rc_context(_STYLE):
            n_panels = min(6, len(candidates))
            if n_panels == 0:
                fig, ax = plt.subplots(figsize=(10, 8))
                _placeholder_text(ax, "No candidates available")
                _save_figure(fig, output_path)
                return

            fig = plt.figure(figsize=(15, 9))
            gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.30)

            for panel_idx in range(n_panels):
                row_i = panel_idx // 3
                col_i = panel_idx % 3
                ax = fig.add_subplot(gs[row_i, col_i])

                src_row = candidates.iloc[panel_idx]
                source_id = str(src_row.get('source_id', f'src_{panel_idx}'))

                # Try to load light curve
                lc_df = None
                results_dir = Path(output_path).parent.parent

                for lc_path in [
                    results_dir / 'wise_enhanced' / f'enhanced_wise_{source_id}.pkl',
                    results_dir / f'wise_{source_id}.pkl',
                ]:
                    if lc_path.exists():
                        try:
                            with open(lc_path, 'rb') as f:
                                wise_data = pickle.load(f)
                            lc_df = wise_data.get('lc', None)
                        except Exception:
                            pass
                        break

                if lc_df is None or lc_df.empty:
                    _placeholder_text(ax, f"{source_id}\nNo light curve data")
                    ax.set_title(source_id, fontsize=10)
                    continue

                # Extract arrays
                mjd = pd.to_numeric(lc_df['mjd'], errors='coerce').values
                if 'w1_flux_mjy' in lc_df.columns:
                    flux = pd.to_numeric(lc_df['w1_flux_mjy'], errors='coerce').values
                    ferr = pd.to_numeric(lc_df.get('w1_flux_err_mjy', pd.Series(np.ones(len(lc_df)) * 0.01)),
                                         errors='coerce').values
                    ylabel = 'W1 flux [mJy]'
                elif 'w1_mag' in lc_df.columns:
                    flux = pd.to_numeric(lc_df['w1_mag'], errors='coerce').values
                    ferr = pd.to_numeric(lc_df.get('w1_err', pd.Series(np.ones(len(lc_df)) * 0.05)),
                                         errors='coerce').values
                    ax.invert_yaxis()
                    ylabel = 'W1 magnitude [Vega]'
                else:
                    _placeholder_text(ax, f"{source_id}\nMissing flux columns")
                    continue

                # Plot by dataset
                dataset_col = 'dataset' if 'dataset' in lc_df.columns else 'dataset_flag'
                if dataset_col in lc_df.columns:
                    datasets = lc_df[dataset_col].unique()
                    for ds in datasets:
                        ds_mask = lc_df[dataset_col] == ds
                        ax.errorbar(
                            mjd[ds_mask], flux[ds_mask], yerr=ferr[ds_mask],
                            fmt='o', markersize=3, linewidth=0.5, capsize=1.5,
                            color=_dataset_color(ds), alpha=0.7,
                            label=str(ds), zorder=3,
                        )
                else:
                    ax.errorbar(mjd, flux, yerr=ferr, fmt='o', markersize=3,
                                color='steelblue', alpha=0.7)

                # DRW posterior
                drw_path = results_dir / 'advanced_drw' / f'advanced_drw_{source_id}.pkl'
                if drw_path.exists():
                    try:
                        with open(drw_path, 'rb') as f:
                            drw_data = pickle.load(f)
                        pred_mean = drw_data.get('mcmc', drw_data).get('pred_mean', None)
                        pred_std = drw_data.get('mcmc', drw_data).get('pred_std', None)
                        times_rest = drw_data.get('mcmc', drw_data).get('times_rest', None)

                        if (pred_mean is not None and pred_std is not None and
                                times_rest is not None):
                            z = float(src_row.get('z', src_row.get('redshift', 0.1)))
                            times_obs = times_rest * (1.0 + z)
                            sort_idx = np.argsort(times_obs)
                            ax.plot(times_obs[sort_idx], pred_mean[sort_idx],
                                    color=_COLORS['drw_fit'], lw=1.5,
                                    label='DRW mean', zorder=4)
                            ax.fill_between(
                                times_obs[sort_idx],
                                (pred_mean - pred_std)[sort_idx],
                                (pred_mean + pred_std)[sort_idx],
                                color=_COLORS['drw_fit'], alpha=0.15,
                                zorder=2, label='DRW 1σ',
                            )
                    except Exception:
                        pass

                # Changepoint vertical line
                cp_path = results_dir / f'changepoint_{source_id}.pkl'
                if cp_path.exists():
                    try:
                        with open(cp_path, 'rb') as f:
                            cp_data = pickle.load(f)
                        t_break = cp_data.get('t_break_mjd', np.nan)
                        if np.isfinite(float(t_break)):
                            ax.axvline(float(t_break), color=_COLORS['changepoint'],
                                       lw=1.5, ls='--', alpha=0.85,
                                       label=f'Break MJD {t_break:.0f}', zorder=5)
                    except Exception:
                        pass

                # WISE hibernation gap
                ax.axvspan(55593, 56141, alpha=0.08, color='gray', label='Hibernation')

                # Axes labels and title
                ax.set_xlabel('MJD', fontsize=9)
                ax.set_ylabel(ylabel, fontsize=9)
                z_val = _safe_float(src_row.get('z', src_row.get('redshift', np.nan)))
                title = f"{source_id}"
                if np.isfinite(z_val):
                    title += f" (z={z_val:.3f})"
                ax.set_title(title, fontsize=9, pad=3)

                # Dual x-axis (calendar year)
                try:
                    _make_dual_xaxis(ax)
                except Exception:
                    pass

                # Legend (only first panel gets full legend)
                if panel_idx == 0:
                    ax.legend(fontsize=7, loc='upper left', markerscale=0.8,
                              ncol=2, framealpha=0.7)

            fig.suptitle('Changing-Look AGN Candidates: WISE/NEOWISE-R Light Curves',
                         fontsize=14, y=1.01, fontweight='bold')

            _save_figure(fig, output_path)

    except Exception as exc:
        logger.error(f"Figure 1 failed: {exc}", exc_info=True)
    finally:
        if fig is not None:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Population distributions
# ---------------------------------------------------------------------------

def plot_figure2_population(all_sources: pd.DataFrame, candidates: pd.DataFrame,
                             output_path: str) -> None:
    """
    Figure 2: Population comparison between CLAGN candidates and parent sample.

    Panels: redshift, W1-W2 color, W1 magnitude, CLAGN score distribution.
    """
    fig = None
    try:
        with plt.rc_context(_STYLE):
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))

            plot_configs = [
                ('z', 'redshift', np.linspace(0, 3, 30), 'Redshift $z$'),
                ('w1_minus_w2', 'W1$-$W2 color [Vega mag]', np.linspace(-0.5, 3.0, 30), 'W1$-$W2 color [Vega mag]'),
                ('w1_mag', 'W1 magnitude [Vega mag]', np.linspace(12, 19, 30), 'W1 magnitude [Vega]'),
                ('clagn_score', 'CLAGN score', np.linspace(0, 14, 28), 'CLAGN score'),
            ]

            for ax, (col_all, col_label, bins, xlabel) in zip(axes.flat, plot_configs):
                # Use 'z' or 'redshift' as column name
                col_candidates = col_all
                col_sources = col_all

                for c_try in [col_all, col_all.replace('z', 'redshift'), 'score']:
                    if c_try in all_sources.columns:
                        col_sources = c_try
                        break
                for c_try in [col_all, col_all.replace('z', 'redshift'), 'score']:
                    if c_try in candidates.columns:
                        col_candidates = c_try
                        break

                vals_all = pd.to_numeric(
                    all_sources.get(col_sources, pd.Series(dtype=float)),
                    errors='coerce'
                ).dropna()
                vals_cand = pd.to_numeric(
                    candidates.get(col_candidates, pd.Series(dtype=float)),
                    errors='coerce'
                ).dropna()

                if len(vals_all) > 0:
                    ax.hist(vals_all, bins=bins, alpha=0.5, density=True,
                            color=_COLORS['all_sources'],
                            label=f'All AGN (N={len(vals_all)})',
                            histtype='stepfilled')
                else:
                    _placeholder_text(ax, f'No data for {col_all}')
                    continue

                if len(vals_cand) > 0:
                    ax.hist(vals_cand, bins=bins, alpha=0.85, density=True,
                            color=_COLORS['candidate'],
                            label=f'CLAGN (N={len(vals_cand)})',
                            histtype='step', linewidth=2)

                ax.set_xlabel(xlabel, fontsize=11)
                ax.set_ylabel('Normalized count', fontsize=11)
                ax.legend(fontsize=9)

            fig.suptitle('Population Statistics: CLAGN vs Parent AGN Sample',
                         fontsize=13, fontweight='bold')
            fig.tight_layout()
            _save_figure(fig, output_path)

    except Exception as exc:
        logger.error(f"Figure 2 failed: {exc}", exc_info=True)
    finally:
        if fig is not None:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Physical parameters
# ---------------------------------------------------------------------------

def plot_figure3_physics(candidates: pd.DataFrame, physics_results: dict,
                          output_path: str) -> None:
    """
    Figure 3: Physical parameters of CLAGN candidates.

    Panels: M_BH vs L_bol, lambda_Edd distribution, tau_rest vs M_BH,
    mechanism identification.
    """
    fig = None
    try:
        with plt.rc_context(_STYLE):
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))

            # Panel 1: M_BH vs L_bol
            ax = axes[0, 0]
            M_BH = pd.to_numeric(
                candidates.get('M_BH_solar', pd.Series(dtype=float)),
                errors='coerce'
            ).values
            L_bol = pd.to_numeric(
                candidates.get('L_bol_erg_s', pd.Series(dtype=float)),
                errors='coerce'
            ).values

            valid = np.isfinite(M_BH) & np.isfinite(L_bol) & (M_BH > 0) & (L_bol > 0)
            if valid.sum() >= 3:
                sc = ax.scatter(np.log10(M_BH[valid]), np.log10(L_bol[valid]),
                                c=_safe_float_array(candidates.get('lambda_Edd', pd.Series())),
                                cmap='plasma', s=60, alpha=0.8, vmin=-3, vmax=0)
                plt.colorbar(sc, ax=ax, label=r'$\log \lambda_\mathrm{Edd}$')
                ax.set_xlabel(r'$\log M_\mathrm{BH}\ [M_\odot]$', fontsize=11)
                ax.set_ylabel(r'$\log L_\mathrm{bol}\ [\mathrm{erg\,s}^{-1}]$', fontsize=11)
                ax.set_title('Black Hole Mass vs Luminosity', fontsize=11)

                # Eddington limits
                m_range = np.linspace(5, 11, 100)
                for lam, ls in [(-1, '--'), (-2, ':'), (-3, '-.')]:
                    L_edd_line = np.log10(1.26e38 * 10 ** m_range) + lam
                    ax.plot(m_range, L_edd_line, color='gray', ls=ls, lw=1, alpha=0.6,
                            label=rf'$\lambda_\mathrm{{Edd}} = 10^{{{lam}}}$')
                ax.legend(fontsize=8)
            else:
                _placeholder_text(ax, 'Insufficient M_BH / L_bol data')

            # Panel 2: Eddington ratio distribution
            ax = axes[0, 1]
            lambda_edd = pd.to_numeric(
                candidates.get('lambda_Edd', pd.Series(dtype=float)),
                errors='coerce'
            ).dropna()
            lambda_edd = lambda_edd[lambda_edd > 0]
            if len(lambda_edd) > 0:
                ax.hist(np.log10(lambda_edd), bins=15, color=_COLORS['candidate'],
                        edgecolor='black', linewidth=0.5, alpha=0.8)
                ax.set_xlabel(r'$\log \lambda_\mathrm{Edd}$', fontsize=11)
                ax.set_ylabel('Count', fontsize=11)
                ax.set_title('Eddington Ratio Distribution', fontsize=11)
            else:
                _placeholder_text(ax, 'No Eddington ratio data')

            # Panel 3: tau_rest vs M_BH
            ax = axes[1, 0]
            tau_vals = pd.to_numeric(
                candidates.get('tau_rest_days', pd.Series(dtype=float)),
                errors='coerce'
            ).values
            valid = np.isfinite(M_BH) & np.isfinite(tau_vals) & (M_BH > 0) & (tau_vals > 0)
            if valid.sum() >= 3:
                ax.scatter(np.log10(M_BH[valid]), np.log10(tau_vals[valid]),
                           s=50, color=_COLORS['candidate'], alpha=0.8)
                ax.set_xlabel(r'$\log M_\mathrm{BH}\ [M_\odot]$', fontsize=11)
                ax.set_ylabel(r'$\log \tau_\mathrm{rest}$ [days]', fontsize=11)
                ax.set_title('DRW Timescale vs Black Hole Mass', fontsize=11)
            else:
                _placeholder_text(ax, 'Insufficient tau / M_BH data')

            # Panel 4: Mechanism pie chart
            ax = axes[1, 1]
            if 'mechanism' in candidates.columns:
                mech_counts = candidates['mechanism'].value_counts()
                if len(mech_counts) > 0:
                    ax.pie(mech_counts.values, labels=mech_counts.index,
                           autopct='%1.0f%%', startangle=90,
                           colors=plt.cm.Set3(np.linspace(0, 1, len(mech_counts))))
                    ax.set_title('Variability Mechanisms', fontsize=11)
                else:
                    _placeholder_text(ax, 'No mechanism data')
            else:
                _placeholder_text(ax, 'No mechanism column')

            fig.suptitle('Physical Properties of CLAGN Candidates', fontsize=13, fontweight='bold')
            fig.tight_layout()
            _save_figure(fig, output_path)

    except Exception as exc:
        logger.error(f"Figure 3 failed: {exc}", exc_info=True)
    finally:
        if fig is not None:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Dust temperature evolution
# ---------------------------------------------------------------------------

def plot_figure4_dust_temperature(candidates: pd.DataFrame, physics_results: dict,
                                   output_path: str) -> None:
    """
    Figure 4: Dust temperature evolution across the CLAGN transition.

    Left: T_dust_pre vs T_dust_post scatter diagram
    Right: ΔT_dust distribution
    """
    fig = None
    try:
        with plt.rc_context(_STYLE):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Panel 1: T_dust pre vs post
            ax = axes[0]
            T_pre = pd.to_numeric(
                candidates.get('T_dust_pre_K', candidates.get('T_dust_pre', pd.Series())),
                errors='coerce'
            ).values
            T_post = pd.to_numeric(
                candidates.get('T_dust_post_K', candidates.get('T_dust_post', pd.Series())),
                errors='coerce'
            ).values

            valid = np.isfinite(T_pre) & np.isfinite(T_post)
            if valid.sum() >= 3:
                colors = np.where(T_post > T_pre, '#F44336', '#2196F3')
                ax.scatter(T_pre[valid], T_post[valid], c=colors[valid], s=60, alpha=0.8)
                t_range = np.linspace(
                    min(T_pre[valid].min(), T_post[valid].min()) * 0.95,
                    max(T_pre[valid].max(), T_post[valid].max()) * 1.05, 2
                )
                ax.plot(t_range, t_range, 'k--', lw=1, alpha=0.5, label='T_pre = T_post')
                ax.set_xlabel('T_dust pre-transition [K]', fontsize=11)
                ax.set_ylabel('T_dust post-transition [K]', fontsize=11)
                ax.set_title('Dust Temperature Evolution', fontsize=11)
                ax.legend(fontsize=9)
            else:
                _placeholder_text(ax, 'Insufficient T_dust data')

            # Panel 2: ΔT_dust distribution
            ax = axes[1]
            delta_T = pd.to_numeric(
                candidates.get('delta_T_dust_K', candidates.get('delta_T_dust', pd.Series())),
                errors='coerce'
            ).dropna()

            if len(delta_T) > 0:
                n_heat = (delta_T > 0).sum()
                n_cool = (delta_T < 0).sum()
                ax.hist(delta_T, bins=15, color='steelblue',
                        edgecolor='black', linewidth=0.5, alpha=0.8)
                ax.axvline(0, color='black', lw=1.5, ls='--')
                ax.set_xlabel(r'$\Delta T_\mathrm{dust}$ [K]', fontsize=11)
                ax.set_ylabel('Count', fontsize=11)
                ax.set_title(
                    f'ΔT_dust Distribution\n(heated: {n_heat}, cooled: {n_cool})',
                    fontsize=11
                )
            else:
                _placeholder_text(ax, 'No ΔT_dust data')

            fig.suptitle('Dust Temperature Evolution in Changing-Look AGN',
                         fontsize=13, fontweight='bold')
            fig.tight_layout()
            _save_figure(fig, output_path)

    except Exception as exc:
        logger.error(f"Figure 4 failed: {exc}", exc_info=True)
    finally:
        if fig is not None:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Validation
# ---------------------------------------------------------------------------

def plot_figure5_validation(injection_results: dict, far_results: dict,
                             output_path: str) -> None:
    """
    Figure 5: Pipeline validation results.

    Left: Injection-recovery curves (recovery fraction vs amplitude)
    Right: False alarm rate histogram
    """
    fig = None
    try:
        with plt.rc_context(_STYLE):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Panel 1: Injection-recovery
            ax = axes[0]
            recovery = injection_results.get('recovery_fractions', {})
            avg_recovery = injection_results.get('avg_recovery', {})

            if avg_recovery:
                amplitudes = sorted(avg_recovery.keys())
                avg_fracs = [avg_recovery[a] for a in amplitudes]
                ax.plot(amplitudes, avg_fracs, 'o-', color='steelblue', lw=2,
                        markersize=6, label='Average (all types)')

                # Per-type lines
                injection_types = ['step', 'ramp', 'gaussian', 'sinusoidal']
                type_colors = ['#F44336', '#4CAF50', '#FF9800', '#9C27B0']
                for inj_type, color in zip(injection_types, type_colors):
                    fracs = [recovery.get(a, {}).get(inj_type, np.nan) for a in amplitudes]
                    if any(np.isfinite(f) for f in fracs):
                        ax.plot(amplitudes, fracs, 's--', color=color,
                                lw=1.2, markersize=4, alpha=0.7, label=inj_type)

                ax.axhline(0.5, color='gray', ls=':', lw=1, label='50% threshold')
                ax.axhline(0.9, color='gray', ls='--', lw=1, label='90% threshold')
                ax.set_xlabel('Injection amplitude [mag]', fontsize=11)
                ax.set_ylabel('Recovery fraction', fontsize=11)
                ax.set_ylim(-0.05, 1.1)
                ax.set_title('Injection-Recovery Test', fontsize=11)
                ax.legend(fontsize=8, loc='lower right')
            else:
                _placeholder_text(ax, 'No injection-recovery results')

            # Panel 2: FAR histogram
            ax = axes[1]
            scores = far_results.get('scores', np.array([]))

            if len(scores) > 0:
                ax.hist(scores, bins=40, color='steelblue',
                        edgecolor='black', linewidth=0.3, alpha=0.8,
                        label=f'DRW simulations (N={len(scores)})')
                ax.axvline(0.5, color='tomato', lw=2, ls='--',
                           label=f"0.5 threshold\n(FAR={far_results.get('far_at_05', 0):.3f})")
                ax.axvline(0.7, color='orangered', lw=2, ls=':',
                           label=f"0.7 threshold\n(FAR={far_results.get('far_at_07', 0):.3f})")
                ax.set_xlabel('CLAGN score', fontsize=11)
                ax.set_ylabel('Count', fontsize=11)
                ax.set_title('False Alarm Rate Distribution', fontsize=11)
                ax.legend(fontsize=8)
            else:
                _placeholder_text(ax, 'No FAR simulation results')

            fig.suptitle('Pipeline Validation: Injection-Recovery and False Alarm Rate',
                         fontsize=13, fontweight='bold')
            fig.tight_layout()
            _save_figure(fig, output_path)

    except Exception as exc:
        logger.error(f"Figure 5 failed: {exc}", exc_info=True)
    finally:
        if fig is not None:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: Gaia-WISE comparison
# ---------------------------------------------------------------------------

def plot_figure6_gaia_wise(candidates: pd.DataFrame, gaia_results: dict,
                            output_path: str) -> None:
    """
    Figure 6: Optical (Gaia G) vs infrared (WISE W1) variability.

    Left: G-band RMS vs W1 RMS scatter plot with R_var lines
    Right: Example Gaia epoch light curve for top candidate
    """
    fig = None
    try:
        with plt.rc_context(_STYLE):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Panel 1: G-RMS vs W1-RMS
            ax = axes[0]
            g_rms = pd.to_numeric(
                candidates.get('g_rms', pd.Series(dtype=float)),
                errors='coerce'
            ).values
            w1_rms = pd.to_numeric(
                candidates.get('w1_rms', candidates.get('sigma_drw', pd.Series())),
                errors='coerce'
            ).values

            valid = np.isfinite(g_rms) & np.isfinite(w1_rms) & (g_rms > 0) & (w1_rms > 0)
            if valid.sum() >= 3:
                ax.scatter(w1_rms[valid], g_rms[valid],
                           s=50, color=_COLORS['candidate'], alpha=0.8)

                # R_var = 1 line
                x_range = np.linspace(w1_rms[valid].min() * 0.8, w1_rms[valid].max() * 1.2, 100)
                for r_var, ls, label in [(1.0, '--', 'R_var=1'), (2.0, ':', 'R_var=2'),
                                          (0.5, '-.', 'R_var=0.5')]:
                    ax.plot(x_range, r_var * x_range, color='gray', ls=ls, lw=1.2,
                            alpha=0.6, label=label)

                ax.set_xlabel('W1 RMS variability [mJy]', fontsize=11)
                ax.set_ylabel('Gaia G RMS variability [mag]', fontsize=11)
                ax.set_title('Optical vs Infrared Variability', fontsize=11)
                ax.legend(fontsize=9)
                ax.set_xscale('log')
                ax.set_yscale('log')
            else:
                _placeholder_text(ax, 'Insufficient Gaia + WISE variability data')

            # Panel 2: Example Gaia light curve
            ax = axes[1]
            gaia_lc_plotted = False

            if isinstance(gaia_results, dict):
                for source_id, gaia_src in gaia_results.items():
                    if isinstance(gaia_src, dict):
                        lc = gaia_src.get('lc', {})
                        if isinstance(lc, dict) and lc.get('available', False):
                            times = lc.get('times_tcb', np.array([]))
                            mags = lc.get('g_mag', np.array([]))
                            errs = lc.get('g_mag_err', np.ones_like(mags) * 0.01)

                            if len(times) >= 5:
                                ax.errorbar(times, mags, yerr=errs,
                                            fmt='o', markersize=3, color='steelblue',
                                            capsize=1.5, linewidth=0.5, alpha=0.8)
                                ax.invert_yaxis()
                                ax.set_xlabel('Time [TCB days]', fontsize=11)
                                ax.set_ylabel('Gaia G [mag]', fontsize=11)
                                ax.set_title(f'Gaia G Light Curve: {source_id}', fontsize=11)
                                gaia_lc_plotted = True
                                break

            if not gaia_lc_plotted:
                _placeholder_text(ax, 'No Gaia epoch photometry available')

            fig.suptitle('Gaia Optical + WISE Infrared Joint Variability',
                         fontsize=13, fontweight='bold')
            fig.tight_layout()
            _save_figure(fig, output_path)

    except Exception as exc:
        logger.error(f"Figure 6 failed: {exc}", exc_info=True)
    finally:
        if fig is not None:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Helper: safe float array conversion
# ---------------------------------------------------------------------------

def _safe_float_array(series) -> np.ndarray:
    """Convert a pandas Series to float array, NaN for non-numeric."""
    try:
        return pd.to_numeric(series, errors='coerce').values.astype(float)
    except Exception:
        return np.full(len(series) if hasattr(series, '__len__') else 1, np.nan)


def _safe_float(val, default=np.nan) -> float:
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_all_figures(results_dir: str = './results/') -> None:
    """
    Generate all six publication figures.

    Loads data from results_dir and saves figures to results_dir/figures/.
    All failures are caught and logged; the pipeline never aborts on a figure error.

    Parameters
    ----------
    results_dir : str, path to results directory
    """
    import pickle

    results_path = Path(results_dir)
    figures_dir = results_path / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load candidates
    candidates_df = pd.DataFrame()
    candidates_file = results_path / 'top_candidates.csv'
    if candidates_file.exists():
        try:
            candidates_df = pd.read_csv(candidates_file)
        except Exception as exc:
            logger.error(f"Failed to load candidates: {exc}")

    # Merge physics
    physics_summary = results_path / 'physics_summary.csv'
    if physics_summary.exists():
        try:
            phys_df = pd.read_csv(physics_summary)
            if 'source_id' in candidates_df.columns and 'source_id' in phys_df.columns:
                candidates_df = candidates_df.merge(phys_df, on='source_id', how='left', suffixes=('', '_phys'))
        except Exception:
            pass

    # Load parent sample
    all_sources_df = pd.DataFrame()
    for fname in ['all_sources.csv', 'catalog_output.csv']:
        p = results_path / fname
        if p.exists():
            try:
                all_sources_df = pd.read_csv(p)
                break
            except Exception:
                pass
    if all_sources_df.empty:
        all_sources_df = candidates_df.copy()

    # Load validation results
    injection_results = {}
    far_results = {}
    val_dir = results_path / 'validation'
    if (val_dir / 'injection_recovery_results.csv').exists():
        try:
            inj_df = pd.read_csv(val_dir / 'injection_recovery_results.csv')
            from ..config import INJECTION_AMPLITUDES
            recovery_fractions = {}
            avg_recovery = {}
            injection_types = ['step', 'ramp', 'gaussian', 'sinusoidal']
            for amp in INJECTION_AMPLITUDES:
                recovery_fractions[amp] = {}
                fracs = []
                for inj_type in injection_types:
                    mask = (inj_df['amplitude_mag'] == amp) & (inj_df['injection_type'] == inj_type)
                    if mask.sum() > 0:
                        frac = float(inj_df[mask]['recovered'].mean())
                        recovery_fractions[amp][inj_type] = frac
                        fracs.append(frac)
                avg_recovery[amp] = float(np.nanmean(fracs)) if fracs else np.nan
            injection_results = {
                'recovery_fractions': recovery_fractions,
                'avg_recovery': avg_recovery,
            }
        except Exception:
            pass

    if (val_dir / 'false_positive_scores.csv').exists():
        try:
            far_df = pd.read_csv(val_dir / 'false_positive_scores.csv')
            scores = far_df['score'].values
            far_results = {
                'scores': scores,
                'far_at_05': float(np.mean(scores > 0.5)),
                'far_at_07': float(np.mean(scores > 0.7)),
            }
        except Exception:
            pass

    # Load Gaia results
    gaia_results = {}
    gaia_dir = results_path / 'gaia_photometry'
    if gaia_dir.exists():
        for pkl_file in gaia_dir.glob('gaia_*.pkl'):
            try:
                with open(pkl_file, 'rb') as f:
                    gaia_data = pickle.load(f)
                src_id = pkl_file.stem.replace('gaia_', '')
                gaia_results[src_id] = gaia_data
            except Exception:
                pass

    # Generate all figures
    logger.info("Generating Figure 1: Discovery light curves...")
    plot_figure1_discovery(candidates_df, str(figures_dir / 'figure1_discovery'))

    logger.info("Generating Figure 2: Population distributions...")
    plot_figure2_population(all_sources_df, candidates_df, str(figures_dir / 'figure2_population'))

    logger.info("Generating Figure 3: Physical parameters...")
    plot_figure3_physics(candidates_df, {}, str(figures_dir / 'figure3_physics'))

    logger.info("Generating Figure 4: Dust temperature evolution...")
    plot_figure4_dust_temperature(candidates_df, {}, str(figures_dir / 'figure4_dust_temperature'))

    logger.info("Generating Figure 5: Validation results...")
    plot_figure5_validation(injection_results, far_results, str(figures_dir / 'figure5_validation'))

    logger.info("Generating Figure 6: Gaia-WISE comparison...")
    plot_figure6_gaia_wise(candidates_df, gaia_results, str(figures_dir / 'figure6_gaia_wise'))

    logger.info(f"All figures saved to {figures_dir}")
