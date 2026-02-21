"""
plots.py — Publication-quality light curve and diagnostic plots.

Every plot must be camera-ready (300 dpi, seaborn-v0_8-paper style).
Always call plt.close(fig) after saving to prevent memory leaks.
"""
import logging
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend for batch processing
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from ..config import WISE_HIBERNATION_MJD_START, WISE_HIBERNATION_MJD_END

logger = logging.getLogger(__name__)

# Suppress matplotlib noisy warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# MJD to calendar year conversion: JD = MJD + 2400000.5; year ≈ 1858.8792 + MJD/365.25
_MJD_TO_YEAR_OFFSET = 1858.8792


def _mjd_to_year(mjd):
    return _MJD_TO_YEAR_OFFSET + np.asarray(mjd) / 365.25


def plot_lightcurve_panel(source_id, lc_df, drw_pred_mean, drw_pred_std,
                           drw_residuals, sf_result, break_mjd,
                           z, score, label, output_path):
    """
    4-panel publication-quality figure per CLAGN candidate.

    Panel 1 (top, largest): W1 and W2 flux vs MJD
        - W1 in blue (AllWISE circles, NEOWISE diamonds), W2 in orange
        - DRW posterior mean ± 1σ shaded
        - Vertical dashed line at changepoint break MJD
        - Dual x-axis: MJD (bottom) + calendar year (top)
        - WISE hibernation gap shaded gray

    Panel 2: W1-W2 color evolution
        - Horizontal line at W1-W2 = 0.8 (AGN boundary, Stern+2012)

    Panel 3: Standardized DRW residuals
        - Horizontal lines at ±2σ and ±3σ

    Panel 4: Structure Function SF(Δt)
        - Filled circles with bootstrap error bars
        - DRW theoretical SF as dashed line

    Parameters
    ----------
    source_id, lc_df, drw_pred_mean, drw_pred_std : ...
    drw_residuals : array, standardized residuals
    sf_result     : dict from structure_function module
    break_mjd     : float, MJD of best-fit changepoint
    z, score, label, output_path : metadata
    """
    try:
        plt.style.use('seaborn-v0_8-paper')
    except OSError:
        plt.style.use('seaborn-paper')   # Fallback for older matplotlib

    fig = plt.figure(figsize=(12, 14))

    # Row heights: panel 1 is largest
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1.2, 1.2, 1.6],
                           hspace=0.45)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3])

    if lc_df is None or len(lc_df) == 0:
        fig.text(0.5, 0.5, f'No data for {source_id}', ha='center', va='center')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return

    mjd = lc_df['mjd'].values
    w1_flux  = lc_df['w1_flux_mjy'].values  if 'w1_flux_mjy'     in lc_df.columns else None
    w1_err   = lc_df['w1_flux_err_mjy'].values if 'w1_flux_err_mjy' in lc_df.columns else None
    w2_flux  = lc_df['w2_flux_mjy'].values  if 'w2_flux_mjy'     in lc_df.columns else None
    w2_err   = lc_df['w2_flux_err_mjy'].values if 'w2_flux_err_mjy' in lc_df.columns else None
    w1_minus_w2 = lc_df['w1_minus_w2'].values if 'w1_minus_w2' in lc_df.columns else None
    dataset_flag = lc_df['dataset_flag'].values if 'dataset_flag' in lc_df.columns else None

    allwise_mask = (dataset_flag == 'allwise') if dataset_flag is not None else np.zeros(len(mjd), bool)
    neowise_mask = (dataset_flag == 'neowise') if dataset_flag is not None else np.ones(len(mjd), bool)

    # -------------------------------------------------------------------------
    # Panel 1: W1 and W2 flux light curve
    # -------------------------------------------------------------------------
    ax1.axvspan(WISE_HIBERNATION_MJD_START, WISE_HIBERNATION_MJD_END,
                alpha=0.12, color='gray', zorder=0, label='WISE hibernation')

    if w1_flux is not None and w1_err is not None:
        if allwise_mask.any():
            ax1.errorbar(
                mjd[allwise_mask], w1_flux[allwise_mask], yerr=w1_err[allwise_mask],
                fmt='o', color='steelblue', ms=6, lw=0.7, capsize=2,
                label='W1 AllWISE', zorder=3, alpha=0.85,
            )
        if neowise_mask.any():
            ax1.errorbar(
                mjd[neowise_mask], w1_flux[neowise_mask], yerr=w1_err[neowise_mask],
                fmt='D', color='royalblue', ms=4, lw=0.5, capsize=2,
                label='W1 NEOWISE-R', zorder=3, alpha=0.75,
            )

    if w2_flux is not None and w2_err is not None:
        if allwise_mask.any():
            ax1.errorbar(
                mjd[allwise_mask], w2_flux[allwise_mask], yerr=w2_err[allwise_mask],
                fmt='s', color='darkorange', ms=5, lw=0.7, capsize=2,
                label='W2 AllWISE', zorder=2, alpha=0.75,
            )
        if neowise_mask.any():
            ax1.errorbar(
                mjd[neowise_mask], w2_flux[neowise_mask], yerr=w2_err[neowise_mask],
                fmt='^', color='orangered', ms=4, lw=0.5, capsize=2,
                label='W2 NEOWISE-R', zorder=2, alpha=0.65,
            )

    # DRW GP prediction
    if drw_pred_mean is not None and len(drw_pred_mean) == len(mjd):
        idx_sort = np.argsort(mjd)
        t_plot = mjd[idx_sort]
        m_plot = drw_pred_mean[idx_sort]
        s_plot = drw_pred_std[idx_sort] if drw_pred_std is not None else np.zeros_like(m_plot)
        ax1.plot(t_plot, m_plot, '-', color='navy', lw=1.5, alpha=0.7, label='DRW GP mean')
        ax1.fill_between(t_plot, m_plot - s_plot, m_plot + s_plot,
                          alpha=0.18, color='navy', label='DRW ±1σ')

    # Changepoint break line
    if break_mjd is not None and np.isfinite(float(break_mjd)):
        ax1.axvline(float(break_mjd), color='crimson', ls='--', lw=1.5,
                    alpha=0.8, label=f'Breakpoint MJD={break_mjd:.0f}')

    ax1.set_ylabel('Flux density (mJy)', fontsize=11)
    ax1.legend(loc='upper right', fontsize=8, ncol=3)

    # ---- Dual x-axis: MJD (bottom) + calendar year (top) ------------------
    ax1_top = ax1.twiny()
    xlim = ax1.get_xlim()
    ax1_top.set_xlim(xlim)
    mjd_range = np.linspace(xlim[0], xlim[1], 6)
    year_labels = [f"{_mjd_to_year(m):.0f}" for m in mjd_range]
    ax1_top.set_xticks(mjd_range)
    ax1_top.set_xticklabels(year_labels, fontsize=9)
    ax1_top.set_xlabel('Calendar year', fontsize=10)

    # -------------------------------------------------------------------------
    # Panel 2: W1-W2 color evolution
    # -------------------------------------------------------------------------
    if w1_minus_w2 is not None:
        ax2.errorbar(mjd, w1_minus_w2,
                     fmt='o', color='mediumseagreen', ms=4, lw=0.5,
                     alpha=0.8, label='W1-W2')
        ax2.axhline(0.8, color='red', ls='--', lw=1.2, alpha=0.7,
                    label='AGN boundary (Stern+2012)')

        # Annotate color direction
        idx_sort = np.argsort(mjd)
        color_early = float(np.nanmedian(w1_minus_w2[idx_sort[:max(3, len(idx_sort)//5)]]))
        color_late  = float(np.nanmedian(w1_minus_w2[idx_sort[-max(3, len(idx_sort)//5):]]))
        if np.isfinite(color_early) and np.isfinite(color_late):
            direction = '→ bluer (turn-off)' if color_late < color_early else '→ redder (turn-on)'
            ax2.text(0.98, 0.88, direction, transform=ax2.transAxes,
                     ha='right', va='top', fontsize=8, color='darkgreen')

    ax2.set_ylabel('W1-W2 (Vega mag)', fontsize=10)
    ax2.legend(loc='upper left', fontsize=8)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # -------------------------------------------------------------------------
    # Panel 3: Standardized DRW residuals
    # -------------------------------------------------------------------------
    if drw_residuals is not None and len(drw_residuals) == len(mjd):
        ax3.scatter(mjd, drw_residuals, c='dimgray', s=12, alpha=0.7, zorder=2)
        ax3.axhline(0,  color='black', lw=0.8, alpha=0.5)
        for level, color, ls in [(2.0, 'orange', '--'), (3.0, 'red', ':')]:
            ax3.axhline( level, color=color, ls=ls, lw=1.2, alpha=0.7, label=f'+{level:.0f}σ')
            ax3.axhline(-level, color=color, ls=ls, lw=1.2, alpha=0.7)
        ax3.set_ylim(-6, 6)
        ax3.legend(loc='upper right', fontsize=7, ncol=2)

    ax3.set_ylabel('DRW residuals\n(standardized)', fontsize=9)
    ax3.set_xlabel('MJD', fontsize=11)

    # -------------------------------------------------------------------------
    # Panel 4: Structure Function
    # -------------------------------------------------------------------------
    if sf_result is not None:
        lag_centers = sf_result.get('lag_centers')
        sf_values   = sf_result.get('sf_values')
        sf_errors   = sf_result.get('sf_errors')
        n_bins      = len(lag_centers) if lag_centers is not None else 0
        valid       = sf_result.get('valid_bins', np.zeros(n_bins, bool))

        if lag_centers is not None and sf_values is not None:
            valid_idx = valid & np.isfinite(sf_values)
            if valid_idx.any():
                ax4.errorbar(
                    lag_centers[valid_idx], sf_values[valid_idx],
                    yerr=(sf_errors[valid_idx] if sf_errors is not None else None),
                    fmt='o', color='indigo', ms=6, lw=1.0, capsize=3,
                    label='Observed SF', zorder=3,
                )

                # Annotate SF excess region
                sf_excess = sf_result.get('sf_excess_ratio', 1.0)
                if np.isfinite(sf_excess) and sf_excess > 1.5:
                    ax4.text(0.98, 0.88,
                             f'SF excess ratio = {sf_excess:.2f}×',
                             transform=ax4.transAxes, ha='right', va='top',
                             fontsize=9, color='indigo')

    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xlabel('Lag Δt (days)', fontsize=11)
    ax4.set_ylabel('SF(Δt) (mag²)', fontsize=10)
    ax4.legend(loc='upper left', fontsize=8)

    # -------------------------------------------------------------------------
    # Title
    # -------------------------------------------------------------------------
    baseline = (mjd.max() - mjd.min()) / 365.25 if len(mjd) > 0 else 0.0
    title = (
        f"Source: {source_id}  |  z = {z:.3f}  |  "
        f"Baseline = {baseline:.1f} yr  |  Score = {score:.3f}\n"
        f"{label}"
    )
    fig.suptitle(title, fontsize=10, fontweight='bold', y=0.995)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved plot: {output_path}")


def plot_summary_grid(all_candidates, output_path):
    """
    Single-page summary figure showing W1 light curves for all top candidates
    in a 2×3 grid. Each panel annotated with rank, score, and classification.

    Parameters
    ----------
    all_candidates : list of result dicts (sorted by composite score, best first)
    output_path    : str, file path for the output PNG
    """
    try:
        plt.style.use('seaborn-v0_8-paper')
    except OSError:
        plt.style.use('seaborn-paper')

    n = min(6, len(all_candidates))
    if n == 0:
        logger.warning("No candidates for summary grid.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for rank_idx in range(n):
        ax = axes[rank_idx]
        candidate = all_candidates[rank_idx]
        source_id = candidate['source_row']['source_id']
        composite = candidate['score'].get('composite', 0.0)
        label     = candidate['score'].get('label', 'Unknown')
        z         = candidate['source_row']['redshift']
        lc        = candidate['wise'].get('lc')

        if lc is None or len(lc) == 0:
            ax.text(0.5, 0.5, f'{source_id}\n(no data)', transform=ax.transAxes,
                    ha='center', va='center')
            continue

        mjd = lc['mjd'].values
        w1_flux = lc['w1_flux_mjy'].values if 'w1_flux_mjy' in lc.columns else None
        w1_err  = lc['w1_flux_err_mjy'].values if 'w1_flux_err_mjy' in lc.columns else None
        dataset_flag = lc['dataset_flag'].values if 'dataset_flag' in lc.columns else None

        if w1_flux is not None:
            allwise = (dataset_flag == 'allwise') if dataset_flag is not None else np.zeros(len(mjd), bool)
            neowise = (dataset_flag == 'neowise') if dataset_flag is not None else np.ones(len(mjd), bool)

            if allwise.any():
                ax.errorbar(mjd[allwise], w1_flux[allwise],
                             yerr=w1_err[allwise] if w1_err is not None else None,
                             fmt='o', color='steelblue', ms=4, lw=0.5, alpha=0.8)
            if neowise.any():
                ax.errorbar(mjd[neowise], w1_flux[neowise],
                             yerr=w1_err[neowise] if w1_err is not None else None,
                             fmt='D', color='royalblue', ms=3, lw=0.4, alpha=0.7)

        ax.axvspan(WISE_HIBERNATION_MJD_START, WISE_HIBERNATION_MJD_END,
                   alpha=0.1, color='gray')

        # Changepoint break
        break_mjd = candidate.get('cp', {}).get('best_break_mjd')
        if break_mjd and np.isfinite(break_mjd):
            ax.axvline(break_mjd, color='crimson', ls='--', lw=1.0, alpha=0.7)

        # Compact label
        label_short = label[:50] + '…' if len(label) > 50 else label
        ax.set_title(
            f"#{rank_idx+1} {source_id}  z={z:.3f}\nScore={composite:.3f}  {label_short}",
            fontsize=7.5, pad=3,
        )
        ax.set_xlabel('MJD', fontsize=8)
        ax.set_ylabel('W1 (mJy)', fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused panels
    for i in range(n, 6):
        axes[i].set_visible(False)

    fig.suptitle('CLAGN Detection Pipeline — Top Candidates Summary', fontsize=12,
                  fontweight='bold')
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved summary grid: {output_path}")
