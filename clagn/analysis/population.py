"""
population.py — Population statistics for CLAGN vs parent AGN sample.

Computes occurrence rates, redshift distributions, correlations between
variability parameters and physical quantities, and population comparisons.

References:
    Ricci & Trakhtenbrot 2022, arXiv:2211.05132 (CLAGN occurrence rate)
    MacLeod et al. 2019, ApJ 874, 8 (SDSS CLAGN demographics)
    Yang et al. 2023, ApJ 960, 29 (WISE CLAGN demographics)
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _bootstrap_ci(data: np.ndarray, statistic_func, n_boot: int = 1000,
                  ci: float = 0.68, seed: int = 42) -> tuple:
    """
    Bootstrap confidence interval for a statistic.

    Returns (stat, lo, hi) where lo and hi are the 1-sigma CI bounds.
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]

    if len(data) < 2:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed=seed)
    stat_obs = statistic_func(data)

    boot_stats = []
    for _ in range(n_boot):
        boot = rng.choice(data, size=len(data), replace=True)
        boot_stats.append(statistic_func(boot))

    boot_stats = np.array(boot_stats)
    lo_pct = 100.0 * (1.0 - ci) / 2.0
    hi_pct = 100.0 - lo_pct

    return (
        float(stat_obs),
        float(np.percentile(boot_stats, lo_pct)),
        float(np.percentile(boot_stats, hi_pct)),
    )


# ---------------------------------------------------------------------------
# Population analysis
# ---------------------------------------------------------------------------

def run_full_population_analysis(all_sources_df: pd.DataFrame,
                                  candidates_df: pd.DataFrame) -> dict:
    """
    Run complete population statistics comparing CLAGN candidates to
    the parent AGN sample.

    Analyses:
        1. Occurrence rate with Poisson uncertainty
        2. KS test on redshift distribution
        3. Spearman correlation: tau vs L_bol, sigma vs L_bol
        4. CLAGN fraction as function of redshift
        5. Luminosity function comparison

    Parameters
    ----------
    all_sources_df : DataFrame, full parent AGN sample
    candidates_df  : DataFrame, confirmed CLAGN candidates

    Returns
    -------
    stats : dict with all population statistics
    """
    from scipy import stats as scipy_stats

    n_total = len(all_sources_df)
    n_candidates = len(candidates_df)

    if n_total == 0:
        logger.warning("Empty parent sample — cannot compute occurrence rate")
        return {
            'n_total': 0, 'n_candidates': 0,
            'occurrence_rate': np.nan,
            'occurrence_rate_lo': np.nan,
            'occurrence_rate_hi': np.nan,
        }

    # ---- Occurrence rate -------------------------------------------------------
    rate = n_candidates / n_total
    # Poisson confidence interval on numerator
    # Exact Poisson CI: Wilson score or Clopper-Pearson
    from scipy.stats import chi2 as _chi2

    alpha = 0.32  # 1-sigma
    if n_candidates == 0:
        rate_lo = 0.0
        rate_hi = float(_chi2.ppf(1.0 - alpha / 2.0, 2) / 2.0 / n_total)
    else:
        rate_lo = float(_chi2.ppf(alpha / 2.0, 2 * n_candidates) / 2.0 / n_total)
        rate_hi = float(_chi2.ppf(1.0 - alpha / 2.0, 2 * (n_candidates + 1)) / 2.0 / n_total)

    logger.info(
        f"Occurrence rate: {n_candidates}/{n_total} = {rate:.4f} "
        f"[{rate_lo:.4f}, {rate_hi:.4f}]"
    )

    result = {
        'n_total': int(n_total),
        'n_candidates': int(n_candidates),
        'occurrence_rate': float(rate),
        'occurrence_rate_lo': float(rate_lo),
        'occurrence_rate_hi': float(rate_hi),
        'occurrence_rate_pct': float(rate * 100.0),
    }

    # ---- Redshift KS test -------------------------------------------------------
    z_all = pd.to_numeric(
        all_sources_df.get('z', all_sources_df.get('redshift', pd.Series(dtype=float))),
        errors='coerce'
    ).dropna().values

    z_cand = pd.to_numeric(
        candidates_df.get('z', candidates_df.get('redshift', pd.Series(dtype=float))),
        errors='coerce'
    ).dropna().values

    if len(z_all) >= 5 and len(z_cand) >= 3:
        ks_stat, ks_pval = scipy_stats.ks_2samp(z_all, z_cand)
        result['ks_redshift_stat'] = float(ks_stat)
        result['ks_redshift_pval'] = float(ks_pval)
        result['ks_redshift_significant'] = bool(ks_pval < 0.05)

        result['z_all_median'] = float(np.nanmedian(z_all))
        result['z_cand_median'] = float(np.nanmedian(z_cand))

        logger.info(
            f"KS test (redshift): D={ks_stat:.4f}, p={ks_pval:.4f} "
            f"(all: z_med={result['z_all_median']:.3f}, "
            f"CLAGN: z_med={result['z_cand_median']:.3f})"
        )
    else:
        result['ks_redshift_stat'] = np.nan
        result['ks_redshift_pval'] = np.nan
        result['ks_redshift_significant'] = False

    # ---- Spearman correlations -----------------------------------------------
    # tau vs L_bol
    tau_vals = pd.to_numeric(
        candidates_df.get('tau_rest_days', pd.Series(dtype=float)),
        errors='coerce'
    ).values
    L_bol_vals = pd.to_numeric(
        candidates_df.get('L_bol_erg_s', candidates_df.get('log_L_bol', pd.Series(dtype=float))),
        errors='coerce'
    ).values

    if np.sum(np.isfinite(tau_vals) & np.isfinite(L_bol_vals)) >= 5:
        valid = np.isfinite(tau_vals) & np.isfinite(L_bol_vals)
        tau_v = tau_vals[valid]
        L_v = L_bol_vals[valid]

        # Use log values for correlation
        log_tau = np.log10(np.maximum(tau_v, 1.0))
        log_L = np.log10(np.maximum(L_v, 1.0))

        spear_tau_L, p_tau_L = scipy_stats.spearmanr(log_tau, log_L)
        result['spearman_tau_vs_Lbol'] = float(spear_tau_L)
        result['spearman_tau_vs_Lbol_pval'] = float(p_tau_L)

        logger.info(
            f"Spearman tau vs L_bol: r={spear_tau_L:.3f}, p={p_tau_L:.4f}"
        )
    else:
        result['spearman_tau_vs_Lbol'] = np.nan
        result['spearman_tau_vs_Lbol_pval'] = np.nan

    # sigma vs L_bol
    sigma_vals = pd.to_numeric(
        candidates_df.get('sigma_drw', pd.Series(dtype=float)),
        errors='coerce'
    ).values

    if np.sum(np.isfinite(sigma_vals) & np.isfinite(L_bol_vals)) >= 5:
        valid = np.isfinite(sigma_vals) & np.isfinite(L_bol_vals)
        sig_v = sigma_vals[valid]
        L_v2 = L_bol_vals[valid]

        log_sig = np.log10(np.maximum(sig_v, 1e-10))
        log_L2 = np.log10(np.maximum(L_v2, 1.0))

        spear_sig_L, p_sig_L = scipy_stats.spearmanr(log_sig, log_L2)
        result['spearman_sigma_vs_Lbol'] = float(spear_sig_L)
        result['spearman_sigma_vs_Lbol_pval'] = float(p_sig_L)

        logger.info(
            f"Spearman sigma vs L_bol: r={spear_sig_L:.3f}, p={p_sig_L:.4f}"
        )
    else:
        result['spearman_sigma_vs_Lbol'] = np.nan
        result['spearman_sigma_vs_Lbol_pval'] = np.nan

    # ---- Luminosity distribution comparison ----------------------------------
    L_all = pd.to_numeric(
        all_sources_df.get('L_bol_erg_s', all_sources_df.get('log_L_bol', pd.Series(dtype=float))),
        errors='coerce'
    ).dropna().values

    L_cand = pd.to_numeric(
        candidates_df.get('L_bol_erg_s', candidates_df.get('log_L_bol', pd.Series(dtype=float))),
        errors='coerce'
    ).dropna().values

    if len(L_all) >= 5 and len(L_cand) >= 3:
        ks_L, p_L = scipy_stats.ks_2samp(np.log10(np.maximum(L_all, 1.0)),
                                           np.log10(np.maximum(L_cand, 1.0)))
        result['ks_luminosity_stat'] = float(ks_L)
        result['ks_luminosity_pval'] = float(p_L)
        result['L_all_median'] = float(np.nanmedian(L_all))
        result['L_cand_median'] = float(np.nanmedian(L_cand))

        logger.info(
            f"KS test (luminosity): D={ks_L:.4f}, p={p_L:.4f}"
        )
    else:
        result['ks_luminosity_stat'] = np.nan
        result['ks_luminosity_pval'] = np.nan

    # ---- CLAGN fraction vs redshift bins ------------------------------------
    if len(z_all) > 0 and len(z_cand) > 0:
        z_edges = np.array([0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0])
        clagn_fractions = []

        for i in range(len(z_edges) - 1):
            z_lo, z_hi = z_edges[i], z_edges[i + 1]
            n_all_bin = int(np.sum((z_all >= z_lo) & (z_all < z_hi)))
            n_cand_bin = int(np.sum((z_cand >= z_lo) & (z_cand < z_hi)))
            frac = n_cand_bin / max(n_all_bin, 1)
            clagn_fractions.append({
                'z_lo': z_lo, 'z_hi': z_hi,
                'n_all': n_all_bin, 'n_clagn': n_cand_bin,
                'fraction': frac,
            })

        result['clagn_fraction_vs_z'] = clagn_fractions

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_population_analysis(results_dir: str = './results/') -> None:
    """
    Run population statistics analysis and save outputs.

    Loads:
        - all_sources.csv or catalog output
        - top_candidates.csv
        - physics_summary.csv (for physical parameters)

    Saves:
        - population_stats.json
        - figures/population_*.png

    Parameters
    ----------
    results_dir : str, path to results directory
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    results_path = Path(results_dir)
    figures_dir = results_path / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load parent sample
    all_sources_df = pd.DataFrame()
    for fname in ['all_sources.csv', 'catalog_output.csv', 'catalog.csv']:
        p = results_path / fname
        if p.exists():
            try:
                all_sources_df = pd.read_csv(p)
                logger.info(f"Loaded parent sample: {p} ({len(all_sources_df)} sources)")
                break
            except Exception:
                pass

    # Load candidates
    candidates_df = pd.DataFrame()
    candidates_file = results_path / 'top_candidates.csv'
    if candidates_file.exists():
        try:
            candidates_df = pd.read_csv(candidates_file)
            logger.info(f"Loaded candidates: {len(candidates_df)}")
        except Exception as exc:
            logger.error(f"Failed to load candidates: {exc}")

    # Merge physics parameters into candidates
    physics_summary = results_path / 'physics_summary.csv'
    if physics_summary.exists():
        try:
            physics_df = pd.read_csv(physics_summary)
            if 'source_id' in candidates_df.columns and 'source_id' in physics_df.columns:
                candidates_df = candidates_df.merge(physics_df, on='source_id', how='left')
                logger.info("Merged physics parameters into candidates DataFrame")
        except Exception as exc:
            logger.warning(f"Could not merge physics summary: {exc}")

    if all_sources_df.empty:
        logger.warning("No parent sample found — using candidates as parent sample for statistics")
        all_sources_df = candidates_df.copy()

    # Run analysis
    stats = run_full_population_analysis(all_sources_df, candidates_df)

    # Save JSON
    json_path = results_path / 'population_stats.json'
    try:
        with open(json_path, 'w') as f:
            # Convert non-serializable values
            clean_stats = {}
            for k, v in stats.items():
                if isinstance(v, float) and not np.isfinite(v):
                    clean_stats[k] = None
                elif isinstance(v, (np.integer, np.floating)):
                    clean_stats[k] = float(v)
                elif isinstance(v, np.ndarray):
                    clean_stats[k] = v.tolist()
                else:
                    clean_stats[k] = v
            json.dump(clean_stats, f, indent=2)
        logger.info(f"Population statistics saved to {json_path}")
    except Exception as exc:
        logger.error(f"Failed to save population stats JSON: {exc}")

    # ---- Figures ---------------------------------------------------------------

    # Figure 1: Redshift distribution comparison
    fig = None
    try:
        fig, ax = plt.subplots(figsize=(8, 5))

        z_all = pd.to_numeric(
            all_sources_df.get('z', all_sources_df.get('redshift', pd.Series())),
            errors='coerce'
        ).dropna()
        z_cand = pd.to_numeric(
            candidates_df.get('z', candidates_df.get('redshift', pd.Series())),
            errors='coerce'
        ).dropna()

        bins = np.linspace(0, max(z_all.max() if len(z_all) > 0 else 3.0, 0.1), 25)

        if len(z_all) > 0:
            ax.hist(z_all, bins=bins, alpha=0.5, label=f'All AGN (N={len(z_all)})',
                    density=True, color='steelblue', histtype='stepfilled')
        if len(z_cand) > 0:
            ax.hist(z_cand, bins=bins, alpha=0.8, label=f'CLAGN (N={len(z_cand)})',
                    density=True, color='tomato', histtype='step', linewidth=2)

        ax.set_xlabel('Redshift $z$', fontsize=12)
        ax.set_ylabel('Normalized count', fontsize=12)
        ax.set_title('Redshift Distribution: CLAGN vs Parent AGN Sample', fontsize=12)
        ax.legend(fontsize=11)

        ks_stat = stats.get('ks_redshift_stat', np.nan)
        ks_pval = stats.get('ks_redshift_pval', np.nan)
        if np.isfinite(ks_stat):
            ax.text(0.97, 0.97,
                    f'KS: D={ks_stat:.3f}, p={ks_pval:.3f}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.tight_layout()
        for ext in ['png', 'pdf']:
            fig.savefig(figures_dir / f'population_redshift.{ext}', dpi=300)
    except Exception as exc:
        logger.warning(f"Redshift figure failed: {exc}")
    finally:
        if fig is not None:
            plt.close(fig)
            fig = None

    # Figure 2: Occurrence rate bar chart
    try:
        fig, ax = plt.subplots(figsize=(6, 4))

        rate = stats.get('occurrence_rate', np.nan)
        rate_lo = stats.get('occurrence_rate_lo', np.nan)
        rate_hi = stats.get('occurrence_rate_hi', np.nan)

        if np.isfinite(rate):
            yerr_lo = max(rate - rate_lo, 0.0)
            yerr_hi = max(rate_hi - rate, 0.0)
            ax.bar(['CLAGN\nOccurrence Rate'], [rate * 100.0],
                   yerr=[[yerr_lo * 100.0], [yerr_hi * 100.0]],
                   color='steelblue', capsize=8, width=0.4)
            ax.set_ylabel('Rate (%)', fontsize=12)
            ax.set_title(f'CLAGN Occurrence Rate\n({stats["n_candidates"]}/{stats["n_total"]} sources)',
                         fontsize=12)

        fig.tight_layout()
        for ext in ['png', 'pdf']:
            fig.savefig(figures_dir / f'population_occurrence.{ext}', dpi=300)
    except Exception as exc:
        logger.warning(f"Occurrence rate figure failed: {exc}")
    finally:
        if fig is not None:
            plt.close(fig)
            fig = None

    # Figure 3: Spearman correlation (tau vs L_bol)
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, x_col, y_col, x_label, y_label, spear_key in [
            (axes[0], 'L_bol_erg_s', 'tau_rest_days',
             r'$L_\mathrm{bol}$ [erg s$^{-1}$]', r'$\tau_\mathrm{rest}$ [days]',
             'spearman_tau_vs_Lbol'),
            (axes[1], 'L_bol_erg_s', 'sigma_drw',
             r'$L_\mathrm{bol}$ [erg s$^{-1}$]', r'$\sigma_\mathrm{DRW}$ [mJy]',
             'spearman_sigma_vs_Lbol'),
        ]:
            x = pd.to_numeric(candidates_df.get(x_col, pd.Series()), errors='coerce').values
            y = pd.to_numeric(candidates_df.get(y_col, pd.Series()), errors='coerce').values
            valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)

            if valid.sum() >= 3:
                ax.scatter(x[valid], y[valid], alpha=0.7, s=40, color='steelblue')
                ax.set_xscale('log')
                ax.set_yscale('log')

            ax.set_xlabel(x_label, fontsize=11)
            ax.set_ylabel(y_label, fontsize=11)

            r_val = stats.get(spear_key, np.nan)
            p_val = stats.get(f'{spear_key}_pval', np.nan)
            if np.isfinite(r_val):
                ax.set_title(f'r = {r_val:.3f}, p = {p_val:.3f}', fontsize=11)

        fig.tight_layout()
        for ext in ['png', 'pdf']:
            fig.savefig(figures_dir / f'population_correlations.{ext}', dpi=300)
    except Exception as exc:
        logger.warning(f"Correlation figure failed: {exc}")
    finally:
        if fig is not None:
            plt.close(fig)

    logger.info("Population analysis complete.")
