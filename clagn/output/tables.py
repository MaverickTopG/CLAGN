"""
tables.py — CSV and FITS output with all required columns.

All 46 columns from the CLAGN_RESEARCH_GRADE_PROMPT.md specification.
"""
import logging
import os
import numpy as np
import pandas as pd

from ..config import OUTPUT_CSV_FLOAT_FORMAT

logger = logging.getLogger(__name__)

# Canonical output column order (46 columns as specified)
OUTPUT_COLUMNS = [
    'rank', 'source_id', 'ra', 'dec', 'redshift',
    'baseline_years', 'n_epochs_total', 'n_epochs_pre_gap', 'n_epochs_post_gap',
    'w1_delta_mag', 'w2_delta_mag', 'w1_flux_ratio',
    'w1_minus_w2_early', 'w1_minus_w2_late', 'delta_color',
    'drw_tau_days', 'drw_tau_err_lo', 'drw_tau_err_hi',
    'drw_sigma_mjy', 'drw_sigma_err_lo', 'drw_sigma_err_hi',
    'drw_nonstationarity_sigma',
    'sf_excess_ratio', 'sf_break_detected',
    'changepoint_mjd', 'changepoint_delta_bic', 'changepoint_flux_ratio',
    'mannkendall_pvalue', 'trend_direction',
    'score_drw_nonstat', 'score_delta_mag', 'score_changepoint',
    'score_sf_break', 'score_color', 'score_sigma_excess', 'score_gaia',
    'composite_score',
    'gaia_ruwe', 'gaia_pm_sig', 'gaia_variable_flag', 'gaia_quasar_prob',
    'host_contamination_risk', 'contamination_flag',
    'clagn_type_label',
    'changepoint_break_duration_days',
    'changepoint_pre_break_mean',
    'changepoint_post_break_mean',
]


def _safe_val(val, default=np.nan):
    """Return default for None and NaN-like values, otherwise the value."""
    if val is None:
        return default
    try:
        v = float(val)
        return v
    except (TypeError, ValueError):
        # Non-numeric — return as-is (string columns)
        return val


def build_candidates_dataframe(all_results, top_n=6):
    """
    Build the output DataFrame from all pipeline results.

    Parameters
    ----------
    all_results : list of result dicts (from process_single_source)
                  Each has keys: source_row, wise, gaia, drw, sf, cp, score
    top_n       : int, number of top candidates to include

    Returns
    -------
    df : pd.DataFrame with OUTPUT_COLUMNS
    """
    # Sort by composite score (descending), filter out None/failed results
    valid_results = [r for r in all_results if r is not None]
    valid_results.sort(key=lambda r: r.get('score', {}).get('composite', 0.0),
                        reverse=True)
    top_results = valid_results[:top_n]

    rows = []
    for rank, res in enumerate(top_results, start=1):
        row = _extract_row(rank, res)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure all required columns are present (fill missing with NaN)
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Reorder to canonical order (only columns that exist)
    present_cols = [c for c in OUTPUT_COLUMNS if c in df.columns]
    extra_cols   = [c for c in df.columns if c not in OUTPUT_COLUMNS]
    df = df[present_cols + extra_cols]

    return df


def _extract_row(rank, res):
    """Extract all output columns from a single pipeline result dict."""
    src   = res.get('source_row', {})
    wise  = res.get('wise', {})
    gaia  = res.get('gaia', {})
    drw   = res.get('drw', {})
    sf    = res.get('sf', {})
    cp    = res.get('cp', {})
    score = res.get('score', {})
    nonst = res.get('nonstat', (np.nan, np.nan, np.nan, np.nan))

    # DRW nonstationarity — handle tuple or dict return
    if isinstance(nonst, (tuple, list)):
        ns_sigma = _safe_val(nonst[0])
    elif isinstance(nonst, dict):
        ns_sigma = _safe_val(nonst.get('nonstationarity_sigma'))
    else:
        ns_sigma = np.nan

    # MCMC uncertainties (if available)
    drw_mcmc = res.get('drw_mcmc', {})
    tau_lo   = _safe_val(drw_mcmc.get('tau_lo',  drw.get('tau_rest_days', np.nan)))
    tau_hi   = _safe_val(drw_mcmc.get('tau_hi',  drw.get('tau_rest_days', np.nan)))
    sig_lo   = _safe_val(drw_mcmc.get('sigma_lo', drw.get('sigma_drw', np.nan)))
    sig_hi   = _safe_val(drw_mcmc.get('sigma_hi', drw.get('sigma_drw', np.nan)))

    return {
        'rank':                    rank,
        'source_id':               str(src.get('source_id', '')),
        'ra':                      _safe_val(src.get('ra')),
        'dec':                     _safe_val(src.get('dec')),
        'redshift':                _safe_val(src.get('redshift')),
        'baseline_years':          _safe_val(wise.get('baseline_years')),
        'n_epochs_total':          _safe_val(wise.get('n_epochs_total')),
        'n_epochs_pre_gap':        _safe_val(wise.get('n_epochs_pre_gap')),
        'n_epochs_post_gap':       _safe_val(wise.get('n_epochs_post_gap')),
        'w1_delta_mag':            _safe_val(score.get('delta_mag_w1')),
        'w2_delta_mag':            _safe_val(score.get('delta_mag_w2')),
        'w1_flux_ratio':           _safe_val(score.get('w1_flux_ratio')),
        'w1_minus_w2_early':       _safe_val(wise.get('w1_minus_w2_early')),
        'w1_minus_w2_late':        _safe_val(wise.get('w1_minus_w2_late')),
        'delta_color':             _safe_val(score.get('delta_color')),
        'drw_tau_days':            _safe_val(drw.get('tau_rest_days')),
        'drw_tau_err_lo':          tau_lo,
        'drw_tau_err_hi':          tau_hi,
        'drw_sigma_mjy':           _safe_val(drw.get('sigma_drw')),
        'drw_sigma_err_lo':        sig_lo,
        'drw_sigma_err_hi':        sig_hi,
        'drw_nonstationarity_sigma': ns_sigma,
        'sf_excess_ratio':         _safe_val(sf.get('sf_excess_ratio')),
        'sf_break_detected':       bool(sf.get('break_detected', False)),
        'changepoint_mjd':         _safe_val(cp.get('best_break_mjd')),
        'changepoint_delta_bic':   _safe_val(cp.get('delta_bic')),
        'changepoint_flux_ratio':  _safe_val(cp.get('mean_ratio')),
        'mannkendall_pvalue':      _safe_val(cp.get('mk_pvalue')),
        'trend_direction':         str(cp.get('trend_dir', 'none')),
        'score_drw_nonstat':       _safe_val(score.get('score_drw_nonstat')),
        'score_delta_mag':         _safe_val(score.get('score_delta_mag')),
        'score_changepoint':       _safe_val(score.get('score_changepoint')),
        'score_sf_break':          _safe_val(score.get('score_sf_break')),
        'score_color':             _safe_val(score.get('score_color')),
        'score_sigma_excess':      _safe_val(score.get('score_sigma_excess')),
        'score_gaia':              _safe_val(score.get('score_gaia')),
        'composite_score':         _safe_val(score.get('composite')),
        'gaia_ruwe':               _safe_val(gaia.get('ruwe')),
        'gaia_pm_sig':             _safe_val(gaia.get('pm_sig')),
        'gaia_variable_flag':      str(gaia.get('phot_variable_flag', 'NOT_AVAILABLE')),
        'gaia_quasar_prob':        _safe_val(gaia.get('classprob_quasar')),
        'host_contamination_risk': bool(wise.get('host_contamination_risk', False)),
        'contamination_flag':      str(res.get('contamination_flag', 'none')),
        'clagn_type_label':        str(score.get('label', 'Unknown')),
        'changepoint_break_duration_days': _safe_val(cp.get('break_duration_days')),
        'changepoint_pre_break_mean':  _safe_val(cp.get('pre_break_mean')),
        'changepoint_post_break_mean': _safe_val(cp.get('post_break_mean')),
    }


def write_output_csv(candidates_df, output_path):
    """
    Write final candidates table to CSV.

    Parameters
    ----------
    candidates_df : pd.DataFrame
    output_path   : str
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    candidates_df.to_csv(output_path, index=False, float_format=OUTPUT_CSV_FLOAT_FORMAT)
    logger.info(f"Wrote {len(candidates_df)} candidates to {output_path}")


def write_output_fits(candidates_df, output_path):
    """
    Write final candidates table to FITS format.

    Parameters
    ----------
    candidates_df : pd.DataFrame
    output_path   : str
    """
    from astropy.table import Table
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # FITS character columns must be ASCII only; strip/replace any non-ASCII
    # characters (e.g. em-dash U+2014 in classification labels).
    df_fits = candidates_df.copy()
    for col in df_fits.select_dtypes(include='object').columns:
        df_fits[col] = df_fits[col].apply(
            lambda v: v.encode('ascii', errors='replace').decode('ascii')
            if isinstance(v, str) else v
        )

    tbl = Table.from_pandas(df_fits)
    tbl.write(output_path, format='fits', overwrite=True)
    logger.info(f"Wrote {len(candidates_df)} candidates to {output_path} (FITS)")
