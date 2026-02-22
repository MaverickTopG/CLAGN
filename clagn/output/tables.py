"""
tables.py — CSV and FITS output with all required columns.

FIX 12: Column names renamed to be unambiguous about method and units.

Column naming convention:
    delta_mag_w1_seasonal_pogson  — signed, seasonal median flux, +2.5*log10(F_late/F_early)
    drw_tau_rest_days             — rest-frame timescale (not observer-frame)
    drw_sigma_flux_mjy            — DRW amplitude in flux units (not magnitudes)
"""
import logging
import os
import numpy as np
import pandas as pd

from ..config import OUTPUT_CSV_FLOAT_FORMAT

logger = logging.getLogger(__name__)

# FIX 12: Mapping from old column names to new unambiguous names
COLUMN_RENAME_MAP = {
    'w1_delta_mag':  'delta_mag_w1_seasonal_pogson',
    'w2_delta_mag':  'delta_mag_w2_seasonal_pogson',
    'w1_flux_ratio': 'flux_ratio_w1_seasonal',
    'drw_tau_days':  'drw_tau_rest_days',
    'drw_sigma_mjy': 'drw_sigma_flux_mjy',
}

# Canonical output column order (with FIX 12 renamed columns + new FIX columns)
OUTPUT_COLUMNS = [
    'rank', 'source_id', 'ra', 'dec', 'redshift',
    'baseline_years', 'n_epochs_total', 'n_epochs_pre_gap', 'n_epochs_post_gap',
    # FIX 12: renamed amplitude columns
    'delta_mag_w1_seasonal_pogson', 'delta_mag_w2_seasonal_pogson',
    'flux_ratio_w1_seasonal',
    # FIX 1: new column for method documentation
    'delta_mag_method',
    # FIX 6: host contamination
    'host_contamination_flag', 'delta_mag_host_corrected',
    'w1_minus_w2_early', 'w1_minus_w2_late', 'delta_color',
    # FIX 12: renamed DRW columns
    'drw_tau_rest_days', 'drw_tau_err_lo', 'drw_tau_err_hi',
    'drw_sigma_flux_mjy', 'drw_sigma_err_lo', 'drw_sigma_err_hi',
    'drw_sigma_mag_equiv',
    # FIX 5: reliability flag
    'tau_reliable',
    'drw_nonstationarity_sigma',
    'sf_excess_ratio', 'sf_break_detected',
    'changepoint_mjd', 'changepoint_delta_bic', 'changepoint_flux_ratio',
    'mannkendall_pvalue', 'trend_direction',
    'score_drw_nonstat', 'score_delta_mag', 'score_changepoint',
    'score_sf_break', 'score_color', 'score_sigma_excess', 'score_gaia',
    'composite_score',
    # FIX 10: group breakdown
    'group_A_amplitude', 'group_B_temporal', 'group_C_color',
    'group_D_statistical', 'group_E_astrometric',
    'gaia_ruwe', 'gaia_pm_sig', 'gaia_variable_flag', 'gaia_quasar_prob',
    # FIX 9: parallax
    'gaia_parallax_sig',
    'host_contamination_risk', 'contamination_flag',
    'clagn_type_label',
    'changepoint_break_duration_days',
    'changepoint_pre_break_mean',
    'changepoint_post_break_mean',
    # FIX 4: W2 systematic flag
    'w2_systematic_flag',
    # FIX 13: W1/W2 coherence
    'w1_w2_coherence_score', 'w1_w2_pearson_r', 'w1_w2_same_direction',
    'w1_w2_coherence_flag',
    # FIX 14: saturation
    'saturation_flag',
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
        # FIX 12: renamed amplitude columns
        'delta_mag_w1_seasonal_pogson': _safe_val(score.get('delta_mag_w1')),
        'delta_mag_w2_seasonal_pogson': _safe_val(score.get('delta_mag_w2')),
        'flux_ratio_w1_seasonal':  _safe_val(score.get('w1_flux_ratio')),
        # FIX 1: method documentation
        'delta_mag_method':        str(score.get('delta_mag_method', 'seasonal_median_flux')),
        # FIX 6: host contamination
        'host_contamination_flag': str(score.get('host_contamination_flag', 'negligible')),
        'delta_mag_host_corrected': _safe_val(score.get('delta_mag_host_corrected')),
        'w1_minus_w2_early':       _safe_val(wise.get('w1_minus_w2_early')),
        'w1_minus_w2_late':        _safe_val(wise.get('w1_minus_w2_late')),
        'delta_color':             _safe_val(score.get('delta_color')),
        # FIX 12: renamed DRW columns
        'drw_tau_rest_days':       _safe_val(drw.get('tau_rest_days')),
        'drw_tau_err_lo':          tau_lo,
        'drw_tau_err_hi':          tau_hi,
        'drw_sigma_flux_mjy':      _safe_val(drw.get('sigma_drw')),
        'drw_sigma_err_lo':        sig_lo,
        'drw_sigma_err_hi':        sig_hi,
        'drw_sigma_mag_equiv':     _safe_val(drw.get('sigma_drw_mag_equiv')),
        # FIX 5: DRW reliability
        'tau_reliable':            bool(drw.get('tau_reliable', True)),
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
        # FIX 10: group breakdown
        'group_A_amplitude':       _safe_val(score.get('group_A_amplitude')),
        'group_B_temporal':        _safe_val(score.get('group_B_temporal')),
        'group_C_color':           _safe_val(score.get('group_C_color')),
        'group_D_statistical':     _safe_val(score.get('group_D_statistical')),
        'group_E_astrometric':     _safe_val(score.get('group_E_astrometric')),
        'gaia_ruwe':               _safe_val(gaia.get('ruwe')),
        'gaia_pm_sig':             _safe_val(gaia.get('pm_sig')),
        'gaia_variable_flag':      str(gaia.get('phot_variable_flag', 'NOT_AVAILABLE')),
        'gaia_quasar_prob':        _safe_val(gaia.get('classprob_quasar')),
        # FIX 9: parallax significance
        'gaia_parallax_sig':       _safe_val(gaia.get('parallax_sig', gaia.get('plx_sig'))),
        'host_contamination_risk': bool(wise.get('host_contamination_risk', False)),
        'contamination_flag':      str(res.get('contamination_flag', 'none')),
        'clagn_type_label':        str(score.get('label', 'Unknown')),
        'changepoint_break_duration_days': _safe_val(cp.get('break_duration_days')),
        'changepoint_pre_break_mean':  _safe_val(cp.get('pre_break_mean')),
        'changepoint_post_break_mean': _safe_val(cp.get('post_break_mean')),
        # FIX 4: W2 systematic flag count
        'w2_systematic_flag':      _safe_val(wise.get('w2_systematic_flag_count', 0)),
        # FIX 13: W1/W2 coherence
        'w1_w2_coherence_score':   _safe_val(score.get('w1_w2_coherence_score')),
        'w1_w2_pearson_r':         _safe_val(score.get('w1_w2_pearson_r')),
        'w1_w2_same_direction':    score.get('w1_w2_same_direction'),
        'w1_w2_coherence_flag':    str(score.get('w1_w2_coherence_flag', 'insufficient_data')),
        # FIX 14: saturation flag
        'saturation_flag':         str(wise.get('saturation_flag', '')),
    }


def apply_column_renames(df):
    """
    Apply COLUMN_RENAME_MAP to ensure output uses unambiguous column names.

    FIX 12: Rename ambiguous columns (e.g. w1_delta_mag → delta_mag_w1_seasonal_pogson).
    Safe to call multiple times — idempotent.
    """
    # Only rename columns that exist in the DataFrame
    rename = {old: new for old, new in COLUMN_RENAME_MAP.items() if old in df.columns}
    if rename:
        df = df.rename(columns=rename)
        logger.debug(f"Renamed columns: {rename}")
    return df


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
