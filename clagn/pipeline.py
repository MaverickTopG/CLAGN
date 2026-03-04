"""
Lightweight pipeline facade for validation workflows.

Provides:
- run(df): run full pipeline on a DataFrame of sources
- score_single(...): score a single light curve without IRSA/GAIA queries
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import MIN_BASELINE_YEARS
from .ingestion.wise import query_wise_lightcurve
from .ingestion.gaia import query_gaia_dr3
from .models.drw import fit_drw_map, compute_drw_nonstationarity
from .models.structure_function import compute_structure_function_from_flux, detect_sf_break
from .models.changepoint import bayesian_changepoint_detection, test_monotonic_trend
from .scoring.clagn_score import compute_composite_clagn_score
from .utils.photometry import check_host_contamination

logger = logging.getLogger(__name__)


def _band_coverage_from_lc(lc_df: Optional[pd.DataFrame]) -> str:
    if lc_df is None or len(lc_df) == 0:
        return 'none'
    has_w1 = 'w1_flux_mjy' in lc_df.columns and np.isfinite(pd.to_numeric(lc_df['w1_flux_mjy'], errors='coerce')).any()
    has_w2 = 'w2_flux_mjy' in lc_df.columns and np.isfinite(pd.to_numeric(lc_df['w2_flux_mjy'], errors='coerce')).any()
    if has_w1 and has_w2:
        return 'W1,W2'
    if has_w1:
        return 'W1_only'
    if has_w2:
        return 'W2_only'
    return 'none'


class CLAGNPipeline:
    def __init__(self, results_dir: str = './results/') -> None:
        self.results_dir = Path(results_dir)
        self.operating_threshold = self._load_operating_threshold()

    def _load_operating_threshold(self) -> float:
        op_path = self.results_dir / 'validation' / 'operating_threshold.json'
        if op_path.exists():
            try:
                data = json.loads(op_path.read_text())
                return float(data.get('threshold', 0.5))
            except Exception:
                return 0.5
        return 0.5

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run full pipeline on a DataFrame of sources (requires IRSA + Gaia).

        Returns a DataFrame with composite scores and key metrics.
        """
        rows = []
        total = len(df)
        for idx, row in df.iterrows():
            source_id = str(row.get('name', row.get('source_id', f'src_{idx}')))
            ra = float(row['ra'])
            dec = float(row['dec'])
            z = float(row.get('redshift', row.get('z', 0.1)))
            try:
                wise_result = query_wise_lightcurve(ra, dec, source_id, z=z)
                lc_df = wise_result.get('lc')
                n_points = int(len(lc_df)) if lc_df is not None else 0
                baseline_years = float(wise_result.get('baseline_years', 0.0) or 0.0)
                band_coverage = _band_coverage_from_lc(lc_df)

                if not wise_result.get('passes_baseline', True):
                    rejection_reason = str(wise_result.get('rejection_reason', '') or '')
                    status = 'NO_DATA' if ('no data' in rejection_reason.lower()) else 'TOO_FEW_POINTS'
                    rows.append({
                        'name': source_id,
                        'ra': ra,
                        'dec': dec,
                        'composite_score_v2': np.nan,
                        'baseline_years': baseline_years,
                        'baseline_days': float(baseline_years * 365.25),
                        'delta_mag': np.nan,
                        'delta_mag_method': None,
                        'rejection_reason': rejection_reason or 'baseline',
                        'status': status,
                        'n_points': n_points,
                        'band_coverage': band_coverage,
                    })
                    continue

                if lc_df is None or len(lc_df) < 6:
                    rows.append({
                        'name': source_id,
                        'ra': ra,
                        'dec': dec,
                        'composite_score_v2': np.nan,
                        'baseline_years': baseline_years,
                        'baseline_days': float(baseline_years * 365.25),
                        'delta_mag': np.nan,
                        'delta_mag_method': None,
                        'rejection_reason': 'insufficient_lc',
                        'status': 'TOO_FEW_POINTS',
                        'n_points': n_points,
                        'band_coverage': band_coverage,
                    })
                    continue

                gaia_result = query_gaia_dr3(ra, dec, source_id)

                times = lc_df['mjd'].values
                w1_flux = lc_df['w1_flux_mjy'].values
                w1_err = lc_df['w1_flux_err_mjy'].values

                drw_map = fit_drw_map(times, w1_flux, w1_err, z=z)
                nonstat = compute_drw_nonstationarity(
                    times, w1_flux, w1_err, z,
                    drw_map['tau_rest_days'], drw_map['sigma_drw'],
                )
                drw_map['nonstationarity_sigma'] = float(nonstat.get('nonstationarity_sigma', 0.0))

                sf_raw = compute_structure_function_from_flux(times, w1_flux, w1_err, band='W1')
                sf_break = detect_sf_break(
                    sf_raw['lag_centers'], sf_raw['sf_values'], sf_raw['sf_errors'],
                    drw_map['tau_rest_days'],
                )
                sf_combined = {**sf_raw, **sf_break}

                cp_results = bayesian_changepoint_detection(times, w1_flux, w1_err)
                mk_results = test_monotonic_trend(times, w1_flux, w1_err)
                cp_combined = {**cp_results, **mk_results}

                score = compute_composite_clagn_score(row, wise_result, drw_map, sf_combined, cp_combined, gaia_result)

                rows.append({
                    'name': source_id,
                    'ra': ra,
                    'dec': dec,
                    'composite_score_v2': score.get('composite', np.nan),
                    'baseline_years': baseline_years,
                    'baseline_days': float(baseline_years * 365.25),
                    'delta_mag': score.get('delta_mag_w1', np.nan),
                    'delta_mag_method': score.get('delta_mag_method', 'seasonal_median_flux'),
                    'status': 'OK',
                    'n_points': n_points,
                    'band_coverage': band_coverage,
                    'rejection_reason': None,
                })
            except Exception as exc:
                logger.exception("Pipeline scoring failed for %s", source_id)
                rows.append({
                    'name': source_id,
                    'ra': ra,
                    'dec': dec,
                    'composite_score_v2': np.nan,
                    'baseline_years': np.nan,
                    'baseline_days': np.nan,
                    'delta_mag': np.nan,
                    'delta_mag_method': None,
                    'status': 'ERROR',
                    'n_points': 0,
                    'band_coverage': 'none',
                    'rejection_reason': f'error:{type(exc).__name__}',
                })

        return pd.DataFrame(rows)

    def score_single(self, times: np.ndarray, flux: np.ndarray,
                     flux_err: np.ndarray, z: float = 0.1) -> dict:
        """
        Score a single light curve without external queries.
        """
        times = np.asarray(times, dtype=float)
        flux = np.asarray(flux, dtype=float)
        flux_err = np.asarray(flux_err, dtype=float)

        lc_df = pd.DataFrame({
            'mjd': times,
            'w1_flux_mjy': flux,
            'w1_flux_err_mjy': flux_err,
        })

        baseline_years = (np.nanmax(times) - np.nanmin(times)) / 365.25 if len(times) else 0.0

        wise_result = {
            'lc': lc_df,
            'baseline_years': baseline_years,
            'n_epochs_total': len(times),
            'w1_minus_w2_early': np.nan,
            'w1_minus_w2_late': np.nan,
            'host_contamination_risk': check_host_contamination(z),
            'passes_baseline': baseline_years >= MIN_BASELINE_YEARS,
        }

        drw_map = fit_drw_map(times, flux, flux_err, z=z)
        nonstat = compute_drw_nonstationarity(
            times, flux, flux_err, z,
            drw_map['tau_rest_days'], drw_map['sigma_drw'],
        )
        drw_map['nonstationarity_sigma'] = float(nonstat.get('nonstationarity_sigma', 0.0))

        sf_raw = compute_structure_function_from_flux(times, flux, flux_err, band='W1')
        sf_break = detect_sf_break(
            sf_raw['lag_centers'], sf_raw['sf_values'], sf_raw['sf_errors'],
            drw_map['tau_rest_days'],
        )
        sf_combined = {**sf_raw, **sf_break}

        cp_results = bayesian_changepoint_detection(times, flux, flux_err)
        mk_results = test_monotonic_trend(times, flux, flux_err)
        cp_combined = {**cp_results, **mk_results}

        gaia_null = {
            'phot_variable_flag': 'NOT_AVAILABLE',
            'classprob_quasar': np.nan,
            'pm_sig': np.nan,
            'parallax_sig': np.nan,
            'ruwe': np.nan,
        }

        score = compute_composite_clagn_score(
            {'redshift': z}, wise_result, drw_map, sf_combined, cp_combined, gaia_null
        )

        return score
