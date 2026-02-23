"""
main.py — CLI entry point for the CLAGN detection pipeline.

Usage:
    python main.py --input catalog.csv --output ./results/ --top_n 6 --test_n 5

Arguments:
    --input     : Path to input catalog (CSV or FITS)
    --output    : Output directory (created if not exists)
    --top_n     : Number of top candidates to output (default: 6)
    --test_n    : If set, only process the first N sources (for debugging)
    --mcmc      : Run full MCMC on top candidates (slower, more rigorous)
    --resume    : Resume from checkpoint if previous run was interrupted
    --workers   : Number of parallel workers (default: 4)

Scientific basis: Ricci & Trakhtenbrot 2022 (arXiv:2211.05132)
"""
import argparse
import logging
import os
import pickle
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# Suppress noisy upstream warnings
warnings.filterwarnings('ignore', message='.*passwords of all user accounts.*')
warnings.filterwarnings('ignore', category=FutureWarning, module='astroquery')

from .config import (
    CHECKPOINT_DIR_DEFAULT, PARALLEL_WORKERS_DEFAULT,
    MIN_MAG_CHANGE_W1, MIN_FLUX_RATIO_CHANGE, MIN_BASELINE_YEARS, MIN_REDSHIFT,
    MAX_PROPER_MOTION_SIG,
)
from .ingestion.catalog import load_and_validate_catalog
from .ingestion.wise import query_wise_lightcurve
from .ingestion.gaia import query_gaia_dr3
from .models.drw import fit_drw_map, fit_drw_mcmc, compute_drw_nonstationarity
from .models.structure_function import (
    compute_structure_function_from_flux, detect_sf_break,
)
from .models.changepoint import bayesian_changepoint_detection, test_monotonic_trend
from .scoring.clagn_score import (
    compute_composite_clagn_score, apply_false_positive_rejection,
)
from .output.plots import plot_lightcurve_panel, plot_summary_grid
from .output.tables import build_candidates_dataframe, write_output_csv, write_output_fits
from .utils.provenance import write_provenance_sidecar
from .utils.validation import validate_against_known_clagn

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(output_dir, verbose=False):
    """Configure logging to both console and file.

    Note: Do NOT use force=True with basicConfig — it destroys astroquery's
    AstropyLogger which has custom methods (_set_defaults, etc.) that standard
    logging.Logger does not have. Instead we add handlers directly.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'pipeline.log')

    level = logging.DEBUG if verbose else logging.INFO
    fmt   = '%(asctime)s [%(levelname)s] %(name)s — %(message)s'
    formatter = logging.Formatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)

    # Add handlers only if not already present (avoids duplicates on re-runs)
    handler_types = {type(h) for h in root.handlers}

    if logging.StreamHandler not in handler_types:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        root.addHandler(sh)

    # Always add a fresh file handler for this run
    fh = logging.FileHandler(log_path, mode='a')
    fh.setFormatter(formatter)
    root.addHandler(fh)

    # Reduce verbosity of non-astropy libraries only.
    # IMPORTANT: do NOT call logging.getLogger('astropy') or logging.getLogger('astroquery')
    # here — those libraries use AstropyLogger (a Logger subclass with custom methods).
    # Pre-creating a standard Logger under those names before the libraries are imported
    # blocks AstropyLogger initialisation and causes AttributeError on _set_defaults.
    for noisy in ('matplotlib', 'urllib3', 'requests', 'pyvo'):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger('clagn.main')


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(source_id, result, checkpoint_dir):
    """Save per-source results to disk as we go — never lose work."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Sanitize source_id for use as a filename
    safe_id = str(source_id).replace('/', '_').replace(':', '_').replace(' ', '_')
    path = os.path.join(checkpoint_dir, f"{safe_id}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(source_id, checkpoint_dir):
    """Load previously computed result. Returns None if not found."""
    safe_id = str(source_id).replace('/', '_').replace(':', '_').replace(' ', '_')
    path = os.path.join(checkpoint_dir, f"{safe_id}.pkl")
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Per-source processing (top-level function for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def process_single_source(args_tuple):
    """
    Process a single source through the full CLAGN detection pipeline.

    Must be a top-level function (not a closure) for ProcessPoolExecutor.

    Parameters
    ----------
    args_tuple : (source_dict, idx, total, checkpoint_dir, resume, worker_delay)

    Returns
    -------
    result : dict or None (None = source rejected or failed)
    """
    source_row, idx, total, checkpoint_dir, resume, worker_delay = args_tuple

    # Stagger workers to avoid IRSA rate limiting
    if worker_delay > 0:
        time.sleep(worker_delay)

    source_id = source_row['source_id']
    ra        = float(source_row['ra'])
    dec       = float(source_row['dec'])
    z         = float(source_row['redshift'])

    # Set up per-process logger
    logger = logging.getLogger(f'clagn.worker.{source_id}')

    def log_status(msg, level='info'):
        prefix = f"[{idx:03d}/{total}] {source_id}"
        full_msg = f"{prefix} | {msg}"
        getattr(logger, level)(full_msg)
        print(full_msg)   # Also print for real-time progress

    try:
        # ---- 1. Check checkpoint (resume mode) --------------------------------
        if resume:
            cached = load_checkpoint(source_id, checkpoint_dir)
            if cached is not None:
                log_status(
                    f"z={z:.3f} | RESUMED from checkpoint | "
                    f"Score={cached.get('score', {}).get('composite', 0.0):.3f}"
                )
                return cached

        # ---- 2. Query WISE light curve ----------------------------------------
        wise_result = query_wise_lightcurve(ra, dec, source_id, z=z)

        if not wise_result['passes_baseline']:
            reason = wise_result.get('rejection_reason', 'unknown')
            log_status(f"z={z:.3f} | Rejected: {reason}")
            return None

        lc_df = wise_result['lc']
        baseline = wise_result['baseline_years']

        # ---- 3. Query GAIA DR3 -----------------------------------------------
        gaia_result = query_gaia_dr3(ra, dec, source_id)

        # Reject confirmed stars (significant proper motion)
        if (gaia_result.get('gaia_found') and
                not gaia_result.get('passes_pm_filter', True)):
            pm_sig = gaia_result.get('pm_sig', 0.0)
            log_status(
                f"z={z:.3f} | Rejected: stellar PM = {pm_sig:.1f}σ "
                f"> {MAX_PROPER_MOTION_SIG}σ threshold"
            )
            return None

        # ---- 4. DRW MAP fitting -----------------------------------------------
        times      = lc_df['mjd'].values
        w1_flux    = lc_df['w1_flux_mjy'].values
        w1_flux_err = lc_df['w1_flux_err_mjy'].values

        drw_map = fit_drw_map(times, w1_flux, w1_flux_err, z=z)

        # ---- 5. DRW nonstationarity ------------------------------------------
        nonstat = compute_drw_nonstationarity(
            times, w1_flux, w1_flux_err, z,
            drw_map['tau_rest_days'], drw_map['sigma_drw'],
        )
        nonstationarity_sigma = nonstat.get('nonstationarity_sigma', 0.0)
        drw_map['nonstationarity_sigma'] = (
            float(nonstationarity_sigma) if np.isfinite(float(nonstationarity_sigma or 0))
            else 0.0
        )

        # ---- 6. Structure function -------------------------------------------
        if 'w1_mag' in lc_df.columns and 'w1_err' in lc_df.columns:
            sf_raw = compute_structure_function_from_flux(
                times, w1_flux, w1_flux_err, band='W1'
            )
        else:
            from .models.structure_function import _empty_sf_result, _empty_sf_break_result
            sf_raw = _empty_sf_result()

        sf_break = detect_sf_break(
            sf_raw['lag_centers'], sf_raw['sf_values'], sf_raw['sf_errors'],
            drw_map['tau_rest_days'],
        )
        sf_combined = {**sf_raw, **sf_break}

        # ---- 7. Changepoint detection ----------------------------------------
        cp_results = bayesian_changepoint_detection(times, w1_flux, w1_flux_err)
        mk_results = test_monotonic_trend(times, w1_flux, w1_flux_err)
        cp_combined = {**cp_results, **mk_results}

        # ---- 8. Composite CLAGN score ----------------------------------------
        score_result = compute_composite_clagn_score(
            source_row, wise_result, drw_map, sf_break, cp_combined, gaia_result
        )

        composite = score_result.get('composite', 0.0)

        log_status(
            f"z={z:.3f} | Baseline={baseline:.1f}yr | "
            f"n_epochs={wise_result['n_epochs_total']} | "
            f"Score={composite:.3f} | {score_result.get('label', '')[:50]}"
        )

        result = {
            'source_row':       source_row,
            'wise':             wise_result,
            'gaia':             gaia_result,
            'drw':              drw_map,
            'sf':               sf_combined,
            'cp':               cp_combined,
            'nonstat':          nonstat,
            'score':            score_result,
            'contamination_flag': 'none',
        }

        save_checkpoint(source_id, result, checkpoint_dir)
        return result

    except Exception as exc:
        logger.error(
            f"[{idx:03d}/{total}] {source_id} | ERROR: {exc}\n"
            f"{traceback.format_exc()}"
        )
        print(f"[{idx:03d}/{total}] {source_id} | ERROR: {exc}")
        return None


# ---------------------------------------------------------------------------
# Final sanity checks
# ---------------------------------------------------------------------------

def apply_final_sanity_checks(top_results, all_results, top_n, logger):
    """
    Verify each top candidate meets minimum CLAGN criteria.

    Criteria (from spec):
    - delta_mag_w1 > MIN_MAG_CHANGE_W1 (0.3 mag)
    - flux_ratio_w1 > MIN_FLUX_RATIO_CHANGE (factor 2)
    - baseline_years >= MIN_BASELINE_YEARS (10.0 yr)
    - redshift > MIN_REDSHIFT
    - gaia_pm_sig < MAX_PROPER_MOTION_SIG (or flagged as unknown)

    If a candidate fails, replace with next ranked source.

    Returns
    -------
    final_top : list of result dicts, length <= top_n
    """
    all_valid = [r for r in all_results if r is not None]
    all_valid.sort(key=lambda r: r.get('score', {}).get('composite', 0.0),
                    reverse=True)

    final_top = []
    used_ids = set()

    for candidate in all_valid:
        if len(final_top) >= top_n:
            break

        src = candidate['source_row']
        wise = candidate['wise']
        gaia = candidate['gaia']
        score = candidate['score']
        source_id = str(src['source_id'])

        if source_id in used_ids:
            continue

        failures = []

        # Check delta_mag — use abs() so turn-off CLAGNs (negative delta_mag) pass
        delta_mag = score.get('delta_mag_w1', 0.0) or 0.0
        if abs(delta_mag) < MIN_MAG_CHANGE_W1:
            failures.append(f"|delta_mag_w1|={abs(delta_mag):.3f} < {MIN_MAG_CHANGE_W1}")

        # Check flux ratio — use max(r, 1/r) so both brightening and fading qualify
        flux_ratio = score.get('w1_flux_ratio', 1.0) or 1.0
        effective_ratio = max(flux_ratio, 1.0 / flux_ratio) if flux_ratio > 0 else 0.0
        if effective_ratio < MIN_FLUX_RATIO_CHANGE:
            failures.append(f"flux_ratio={flux_ratio:.2f} (effective={effective_ratio:.2f}) < {MIN_FLUX_RATIO_CHANGE}")

        # Check baseline
        baseline = wise.get('baseline_years', 0.0) or 0.0
        if baseline < MIN_BASELINE_YEARS:
            failures.append(f"baseline={baseline:.1f}yr < {MIN_BASELINE_YEARS}yr")

        # Check redshift
        z = float(src.get('redshift', 0.0))
        if z <= MIN_REDSHIFT:
            failures.append(f"z={z:.4f} <= {MIN_REDSHIFT}")

        # Check GAIA PM
        pm_sig = gaia.get('pm_sig', np.nan)
        gaia_pm_unknown = gaia.get('gaia_pm_unknown', True)
        if (not gaia_pm_unknown and np.isfinite(pm_sig) and
                pm_sig >= MAX_PROPER_MOTION_SIG):
            failures.append(f"pm_sig={pm_sig:.1f} >= {MAX_PROPER_MOTION_SIG}")

        if failures:
            logger.warning(
                f"Sanity check: {source_id} FAILS criteria — "
                f"{'; '.join(failures)} — replacing with next ranked source"
            )
            continue

        final_top.append(candidate)
        used_ids.add(source_id)

    return final_top


# ---------------------------------------------------------------------------
# Main pipeline orchestration
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='CLAGN Detection Pipeline (Ricci & Trakhtenbrot 2022)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input',   required=True,
                        help='Path to input catalog (CSV or FITS)')
    parser.add_argument('--output',  required=True,
                        help='Output directory')
    parser.add_argument('--top_n',   type=int, default=6,
                        help='Number of top candidates to output')
    parser.add_argument('--test_n',  type=int, default=None,
                        help='Process only first N sources (test mode)')
    parser.add_argument('--mcmc',    action='store_true',
                        help='Run full MCMC on top candidates (slower)')
    parser.add_argument('--resume',  action='store_true',
                        help='Resume from checkpoints if available')
    parser.add_argument('--workers', type=int, default=PARALLEL_WORKERS_DEFAULT,
                        help='Number of parallel workers')
    parser.add_argument('--min_baseline', type=float, default=None,
                        help='Override minimum baseline in years (default: config value)')
    parser.add_argument('--min_epochs', type=int, default=None,
                        help='Override minimum number of epochs (default: config value)')
    parser.add_argument('--strict_quality', action='store_true',
                        help='Enforce strict quality gates (absolute threshold checks)')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    # ---- Apply CLI overrides to module-level constants ----------------------
    import clagn.config as _cfg
    if args.min_baseline is not None:
        _cfg.MIN_BASELINE_YEARS = args.min_baseline
    if args.min_epochs is not None:
        _cfg.MIN_EPOCHS = args.min_epochs

    # ---- Setup --------------------------------------------------------------
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, 'plots'), exist_ok=True)
    checkpoint_dir = os.path.join(args.output, CHECKPOINT_DIR_DEFAULT)
    logger = setup_logging(args.output, verbose=args.verbose)

    logger.info("=" * 70)
    logger.info("CLAGN DETECTION PIPELINE")
    logger.info("Scientific basis: Ricci & Trakhtenbrot 2022 (arXiv:2211.05132)")
    logger.info("=" * 70)
    logger.info(f"Input:   {args.input}")
    logger.info(f"Output:  {args.output}")
    logger.info(f"Top N:   {args.top_n}")
    logger.info(f"Workers: {args.workers}")
    if args.test_n:
        logger.info(f"TEST MODE: processing first {args.test_n} sources only")
    if args.mcmc:
        logger.info("MCMC mode enabled — will run full MCMC on top candidates")

    # ---- Load and validate catalog ------------------------------------------
    logger.info(f"\nLoading catalog: {args.input}")
    catalog_df = load_and_validate_catalog(args.input)

    if args.test_n is not None:
        catalog_df = catalog_df.head(args.test_n)
        logger.info(f"Test mode: using first {args.test_n} sources")

    total = len(catalog_df)
    logger.info(f"Processing {total} sources\n")

    # Build list of source dicts for the executor
    source_list = catalog_df.to_dict(orient='records')

    # Build args tuples: (source_row, idx, total, checkpoint_dir, resume, worker_delay)
    # Stagger workers by 0.5s each to avoid IRSA rate-limiting burst
    task_args = [
        (src, i + 1, total, checkpoint_dir, args.resume,
         (i % args.workers) * 0.5)
        for i, src in enumerate(source_list)
    ]

    # ---- Process sources ----------------------------------------------------
    t_start = time.time()
    all_results = []

    if args.workers == 1:
        # Single-process mode (easier debugging)
        for task in task_args:
            result = process_single_source(task)
            all_results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_single_source, task): task[1]
                for task in task_args
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result(timeout=300)   # 5 min per source
                    all_results.append(result)
                except Exception as exc:
                    logger.error(f"Worker for source {idx} failed: {exc}")
                    all_results.append(None)

    elapsed = time.time() - t_start
    n_processed = sum(1 for r in all_results if r is not None)
    logger.info(
        f"\n{'='*60}\n"
        f"Processing complete: {n_processed}/{total} sources succeeded "
        f"in {elapsed:.0f}s\n"
        f"{'='*60}"
    )

    if n_processed == 0:
        logger.error("No sources successfully processed. Exiting.")
        sys.exit(1)

    # ---- Sort by score and apply false positive rejection -------------------
    valid_results = [r for r in all_results if r is not None]
    valid_results.sort(key=lambda r: r.get('score', {}).get('composite', 0.0),
                        reverse=True)

    # Build lc_dict and wise_dict_map for FP rejection
    lc_dict      = {r['source_row']['source_id']: r['wise'].get('lc')
                    for r in valid_results}
    wise_dict_map = {r['source_row']['source_id']: r['wise']
                     for r in valid_results}

    # Apply false positive rejection (modifies composite scores in-place)
    # Build a temporary DataFrame for FP rejection then apply back
    import pandas as pd
    fp_rows = []
    for r in valid_results:
        src = r['source_row']
        cp  = r.get('cp', {})
        wise = r.get('wise', {})
        gaia = r.get('gaia', {})
        score = r.get('score', {})
        fp_rows.append({
            'source_id': src['source_id'],
            'ra': src.get('ra'), 'dec': src.get('dec'),
            'composite_score': score.get('composite', 0.0),
            'w1_delta_mag': score.get('delta_mag_w1', 0.0),
            'changepoint_break_duration_days': cp.get('break_duration_days', np.nan),
            'changepoint_pre_break_mean': cp.get('pre_break_mean', np.nan),
            'changepoint_post_break_mean': cp.get('post_break_mean', np.nan),
            'gaia_quasar_prob': gaia.get('classprob_quasar', np.nan),
            'host_contamination_risk': wise.get('host_contamination_risk', False),
        })

    fp_df = pd.DataFrame(fp_rows)
    fp_df_updated = apply_false_positive_rejection(fp_df, lc_dict, wise_dict_map)

    # Write back updated scores and contamination flags
    score_map = dict(zip(fp_df_updated['source_id'],
                          fp_df_updated['composite_score']))
    flag_map  = dict(zip(fp_df_updated['source_id'],
                          fp_df_updated['contamination_flag']))

    for r in valid_results:
        sid = r['source_row']['source_id']
        if sid in score_map:
            r['score']['composite'] = score_map[sid]
        r['contamination_flag'] = flag_map.get(sid, 'none')

    # Re-sort after FP penalty application
    valid_results.sort(key=lambda r: r.get('score', {}).get('composite', 0.0),
                        reverse=True)

    # ---- Apply final sanity checks ------------------------------------------
    final_top = apply_final_sanity_checks(
        valid_results[:args.top_n * 2], valid_results, args.top_n, logger
    )

    if not final_top:
        logger.warning("No sources passed final sanity checks.")
        # Fall back to best scored sources even if they fail some criteria
        final_top = valid_results[:args.top_n]

    logger.info(f"\nTop {len(final_top)} CLAGN candidates:")
    logger.info("-" * 60)
    for rank, r in enumerate(final_top, 1):
        composite = r['score'].get('composite', 0.0)
        label     = r['score'].get('label', '')
        source_id = r['source_row']['source_id']
        z         = r['source_row']['redshift']
        logger.info(f"#{rank:2d} | {source_id} | z={z:.3f} | Score={composite:.3f} | {label[:60]}")

    # ---- Run MCMC on top candidates (optional) ------------------------------
    if args.mcmc:
        logger.info("\nRunning MCMC on top candidates...")
        for r in final_top:
            src   = r['source_row']
            lc    = r['wise']['lc']
            z_src = float(src['redshift'])
            sid   = src['source_id']
            if lc is None or len(lc) < 20:
                continue
            try:
                logger.info(f"  MCMC: {sid}")
                mcmc_result = fit_drw_mcmc(
                    lc['mjd'].values,
                    lc['w1_flux_mjy'].values,
                    lc['w1_flux_err_mjy'].values,
                    z=z_src,
                )
                r['drw_mcmc'] = mcmc_result
                # Update DRW tau to MCMC median (more accurate)
                r['drw']['tau_rest_days'] = mcmc_result['tau_rest_days']
                logger.info(
                    f"  MCMC done: τ={mcmc_result['tau_rest_days']:.1f} d "
                    f"[{mcmc_result['tau_lo']:.1f}–{mcmc_result['tau_hi']:.1f}], "
                    f"acceptance={mcmc_result['acceptance_fraction']:.2f}"
                )
                # Re-save checkpoint with MCMC results
                save_checkpoint(sid, r, checkpoint_dir)
            except Exception as exc:
                logger.error(f"  MCMC failed for {sid}: {exc}")

    # ---- Generate plots -----------------------------------------------------
    logger.info("\nGenerating plots...")
    for rank, r in enumerate(final_top, 1):
        src     = r['source_row']
        lc      = r['wise']['lc']
        drw     = r['drw']
        sf      = r['sf']
        cp      = r['cp']
        score   = r['score']
        gaia    = r['gaia']

        plot_path = os.path.join(
            args.output, 'plots',
            f"rank{rank:02d}_{src['source_id']}.png"
        )
        try:
            plot_lightcurve_panel(
                source_id=src['source_id'],
                lc_df=lc,
                drw_pred_mean=drw.get('pred_mean'),
                drw_pred_std=drw.get('pred_std'),
                drw_residuals=drw.get('residuals'),
                sf_result=sf,
                break_mjd=cp.get('best_break_mjd'),
                z=float(src['redshift']),
                score=float(score.get('composite', 0.0)),
                label=str(score.get('label', '')),
                output_path=plot_path,
            )
        except Exception as exc:
            logger.error(f"Plot failed for {src['source_id']}: {exc}")

    # Summary grid
    try:
        grid_path = os.path.join(args.output, 'summary_grid.png')
        plot_summary_grid(final_top, grid_path)
    except Exception as exc:
        logger.error(f"Summary grid failed: {exc}")

    # ---- Write output table -------------------------------------------------
    logger.info("\nWriting output tables...")
    candidates_df = build_candidates_dataframe(final_top, top_n=args.top_n)

    csv_path  = os.path.join(args.output, 'top_candidates.csv')
    fits_path = os.path.join(args.output, 'top_candidates.fits')
    write_output_csv(candidates_df, csv_path)
    try:
        write_output_fits(candidates_df, fits_path)
    except Exception as exc:
        logger.warning(f"FITS output failed: {exc}")

    # FLAW C1: Write provenance sidecar for reproducibility
    try:
        pipeline_config = vars(args)
        sidecar_path = write_provenance_sidecar(csv_path, pipeline_config)
        logger.info(f"Provenance sidecar: {sidecar_path}")
    except Exception as exc:
        logger.warning(f"Provenance sidecar failed: {exc}")

    # ---- Cross-validation ---------------------------------------------------
    logger.info("\nCross-validating against known CLAGN...")
    try:
        val_report = validate_against_known_clagn(
            valid_results, search_radius_arcsec=10.0
        )
        logger.info(
            f"Recovery: {val_report['n_recovered']}/{val_report['n_known_in_catalog']} "
            f"known CLAGN recovered "
            f"(rate={100*val_report['recovery_rate']:.0f}%)"
        )
        for detail in val_report['recovery_details']:
            logger.info(
                f"  {detail['known_name']}: rank #{detail['pipeline_rank']}, "
                f"score={detail['composite_score']:.3f}"
            )
    except Exception as exc:
        logger.warning(f"Cross-validation failed: {exc}")

    # ---- Final summary ------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Output CSV:  {csv_path}")
    logger.info(f"  Plots:       {os.path.join(args.output, 'plots/')}")
    logger.info(f"  Log:         {os.path.join(args.output, 'pipeline.log')}")
    logger.info("=" * 70)

    return final_top, candidates_df


if __name__ == '__main__':
    main()
