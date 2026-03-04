from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy

from clagn_audit.constants import (
    DEFAULT_BIAS_BINS,
    DEFAULT_HIST_BINS,
    RUNBOOK_TEMPLATE,
    SCORE_DEFINITION_TEMPLATE,
    SORT_COLUMNS_DEFAULT,
)
from clagn_audit.io import read_results_csv, write_csv, write_json, safe_mkdir
from clagn_audit.schema import validate_required_columns, coerce_types, warn_duplicate_canonical_ids
from clagn_audit.cleaning import add_derived_columns
from clagn_audit.quality import build_quality_tables, write_quality_report_markdown, write_quality_report_csv
from clagn_audit.stats import compute_internal_correlations
from clagn_audit.plotting import (
    plot_score_hist,
    plot_score_vs_baseline,
    plot_score_vs_npoints,
    plot_score_vs_deltamag,
)
from clagn_audit.mining import build_case_mining_tables, write_case_mining_md, write_top_candidates_csv
from clagn_audit.utils import stable_sort_df

from analysis.bias_audit import run_bias_audit



def _write_docs_templates(root: Path) -> None:
    docs_dir = root / 'docs'
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / 'score_definition.md').write_text(SCORE_DEFINITION_TEMPLATE)
    (docs_dir / 'TOP_TIER_ANALYSIS_RUNBOOK.md').write_text(RUNBOOK_TEMPLATE)



def _build_manifest(args: argparse.Namespace, outdir: Path, df: pd.DataFrame, duplicate_ids: list[str],
                    status: str, notes: list[str]) -> dict[str, Any]:
    return {
        'command': ' '.join(sys.argv),
        'input_path': str(Path(args.input)),
        'output_path': str(outdir),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'row_count': int(len(df)),
        'duplicate_canonical_id_count': int(len(duplicate_ids)),
        'status': status,
        'notes': notes,
        'versions': {
            'python': sys.version.split()[0],
            'pandas': pd.__version__,
            'numpy': np.__version__,
            'matplotlib': mpl.__version__,
            'scipy': scipy.__version__,
        },
        'deterministic_settings': {
            'sort_key': list(SORT_COLUMNS_DEFAULT),
            'hist_bins': int(args.n_bins_hist),
            'bias_qcut_bins': int(args.n_bins_bias),
            'matplotlib_backend': 'Agg',
        },
    }



def _write_console_summary(outdir: Path, summary: dict[str, Any]) -> str:
    text = (
        f"rows={summary['rows']} | ok={summary['ok_rows']} | rejected={summary['rejected_rows']} | "
        f"coverage={summary['coverage_rate']:.3f} | strict_coverage={summary['strict_coverage_rate']:.3f} | "
        f"duplicate_canonical_ids={summary['duplicate_ids']} | id_mismatches={summary['id_mismatches']} | "
        f"ambiguous_ids={summary['ambiguous_ids']} | baseline_bad={summary['baseline_bad']} | "
        f"missing_score={summary['missing_score_count']} | "
        f"top_candidate={summary['top_candidate']} | outdir={outdir}"
    )
    (outdir / 'console_summary.txt').write_text(text + '\n')
    return text



def main() -> None:
    parser = argparse.ArgumentParser(description='Deterministic offline audit for CLAGN result tables')
    parser.add_argument('--input', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--overwrite', action='store_true', help='Accepted for interface compatibility; outputs are overwritten deterministically')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--n_bins_bias', type=int, default=DEFAULT_BIAS_BINS)
    parser.add_argument('--n_bins_hist', type=int, default=DEFAULT_HIST_BINS)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    safe_mkdir(outdir)
    notes: list[str] = []

    try:
        df = read_results_csv(Path(args.input))
        validate_required_columns(df)
        df = coerce_types(df)
        if int((df['status'].astype('string') == 'UNKNOWN').sum()) == len(df):
            raise ValueError('All rows have missing/blank status (normalized to UNKNOWN).')
        duplicate_ids = warn_duplicate_canonical_ids(df)
        if duplicate_ids:
            notes.append(f'duplicate canonical_id values: {len(duplicate_ids)}')

        df = stable_sort_df(df, SORT_COLUMNS_DEFAULT)
        df = add_derived_columns(df)
        write_csv(df, outdir / 'cleaned_results.csv')

        _write_docs_templates(ROOT)

        correlations = compute_internal_correlations(df)
        quality_bundle = build_quality_tables(df, correlations=correlations, duplicate_canonical_ids=duplicate_ids)
        write_quality_report_markdown(df, quality_bundle, outdir / 'quality_report.md')
        write_quality_report_csv(quality_bundle, outdir / 'quality_report.csv')

        # Always generate requested figures; placeholder text used when data unavailable.
        plot_score_hist(df, outdir / 'fig_score_hist.png', bins=args.n_bins_hist)
        plot_score_vs_baseline(df, outdir / 'fig_score_vs_baseline.png')
        plot_score_vs_npoints(df, outdir / 'fig_score_vs_npoints.png')
        plot_score_vs_deltamag(df, outdir / 'fig_score_vs_deltamag.png')

        finite_scores = int(df['score_is_finite'].sum())
        if finite_scores == 0:
            notes.append('All score values missing/NaN; skipping bias audit and case mining.')
            manifest = _build_manifest(args, outdir, df, duplicate_ids, status='incomplete_no_scores', notes=notes)
            write_json(manifest, outdir / 'run_manifest.json')
            summary_line = _write_console_summary(outdir, {
                'rows': int(len(df)),
                'ok_rows': int(df['is_ok'].sum()),
                'rejected_rows': int((~df['is_ok']).sum()),
                'coverage_rate': float(df['is_ok'].mean()),
                'strict_coverage_rate': float(df['strict_quality'].mean()),
                'duplicate_ids': len(duplicate_ids),
                'id_mismatches': int(pd.Series(df['id_mismatch_reason']).notna().sum()),
                'ambiguous_ids': int(df['is_ambiguous_id'].astype(bool).sum()),
                'baseline_bad': int((pd.to_numeric(df['baseline_years_diff'], errors='coerce') > 0.01).sum()),
                'missing_score_count': int(df['score_is_missing'].sum()),
                'top_candidate': 'NA',
            })
            if not args.quiet:
                print(summary_line)
            raise SystemExit('All rows missing/NaN score. Quality report generated; downstream analyses skipped.')

        run_bias_audit(df, outdir, n_bins=args.n_bins_bias)

        mining_tables = build_case_mining_tables(df, top_n=30)
        write_top_candidates_csv(mining_tables, outdir / 'top_candidates.csv')
        write_case_mining_md(mining_tables, outdir / 'case_mining.md', top_n=30)

        top_ok = mining_tables.get('top_ok', pd.DataFrame())
        if not top_ok.empty:
            top_id = str(top_ok.iloc[0].get('canonical_id') or top_ok.iloc[0].get('source_id'))
            top_score = pd.to_numeric(pd.Series([top_ok.iloc[0].get('score')]), errors='coerce').iloc[0]
            top_candidate = f"{top_id} ({top_score:.4f})" if pd.notna(top_score) else top_id
        else:
            top_candidate = 'NA'

        manifest = _build_manifest(args, outdir, df, duplicate_ids, status='ok', notes=notes)
        write_json(manifest, outdir / 'run_manifest.json')

        summary_line = _write_console_summary(outdir, {
            'rows': int(len(df)),
            'ok_rows': int(df['is_ok'].sum()),
            'rejected_rows': int((~df['is_ok']).sum()),
            'coverage_rate': float(df['is_ok'].mean()),
            'strict_coverage_rate': float(df['strict_quality'].mean()),
            'duplicate_ids': len(duplicate_ids),
            'id_mismatches': int(pd.Series(df['id_mismatch_reason']).notna().sum()),
            'ambiguous_ids': int(df['is_ambiguous_id'].astype(bool).sum()),
            'baseline_bad': int((pd.to_numeric(df['baseline_years_diff'], errors='coerce') > 0.01).sum()),
            'missing_score_count': int(df['score_is_missing'].sum()),
            'top_candidate': top_candidate,
        })
        if not args.quiet:
            print(summary_line)
    except SystemExit:
        raise
    except Exception as exc:
        err = {
            'command': ' '.join(sys.argv),
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'error',
            'error_type': type(exc).__name__,
            'error': str(exc),
        }
        write_json(err, outdir / 'run_manifest.json')
        raise


if __name__ == '__main__':
    main()
