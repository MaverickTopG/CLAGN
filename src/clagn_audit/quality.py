from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .constants import OPTIONAL_COLUMNS
from .markdown import df_to_markdown_table
from .utils import ensure_parent_dir, format_pct



def _count_series(series: pd.Series, colname: str = 'count') -> pd.DataFrame:
    counts = series.astype('string').fillna('<NA>').value_counts(dropna=False).rename(colname).reset_index()
    counts = counts.rename(columns={'index': series.name or 'value'})
    if counts.columns[0] != 'value':
        counts = counts.rename(columns={counts.columns[0]: 'value'})
    return counts



def build_quality_tables(df: pd.DataFrame, correlations: dict[str, dict[str, Any]] | None = None,
                         duplicate_canonical_ids: list[str] | None = None) -> dict[str, Any]:
    duplicate_canonical_ids = duplicate_canonical_ids or []
    status_counts = _count_series(df['status_normalized'])
    band_counts = _count_series(df['band_coverage'])
    rejection_counts = _count_series(df['rejection_bucket'])
    rejection_counts = rejection_counts.rename(columns={'value': 'rejection_bucket'})
    id_mismatch_counts = _count_series(pd.Series(df['id_mismatch_reason'], name='id_mismatch_reason').fillna('<none>'))
    id_mismatch_counts = id_mismatch_counts.rename(columns={'value': 'id_mismatch_reason'})

    total = int(len(df))
    ok_count = int(df['is_ok'].sum())
    strict_count = int(df['strict_quality'].sum())
    missing_score = int(df['score_is_missing'].sum())
    id_mismatch_count = int((~df['source_id_matches_canonical'].astype(bool)).sum())
    raw_id_difference_count = int(
        (df['source_id'].astype('string').fillna('').str.strip() !=
         df['canonical_id'].astype('string').fillna('').str.strip()).sum()
    )
    ambiguous_id_count = int(df['is_ambiguous_id'].astype(bool).sum()) if 'is_ambiguous_id' in df.columns else 0
    baseline_diff = pd.to_numeric(df.get('baseline_years_diff', pd.Series(dtype=float)), errors='coerce')
    baseline_diff_finite = int(baseline_diff.notna().sum())
    baseline_diff_bad = int((baseline_diff > 0.01).sum()) if baseline_diff_finite else 0
    baseline_diff_max = float(baseline_diff.max()) if baseline_diff_finite else None
    baseline_given_count = int(pd.to_numeric(df.get('baseline_years', pd.Series(dtype=float)), errors='coerce').notna().sum()) if 'baseline_years' in df.columns else 0

    missing_optional = [c for c in OPTIONAL_COLUMNS if c not in df.columns]
    warnings: list[str] = []
    if duplicate_canonical_ids:
        warnings.append(f'Duplicate canonical_id values detected ({len(duplicate_canonical_ids)} unique duplicates).')
    if missing_optional:
        warnings.append('Missing optional columns: ' + ', '.join(missing_optional))
    if int(df['score_is_finite'].sum()) == 0:
        warnings.append('All score values are missing/NaN.')
    if 'delta_mag' not in df.columns or int(pd.to_numeric(df.get('delta_mag', pd.Series(dtype=float)), errors='coerce').notna().sum()) == 0:
        warnings.append('delta_mag unavailable or all NaN.')

    summary = {
        'total_rows': total,
        'ok_rows': ok_count,
        'rejected_rows': total - ok_count,
        'coverage_rate': (ok_count / total) if total else 0.0,
        'strict_quality_rows': strict_count,
        'strict_coverage_rate': (strict_count / total) if total else 0.0,
        'duplicate_canonical_id_count': len(duplicate_canonical_ids),
        'missing_score_count': missing_score,
        'source_id_mismatch_count': id_mismatch_count,
        'source_id_raw_difference_count': raw_id_difference_count,
        'ambiguous_id_count': ambiguous_id_count,
        'baseline_years_provided_count': baseline_given_count,
        'baseline_years_diff_finite_count': baseline_diff_finite,
        'baseline_diff_gt_0p01_count': baseline_diff_bad,
        'baseline_diff_max': baseline_diff_max,
    }
    return {
        'summary': summary,
        'status_counts': status_counts,
        'band_coverage_counts': band_counts,
        'rejection_bucket_counts': rejection_counts,
        'id_mismatch_counts': id_mismatch_counts,
        'correlations': correlations or {},
        'warnings': warnings,
        'duplicate_canonical_ids': duplicate_canonical_ids,
    }



def write_quality_report_markdown(df: pd.DataFrame, stats_bundle: dict[str, Any], out_path: Path) -> None:
    ensure_parent_dir(out_path)
    s = stats_bundle['summary']
    status_counts = stats_bundle['status_counts']
    band_counts = stats_bundle['band_coverage_counts']
    rejection_counts = stats_bundle['rejection_bucket_counts']
    id_mismatch_counts = stats_bundle['id_mismatch_counts']
    correlations = stats_bundle.get('correlations', {})
    warnings = stats_bundle.get('warnings', [])

    lines: list[str] = []
    lines.append('# Quality Report')
    lines.append('')
    lines.append('## Dataset Summary')
    lines.append('')
    lines.append(f"- Total rows: {s['total_rows']}")
    lines.append(f"- Duplicate `canonical_id` count: {s['duplicate_canonical_id_count']}")
    lines.append(f"- Missing score count: {s['missing_score_count']}")
    lines.append('')
    lines.append('## Counts by Status')
    lines.append('')
    lines.append(df_to_markdown_table(status_counts))
    lines.append('')
    lines.append('## Counts by Band Coverage')
    lines.append('')
    lines.append(df_to_markdown_table(band_counts))
    lines.append('')
    lines.append('## ID Integrity')
    lines.append('')
    lines.append(f"- Rows where `source_id_normalized != canonical_id`: {s['source_id_mismatch_count']}")
    lines.append(f"- Rows where raw `source_id != canonical_id`: {s['source_id_raw_difference_count']}")
    lines.append(f"- Ambiguous ID count (e.g., PKS/PG/MR/HS/B2): {s['ambiguous_id_count']}")
    lines.append('')
    lines.append('### ID Mismatch Breakdown')
    lines.append('')
    lines.append(df_to_markdown_table(id_mismatch_counts))
    lines.append('')
    mismatch_preview = df.loc[pd.Series(df['id_mismatch_reason']).notna(), [
        c for c in ['source_id', 'source_id_normalized', 'canonical_id', 'id_mismatch_reason'] if c in df.columns
    ]].copy()
    lines.append('### Mismatch Preview (first 20)')
    lines.append('')
    lines.append(df_to_markdown_table(mismatch_preview.head(20)))
    lines.append('')
    lines.append('## Baseline Consistency Check')
    lines.append('')
    lines.append(f"- Rows with provided `baseline_years`: {s['baseline_years_provided_count']}")
    lines.append(f"- Rows with finite baseline diff: {s['baseline_years_diff_finite_count']}")
    lines.append(f"- Rows with `abs(baseline_years - baseline_years_recomputed) > 0.01`: {s['baseline_diff_gt_0p01_count']}")
    lines.append(f"- Max baseline diff: {s['baseline_diff_max'] if s['baseline_diff_max'] is not None else 'NA'}")
    lines.append('- Downstream analysis uses `baseline_years_recomputed` as the trusted value.')
    lines.append('')
    lines.append('## Coverage Metrics')
    lines.append('')
    lines.append(f"- coverage_rate = OK / total = {format_pct(s['coverage_rate'])}")
    lines.append(f"- strict_coverage_rate = strict_quality / total = {format_pct(s['strict_coverage_rate'])}")
    lines.append('')
    lines.append('## Rejection Bucket Histogram')
    lines.append('')
    lines.append(df_to_markdown_table(rejection_counts))
    lines.append('')
    lines.append('## Correlation Summary')
    lines.append('')
    if correlations:
        corr_rows = []
        for key, val in correlations.items():
            if val.get('available'):
                corr_rows.append({
                    'pair': key,
                    'available': True,
                    'rho_spearman': f"{val['rho']:.4f}",
                    'pvalue': f"{val['pvalue']:.3e}",
                    'n': int(val['n']),
                    'note': '',
                })
            else:
                corr_rows.append({
                    'pair': key,
                    'available': False,
                    'rho_spearman': '',
                    'pvalue': '',
                    'n': int(val.get('n', 0)),
                    'note': val.get('reason', 'unavailable'),
                })
        lines.append(df_to_markdown_table(pd.DataFrame(corr_rows)))
    else:
        lines.append('(no correlations computed)')
    lines.append('')
    lines.append('## Warnings')
    lines.append('')
    if warnings:
        for w in warnings:
            lines.append(f'- {w}')
    else:
        lines.append('- None')
    lines.append('')
    out_path.write_text('\n'.join(lines))



def write_quality_report_csv(stats_bundle: dict[str, Any], out_path: Path) -> None:
    ensure_parent_dir(out_path)
    rows: list[dict[str, Any]] = []
    s = stats_bundle['summary']
    for k, v in s.items():
        rows.append({'section': 'summary', 'metric': k, 'value': v, 'note': ''})
    for _, r in stats_bundle['status_counts'].iterrows():
        rows.append({'section': 'status_count', 'metric': str(r['value']), 'value': int(r['count']), 'note': ''})
    for _, r in stats_bundle['band_coverage_counts'].iterrows():
        rows.append({'section': 'band_coverage_count', 'metric': str(r['value']), 'value': int(r['count']), 'note': ''})
    for _, r in stats_bundle['rejection_bucket_counts'].iterrows():
        rows.append({'section': 'rejection_bucket_count', 'metric': str(r['rejection_bucket']), 'value': int(r['count']), 'note': ''})
    for _, r in stats_bundle['id_mismatch_counts'].iterrows():
        rows.append({'section': 'id_mismatch_count', 'metric': str(r['id_mismatch_reason']), 'value': int(r['count']), 'note': ''})
    rows.append({'section': 'id_integrity', 'metric': 'source_id_mismatch_count', 'value': s.get('source_id_mismatch_count', ''), 'note': ''})
    rows.append({'section': 'id_integrity', 'metric': 'source_id_raw_difference_count', 'value': s.get('source_id_raw_difference_count', ''), 'note': ''})
    rows.append({'section': 'id_integrity', 'metric': 'ambiguous_id_count', 'value': s.get('ambiguous_id_count', ''), 'note': ''})
    rows.append({'section': 'baseline_check', 'metric': 'baseline_diff_gt_0p01_count', 'value': s.get('baseline_diff_gt_0p01_count', ''), 'note': ''})
    rows.append({'section': 'baseline_check', 'metric': 'baseline_diff_max', 'value': s.get('baseline_diff_max', ''), 'note': ''})
    for key, val in stats_bundle.get('correlations', {}).items():
        if val.get('available'):
            rows.append({'section': 'correlation', 'metric': key, 'value': val['rho'], 'note': f"p={val['pvalue']:.3e}; n={val['n']}"})
        else:
            rows.append({'section': 'correlation', 'metric': key, 'value': '', 'note': val.get('reason', 'unavailable')})
    for w in stats_bundle.get('warnings', []):
        rows.append({'section': 'warning', 'metric': 'warning', 'value': '', 'note': w})
    pd.DataFrame(rows, columns=['section', 'metric', 'value', 'note']).to_csv(out_path, index=False)
