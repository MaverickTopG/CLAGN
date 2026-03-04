from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .constants import CASE_MINING_COLUMNS
from .markdown import df_to_markdown_table
from .utils import ensure_parent_dir



def _finite_score_ok(df: pd.DataFrame) -> pd.Series:
    return df['score_is_finite'].astype(bool) & df['is_ok'].astype(bool)



def build_case_mining_tables(df: pd.DataFrame, top_n: int = 30) -> dict[str, pd.DataFrame]:
    base = df.copy()
    if 'score' in base.columns:
        base['score'] = pd.to_numeric(base['score'], errors='coerce')

    ok_finite = base[_finite_score_ok(base)].copy()
    top_ok = ok_finite.sort_values(['score', 'canonical_id', 'source_id'], ascending=[False, True, True], kind='mergesort').head(top_n)
    bottom_ok = ok_finite.sort_values(['score', 'canonical_id', 'source_id'], ascending=[True, True, True], kind='mergesort').head(top_n)

    rejected_with_score = base[(~base['is_ok'].astype(bool)) & base['score_is_finite'].astype(bool)].copy()
    rejected_with_score = rejected_with_score.sort_values(['score', 'canonical_id', 'source_id'], ascending=[False, True, True], kind='mergesort').head(top_n)

    return {
        'top_ok': top_ok,
        'bottom_ok': bottom_ok,
        'rejected_with_score': rejected_with_score,
    }



def write_top_candidates_csv(tables: dict[str, pd.DataFrame], out_path: Path) -> None:
    ensure_parent_dir(out_path)
    chunks = []
    for subset in ['top_ok', 'bottom_ok', 'rejected_with_score']:
        df = tables.get(subset, pd.DataFrame()).copy()
        if df.empty:
            continue
        df['subset'] = subset
        cols = ['subset'] + [c for c in CASE_MINING_COLUMNS if c in df.columns]
        chunks.append(df[cols])
    if chunks:
        out = pd.concat(chunks, ignore_index=True)
        out = out.sort_values(['subset', 'canonical_id', 'source_id'], kind='mergesort').reset_index(drop=True)
    else:
        out = pd.DataFrame(columns=['subset'] + list(CASE_MINING_COLUMNS))
    out.to_csv(out_path, index=False)



def write_case_mining_md(tables: dict[str, pd.DataFrame], out_path: Path, top_n: int = 30) -> None:
    ensure_parent_dir(out_path)
    lines: list[str] = []
    lines.append('# Case Mining')
    lines.append('')
    sections = [
        ('Top 30 OK by score', 'top_ok'),
        ('Bottom 30 OK by score', 'bottom_ok'),
        ('Rejected rows with non-null score (top by score)', 'rejected_with_score'),
    ]
    for title, key in sections:
        lines.append(f'## {title}')
        lines.append('')
        df = tables.get(key, pd.DataFrame()).copy()
        if not df.empty:
            preview_cols = [c for c in ['source_id', 'canonical_id', 'score', 'status', 'n_points', 'baseline_years', 'band_coverage', 'delta_mag', 'rejection_reason'] if c in df.columns]
            lines.append(df_to_markdown_table(df[preview_cols].head(top_n)))
        else:
            lines.append('(none)')
        lines.append('')

    lines.append('## Next Manual Checks Checklist')
    lines.append('')
    checklist = [
        'crossmatch blazars / radio catalogs',
        'inspect WISE light curves (seasonal medians + raw points)',
        'check image blending / host contamination',
        'inspect W1/W2 color evolution',
        'verify baseline and pre/post-hibernation coverage',
        'check external transient catalogs / SNe contamination',
        'look for Gaia stellar contamination flags',
        'prioritize spectroscopy availability',
    ]
    for item in checklist:
        lines.append(f'- [ ] {item}')
    lines.append('')
    out_path.write_text('\n'.join(lines))
