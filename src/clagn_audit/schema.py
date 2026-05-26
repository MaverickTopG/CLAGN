from __future__ import annotations

from typing import Iterable

import pandas as pd

from .constants import OPTIONAL_COLUMNS, REQUIRED_COLUMNS


class SchemaError(ValueError):
    """Raised when the input results table does not satisfy required schema."""



def validate_required_columns(df: pd.DataFrame, required: Iterable[str] = REQUIRED_COLUMNS) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {', '.join(missing)}")
    if len(df) == 0:
        raise SchemaError('Input CSV has zero rows')



def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ['source_id', 'canonical_id', 'status', 'band_coverage', 'delta_mag_method', 'rejection_reason']:
        if col in out.columns:
            out[col] = out[col].astype('string')

    for col in ['score', 'baseline_days', 'baseline_years', 'delta_mag']:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce')

    if 'n_points' in out.columns:
        out['n_points'] = pd.to_numeric(out['n_points'], errors='coerce').round().astype('Int64')

    if 'status' in out.columns:
        s = out['status'].astype('string').fillna('')
        s = s.str.strip().str.upper()
        s = s.replace({'': 'UNKNOWN'})
        out['status'] = s

    if 'band_coverage' in out.columns:
        out['band_coverage'] = out['band_coverage'].astype('string').fillna('none').str.strip().replace({'': 'none'})

    for col in OPTIONAL_COLUMNS:
        if col not in out.columns:
            # Keep absent optional columns absent; caller may warn/report.
            continue

    return out



def warn_duplicate_canonical_ids(df: pd.DataFrame) -> list[str]:
    if 'canonical_id' not in df.columns:
        return []
    s = df['canonical_id'].astype('string').fillna('')
    dupes = sorted(set(s[s.duplicated(keep=False)].tolist()))
    return [d for d in dupes if d]
