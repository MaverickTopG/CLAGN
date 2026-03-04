from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def stable_sort_df(df: pd.DataFrame, by: Iterable[str]) -> pd.DataFrame:
    cols = [c for c in by if c in df.columns]
    if not cols:
        return df.copy()
    return df.sort_values(cols, kind='mergesort').reset_index(drop=True)


def format_pct(value: float | int | None, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return 'NA'
    return f"{100.0 * float(value):.{digits}f}%"


def finite_mask(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors='coerce')
    return pd.Series(np.isfinite(vals.to_numpy(float)), index=series.index)


def maybe_float(x: object) -> float | None:
    try:
        val = float(x)
    except Exception:
        return None
    if not np.isfinite(val):
        return None
    return val
