from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import ensure_parent_dir


NA_VALUES = ['', ' ', 'NA', 'NaN', 'null', 'None']


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_results_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Missing results CSV: {path}')
    df = pd.read_csv(
        path,
        engine='python',
        dtype=str,
        keep_default_na=True,
        na_values=NA_VALUES,
    )
    # Normalize column names and strip string content.
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
    return df


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_parent_dir(path)
    df.to_csv(path, index=False)


def write_json(payload: dict[str, Any], path: Path) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
