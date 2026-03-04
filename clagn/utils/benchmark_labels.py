from __future__ import annotations

import pandas as pd


POSITIVE_CLASS_NAMES = {"clagn", "positive"}


def class_column_name(df: pd.DataFrame) -> str:
    if "class" in df.columns:
        return "class"
    if "label" in df.columns:
        return "label"
    raise KeyError("Benchmark dataframe missing class/label column")


def normalized_class_series(df: pd.DataFrame) -> pd.Series:
    col = class_column_name(df)
    s = df[col].astype(str).str.strip()
    return s.replace({"positive": "clagn"})


def ensure_class_ytrue(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    s = normalized_class_series(out)
    if "label" in out.columns and "class" not in out.columns:
        out = out.rename(columns={"label": "class"})
    out["class"] = s
    out["y_true"] = out["class"].astype(str).str.lower().isin(POSITIVE_CLASS_NAMES).astype(int)
    return out


def ensure_y_true(df: pd.DataFrame) -> pd.Series:
    if "y_true" in df.columns:
        return pd.to_numeric(df["y_true"], errors="coerce").fillna(0).astype(int)
    return normalized_class_series(df).astype(str).str.lower().isin(POSITIVE_CLASS_NAMES).astype(int)
