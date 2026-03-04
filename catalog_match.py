from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_AMBIG_ALIAS_SPLIT = re.compile(r"[;,|]")


def normalize_identifier(value: str | object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return ""
    if s.isdigit():
        return s
    s = re.sub(r"^SDSSJ{2,}", "SDSSJ", s, flags=re.IGNORECASE)
    s = re.sub(r"^NGC\s*(\d+)$", r"NGC \1", s, flags=re.IGNORECASE)
    s = re.sub(r"^MRK\s*(\d+)$", r"Mrk \1", s, flags=re.IGNORECASE)
    s = re.sub(r"^IC\s*(\d+)$", r"IC \1", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s)
    return s


def _norm_upper(value: str | object) -> str:
    return normalize_identifier(value).upper()


def _read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return pd.read_csv(
        p,
        dtype=str,
        keep_default_na=True,
        na_values=["", " ", "NA", "NaN", "null", "None"],
    )


def load_known_blazars(path: str) -> pd.DataFrame:
    df = _read_csv(path)
    if df.empty:
        raise ValueError(f"Known blazar catalog is empty: {path}")
    df = df.copy()
    id_cols = [c for c in df.columns if c.lower() in {"canonical_id", "source_id", "name", "alias", "aliases"}]
    if not id_cols:
        raise ValueError(
            "known_blazars.csv must contain at least one identifier column among: "
            "canonical_id, source_id, name, alias, aliases"
        )
    token_rows: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        seen = set()
        for col in id_cols:
            val = row.get(col)
            if pd.isna(val):
                continue
            tokens = [str(val)]
            if col.lower() == "aliases":
                tokens = [t for t in _AMBIG_ALIAS_SPLIT.split(str(val)) if t.strip()]
            for tok in tokens:
                norm = _norm_upper(tok)
                if not norm or norm in seen:
                    continue
                seen.add(norm)
                token_rows.append({
                    "catalog_index": int(idx),
                    "match_token": norm,
                    "match_field": col,
                })
    token_df = pd.DataFrame(token_rows)
    if token_df.empty:
        raise ValueError("known_blazars.csv contained no usable identifier tokens")
    return df, token_df


def match_blazar(candidate_row: pd.Series, blazar_df: tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame) -> dict:
    raw_df: pd.DataFrame
    token_df: pd.DataFrame
    if isinstance(blazar_df, tuple):
        raw_df, token_df = blazar_df
    else:
        raise ValueError("match_blazar expects the tuple returned by load_known_blazars")

    candidate_tokens = []
    for field in ["canonical_id", "source_id"]:
        v = candidate_row.get(field)
        norm = _norm_upper(v)
        if norm:
            candidate_tokens.append((field, norm))
    for field, norm in candidate_tokens:
        hit = token_df[token_df["match_token"] == norm]
        if not hit.empty:
            first = hit.iloc[0]
            catalog_row = raw_df.iloc[int(first["catalog_index"])]
            return {
                "matched": True,
                "match_field_candidate": field,
                "match_field_catalog": str(first["match_field"]),
                "matched_token": norm,
                "catalog_row": catalog_row.to_dict(),
            }
    return {"matched": False}


def load_known_clagn(path: str) -> pd.DataFrame:
    df = _read_csv(path)
    if df.empty:
        raise ValueError(f"Known CLAGN catalog is empty: {path}")
    return df


def _extract_known_clagn_tokens(df: pd.DataFrame) -> pd.DataFrame:
    name_cols = [c for c in df.columns if c.lower() in {"canonical_id", "source_id", "name", "alias", "aliases"}]
    rows = []
    for idx, row in df.iterrows():
        for col in name_cols:
            val = row.get(col)
            if pd.isna(val):
                continue
            toks = [str(val)] if col.lower() != "aliases" else [t for t in _AMBIG_ALIAS_SPLIT.split(str(val)) if t.strip()]
            for t in toks:
                norm = _norm_upper(t)
                if norm:
                    rows.append({"catalog_index": int(idx), "match_token": norm, "match_field": col})
    return pd.DataFrame(rows)


def _ang_sep_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    ra1r, dec1r, ra2r, dec2r = map(math.radians, [ra1, dec1, ra2, dec2])
    cosd = math.sin(dec1r) * math.sin(dec2r) + math.cos(dec1r) * math.cos(dec2r) * math.cos(ra1r - ra2r)
    cosd = max(-1.0, min(1.0, cosd))
    return math.degrees(math.acos(cosd)) * 3600.0


def match_known_clagn(candidate_row: pd.Series, known_clagn_df: pd.DataFrame, tol_arcsec: float = 2.0) -> dict:
    tokens_df = _extract_known_clagn_tokens(known_clagn_df)
    for field in ["canonical_id", "source_id"]:
        tok = _norm_upper(candidate_row.get(field))
        if not tok or tokens_df.empty:
            continue
        hit = tokens_df[tokens_df["match_token"] == tok]
        if not hit.empty:
            first = hit.iloc[0]
            return {
                "matched": True,
                "method": "identifier",
                "matched_token": tok,
                "match_field_catalog": str(first["match_field"]),
                "catalog_row": known_clagn_df.iloc[int(first["catalog_index"])].to_dict(),
            }
    # coordinate fallback
    ra = pd.to_numeric(pd.Series([candidate_row.get("ra")]), errors="coerce").iloc[0]
    dec = pd.to_numeric(pd.Series([candidate_row.get("dec")]), errors="coerce").iloc[0]
    if np.isfinite(ra) and np.isfinite(dec) and {"ra", "dec"}.issubset({c.lower() for c in known_clagn_df.columns}):
        km = known_clagn_df.copy()
        ra_col = next(c for c in km.columns if c.lower() == "ra")
        dec_col = next(c for c in km.columns if c.lower() == "dec")
        km["_ra"] = pd.to_numeric(km[ra_col], errors="coerce")
        km["_dec"] = pd.to_numeric(km[dec_col], errors="coerce")
        best = None
        best_sep = float("inf")
        for idx, r in km.dropna(subset=["_ra", "_dec"]).iterrows():
            sep = _ang_sep_arcsec(float(ra), float(dec), float(r["_ra"]), float(r["_dec"]))
            if sep < best_sep:
                best_sep = sep
                best = (idx, r)
        if best is not None and best_sep <= tol_arcsec:
            return {
                "matched": True,
                "method": "coordinate",
                "separation_arcsec": float(best_sep),
                "catalog_row": known_clagn_df.loc[best[0]].to_dict(),
            }
    return {"matched": False}


def load_gaia_metrics(path: str) -> pd.DataFrame:
    df = _read_csv(path)
    if df.empty:
        return df
    df = df.copy()
    keys = []
    for key in ["canonical_id", "source_id"]:
        if key in df.columns:
            keys.append(key)
            df[f"_{key}_norm"] = df[key].map(_norm_upper)
    if not keys:
        raise ValueError("Gaia metrics file must contain canonical_id or source_id")
    return df


def lookup_gaia_metrics(candidate_row: pd.Series, gaia_df: pd.DataFrame | None) -> dict | None:
    if gaia_df is None or gaia_df.empty:
        return None
    for key in ["canonical_id", "source_id"]:
        if f"_{key}_norm" not in gaia_df.columns:
            continue
        tok = _norm_upper(candidate_row.get(key))
        if not tok:
            continue
        hit = gaia_df[gaia_df[f"_{key}_norm"] == tok]
        if not hit.empty:
            return hit.iloc[0].drop(labels=[c for c in hit.columns if c.startswith("_")], errors="ignore").to_dict()
    return None
