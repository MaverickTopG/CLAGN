from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


AMBIGUOUS_BARE_PREFIXES = {"PKS", "PG", "HS", "B2"}


def canonicalize_id(source_id: object) -> str:
    """Normalize common catalog naming inconsistencies for join hygiene."""
    sid = "" if source_id is None else str(source_id).strip()
    if not sid:
        return ""

    # Preserve all-digit IDs (e.g., Gaia-like numeric identifiers).
    if sid.isdigit():
        return sid

    sid = re.sub(r"^SDSSJ{2,}", "SDSSJ", sid)
    sid = re.sub(r"^NGC\s*(\d+)$", r"NGC \1", sid, flags=re.IGNORECASE)
    sid = re.sub(r"^Mrk\s*(\d+)$", r"Mrk \1", sid, flags=re.IGNORECASE)
    sid = re.sub(r"^IC\s*(\d+)$", r"IC \1", sid, flags=re.IGNORECASE)

    # Standardize capitalization for the above replacements.
    sid = re.sub(r"^ngc ", "NGC ", sid)
    sid = re.sub(r"^mrk ", "Mrk ", sid)
    sid = re.sub(r"^ic ", "IC ", sid)
    return sid


def find_ambiguous_ids(ids: Iterable[object]) -> list[str]:
    flagged: list[str] = []
    seen: set[str] = set()
    for raw in ids:
        sid = canonicalize_id(raw)
        if not sid:
            continue
        if sid in AMBIGUOUS_BARE_PREFIXES:
            if sid not in seen:
                flagged.append(sid)
                seen.add(sid)
            continue
        if re.fullmatch(r"[A-Za-z]{1,3}", sid):
            if sid not in seen:
                flagged.append(sid)
                seen.add(sid)
    return flagged


def add_canonical_id(df: pd.DataFrame, source_col: str = "source_id") -> pd.DataFrame:
    out = df.copy()
    if source_col not in out.columns:
        raise KeyError(f"Missing source column: {source_col}")
    out["canonical_id"] = out[source_col].map(canonicalize_id)
    return out


def best_join_key(left: pd.DataFrame, right: pd.DataFrame) -> str:
    if "canonical_id" in left.columns and "canonical_id" in right.columns:
        return "canonical_id"
    return "source_id"
