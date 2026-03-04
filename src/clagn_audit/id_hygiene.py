from __future__ import annotations

import re


AMBIGUOUS_IDS = {"PKS", "PG", "MR", "HS", "B2"}


def normalize_source_id_for_audit(source_id: object) -> str:
    sid = "" if source_id is None else str(source_id).strip()
    if not sid:
        return ""
    if sid.isdigit():
        return sid
    sid = re.sub(r"^SDSSJ{2,}", "SDSSJ", sid)
    sid = re.sub(r"^NGC\s*(\d+)$", r"NGC \1", sid, flags=re.IGNORECASE)
    sid = re.sub(r"^MRK\s*(\d+)$", r"Mrk \1", sid, flags=re.IGNORECASE)
    sid = re.sub(r"^IC\s*(\d+)$", r"IC \1", sid, flags=re.IGNORECASE)
    return sid


def is_ambiguous_identifier(source_id: str) -> bool:
    sid = normalize_source_id_for_audit(source_id)
    return sid in AMBIGUOUS_IDS


def classify_id_mismatch(source_id: str, canonical_id: str) -> str | None:
    raw = "" if source_id is None else str(source_id).strip()
    canon = "" if canonical_id is None else str(canonical_id).strip()
    norm = normalize_source_id_for_audit(raw)
    if not canon:
        return "other_mismatch"
    if norm == canon:
        if re.match(r"^SDSSJ{2,}", raw):
            return "sdss_double_j"
        if raw != norm:
            return "spacing_normalized"
        return None
    if raw != norm:
        return "other_mismatch"
    return "other_mismatch"
