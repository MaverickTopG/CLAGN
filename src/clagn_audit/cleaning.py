from __future__ import annotations

import pandas as pd

from .id_hygiene import (
    classify_id_mismatch,
    is_ambiguous_identifier,
    normalize_source_id_for_audit,
)


STATUS_OK = 'OK'


def parse_band_coverage(series: pd.Series) -> pd.DataFrame:
    s = series.astype('string').fillna('none').str.strip()

    def _flags(text: str) -> tuple[bool, bool]:
        tokens = {tok.strip().upper() for tok in str(text).split(',') if tok.strip()}
        has_w1 = 'W1' in tokens or 'W1_ONLY' in tokens or 'W1ONLY' in tokens
        has_w2 = 'W2' in tokens or 'W2_ONLY' in tokens or 'W2ONLY' in tokens
        # Support common string form "W1,W2"
        if 'W1,W2' in str(text).upper():
            has_w1, has_w2 = True, True
        return has_w1, has_w2

    parsed = s.map(_flags)
    return pd.DataFrame({
        'has_w1': parsed.map(lambda t: bool(t[0])),
        'has_w2': parsed.map(lambda t: bool(t[1])),
    }, index=series.index)



def assign_rejection_bucket(status: object, rejection_reason: object) -> str:
    st = '' if status is None else str(status).strip().upper()
    rr = '' if rejection_reason is None else str(rejection_reason).strip().lower()

    if st == 'OK':
        return 'ok'
    if st == 'NO_DATA':
        if 'allwise' in rr:
            return 'missing_allwise'
        if 'neowise' in rr:
            return 'missing_neowise'
        if 'no data' in rr:
            return 'no_data'
        return 'no_data_other'
    if st == 'TOO_FEW_POINTS':
        if 'epoch' in rr or 'insufficient' in rr:
            return 'few_epochs'
        if 'baseline' in rr or 'yr <' in rr:
            return 'short_baseline'
        return 'too_few_points_other'
    if st == 'ERROR':
        return 'error'
    if st == 'NAME_RESOLVE_FAIL':
        return 'name_resolve_fail'
    return 'unknown'



def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out['status_normalized'] = out['status'].astype('string').fillna('UNKNOWN').str.strip().str.upper().replace({'': 'UNKNOWN'})
    out['baseline_years_recomputed'] = pd.to_numeric(out['baseline_days'], errors='coerce') / 365.25
    if 'baseline_years' in out.columns:
        base_given = pd.to_numeric(out['baseline_years'], errors='coerce')
        out['baseline_years_diff'] = (base_given - out['baseline_years_recomputed']).abs()
        out['baseline_years_consistent'] = out['baseline_years_diff'] <= 0.01
    else:
        out['baseline_years_diff'] = pd.NA
        out['baseline_years_consistent'] = pd.Series([pd.NA] * len(out), dtype='boolean')
    out['is_ok'] = out['status_normalized'] == STATUS_OK

    out['source_id_normalized'] = out['source_id'].map(normalize_source_id_for_audit)
    canonical = out['canonical_id'].astype('string').fillna('').str.strip()
    src_norm = out['source_id_normalized'].astype('string').fillna('').str.strip()
    out['source_id_matches_canonical'] = (src_norm == canonical)
    out['id_mismatch_reason'] = [
        classify_id_mismatch(s, c) for s, c in zip(out['source_id'], out['canonical_id'])
    ]
    out['is_ambiguous_id'] = out['source_id'].map(lambda x: bool(is_ambiguous_identifier(str(x))))

    cov = parse_band_coverage(out['band_coverage'])
    out['has_w1'] = cov['has_w1'].astype(bool)
    out['has_w2'] = cov['has_w2'].astype(bool)

    n_points = pd.to_numeric(out['n_points'], errors='coerce')
    baseline = pd.to_numeric(out['baseline_years_recomputed'], errors='coerce')
    out['strict_quality'] = (
        (baseline >= 7.0) &
        (n_points >= 50) &
        out['has_w1'].astype(bool) &
        out['has_w2'].astype(bool) &
        out['is_ok'].astype(bool)
    )

    score_numeric = pd.to_numeric(out['score'], errors='coerce')
    out['score_is_missing'] = score_numeric.isna()
    out['score_is_finite'] = score_numeric.notna()

    if 'rejection_reason' not in out.columns:
        out['rejection_reason'] = pd.Series([pd.NA] * len(out), dtype='string')

    out['rejection_bucket'] = [
        assign_rejection_bucket(st, rr)
        for st, rr in zip(out['status_normalized'], out['rejection_reason'])
    ]

    return out
