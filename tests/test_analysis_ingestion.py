from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.clagn_audit.io import read_results_csv
from src.clagn_audit.schema import validate_required_columns, coerce_types
from src.clagn_audit.cleaning import add_derived_columns, assign_rejection_bucket

TEST_DATA = Path('tests/data')


def _load(name: str) -> pd.DataFrame:
    df = read_results_csv(TEST_DATA / name)
    validate_required_columns(df)
    df = coerce_types(df)
    df = add_derived_columns(df)
    return df


def test_parsing_blank_scores() -> None:
    df = _load('sample_results_minimal.csv')
    assert len(df) == 2
    row = df.loc[df['canonical_id'] == 'SRC_A'].iloc[0]
    assert pd.isna(row['score'])
    assert bool(row['score_is_missing']) is True


def test_parsing_quoted_band_coverage() -> None:
    df = _load('sample_results_quoted.csv')
    row = df.loc[df['canonical_id'] == 'NGC 5548'].iloc[0]
    assert row['band_coverage'] == 'W1,W2'
    assert bool(row['has_w1']) is True
    assert bool(row['has_w2']) is True


def test_strict_quality_logic() -> None:
    df = _load('sample_results_minimal.csv')
    ok_row = df.loc[df['canonical_id'] == 'SRC_B'].iloc[0]
    bad_row = df.loc[df['canonical_id'] == 'SRC_A'].iloc[0]
    assert float(ok_row['baseline_years_recomputed']) == 7.0
    assert int(ok_row['n_points']) == 50
    assert bool(ok_row['strict_quality']) is True
    assert bool(bad_row['strict_quality']) is False  # W1_only and too few points


def test_rejection_bucket_mapping() -> None:
    assert assign_rejection_bucket('NO_DATA', 'missing allwise coverage') == 'missing_allwise'
    assert assign_rejection_bucket('TOO_FEW_POINTS', 'Baseline 3.1 yr < 7 yr') == 'short_baseline'


def test_sdss_double_j_normalization_and_mismatch_flag() -> None:
    df = _load('sample_results_minimal.csv').copy()
    df.loc[df['canonical_id'] == 'SRC_A', 'source_id'] = 'SDSSJJ123456.78+123456.7'
    df.loc[df['canonical_id'] == 'SRC_A', 'canonical_id'] = 'SDSSJ123456.78+123456.7'
    from src.clagn_audit.cleaning import add_derived_columns
    df = add_derived_columns(df)
    row = df.loc[df['canonical_id'] == 'SDSSJ123456.78+123456.7'].iloc[0]
    assert row['source_id_normalized'] == row['canonical_id']
    assert bool(row['source_id_matches_canonical']) is True
    assert row['id_mismatch_reason'] == 'sdss_double_j'


def test_ambiguous_id_flagging() -> None:
    df = _load('sample_results_minimal.csv').copy()
    df.loc[df['canonical_id'] == 'SRC_A', 'source_id'] = 'PKS'
    df.loc[df['canonical_id'] == 'SRC_A', 'canonical_id'] = 'PKS'
    from src.clagn_audit.cleaning import add_derived_columns
    df = add_derived_columns(df)
    row = df.loc[df['canonical_id'] == 'PKS'].iloc[0]
    assert bool(row['is_ambiguous_id']) is True


def test_baseline_consistency_flag() -> None:
    df = _load('sample_results_minimal.csv').copy()
    from src.clagn_audit.cleaning import add_derived_columns
    # Row B is exactly consistent in fixture.
    row_b = df.loc[df['canonical_id'] == 'SRC_B'].iloc[0]
    assert bool(row_b['baseline_years_consistent']) is True
    # Force mismatch for row A and recompute.
    df.loc[df['canonical_id'] == 'SRC_A', 'baseline_years'] = 8.5
    df = add_derived_columns(df)
    row_a = df.loc[df['canonical_id'] == 'SRC_A'].iloc[0]
    assert float(row_a['baseline_years_diff']) > 0.01
    assert bool(row_a['baseline_years_consistent']) is False
