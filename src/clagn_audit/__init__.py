"""CLAGN audit toolkit for offline result-table analysis."""

from .io import read_results_csv, write_csv, write_json, safe_mkdir
from .schema import validate_required_columns, coerce_types, warn_duplicate_canonical_ids
from .cleaning import add_derived_columns, parse_band_coverage, assign_rejection_bucket
from .quality import build_quality_tables, write_quality_report_csv, write_quality_report_markdown
from .stats import compute_internal_correlations, prepare_ok_vs_rejected_subsets
from .mining import build_case_mining_tables, write_case_mining_md, write_top_candidates_csv
from .id_hygiene import normalize_source_id_for_audit, classify_id_mismatch, is_ambiguous_identifier

__all__ = [
    'read_results_csv', 'write_csv', 'write_json', 'safe_mkdir',
    'validate_required_columns', 'coerce_types', 'warn_duplicate_canonical_ids',
    'add_derived_columns', 'parse_band_coverage', 'assign_rejection_bucket',
    'build_quality_tables', 'write_quality_report_csv', 'write_quality_report_markdown',
    'compute_internal_correlations', 'prepare_ok_vs_rejected_subsets',
    'build_case_mining_tables', 'write_case_mining_md', 'write_top_candidates_csv',
    'normalize_source_id_for_audit', 'classify_id_mismatch', 'is_ambiguous_identifier',
]
