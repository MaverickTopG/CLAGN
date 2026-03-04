"""
Contaminant performance audit and coverage accounting.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clagn.utils.validation_metadata import build_metadata, write_metadata_sidecar, write_json_with_metadata
from clagn.utils.benchmark_labels import normalized_class_series
from clagn.utils.id_canonicalization import best_join_key, add_canonical_id


def _load_scores(scores_path: Path) -> pd.DataFrame:
    df = pd.read_csv(scores_path)
    if 'source_id' not in df.columns or 'score' not in df.columns:
        raise ValueError("scores_csv must contain source_id and score")
    if 'canonical_id' not in df.columns:
        df = add_canonical_id(df)
    if 'status' not in df.columns:
        df['status'] = 'OK'
    keep = ['source_id', 'canonical_id', 'score', 'status']
    return df[[c for c in keep if c in df.columns]]


def _load_catalogs_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing catalogs config: {path}")
    return json.loads(path.read_text())


def _counts_from_cache(df: pd.DataFrame) -> dict:
    cols = {c.lower(): c for c in df.columns}
    if {'queried_count', 'matched_count', 'no_match_count', 'unavailable_count', 'failure_count'}.issubset(df.columns):
        row = df.iloc[0]
        return {k: int(row[k]) for k in ['queried_count', 'matched_count', 'no_match_count', 'unavailable_count', 'failure_count']}

    if 'status' in cols:
        status = df[cols['status']].astype(str).str.lower()
        return {
            'queried_count': len(df),
            'matched_count': int((status == 'matched').sum()),
            'no_match_count': int((status == 'no_match').sum()),
            'unavailable_count': int((status == 'unavailable').sum()),
            'failure_count': int((status == 'failure').sum()),
        }

    if 'matched' in cols:
        matched = df[cols['matched']].astype(bool)
        return {
            'queried_count': len(df),
            'matched_count': int(matched.sum()),
            'no_match_count': int((~matched).sum()),
            'unavailable_count': 0,
            'failure_count': 0,
        }

    if 'match_id' in cols:
        series = df[cols['match_id']]
        matched = series.notna() & (series.astype(str).str.strip() != '')
        return {
            'queried_count': len(df),
            'matched_count': int(matched.sum()),
            'no_match_count': int((~matched).sum()),
            'unavailable_count': 0,
            'failure_count': 0,
        }

    return {
        'queried_count': len(df),
        'matched_count': 0,
        'no_match_count': len(df),
        'unavailable_count': 0,
        'failure_count': 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/real_clagn')
    parser.add_argument('--scores_csv', default=None)
    parser.add_argument('--output_dir', default='results_real')
    parser.add_argument('--threshold', type=float, default=None)
    args = parser.parse_args()

    base = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    benchmark_path = base / 'benchmark_master.csv'
    if not benchmark_path.exists():
        print(f"Missing benchmark_master.csv: {benchmark_path}")
        sys.exit(1)

    if args.scores_csv:
        scores_path = Path(args.scores_csv)
    else:
        default1 = base / 'benchmark_scores.csv'
        default2 = base / 'scores' / 'benchmark_scores.csv'
        scores_path = default1 if default1.exists() else default2

    if not scores_path.exists():
        print(f"Missing scores file: {scores_path}")
        sys.exit(1)

    if args.threshold is None:
        op_path = Path(args.output_dir) / 'operating_threshold.json'
        if not op_path.exists():
            print("Missing operating_threshold.json. Run dev evaluation first or provide --threshold")
            sys.exit(1)
        args.threshold = json.loads(op_path.read_text()).get('threshold', 0.5)

    bench = pd.read_csv(benchmark_path)
    if 'canonical_id' not in bench.columns:
        bench = add_canonical_id(bench)
    bench = bench.copy()
    bench['class'] = normalized_class_series(bench)
    scores = _load_scores(scores_path)
    join_key = best_join_key(bench, scores)
    merged = bench.merge(scores, on=join_key, how='inner')
    if merged.empty:
        print("No overlap between benchmark and scores")
        sys.exit(1)

    def rejection_rate(label_key: str) -> float:
        subset = merged[merged['class'].astype(str).str.lower().str.contains(label_key)]
        if subset.empty:
            return float('nan')
        return float((subset['score'] < args.threshold).mean())

    def contamination_rate(label_key: str) -> float:
        subset = merged[merged['class'].astype(str).str.lower().str.contains(label_key)]
        if subset.empty:
            return float('nan')
        return float((subset['score'] >= args.threshold).mean())

    report = {
        'sn_rejection_rate': rejection_rate('sn'),
        'blazar_rejection_rate': rejection_rate('blazar'),
        'stellar_contamination_rate': contamination_rate('star'),
        'threshold': args.threshold,
    }

    # Coverage accounting
    catalogs_path = base / 'catalogs' / 'catalogs.json'
    if not catalogs_path.exists():
        print(f"Missing catalogs.json: {catalogs_path}")
        sys.exit(1)

    config = _load_catalogs_config(catalogs_path)
    catalogs = config.get('catalogs', [])
    if not catalogs:
        print("No catalogs defined in catalogs.json")
        sys.exit(1)

    rows = []
    for cat in catalogs:
        name = cat.get('name')
        cache_path = cat.get('cache_path')
        queried_count = cat.get('queried_count')
        if not name or not cache_path:
            print("Catalog config missing name or cache_path")
            sys.exit(1)
        cache_file = Path(cache_path)
        if not cache_file.is_absolute():
            cache_file = base / cache_file
        if not cache_file.exists():
            print(f"Missing cache for {name}: {cache_file}")
            sys.exit(1)
        df = pd.read_csv(cache_file)
        counts = _counts_from_cache(df)
        if queried_count is not None:
            counts['queried_count'] = int(queried_count)
        queried = counts['queried_count']
        unavailable = counts['unavailable_count']
        coverage = (queried - unavailable) / queried if queried else 0.0
        availability = (queried - counts['failure_count']) / queried if queried else 0.0
        rows.append({
            'catalog': name,
            **counts,
            'coverage_fraction': coverage,
            'availability_fraction': availability,
        })

    coverage_df = pd.DataFrame(rows)
    report['coverage_fraction'] = float(coverage_df['coverage_fraction'].mean())
    report['catalog_availability_fraction'] = float(coverage_df['availability_fraction'].mean())

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = output_dir / 'metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    meta = build_metadata(base)

    cov_path = output_dir / 'crossmatch_coverage.csv'
    coverage_df.to_csv(cov_path, index=False)
    write_metadata_sidecar(cov_path, metadata_dir, meta)

    write_json_with_metadata(output_dir / 'contaminant_report.json', report, metadata_dir, meta)

    print("Contaminant audit complete")


if __name__ == '__main__':
    main()
