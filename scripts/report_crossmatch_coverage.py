"""
Report crossmatch coverage using cached outputs.
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
        status_col = cols['status']
        status = df[status_col].astype(str).str.lower()
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
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--catalogs', default=None)
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--benchmark_dir', default='data/benchmark')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    catalogs_path = Path(args.catalogs) if args.catalogs else data_dir / 'catalogs' / 'catalogs.json'
    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / 'metrics'
    metadata_dir = output_dir / 'metadata'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    try:
        config = _load_catalogs_config(catalogs_path)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    catalogs = config.get('catalogs', [])
    if not catalogs:
        print("No catalogs defined in config")
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
            cache_file = data_dir / cache_file

        if not cache_file.exists():
            if args.offline:
                print(f"Missing cache for {name}: {cache_file}")
                print("Provide cached crossmatch CSVs in data/cache/ or update catalogs.json")
                sys.exit(1)
            counts = {
                'queried_count': int(queried_count) if queried_count else 0,
                'matched_count': 0,
                'no_match_count': 0,
                'unavailable_count': int(queried_count) if queried_count else 0,
                'failure_count': 0,
            }
        else:
            df = pd.read_csv(cache_file)
            counts = _counts_from_cache(df)
            if queried_count is not None:
                counts['queried_count'] = int(queried_count)

        queried = counts['queried_count']
        unavailable = counts['unavailable_count']
        coverage = (queried - unavailable) / queried if queried else 0.0

        row = {
            'catalog': name,
            **counts,
            'coverage_fraction': coverage,
        }
        rows.append(row)

    coverage_df = pd.DataFrame(rows)
    meta = build_metadata(Path(args.benchmark_dir))

    if args.dry_run:
        print("Dry run: coverage report skipped")
        return

    csv_path = metrics_dir / 'crossmatch_coverage.csv'
    coverage_df.to_csv(csv_path, index=False)
    write_metadata_sidecar(csv_path, metadata_dir, meta)

    summary = {
        'catalogs': rows,
    }
    write_json_with_metadata(metrics_dir / 'crossmatch_coverage.json', summary, metadata_dir, meta)

    print("Crossmatch coverage report complete")


if __name__ == '__main__':
    main()
