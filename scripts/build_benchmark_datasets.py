"""
Build benchmark datasets with schema validation, deduplication, and frozen splits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    except Exception:
        return 'unavailable'


def _load_schema(schema_path: Path) -> dict:
    if not schema_path.exists():
        raise FileNotFoundError(f"Missing benchmark schema: {schema_path}")
    return json.loads(schema_path.read_text())


def _validate_schema(df: pd.DataFrame, schema: dict) -> list[str]:
    errors = []
    required = schema.get('required', [])
    for col in required:
        if col not in df.columns:
            errors.append(f"missing required column: {col}")
    return errors


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ra'] = pd.to_numeric(df['ra'], errors='coerce')
    df['dec'] = pd.to_numeric(df['dec'], errors='coerce')
    df['source_id'] = df['source_id'].astype(str)
    df['__ra_round'] = df['ra'].round(6)
    df['__dec_round'] = df['dec'].round(6)
    df = df.drop_duplicates(subset=['source_id', '__ra_round', '__dec_round'])
    df = df.drop(columns=['__ra_round', '__dec_round'])
    return df


def _load_sources_from_schema_csvs(sources_dir: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(sources_dir.glob('*.csv')):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if {'source_id', 'ra', 'dec', 'label', 'label_source', 'label_method',
            'independent_of_pipeline', 'notes'}.issubset(df.columns):
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _build_from_legacy(bench_dir: Path) -> pd.DataFrame:
    frames = []

    known = bench_dir / 'known_clagn.csv'
    if known.exists():
        df = pd.read_csv(known)
        out = pd.DataFrame({
            'source_id': df.get('name', df.get('source_id')).astype(str),
            'ra': df['ra'],
            'dec': df['dec'],
            'label': 'positive',
            'label_source': df.get('reference', ''),
            'label_method': df.get('tier', '').apply(lambda x: 'spectroscopic' if str(x) == '1' else 'photometric'),
            'independent_of_pipeline': True,
            'notes': df.get('transition_type', 'clagn').astype(str),
        })
        frames.append(out)

    cont = bench_dir / 'known_contaminants.csv'
    if cont.exists():
        df = pd.read_csv(cont)
        out = pd.DataFrame({
            'source_id': df.get('name', df.get('source_id')).astype(str),
            'ra': df['ra'],
            'dec': df['dec'],
            'label': df.get('contaminant_type', 'contaminant'),
            'label_source': df.get('reference', ''),
            'label_method': 'catalog',
            'independent_of_pipeline': True,
            'notes': df.get('notes', '').astype(str),
        })
        frames.append(out)

    ctrl = bench_dir / 'normal_agn_control.csv'
    if ctrl.exists():
        df = pd.read_csv(ctrl)
        out = pd.DataFrame({
            'source_id': df.get('name', df.get('source_id')).astype(str),
            'ra': df['ra'],
            'dec': df['dec'],
            'label': 'control',
            'label_source': df.get('reference', ''),
            'label_method': 'catalog',
            'independent_of_pipeline': True,
            'notes': '',
        })
        frames.append(out)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _validate_integrity(df: pd.DataFrame) -> list[str]:
    errors = []
    # citations
    if df['label_source'].isna().any() or (df['label_source'].astype(str).str.strip() == '').any():
        errors.append('Missing label_source citations')
    # independent_of_pipeline
    indep_vals = df['independent_of_pipeline'].apply(lambda x: str(x).strip().lower())
    if not indep_vals.isin({'true', '1', 'yes'}).all():
        errors.append('independent_of_pipeline must be True for all rows')
    # duplicates by source_id
    if df['source_id'].duplicated().any():
        errors.append('Duplicate source_id entries found')

    # counts
    n_pos = (df['label'] == 'positive').sum()
    n_cont = (~df['label'].isin(['positive', 'control'])).sum()
    if n_pos < 30:
        errors.append(f'positives < 30 (found {n_pos})')
    if n_cont < 80:
        errors.append(f'contaminants < 80 (found {n_cont})')

    return errors


def _split_train_test(df: pd.DataFrame, seed: int, test_frac: float) -> pd.DataFrame:
    rng = pd.Series(range(len(df))).sample(frac=1, random_state=seed).index
    df = df.loc[rng].reset_index(drop=True)

    split = []
    for label, group in df.groupby('label'):
        n = len(group)
        n_test = max(1, int(round(n * test_frac)))
        n_test = max(n_test, int((0.2 * n) + 0.9999))
        test_idx = group.index[:n_test]
        split.extend([(i, 'test') for i in test_idx])
    split_df = pd.DataFrame(split, columns=['idx', 'split']).set_index('idx')
    df['split'] = 'train_dev'
    df.loc[split_df.index, 'split'] = split_df['split']
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--benchmark_dir', default='data/benchmark')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_frac', type=float, default=0.2)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    bench_dir = Path(args.benchmark_dir)
    sources_dir = bench_dir / 'sources'
    schema_path = bench_dir / 'schema' / 'benchmark_schema.json'

    schema = _load_schema(schema_path)

    # Load schema-compliant sources first
    df = _load_sources_from_schema_csvs(sources_dir)
    if df.empty:
        df = _build_from_legacy(bench_dir)

    if df.empty:
        print("No benchmark sources found. Provide CSVs under data/benchmark/sources/"
              " or legacy known_clagn/known_contaminants/normal_agn_control files.")
        sys.exit(1)

    df = _dedupe(df)

    # Ensure required columns
    for col in schema['required']:
        if col not in df.columns:
            df[col] = '' if col in ('label_source', 'label_method', 'notes') else None

    # Schema validation
    errors = _validate_schema(df, schema)
    if errors:
        print("Schema validation FAILED:")
        for e in errors:
            print(f"- {e}")
        sys.exit(1)

    # Integrity validation
    integ = _validate_integrity(df)
    if integ:
        print("Benchmark integrity FAILED:")
        for e in integ:
            print(f"- {e}")
        sys.exit(1)

    if args.dry_run:
        print("Dry run: benchmark validation PASSED")
        return

    bench_dir.mkdir(parents=True, exist_ok=True)

    # Write master
    master_path = bench_dir / 'benchmark_master.csv'
    df.to_csv(master_path, index=False)

    # Split
    split_df = _split_train_test(df, seed=args.seed, test_frac=args.test_frac)
    train_dev = split_df[split_df['split'] == 'train_dev'].drop(columns=['split'])
    test = split_df[split_df['split'] == 'test'].drop(columns=['split'])

    train_dev.to_csv(bench_dir / 'benchmark_train_dev.csv', index=False)
    test.to_csv(bench_dir / 'benchmark_test.csv', index=False)

    # Split file
    split_file = bench_dir / 'TRAIN_DEV_TEST_SPLIT.csv'
    split_df[['source_id', 'split']].to_csv(split_file, index=False)
    split_hash = _sha256_file(split_file)
    (bench_dir / 'SPLIT_HASH.txt').write_text(split_hash + "\n")

    # Freeze
    freeze = {
        'benchmark_hash': _sha256_file(master_path),
        'counts': split_df['label'].value_counts().to_dict(),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': _git_commit(),
    }
    with open(bench_dir / 'BENCHMARK_FREEZE.json', 'w') as f:
        json.dump(freeze, f, indent=2)

    print("Benchmark build PASS")


if __name__ == '__main__':
    main()
