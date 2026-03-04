"""
Split real CLAGN benchmark into dev/test with stratification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clagn.utils.validation_metadata import git_commit
from clagn.utils.benchmark_labels import ensure_class_ytrue, ensure_y_true
from clagn.utils.id_canonicalization import add_canonical_id


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stratified_split(df: pd.DataFrame, test_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = ensure_y_true(df)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    dev_idx, test_idx = next(sss.split(df, y))
    out = df.copy()
    out['split'] = ''
    out.loc[out.index[dev_idx], 'split'] = 'dev'
    out.loc[out.index[test_idx], 'split'] = 'test'
    dev = out[out['split'] == 'dev'].copy()
    test = out[out['split'] == 'test'].copy()
    return dev, test, out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/real_clagn')
    parser.add_argument('--test_frac', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    base = Path(args.data_dir)
    master_path = base / 'benchmark_master.csv'
    if not master_path.exists():
        print(f"Missing benchmark_master.csv: {master_path}")
        sys.exit(1)

    df = pd.read_csv(master_path)
    df = ensure_class_ytrue(df)
    if 'canonical_id' not in df.columns:
        df = add_canonical_id(df)

    n_pos = int(df['y_true'].sum())
    if n_pos < 20:
        print("Insufficient positives for dev/test split (need >=20)")
        sys.exit(1)

    dev, test, df_with_split = _stratified_split(df, args.test_frac, args.seed)

    if int(dev['y_true'].sum()) < 10:
        print(f"Dev split has {int(dev['y_true'].sum())} positives; need >=10")
        sys.exit(1)
    if int(test['y_true'].sum()) < 10:
        print(f"Test split has {int(test['y_true'].sum())} positives; need >=10")
        sys.exit(1)
    if int(test['y_true'].sum()) < max(10, int(0.2 * n_pos)):
        print("Test split does not contain >=20% positives and >=10 positives")
        sys.exit(1)

    key = 'canonical_id' if 'canonical_id' in df.columns else 'source_id'
    if dev[key].duplicated().any() or test[key].duplicated().any():
        print(f"Duplicate {key} within splits")
        sys.exit(1)

    overlap = set(dev[key]).intersection(set(test[key]))
    if overlap:
        print(f"Duplicate {key} across splits: {len(overlap)}")
        sys.exit(1)

    dev_path = base / 'benchmark_dev.csv'
    test_path = base / 'benchmark_test.csv'
    dev.to_csv(dev_path, index=False)
    test.to_csv(test_path, index=False)

    # Persist split assignments back into master benchmark.
    df_with_split.to_csv(master_path, index=False)

    split_path = base / 'SPLIT_ASSIGNMENTS.csv'
    df_with_split[[c for c in ['source_id', 'canonical_id', 'split'] if c in df_with_split.columns]].to_csv(split_path, index=False)
    split_hash = _sha256_file(split_path)

    class_col = 'class' if 'class' in df_with_split.columns else 'label'
    split_meta = {
        'split_hash': split_hash,
        'counts': {
            'dev': dev[class_col].value_counts().to_dict(),
            'test': test[class_col].value_counts().to_dict(),
        },
        'y_true_counts': {
            'dev': pd.to_numeric(dev['y_true'], errors='coerce').fillna(0).astype(int).value_counts().to_dict(),
            'test': pd.to_numeric(test['y_true'], errors='coerce').fillna(0).astype(int).value_counts().to_dict(),
        },
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': git_commit(),
    }
    (base / 'SPLIT_HASH.json').write_text(json.dumps(split_meta, indent=2))

    print(
        f"Real benchmark split PASS | dev={len(dev)} rows ({int(dev['y_true'].sum())} pos) | "
        f"test={len(test)} rows ({int(test['y_true'].sum())} pos)"
    )


if __name__ == '__main__':
    main()
