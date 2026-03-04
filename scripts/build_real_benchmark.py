"""
Build spectroscopic real-data CLAGN benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clagn.utils.validation_metadata import git_commit
from clagn.utils.benchmark_labels import ensure_class_ytrue
from clagn.utils.id_canonicalization import add_canonical_id

FORBIDDEN_COL_SUBSTR = [
    'score', 'delta_mag', 'drw', 'changepoint', 'sf_', 'composite',
    'variability', 'tau', 'sigma'
]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _angular_sep_arcsec(ra1, dec1, ra2, dec2) -> float:
    ra1 = np.deg2rad(ra1)
    dec1 = np.deg2rad(dec1)
    ra2 = np.deg2rad(ra2)
    dec2 = np.deg2rad(dec2)
    sin_d = np.sin((dec2 - dec1) / 2.0) ** 2
    sin_r = np.sin((ra2 - ra1) / 2.0) ** 2
    a = sin_d + np.cos(dec1) * np.cos(dec2) * sin_r
    angle = 2 * np.arcsin(np.sqrt(a))
    return float(np.rad2deg(angle) * 3600.0)


def _find_duplicates(df: pd.DataFrame, max_arcsec: float) -> list[tuple[int, int, float]]:
    coords = df[['ra', 'dec']].to_numpy(float)
    dups = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            sep = _angular_sep_arcsec(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            if sep < max_arcsec:
                dups.append((i, j, sep))
    return dups


def _require_ref(value: str) -> bool:
    if not value:
        return False
    v = str(value)
    if '10.' in v:
        return True
    if 'bibcode' in v.lower():
        return True
    if len(v) >= 10 and v[:4].isdigit():
        return True
    return False


def _load_json_list(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list of objects")
    return data


def _load_contaminants(raw_path: Path) -> pd.DataFrame:
    rows = []
    for item in _load_json_list(raw_path):
        source_id = str(item.get('source_id', '')).strip()
        ra = item.get('ra')
        dec = item.get('dec')
        ctype = str(item.get('contaminant_type', '')).strip()
        citation = str(item.get('citation', '')).strip()
        method = str(item.get('classification_method', '')).strip()
        indep = item.get('independent_of_pipeline', True)

        if not source_id or ra is None or dec is None:
            raise ValueError("Contaminant row missing source_id/ra/dec")
        if not ctype:
            raise ValueError("Contaminant row missing contaminant_type")
        if not citation:
            raise ValueError("Contaminant row missing citation")
        if not method:
            raise ValueError("Contaminant row missing classification_method")
        if not bool(indep):
            raise ValueError("Contaminant independent_of_pipeline must be True")

        row = {
            'source_id': source_id,
            'ra': float(ra),
            'dec': float(dec),
            'class': ctype,
            'label_source': citation,
            'label_method': method,
            'independent_of_pipeline': True,
            'notes': str(item.get('notes', '')).strip(),
        }
        for key in ['w1_mag', 'w2_mag', 'mag', 'g_mag', 'r_mag', 'i_mag', 'redshift']:
            if key in item and item[key] is not None:
                row[key] = item[key]
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) < 100:
        raise ValueError(f"Contaminants < 100 (found {len(df)})")
    dups = _find_duplicates(df, 1.0)
    if dups:
        raise ValueError(f"Duplicate contaminants within 1 arcsec: {len(dups)} pairs")
    return df


def _load_clagn(raw_path: Path) -> pd.DataFrame:
    rows = []
    for item in _load_json_list(raw_path):
        source_id = str(item.get('source_id', '')).strip()
        ra = item.get('ra')
        dec = item.get('dec')
        redshift = item.get('redshift')
        spec_ref = str(item.get('spectroscopic_reference', '')).strip()
        doi = str(item.get('doi', '')).strip()
        bibcode = str(item.get('ads_bibcode', '')).strip()
        label_method = str(item.get('label_method', '')).strip().lower()
        indep = item.get('independent_of_pipeline', True)

        spec_ref = spec_ref or doi or bibcode

        if not source_id or ra is None or dec is None:
            raise ValueError("CLAGN row missing source_id/ra/dec")
        if redshift is None:
            raise ValueError("CLAGN row missing redshift")
        if not _require_ref(spec_ref):
            raise ValueError(f"CLAGN row missing DOI/ADS bibcode: {source_id}")
        if label_method and label_method != 'spectroscopic':
            raise ValueError(f"CLAGN row label_method must be spectroscopic: {source_id}")
        if 'wise' in label_method:
            raise ValueError(f"CLAGN label appears WISE-derived: {source_id}")
        if not bool(indep):
            raise ValueError("CLAGN independent_of_pipeline must be True")
        basis = str(item.get('label_basis', '')).lower()
        if 'wise' in basis and 'spectro' not in basis:
            raise ValueError(f"CLAGN label appears WISE-derived: {source_id}")

        for field in ['spectral_state_1', 'spectral_state_2', 'epoch_1', 'epoch_2']:
            if not str(item.get(field, '')).strip():
                raise ValueError(f"CLAGN row missing {field}: {source_id}")

        row = {
            'source_id': source_id,
            'ra': float(ra),
            'dec': float(dec),
            'redshift': float(redshift),
            'spectral_state_1': str(item.get('spectral_state_1', '')).strip(),
            'spectral_state_2': str(item.get('spectral_state_2', '')).strip(),
            'epoch_1': str(item.get('epoch_1', '')).strip(),
            'epoch_2': str(item.get('epoch_2', '')).strip(),
            'class': 'clagn',
            'label_source': spec_ref,
            'label_method': 'spectroscopic',
            'independent_of_pipeline': True,
            'confidence': 'gold',
        }
        for key in ['w1_mag', 'w2_mag', 'mag', 'g_mag', 'r_mag', 'i_mag']:
            if key in item and item[key] is not None:
                row[key] = item[key]
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) < 40:
        raise ValueError(f"Spectroscopic CLAGN < 40 (found {len(df)})")
    dups = _find_duplicates(df, 1.0)
    if dups:
        raise ValueError(f"Duplicate CLAGN within 1 arcsec: {len(dups)} pairs")
    if df['label_source'].isna().any() or (df['label_source'].astype(str).str.strip() == '').any():
        raise ValueError("CLAGN rows missing spectroscopic_reference")
    return df


def _validate_schema(df: pd.DataFrame, schema_path: Path) -> None:
    if not schema_path.exists():
        raise FileNotFoundError(f"Missing schema: {schema_path}")
    schema = json.loads(schema_path.read_text())
    for col in schema.get('required', []):
        if col not in df.columns:
            raise ValueError(f"Benchmark missing required column: {col}")


def _check_label_leakage(df: pd.DataFrame) -> None:
    for col in df.columns:
        for sub in FORBIDDEN_COL_SUBSTR:
            if sub in col.lower():
                raise ValueError(f"Label leakage risk: forbidden column {col}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/real_clagn')
    args = parser.parse_args()

    base = Path(args.data_dir)
    raw_dir = base / 'raw'
    proc_dir = base / 'processed'
    proc_dir.mkdir(parents=True, exist_ok=True)

    clagn_raw = raw_dir / 'literature_sources.json'
    cont_raw = raw_dir / 'contaminants_sources.json'

    clagn_df = _load_clagn(clagn_raw)
    cont_df = _load_contaminants(cont_raw)

    clagn_path = proc_dir / 'clagn_spectroscopic.csv'
    cont_path = proc_dir / 'contaminants_master.csv'
    clagn_df.to_csv(clagn_path, index=False)
    cont_df.to_csv(cont_path, index=False)

    # Combine
    benchmark = pd.concat([clagn_df, cont_df], ignore_index=True)
    benchmark = ensure_class_ytrue(benchmark)
    benchmark = add_canonical_id(benchmark)

    if len(benchmark) < 200:
        raise ValueError(f"Benchmark total < 200 (found {len(benchmark)})")

    # Ensure no duplicates within 1 arcsec across combined
    dups = _find_duplicates(benchmark, 1.0)
    if dups:
        raise ValueError(f"Duplicate sources within 1 arcsec in benchmark: {len(dups)} pairs")

    _validate_schema(benchmark, base / 'schema' / 'real_benchmark_schema.json')
    _check_label_leakage(benchmark)

    master_path = base / 'benchmark_master.csv'
    benchmark.to_csv(master_path, index=False)

    freeze = {
        'benchmark_hash': _sha256_file(master_path),
        'counts': benchmark['class'].value_counts().to_dict(),
        'y_true_positive_count': int(benchmark['y_true'].sum()),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': git_commit(),
    }
    (base / 'BENCHMARK_FREEZE.json').write_text(json.dumps(freeze, indent=2))

    print("Real benchmark build PASS")


if __name__ == '__main__':
    main()
