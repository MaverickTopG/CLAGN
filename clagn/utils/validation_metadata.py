from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    except Exception:
        return 'unavailable'


def config_hash(config_path: Path | None = None) -> str:
    cfg = config_path or Path('clagn/config.py')
    if cfg.exists():
        return sha256_file(cfg)
    return 'unavailable'


def benchmark_hash(benchmark_dir: Path) -> str:
    freeze = benchmark_dir / 'BENCHMARK_FREEZE.json'
    if freeze.exists():
        try:
            data = json.loads(freeze.read_text())
            if 'benchmark_hash' in data:
                return data['benchmark_hash']
            if 'sha256' in data:
                return data['sha256']
        except Exception:
            pass
    master = benchmark_dir / 'benchmark_master.csv'
    if master.exists():
        return sha256_file(master)
    return 'unavailable'


def build_metadata(benchmark_dir: Path, extra: dict | None = None) -> dict:
    meta = {
        'git_commit': git_commit(),
        'config_hash': config_hash(),
        'benchmark_hash': benchmark_hash(benchmark_dir),
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        meta.update(extra)
    return meta


def write_metadata_sidecar(target_path: Path, metadata_dir: Path, meta: dict) -> None:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    out = metadata_dir / f"{target_path.name}.metadata.json"
    payload = meta.copy()
    payload['output_path'] = str(target_path)
    out.write_text(json.dumps(payload, indent=2))


def write_json_with_metadata(path: Path, payload: dict, metadata_dir: Path, meta: dict) -> None:
    full = payload.copy()
    full.update(meta)
    path.write_text(json.dumps(full, indent=2))
    write_metadata_sidecar(path, metadata_dir, meta)
