#!/usr/bin/env python3
"""
Fetch WISE NEOWISE-R light curve cards for all sources in benchmark_master_v3.csv
Uses 8 parallel threads with retry, backoff, rate limit handling, and checkpointing.
Fully resumable — skips sources that already have cards.
DO NOT run evaluate_benchmark.py after this.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timedelta, timezone
from threading import Event, Lock
from pathlib import Path

import pandas as pd
import requests

# -- Config -----------------------------------------------------------------
BENCHMARK_CSV = "data/benchmark/benchmark_master_v3.csv"
CARDS_DIR = "cards/benchmark_v3"
FAILED_LOG = "logs/failed_sources.csv"
PROGRESS_LOG = "logs/fetch_progress.json"
N_WORKERS = 8
TIMEOUT_SEC = 60
MAX_RETRIES = 3
BACKOFF_BASE = 5        # seconds, doubles each retry
RATE_LIMIT_WAIT = 30    # seconds to pause all workers on 429
CHECKPOINT_EVERY = 1000
LOG_EVERY = 500
DISK_CHECK_EVERY = 5000
MAX_CARDS_GB = 35.0
IRSA_URL = "https://irsa.ipac.caltech.edu/TAP/sync"
MAX_INFLIGHT = N_WORKERS * 8
HEARTBEAT_EVERY_SEC = 30

Path(CARDS_DIR).mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.WARNING)
print_lock = Lock()
stats_lock = Lock()
stop_event = Event()

stats = {"done": 0, "failed": 0, "skipped": 0, "start": time.time()}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cards_dir_size_gb() -> float:
    total = 0
    base = Path(CARDS_DIR)
    if not base.exists():
        return 0.0
    for p in base.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                continue
    return total / (1024.0 ** 3)


def _normalize_tap_json(data: dict) -> list[dict]:
    """
    Accept both classic {'metadata','data'} and VOTABLE JSON forms.
    """
    if isinstance(data, dict) and data.get("data") and data.get("metadata"):
        cols = [c["name"] for c in data["metadata"]]
        return [dict(zip(cols, row)) for row in data["data"]]

    vot = data.get("VOTABLE", {}) if isinstance(data, dict) else {}
    resources = vot.get("RESOURCE_ARRAY", [])
    if isinstance(resources, dict):
        resources = [resources]
    if not resources:
        return []
    table = resources[0].get("TABLE", {})
    fields = table.get("FIELD_ARRAY", [])
    if isinstance(fields, dict):
        fields = [fields]
    cols = [f.get("<xmlattr>", {}).get("name", "") for f in fields]
    tabledata = table.get("DATA", {}).get("TABLEDATA", [])
    rows = []
    if isinstance(tabledata, list):
        rows = tabledata
    elif isinstance(tabledata, dict):
        tr = tabledata.get("TR_ARRAY", [])
        if isinstance(tr, dict):
            tr = [tr]
        for r in tr:
            td = r.get("TD_ARRAY", [])
            if isinstance(td, list):
                rows.append(td)
    out = []
    for r in rows:
        if isinstance(r, list):
            out.append(dict(zip(cols, r)))
    return out


def query_wise(ra: float, dec: float, radius_arcsec: float = 3.0, timeout: int = TIMEOUT_SEC):
    adql = f"""
    SELECT mjd, w1mpro, w1sigmpro, w2mpro, w2sigmpro, qi_fact, saa_sep, moon_masked
    FROM neowiser_p1bs_psd
    WHERE CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius_arcsec / 3600.0:.8f})
    ) = 1
    AND qi_fact >= 1 AND saa_sep >= 5
    AND w1sigmpro IS NOT NULL AND w1sigmpro < 0.2
    ORDER BY mjd
    """
    resp = requests.get(
        IRSA_URL,
        params={"QUERY": adql, "FORMAT": "json", "LANG": "ADQL"},
        timeout=(10, timeout),
    )
    if resp.status_code == 429:
        raise RuntimeError("RATE_LIMITED")
    resp.raise_for_status()
    data = resp.json()
    rows = _normalize_tap_json(data)
    if not rows:
        return None
    # moon_masked handling can be scalar/string like "00", keep rows where no moon mask if field exists
    cleaned = []
    for row in rows:
        mm = str(row.get("moon_masked", "0")).strip()
        if mm in {"1", "01", "10", "11"}:
            continue
        cleaned.append(row)
    return cleaned


def _safe_card_name(source_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in source_id)


def process_source(row: dict):
    if stop_event.is_set():
        return "stopped", str(row.get("source_id", "")), "STOP_EVENT_SET"

    source_id = str(row["source_id"])
    ra = float(row["ra"])
    dec = float(row["dec"])
    card_path = os.path.join(CARDS_DIR, f"{_safe_card_name(source_id)}.json")

    if os.path.exists(card_path):
        with stats_lock:
            stats["skipped"] += 1
        return "skipped", source_id, None

    last_error = None
    for attempt in range(MAX_RETRIES):
        if stop_event.is_set():
            return "stopped", source_id, "STOP_EVENT_SET"
        try:
            wise_data = query_wise(ra, dec)
            card = {
                "source_id": source_id,
                "ra": ra,
                "dec": dec,
                "z": float(row.get("z", 0) or 0),
                "label": str(row.get("label", "")),
                "reference": str(row.get("reference", "")),
                "wise_epochs": wise_data or [],
                "n_epochs": len(wise_data) if wise_data else 0,
                "fetched_at": _utc_now_iso(),
            }
            with open(card_path, "w", encoding="utf-8") as f:
                json.dump(card, f)
            with stats_lock:
                stats["done"] += 1
            return "success", source_id, None
        except Exception as e:  # noqa: BLE001
            last_error = str(e)
            if "RATE_LIMITED" in last_error:
                with print_lock:
                    print(f"\n  [RATE LIMITED] Pausing all workers {RATE_LIMIT_WAIT}s...")
                time.sleep(RATE_LIMIT_WAIT)
                continue
            wait = BACKOFF_BASE * (2 ** attempt)
            time.sleep(wait)

    with stats_lock:
        stats["failed"] += 1
    return "failed", source_id, last_error


def log_failure(source_id: str, error: str, writer: csv.DictWriter):
    writer.writerow(
        {"source_id": source_id, "error": error, "timestamp": _utc_now_iso()}
    )


def checkpoint(n_total: int):
    elapsed = time.time() - stats["start"]
    done_total = stats["done"] + stats["skipped"]
    rate = done_total / elapsed if elapsed > 0 else 0
    remaining = n_total - done_total
    eta_sec = remaining / rate if rate > 0 else 0
    eta = str(timedelta(seconds=int(eta_sec)))
    cards_gb = _cards_dir_size_gb()
    progress = {
        "completed": done_total,
        "succeeded": stats["done"],
        "failed": stats["failed"],
        "skipped": stats["skipped"],
        "total": n_total,
        "elapsed_sec": int(elapsed),
        "eta": eta,
        "cards_dir_size_gb": round(cards_gb, 3),
        "timestamp": _utc_now_iso(),
    }
    with open(PROGRESS_LOG, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)
    return elapsed, eta, cards_gb


def _submit_next(executor: ThreadPoolExecutor, rows_iter, futures_map: dict) -> bool:
    try:
        row = next(rows_iter)
    except StopIteration:
        return False
    fut = executor.submit(process_source, row)
    futures_map[fut] = row.get("source_id", "")
    return True


def main() -> None:
    df = pd.read_csv(BENCHMARK_CSV, low_memory=False)
    n_total = len(df)
    print(f"Loaded {n_total:,} sources from {BENCHMARK_CSV}")
    print(f"Cards directory: {CARDS_DIR}")
    print(f"Workers: {N_WORKERS} | Timeout: {TIMEOUT_SEC}s | Max retries: {MAX_RETRIES}")
    print(f"Starting at {_utc_now_iso()}")
    print("-" * 60)
    checkpoint(n_total)

    disk_limit_error = None
    processed = 0
    rows_iter = iter(df.to_dict(orient="records"))
    last_heartbeat = time.time()
    with open(FAILED_LOG, "w", newline="", encoding="utf-8") as fail_file:
        writer = csv.DictWriter(fail_file, fieldnames=["source_id", "error", "timestamp"])
        writer.writeheader()

        executor = ThreadPoolExecutor(max_workers=N_WORKERS)
        futures = {}
        for _ in range(min(MAX_INFLIGHT, n_total)):
            if not _submit_next(executor, rows_iter, futures):
                break

        if futures:
            print(f"Queued initial batch: {len(futures)} in-flight requests")
        try:
            while futures:
                done_set, _ = wait(futures.keys(), timeout=5, return_when=FIRST_COMPLETED)
                if not done_set:
                    now = time.time()
                    if now - last_heartbeat >= HEARTBEAT_EVERY_SEC:
                        elapsed, eta, cards_gb = checkpoint(n_total)
                        h, m = divmod(int(elapsed), 3600)
                        m //= 60
                        in_flight = len(futures)
                        with print_lock:
                            print(
                                f"[heartbeat] processed={processed:,}/{n_total:,} "
                                f"in_flight={in_flight} done={stats['done']:,} "
                                f"skipped={stats['skipped']:,} failed={stats['failed']:,} "
                                f"cards={cards_gb:.2f}GB elapsed={h}h{m:02d}m ETA={eta}"
                            )
                        last_heartbeat = now
                    continue
                for future in done_set:
                    futures.pop(future, None)
                    processed += 1

                    status, source_id, error = future.result()
                    if status == "failed" and error:
                        log_failure(source_id, error, writer)

                    if not stop_event.is_set():
                        _submit_next(executor, rows_iter, futures)

                    if processed % LOG_EVERY == 0:
                        elapsed, eta, cards_gb = checkpoint(n_total)
                        h, m = divmod(int(elapsed), 3600)
                        m //= 60
                        with print_lock:
                            print(
                                f"[{processed:,}/{n_total:,}] done={stats['done']:,} "
                                f"skipped={stats['skipped']:,} failed={stats['failed']:,} "
                                f"cards={cards_gb:.2f}GB elapsed={h}h{m:02d}m ETA={eta}"
                            )

                    if processed % CHECKPOINT_EVERY == 0:
                        checkpoint(n_total)

                    if processed % DISK_CHECK_EVERY == 0:
                        _, _, cards_gb = checkpoint(n_total)
                        if cards_gb > MAX_CARDS_GB:
                            disk_limit_error = (
                                f"DISK_LIMIT_EXCEEDED: {CARDS_DIR} is {cards_gb:.2f} GB "
                                f"(>{MAX_CARDS_GB:.2f} GB). Aborting fetch."
                            )
                            stop_event.set()
                            with print_lock:
                                print(f"\n{disk_limit_error}")
                            break

                if stop_event.is_set():
                    break
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    elapsed, _, cards_gb = checkpoint(n_total)
    h, m = divmod(int(elapsed), 3600)
    m //= 60
    attempted = stats["done"] + stats["failed"]
    wise_rate = stats["done"] / max(attempted, 1)
    print("\n" + "=" * 60)
    if disk_limit_error:
        print("FETCH ABORTED")
        print(f"  Reason:           {disk_limit_error}")
    else:
        print("FETCH COMPLETE")
    print(f"  Total attempted:  {n_total:,}")
    print(f"  Succeeded:        {stats['done']:,}")
    print(f"  Skipped (cached): {stats['skipped']:,}")
    print(f"  Failed:           {stats['failed']:,}")
    print(f"  WISE coverage:    {wise_rate:.1%}")
    print(f"  Cards size:       {cards_gb:.2f} GB")
    print(f"  Elapsed:          {h}h {m:02d}m")
    print(f"  Failed log:       {FAILED_LOG}")
    print(f"  Progress log:     {PROGRESS_LOG}")
    print("=" * 60)
    print("DO NOT run evaluate_benchmark.py — fetch only")
    if disk_limit_error:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
