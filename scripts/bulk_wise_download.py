#!/usr/bin/env python3
"""Bulk-download NEOWISE-R photometry by HEALPix cell.

This replaces per-source IRSA requests with one request per occupied HEALPix cell.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from astropy import units as u
from astropy.coordinates import ICRS
from astropy_healpix import HEALPix

IRSA_TAP_SYNC = "https://irsa.ipac.caltech.edu/TAP/sync"
WISE_TABLE = "neowiser_p1bs_psd"

COLUMNS = [
    "ra",
    "dec",
    "mjd",
    "w1mpro",
    "w1sigmpro",
    "w2mpro",
    "w2sigmpro",
    "qi_fact",
    "saa_sep",
    "moon_masked",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bulk NEOWISE-R downloader by HEALPix")
    p.add_argument("--benchmark", required=True, help="Benchmark CSV with ra/dec columns")
    p.add_argument("--output", required=True, help="Output directory for per-cell parquet files")
    p.add_argument("--nside", type=int, default=32, help="HEALPix nside (default: 32)")
    p.add_argument("--workers", type=int, default=16, help="Parallel workers")
    p.add_argument("--cell-radius-deg", type=float, default=1.8, help="Cone radius per cell")
    p.add_argument("--request-timeout", type=int, default=120, help="IRSA read timeout seconds")
    p.add_argument("--retries", type=int, default=4, help="Retries per cell")
    p.add_argument("--overwrite", action="store_true", help="Re-download existing cells")
    p.add_argument("--max-cells", type=int, default=None, help="Optional cap for testing")
    return p.parse_args()


def normalize_tap_json(payload: dict) -> list[dict]:
    if isinstance(payload, dict) and payload.get("metadata") and payload.get("data"):
        cols = [m.get("name") for m in payload["metadata"]]
        return [dict(zip(cols, row)) for row in payload["data"]]

    vot = payload.get("VOTABLE", {}) if isinstance(payload, dict) else {}
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
    rows: list[list[str]] = []
    if isinstance(tabledata, list):
        rows = tabledata
    elif isinstance(tabledata, dict):
        tr = tabledata.get("TR_ARRAY", [])
        if isinstance(tr, dict):
            tr = [tr]
        for row in tr:
            td = row.get("TD_ARRAY", [])
            if isinstance(td, list):
                rows.append(td)

    out = []
    for row in rows:
        if isinstance(row, list):
            out.append(dict(zip(cols, row)))
    return out


def moon_mask_clean(v) -> bool:
    if v is None:
        return True
    s = str(v).strip()
    if s == "":
        return True
    return set(s).issubset({"0"})


def to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def local_quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # numeric conversions
    to_numeric(df, ["ra", "dec", "mjd", "w1mpro", "w1sigmpro", "w2mpro", "w2sigmpro", "qi_fact", "saa_sep"])

    mask = np.isfinite(df["ra"]) & np.isfinite(df["dec"]) & np.isfinite(df["mjd"])

    if "qi_fact" in df.columns:
        mask &= pd.to_numeric(df["qi_fact"], errors="coerce") >= 1
    if "saa_sep" in df.columns:
        mask &= pd.to_numeric(df["saa_sep"], errors="coerce") >= 5
    if "w1sigmpro" in df.columns:
        mask &= pd.to_numeric(df["w1sigmpro"], errors="coerce") < 0.2
    if "moon_masked" in df.columns:
        mask &= df["moon_masked"].map(moon_mask_clean)

    cleaned = df.loc[mask, COLUMNS].copy()
    cleaned.sort_values("mjd", inplace=True)
    cleaned.reset_index(drop=True, inplace=True)
    return cleaned


def build_adql(ra_deg: float, dec_deg: float, radius_deg: float) -> str:
    return (
        "SELECT ra, dec, mjd, w1mpro, w1sigmpro, w2mpro, w2sigmpro, qi_fact, saa_sep, moon_masked "
        f"FROM {WISE_TABLE} "
        "WHERE CONTAINS(POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {ra_deg:.7f}, {dec_deg:.7f}, {radius_deg:.7f})) = 1 "
        "AND qi_fact >= 1 AND saa_sep >= 5 "
        "AND w1sigmpro IS NOT NULL AND w1sigmpro < 0.2 "
        "AND (moon_masked IS NULL OR moon_masked='0' OR moon_masked='00' OR moon_masked='0000') "
        "ORDER BY mjd"
    )


def query_cell(ra_deg: float, dec_deg: float, radius_deg: float, timeout_sec: int, retries: int) -> tuple[pd.DataFrame, int]:
    adql = build_adql(ra_deg, dec_deg, radius_deg)
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(
                IRSA_TAP_SYNC,
                params={"QUERY": adql, "FORMAT": "json", "LANG": "ADQL"},
                timeout=(15, timeout_sec),
            )
            if resp.status_code == 429:
                raise RuntimeError("IRSA rate limited (429)")
            resp.raise_for_status()
            rows = normalize_tap_json(resp.json())
            raw_count = len(rows)
            if raw_count == 0:
                return pd.DataFrame(columns=COLUMNS), 0

            df = pd.DataFrame(rows)
            for col in COLUMNS:
                if col not in df.columns:
                    df[col] = np.nan
            df = local_quality_filter(df)
            return df, raw_count
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            if attempt < retries:
                time.sleep(min(60, 2 ** attempt))

    raise RuntimeError(last_err or "unknown IRSA error")


def main() -> None:
    args = parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    progress_path = outdir / "progress.json"
    failures_path = outdir / "failures.csv"
    manifest_path = outdir / "manifest.json"

    df = pd.read_csv(args.benchmark, low_memory=False)
    required = {"ra", "dec"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns in benchmark: {missing}")

    hp = HEALPix(nside=args.nside, order="ring", frame=ICRS())
    lon = np.asarray(df["ra"], dtype=float) * u.deg
    lat = np.asarray(df["dec"], dtype=float) * u.deg
    pix = hp.lonlat_to_healpix(lon, lat)
    cells = sorted(set(int(x) for x in np.asarray(pix).tolist()))
    if args.max_cells is not None:
        cells = cells[: args.max_cells]

    lon_c, lat_c = hp.healpix_to_lonlat(np.asarray(cells, dtype=int))
    centers = {
        int(cell): (float(lon_c[i].to_value(u.deg)), float(lat_c[i].to_value(u.deg)))
        for i, cell in enumerate(cells)
    }

    jobs = []
    skipped = 0
    for cell in cells:
        path = outdir / f"healpix_{cell}.parquet"
        if path.exists() and not args.overwrite:
            skipped += 1
            continue
        jobs.append(cell)

    total = len(cells)
    print(f"Benchmark rows: {len(df):,}")
    print(f"Occupied HEALPix cells (nside={args.nside}): {total:,}")
    print(f"Cell radius: {args.cell_radius_deg:.3f} deg")
    print(f"Workers: {args.workers} | To query: {len(jobs):,} | Skipping existing: {skipped:,}")

    completed = skipped
    success = 0
    failed = 0
    raw_rows = 0
    clean_rows = 0

    with failures_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["cell_id", "ra", "dec", "error"])
        writer.writeheader()

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            fut_map = {}
            for cell in jobs:
                ra_c, dec_c = centers[cell]
                fut = ex.submit(query_cell, ra_c, dec_c, args.cell_radius_deg, args.request_timeout, args.retries)
                fut_map[fut] = (cell, ra_c, dec_c)

            for i, fut in enumerate(as_completed(fut_map), 1):
                cell, ra_c, dec_c = fut_map[fut]
                out_path = outdir / f"healpix_{cell}.parquet"
                try:
                    cell_df, raw_count = fut.result()
                    cell_df.to_parquet(out_path, index=False)
                    success += 1
                    raw_rows += int(raw_count)
                    clean_rows += int(len(cell_df))
                except Exception as e:  # noqa: BLE001
                    failed += 1
                    writer.writerow({"cell_id": cell, "ra": ra_c, "dec": dec_c, "error": str(e)})

                completed += 1
                if completed % 25 == 0 or completed == total:
                    progress = {
                        "benchmark": args.benchmark,
                        "nside": args.nside,
                        "total_cells": total,
                        "completed_cells": completed,
                        "success_cells": success,
                        "failed_cells": failed,
                        "skipped_cells": skipped,
                        "raw_rows": raw_rows,
                        "clean_rows": clean_rows,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
                    print(
                        f"[{completed:,}/{total:,}] success={success:,} failed={failed:,} "
                        f"skipped={skipped:,} raw_rows={raw_rows:,} clean_rows={clean_rows:,}"
                    )

    manifest = {
        "benchmark": args.benchmark,
        "output": str(outdir),
        "tap_endpoint": IRSA_TAP_SYNC,
        "wise_table": WISE_TABLE,
        "nside": args.nside,
        "cell_radius_deg": args.cell_radius_deg,
        "total_cells": total,
        "success_cells": success,
        "failed_cells": failed,
        "skipped_cells": skipped,
        "raw_rows": raw_rows,
        "clean_rows": clean_rows,
        "failures_csv": str(failures_path),
        "progress_json": str(progress_path),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Done")
    print(f"Manifest: {manifest_path}")
    print(f"Failures: {failures_path}")


if __name__ == "__main__":
    main()
