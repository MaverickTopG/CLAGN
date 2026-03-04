#!/usr/bin/env python3
"""Build per-source WISE cards from locally downloaded HEALPix parquet files.

No IRSA requests are made in this script.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import ICRS, SkyCoord, search_around_sky
from astropy_healpix import HEALPix

WISE_COLUMNS = [
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
    p = argparse.ArgumentParser(description="Build cards from bulk NEOWISE parquet")
    p.add_argument("--benchmark", required=True, help="Benchmark CSV")
    p.add_argument("--wise_dir", required=True, help="Directory with healpix_*.parquet")
    p.add_argument("--output", required=True, help="Output cards directory")
    p.add_argument("--match_radius_arcsec", type=float, default=3.0, help="Crossmatch radius")
    p.add_argument("--nside", type=int, default=None, help="HEALPix nside (auto from manifest.json if omitted)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing cards")
    p.add_argument("--progress_every", type=int, default=500, help="Print progress every N sources")
    p.add_argument("--cell_cache", type=int, default=32, help="Cell parquet LRU cache size")
    return p.parse_args()


def safe_name(source_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", source_id)


def infer_nside(wise_dir: Path, nside_flag: int | None) -> int:
    if nside_flag is not None:
        return nside_flag
    manifest = wise_dir / "manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            if "nside" in data:
                return int(data["nside"])
        except Exception:
            pass
    return 32


def to_epoch_records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    out = df[WISE_COLUMNS].copy()
    out = out.replace({np.nan: None})
    return out.to_dict(orient="records")


def main() -> None:
    args = parse_args()

    benchmark_path = Path(args.benchmark)
    wise_dir = Path(args.wise_dir)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(benchmark_path, low_memory=False)
    required = {"source_id", "ra", "dec"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns in benchmark: {missing}")

    nside = infer_nside(wise_dir, args.nside)
    hp = HEALPix(nside=nside, order="ring", frame=ICRS())

    ra = pd.to_numeric(df["ra"], errors="coerce").to_numpy(dtype=float)
    dec = pd.to_numeric(df["dec"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(ra) & np.isfinite(dec)
    if not np.all(valid):
        bad = int((~valid).sum())
        raise SystemExit(f"Found {bad} rows with invalid ra/dec; fix benchmark first")

    pix = hp.lonlat_to_healpix(ra * u.deg, dec * u.deg)
    df = df.copy()
    df["_cell"] = np.asarray(pix, dtype=int)

    @lru_cache(maxsize=args.cell_cache)
    def load_cell(cell_id: int) -> pd.DataFrame:
        p = wise_dir / f"healpix_{int(cell_id)}.parquet"
        if not p.exists():
            return pd.DataFrame(columns=WISE_COLUMNS)
        x = pd.read_parquet(p)
        for col in WISE_COLUMNS:
            if col not in x.columns:
                x[col] = np.nan
        for col in ["ra", "dec", "mjd", "w1mpro", "w1sigmpro", "w2mpro", "w2sigmpro", "qi_fact", "saa_sep"]:
            x[col] = pd.to_numeric(x[col], errors="coerce")
        x = x.dropna(subset=["ra", "dec", "mjd"]).copy()
        x.sort_values("mjd", inplace=True)
        x.reset_index(drop=True, inplace=True)
        return x[WISE_COLUMNS]

    total = len(df)
    written = 0
    skipped = 0
    start = time.time()
    r = args.match_radius_arcsec * u.arcsec

    grouped = df.groupby("_cell", sort=True).indices

    print(f"Benchmark rows: {total:,}")
    print(f"Using nside={nside} | match radius={args.match_radius_arcsec} arcsec")
    print(f"Occupied cells: {len(grouped):,}")

    for cell, row_idx in grouped.items():
        row_idx = np.asarray(row_idx, dtype=int)
        block = df.iloc[row_idx].copy()

        neigh = hp.neighbours(int(cell))
        neigh = np.asarray(neigh, dtype=int).ravel()
        candidate_cells = [int(cell)] + [int(x) for x in neigh.tolist() if int(x) >= 0]
        candidate_cells = sorted(set(candidate_cells))

        frames = [load_cell(c) for c in candidate_cells]
        frames = [f for f in frames if not f.empty]

        if frames:
            phot = pd.concat(frames, ignore_index=True)
            phot = phot.drop_duplicates(subset=["ra", "dec", "mjd", "w1mpro", "w2mpro"])
            phot_coords = SkyCoord(ra=phot["ra"].to_numpy() * u.deg, dec=phot["dec"].to_numpy() * u.deg)
            src_coords = SkyCoord(ra=block["ra"].to_numpy(dtype=float) * u.deg, dec=block["dec"].to_numpy(dtype=float) * u.deg)
            idx_src, idx_phot, _, _ = search_around_sky(src_coords, phot_coords, r)
            match_map: dict[int, list[int]] = {}
            for s, p in zip(idx_src.tolist(), idx_phot.tolist()):
                match_map.setdefault(int(s), []).append(int(p))
        else:
            phot = pd.DataFrame(columns=WISE_COLUMNS)
            match_map = {}

        for local_i, (_, row) in enumerate(block.iterrows()):
            source_id = str(row["source_id"])
            card_path = outdir / f"{safe_name(source_id)}.json"
            if card_path.exists() and not args.overwrite:
                skipped += 1
                continue

            match_ids = match_map.get(local_i, [])
            if match_ids:
                matched = phot.iloc[match_ids].copy()
                matched = matched.drop_duplicates(subset=["mjd", "ra", "dec"]).sort_values("mjd")
            else:
                matched = pd.DataFrame(columns=WISE_COLUMNS)

            card = {
                "source_id": source_id,
                "ra": float(row["ra"]),
                "dec": float(row["dec"]),
                "z": float(row.get("z", 0.0) or 0.0),
                "label": str(row.get("label", "")),
                "reference": str(row.get("reference", "")),
                "wise_epochs": to_epoch_records(matched),
                "n_epochs": int(len(matched)),
                "built_from": "bulk_local_crossmatch",
                "match_radius_arcsec": args.match_radius_arcsec,
                "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

            card_path.write_text(json.dumps(card), encoding="utf-8")
            written += 1

            processed = written + skipped
            if processed % args.progress_every == 0 or processed == total:
                elapsed = time.time() - start
                rate = processed / elapsed if elapsed > 0 else 0.0
                eta = (total - processed) / rate if rate > 0 else 0.0
                print(
                    f"[{processed:,}/{total:,}] written={written:,} skipped={skipped:,} "
                    f"rate={rate:.1f}/s eta={eta/60:.1f}m"
                )

    elapsed = time.time() - start
    print("Done")
    print(f"Written: {written:,}")
    print(f"Skipped: {skipped:,}")
    print(f"Elapsed: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
