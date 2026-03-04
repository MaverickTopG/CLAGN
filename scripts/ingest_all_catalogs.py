#!/usr/bin/env python3
"""
Ingest all downloaded raw catalogs into a single normalized benchmark CSV.
Run AFTER all Part 1 downloads complete.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u

OUT = Path("data/benchmark/benchmark_master_v3.csv")
COUNTS_OUT = Path("logs/ingest_catalog_counts.json")
DEDUP_RADIUS = 5.0  # arcsec
NORMAL_AGN_TARGET = 150_000


def _safe_float(v, default=np.nan):
    try:
        if v is None or (isinstance(v, str) and not v.strip()):
            return default
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _parse_ra_dec(ra_v, dec_v) -> tuple[float, float]:
    """Parse RA/Dec from decimal or sexagesimal strings."""
    ra = _safe_float(ra_v)
    dec = _safe_float(dec_v)
    if np.isfinite(ra) and np.isfinite(dec):
        return float(ra), float(dec)
    try:
        c = SkyCoord(str(ra_v), str(dec_v), unit=(u.hourangle, u.deg), frame="icrs")
        return float(c.ra.deg), float(c.dec.deg)
    except Exception:
        return np.nan, np.nan


def _read_tsv_loose(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    # Try tab first, then pipe/whitespace fallback.
    for kwargs in (
        {"sep": "\t", "comment": "#"},
        {"sep": "|", "comment": "#"},
        {"delim_whitespace": True, "comment": "#"},
    ):
        try:
            df = pd.read_csv(path, **kwargs)
            if len(df.columns) > 1:
                return df
        except Exception:
            continue
    return pd.read_csv(path, sep="\t", comment="#", engine="python")


def _parse_osc_entry(name: str, entry: dict) -> tuple[float, float, float] | None:
    def _extract(field: str):
        val = entry.get(field)
        if isinstance(val, list) and val:
            first = val[0]
            if isinstance(first, dict):
                return first.get("value")
            return first
        if isinstance(val, dict):
            return val.get("value")
        return val

    ra = _safe_float(_extract("ra"))
    dec = _safe_float(_extract("dec"))
    z = _safe_float(_extract("redshift"))
    if np.isnan(ra) or np.isnan(dec) or np.isnan(z):
        return None
    return ra, dec, z


def _sample_by_z_bins(df: pd.DataFrame, z_col: str, bins: list[tuple[float, float, int]], seed: int) -> pd.DataFrame:
    parts = []
    for zlo, zhi, n in bins:
        sub = df[(df[z_col] >= zlo) & (df[z_col] < zhi)]
        if len(sub) == 0:
            continue
        parts.append(sub.sample(min(n, len(sub)), random_state=seed))
    if not parts:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    rows: list[dict] = []
    counts: dict[str, int] = {}
    Path("logs").mkdir(parents=True, exist_ok=True)
    mq_pool = pd.DataFrame(columns=["source_id", "ra", "dec", "z"])
    dr16_pool = pd.DataFrame(columns=["source_id", "ra", "dec", "z"])

    # 1) Milliquas v8
    print("Ingesting Milliquas v8...")
    try:
        p_csv = Path("data/raw/milliquas_v8.csv")
        p_tsv = Path("data/raw/milliquas_vizier.tsv")
        if p_tsv.exists():
            mq = _read_tsv_loose(p_tsv)
            mq = mq.rename(
                columns={
                    "Name": "name",
                    "RAJ2000": "ra",
                    "DEJ2000": "dec",
                    "Type": "type",
                    "Rmag": "rmag",
                }
            )
        elif p_csv.exists():
            mq = pd.read_csv(
                p_csv,
                header=None,
                names=[
                    "name", "ra", "dec", "type", "rmag", "bmag",
                    "comment", "z", "cite", "zcite", "qpct", "xname",
                    "rname", "2rass",
                ],
            )
            mq["name"] = mq["name"].astype(str)
            mq["type"] = mq["type"].astype(str)
        else:
            raise FileNotFoundError("Neither data/raw/milliquas_v8.csv nor data/raw/milliquas_vizier.tsv found")

        mq["z"] = pd.to_numeric(mq.get("z"), errors="coerce")
        mq["ra"] = pd.to_numeric(mq.get("ra"), errors="coerce")
        mq["dec"] = pd.to_numeric(mq.get("dec"), errors="coerce")
        mq["type"] = mq.get("type", "").astype(str)
        mq = mq[mq["type"].str.contains("Q|A|K|N|q|a", regex=True, na=False)]
        mq = mq[(mq["z"] > 0.01) & (mq["z"] < 3.0)].dropna(subset=["ra", "dec"])

        z_bins = [
            (0.0, 0.3, 30000),
            (0.3, 0.6, 35000),
            (0.6, 1.0, 30000),
            (1.0, 2.0, 35000),
            (2.0, 3.0, 20000),
        ]
        mq_sample = _sample_by_z_bins(mq, "z", z_bins, seed=42)
        for _, r in mq_sample.iterrows():
            rows.append({
                "source_id": str(r["name"]),
                "ra": float(r["ra"]),
                "dec": float(r["dec"]),
                "z": float(r["z"]),
                "label": "normal_agn",
                "reference": "Flesch+2023_Milliquas8",
                "evidence_tier": 3,
            })
        mq_pool = pd.DataFrame(
            {
                "source_id": mq["name"].astype(str),
                "ra": mq["ra"].astype(float),
                "dec": mq["dec"].astype(float),
                "z": mq["z"].astype(float),
            }
        )
        counts["milliquas"] = int(len(mq_sample))
        print(f"  Milliquas: {len(mq_sample):,} normal AGN ingested")
    except Exception as e:
        counts["milliquas"] = 0
        print(f"  Milliquas FAILED: {e}")

    # 2) BZCAT
    print("Ingesting BZCAT 5th Edition...")
    try:
        bz = _read_tsv_loose(Path("data/raw/bzcat5.tsv"))
        n_added = 0
        for _, r in bz.iterrows():
            ra, dec = _parse_ra_dec(r.get("RAJ2000", r.get("ra")), r.get("DEJ2000", r.get("dec")))
            if np.isnan(ra) or np.isnan(dec):
                continue
            rows.append({
                "source_id": str(r.get("Name", r.get("name", ""))),
                "ra": float(ra),
                "dec": float(dec),
                "z": _safe_float(r.get("z", r.get("Redshift", 0.0)), default=0.0),
                "label": "blazar",
                "reference": "Massaro+2015_BZCAT5",
                "evidence_tier": 1,
            })
            n_added += 1
        counts["bzcat"] = int(n_added)
        print(f"  BZCAT: {n_added:,} blazars ingested")
    except Exception as e:
        counts["bzcat"] = 0
        print(f"  BZCAT FAILED: {e}")

    # 3) Fermi 4FGL-DR4
    print("Ingesting Fermi 4FGL-DR4...")
    try:
        with fits.open("data/raw/4fgl_dr4.fit") as h:
            d = h[1].data
            cls = np.array([str(x).strip().lower() for x in d["CLASS1"]])
            mask = np.isin(cls, ["bll", "fsrq", "bcu"])
            if "ASSOC_PROB_BAY" in d.columns.names:
                mask &= np.asarray(d["ASSOC_PROB_BAY"], dtype=float) > 0.8
            d = d[mask]
            n_added = 0
            for row in d:
                z = _safe_float(row["Redshift"], default=0.0) if "Redshift" in d.columns.names else 0.0
                rows.append({
                    "source_id": str(row["Source_Name"]).strip(),
                    "ra": float(row["RAJ2000"]),
                    "dec": float(row["DEJ2000"]),
                    "z": z if z > 0 else 0.0,
                    "label": "blazar",
                    "reference": "Abdollahi+2022_4FGL_DR4",
                    "evidence_tier": 1,
                })
                n_added += 1
        counts["fermi_4fgl"] = int(n_added)
        print(f"  4FGL: {n_added:,} blazars ingested")
    except Exception as e:
        counts["fermi_4fgl"] = 0
        print(f"  4FGL FAILED: {e}")

    # 4) CLAGN tables
    print("Ingesting CLAGN literature tables...")
    clagn_files = [
        ("data/raw/macleod2019.tsv", "MacLeod+2019", 1, "turn_off"),
        ("data/raw/sheng2020.tsv", "Sheng+2020", 1, "mixed"),
        ("data/raw/hon2022.tsv", "Hon+2022", 1, "mixed"),
        ("data/raw/ward2024.tsv", "Ward+2024", 1, "mixed"),
    ]
    for fpath, ref, tier, trans in clagn_files:
        p = Path(fpath)
        key = f"clagn_{ref.lower().replace('+', '_').replace(' ', '_')}"
        if not p.exists():
            counts[key] = 0
            print(f"  SKIP {fpath} — not downloaded")
            continue
        try:
            df = _read_tsv_loose(p)
            n_added = 0
            for _, r in df.iterrows():
                ra, dec = _parse_ra_dec(
                    r.get("RAJ2000", r.get("ra", r.get("RA"))),
                    r.get("DEJ2000", r.get("dec", r.get("Dec"))),
                )
                z = _safe_float(r.get("z", r.get("Redshift", r.get("Z", 0.0))), default=0.0)
                if np.isnan(ra) or np.isnan(dec):
                    continue
                rows.append({
                    "source_id": str(r.get("Name", r.get("name", f"{ra:.4f}_{dec:.4f}"))),
                    "ra": float(ra),
                    "dec": float(dec),
                    "z": float(z),
                    "label": "CLAGN",
                    "reference": ref,
                    "evidence_tier": tier,
                    "transition_type": trans,
                })
                n_added += 1
            counts[key] = int(n_added)
            print(f"  {ref}: {n_added:,} CLAGN ingested")
        except Exception as e:
            counts[key] = 0
            print(f"  {ref} FAILED: {e}")

    # Optional CLAGN fallback from existing local benchmark if literature tables are sparse/unavailable.
    clagn_count_now = sum(1 for r in rows if r.get("label") == "CLAGN")
    if clagn_count_now == 0:
        fallback = Path("data/real_clagn/benchmark_master.csv")
        if fallback.exists():
            try:
                fb = pd.read_csv(fallback)
                fb = fb[fb.get("class", "").astype(str).str.lower() == "clagn"].copy()
                n_fb = 0
                for _, r in fb.iterrows():
                    ra = _safe_float(r.get("ra"))
                    dec = _safe_float(r.get("dec"))
                    z = _safe_float(r.get("redshift", r.get("z", 0.0)), default=0.0)
                    if np.isnan(ra) or np.isnan(dec):
                        continue
                    rows.append({
                        "source_id": str(r.get("source_id", f"clagn_fb_{n_fb}")),
                        "ra": float(ra),
                        "dec": float(dec),
                        "z": float(z),
                        "label": "CLAGN",
                        "reference": str(r.get("label_source", "local_real_clagn_benchmark")),
                        "evidence_tier": 1,
                        "transition_type": "mixed",
                    })
                    n_fb += 1
                counts["clagn_fallback_local"] = int(n_fb)
                print(f"  CLAGN fallback: {n_fb:,} from data/real_clagn/benchmark_master.csv")
            except Exception as e:
                counts["clagn_fallback_local"] = 0
                print(f"  CLAGN fallback FAILED: {e}")
        else:
            counts["clagn_fallback_local"] = 0

    # 5) OSC supernovae
    print("Ingesting Open Supernova Catalog...")
    sn_type_limits = {
        "IIn": 200, "IIP": 200, "Ia": 200, "Ib": 80, "Ic": 80,
        "IIb": 80, "SLSN-I": 80, "LBV": 50, "Ibn": 50,
    }
    for sn_type, n_limit in sn_type_limits.items():
        fname = Path(f"data/raw/osc_{sn_type.replace('-', '')}.json")
        if not fname.exists():
            fname = Path(f"data/raw/osc_{sn_type}.json")
        key = f"osc_{sn_type}"
        if not fname.exists():
            counts[key] = 0
            print(f"  SKIP OSC {sn_type} — file not found")
            continue
        try:
            data = json.loads(fname.read_text())
            n_added = 0
            # OSC API often returns dict keyed by name.
            items = list(data.items()) if isinstance(data, dict) else []
            for name, entry in items[: n_limit * 3]:
                parsed = _parse_osc_entry(str(name), entry if isinstance(entry, dict) else {})
                if parsed is None:
                    continue
                ra, dec, z = parsed
                if z <= 0 or z > 0.3:
                    continue
                rows.append({
                    "source_id": str(name),
                    "ra": float(ra),
                    "dec": float(dec),
                    "z": float(z),
                    "label": "supernova",
                    "sn_type": sn_type,
                    "reference": "OSC_Guillochon+2017",
                    "evidence_tier": 2,
                })
                n_added += 1
                if n_added >= n_limit:
                    break
            counts[key] = int(n_added)
            print(f"  OSC {sn_type}: {n_added} SNe ingested")
        except Exception as e:
            counts[key] = 0
            print(f"  OSC {sn_type} FAILED: {e}")

    # 6) CRTS
    print("Ingesting CRTS variable AGN...")
    try:
        crts = _read_tsv_loose(Path("data/raw/crts_agn.tsv"))
        n_added = 0
        for _, r in crts.iterrows():
            ra, dec = _parse_ra_dec(r.get("RAJ2000", r.get("ra")), r.get("DEJ2000", r.get("dec")))
            if np.isnan(ra) or np.isnan(dec):
                continue
            rows.append({
                "source_id": str(r.get("Name", r.get("name", ""))),
                "ra": float(ra),
                "dec": float(dec),
                "z": _safe_float(r.get("z", 0.0), default=0.0),
                "label": "normal_agn_variable",
                "reference": "Graham+2017_CRTS",
                "evidence_tier": 2,
            })
            n_added += 1
        counts["crts"] = int(n_added)
        print(f"  CRTS: {n_added:,} hard negatives ingested")
    except Exception as e:
        counts["crts"] = 0
        print(f"  CRTS FAILED: {e}")

    # 7) DR16Q
    print("Ingesting SDSS DR16Q stratified sample...")
    try:
        p_tsv = Path("data/raw/dr16q_vizier.tsv")
        p_fits = Path("data/raw/DR16Q_v4.fits")
        if p_tsv.exists():
            df = _read_tsv_loose(p_tsv).rename(
                columns={"SDSS": "source_id", "RAJ2000": "ra", "DEJ2000": "dec"}
            )
            df["source_id"] = df["source_id"].astype(str)
            df["ra"] = pd.to_numeric(df["ra"], errors="coerce")
            df["dec"] = pd.to_numeric(df["dec"], errors="coerce")
            df["z"] = pd.to_numeric(df["z"], errors="coerce")
        elif p_fits.exists():
            with fits.open(p_fits) as h:
                d = h[1].data
                df = pd.DataFrame(
                    {
                        "source_id": [str(x) for x in d["SDSS_NAME"]],
                        "ra": d["RA"].astype(float),
                        "dec": d["DEC"].astype(float),
                        "z": d["Z"].astype(float),
                    }
                )
        else:
            raise FileNotFoundError("Neither data/raw/DR16Q_v4.fits nor data/raw/dr16q_vizier.tsv found")

        df = df[(df["z"] > 0.01) & (df["z"] < 3.0)]
        z_bins = [(0.0, 0.3, 5000), (0.3, 0.6, 6000), (0.6, 1.0, 5000), (1.0, 2.0, 6000), (2.0, 5.0, 3000)]
        dr16_sample = _sample_by_z_bins(df, "z", z_bins, seed=99)
        for _, r in dr16_sample.iterrows():
            rows.append({
                "source_id": str(r["source_id"]),
                "ra": float(r["ra"]),
                "dec": float(r["dec"]),
                "z": float(r["z"]),
                "label": "normal_agn",
                "reference": "Lyke+2020_DR16Q",
                "evidence_tier": 3,
            })
        dr16_pool = df[["source_id", "ra", "dec", "z"]].copy()
        counts["dr16q"] = int(len(dr16_sample))
        print(f"  DR16Q: {len(dr16_sample):,} normal AGN ingested")
    except Exception as e:
        counts["dr16q"] = 0
        print(f"  DR16Q FAILED: {e}")

    # Optional top-up to ensure large benchmark cardinality
    current_normal = sum(1 for r in rows if r.get("label") == "normal_agn")
    needed = max(0, NORMAL_AGN_TARGET - current_normal)
    if needed > 0:
        print(f"Top-up normal_agn pool by {needed:,} rows to target {NORMAL_AGN_TARGET:,}...")
        used_ids = {str(r.get("source_id", "")) for r in rows if r.get("label") == "normal_agn"}
        extras = []
        for pool_df, ref, seed in (
            (mq_pool, "Flesch+2023_Milliquas8", 123),
            (dr16_pool, "Lyke+2020_DR16Q", 456),
        ):
            if needed <= 0 or pool_df.empty:
                continue
            pool = pool_df[~pool_df["source_id"].astype(str).isin(used_ids)].copy()
            if pool.empty:
                continue
            take = min(needed, len(pool))
            pick = pool.sample(take, random_state=seed)
            for _, r in pick.iterrows():
                extras.append({
                    "source_id": str(r["source_id"]),
                    "ra": float(r["ra"]),
                    "dec": float(r["dec"]),
                    "z": float(r["z"]),
                    "label": "normal_agn",
                    "reference": ref,
                    "evidence_tier": 3,
                })
            used_ids.update(pick["source_id"].astype(str).tolist())
            needed -= take
        rows.extend(extras)
        counts["normal_agn_topup"] = int(len(extras))
        print(f"  Top-up added: {len(extras):,}")
    else:
        counts["normal_agn_topup"] = 0

    print(f"\nTotal before dedup: {len(rows):,}")
    counts["total_before_dedup"] = int(len(rows))

    if rows:
        df_all = pd.DataFrame(rows)
        df_all = df_all.dropna(subset=["ra", "dec"])
        df_all = df_all[(df_all["ra"] >= 0) & (df_all["ra"] <= 360)]
        df_all = df_all[(df_all["dec"] >= -90) & (df_all["dec"] <= 90)]
        df_all = df_all.sort_values("evidence_tier")

        if len(df_all) > 0:
            coords = SkyCoord(ra=df_all["ra"].to_numpy() * u.deg, dec=df_all["dec"].to_numpy() * u.deg)
            # Efficient all-pairs within radius. Since df_all is sorted by evidence_tier,
            # we drop later rows when a close earlier row exists.
            idx_i, idx_j, _, _ = coords.search_around_sky(coords, DEDUP_RADIUS * u.arcsec)
            dup_later = idx_j[idx_i < idx_j]
            keep = np.ones(len(df_all), dtype=bool)
            if len(dup_later) > 0:
                keep[np.unique(dup_later)] = False
            df_dedup = df_all[keep].reset_index(drop=True)
        else:
            df_dedup = df_all.copy()
    else:
        df_dedup = pd.DataFrame(
            columns=[
                "source_id", "ra", "dec", "z", "label",
                "reference", "evidence_tier", "transition_type", "sn_type",
            ]
        )
    counts["total_after_dedup"] = int(len(df_dedup))
    print(f"Total after dedup:  {len(df_dedup):,}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df_dedup.to_csv(OUT, index=False)
    print(f"\nSaved to {OUT}")
    print("\nClass breakdown:")
    print(df_dedup["label"].value_counts().to_string())
    print(f"\nTotal: {len(df_dedup):,} sources")

    COUNTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    COUNTS_OUT.write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(f"\nWrote catalog row counts to {COUNTS_OUT}")


if __name__ == "__main__":
    main()
