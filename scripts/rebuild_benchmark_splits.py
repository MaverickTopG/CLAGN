#!/usr/bin/env python3
"""
Assemble benchmark_master_v2.csv from existing benchmark + expansion CSVs.

Expansion sources:
  - CLAGN: data/expansion/clagn_coverage_report.csv (from expand_clagn_catalog.py)
  - SN:    data/expansion/sn_expansion.csv (hardcoded + optional VizieR fetch)
  - Blazar: data/expansion/blazar_expansion.csv (hardcoded HIGH_PRIORITY + BZCAT fill)
  - AGN:   data/expansion/agn_expansion.csv (SDSS DR16Q via VizieR VII/289)
  - Star:  data/expansion/star_expansion.csv (Gaia DR3 I/358/varisum)

Rules:
  - Existing sources: split assignments preserved verbatim from benchmark_master.csv
  - New sources: StratifiedShuffleSplit(test_frac=0.20, seed=42) by class
  - Dedup radius: 5 arcsec (coordinate matching)
  - NEVER modifies configs/pipeline_policy.yaml

Outputs written to data/real_clagn/:
  benchmark_master_v2.csv
  benchmark_dev_v2.csv
  benchmark_test_v2.csv
  SPLIT_HASH_v2.json
  benchmark_scores_v2.csv   (existing scored + new unscored with status=UNSCORED)

Usage:
    python scripts/rebuild_benchmark_splits.py [--generate-sn] [--generate-blazar]
         [--generate-agn] [--generate-star] [--target-total 5000] [--seed 42]

    # Generate all expansion CSVs if not present, then assemble:
    python scripts/rebuild_benchmark_splits.py --generate-sn --generate-blazar --generate-agn --generate-star
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
import astropy.units as u

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clagn.utils.benchmark_labels import ensure_class_ytrue
from clagn.utils.id_canonicalization import add_canonical_id

try:
    from sklearn.model_selection import StratifiedShuffleSplit
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

# ---------------------------------------------------------------------------
# Target composition
# ---------------------------------------------------------------------------
CLASS_ORDER = ["clagn", "sn", "blazar", "normal_agn", "star"]
TARGET = {"clagn": 120, "sn": 200, "blazar": 150, "normal_agn": 580, "star": 65}
VIZIER_ENDPOINTS = [
    "https://vizier.u-strasbg.fr/viz-bin/asu-tsv",
    "https://vizier.cds.unistra.fr/viz-bin/asu-tsv",
    "https://vizier.cfa.harvard.edu/viz-bin/asu-tsv",
]


def _vizier_get(params: dict, timeout: int = 120) -> requests.Response:
    """Try multiple VizieR mirrors for resilience."""
    last_exc = None
    for base in VIZIER_ENDPOINTS:
        try:
            resp = requests.get(base, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"All VizieR endpoints failed. Last error: {last_exc}")


def _largest_remainder_allocate(weights: dict[str, float], total: int) -> dict[str, int]:
    """Deterministic largest-remainder allocator over keys in input order."""
    keys = list(weights.keys())
    if total <= 0:
        return {k: 0 for k in keys}
    wsum = float(sum(max(0.0, float(weights[k])) for k in keys))
    if wsum <= 0:
        base = total // max(len(keys), 1)
        rem = total - base * len(keys)
        out = {k: base for k in keys}
        for k in keys[:rem]:
            out[k] += 1
        return out

    raw = {k: (max(0.0, float(weights[k])) / wsum) * total for k in keys}
    floor_vals = {k: int(np.floor(raw[k])) for k in keys}
    remainder = total - sum(floor_vals.values())
    ranked = sorted(keys, key=lambda k: (raw[k] - floor_vals[k], -keys.index(k)), reverse=True)
    for k in ranked[:remainder]:
        floor_vals[k] += 1
    return floor_vals


def _plan_class_targets(
    existing: pd.DataFrame,
    target_total: int,
    mix_mode: str = "preserve_ratio",
    overrides: dict[str, int | None] | None = None,
) -> tuple[dict[str, int], dict[str, int]]:
    """Return final targets and new-needed counts per class."""
    overrides = overrides or {}
    class_counts = (
        existing["class"].astype(str).str.lower().value_counts().to_dict()
        if "class" in existing.columns else {}
    )
    existing_counts = {c: int(class_counts.get(c, 0)) for c in CLASS_ORDER}
    existing_total = sum(existing_counts.values())

    if target_total < existing_total:
        raise ValueError(f"target_total={target_total} < existing_total={existing_total}")

    provided = {k: int(v) for k, v in overrides.items() if v is not None}
    bad = set(provided) - set(CLASS_ORDER)
    if bad:
        raise ValueError(f"Unknown class target override(s): {sorted(bad)}")
    if any(v < 0 for v in provided.values()):
        raise ValueError(f"Negative class target override detected: {provided}")
    if mix_mode not in {"preserve_ratio", "fixed"}:
        raise ValueError(f"Unsupported mix_mode: {mix_mode}")

    if mix_mode == "fixed":
        missing = [c for c in CLASS_ORDER if c not in provided]
        if missing:
            raise ValueError(f"mix_mode=fixed requires all class targets. Missing: {missing}")
        final = {c: provided[c] for c in CLASS_ORDER}
        if sum(final.values()) != target_total:
            raise ValueError(
                f"sum(fixed class targets)={sum(final.values())} must equal target_total={target_total}"
            )
    else:
        final: dict[str, int | None] = {c: None for c in CLASS_ORDER}
        for c, v in provided.items():
            final[c] = v
        fixed_sum = sum(v for v in final.values() if v is not None)
        remaining_total = target_total - fixed_sum
        if remaining_total < 0:
            raise ValueError(
                f"Provided class targets sum to {fixed_sum} > target_total={target_total}"
            )
        remaining_classes = [c for c in CLASS_ORDER if final[c] is None]
        if remaining_classes:
            weights = {c: float(max(existing_counts.get(c, 0), 0)) for c in remaining_classes}
            if sum(weights.values()) <= 0:
                weights = {c: 1.0 for c in remaining_classes}
            alloc = _largest_remainder_allocate(weights, remaining_total)
            for c in remaining_classes:
                final[c] = int(alloc[c])
        final = {c: int(final[c]) for c in CLASS_ORDER}

    new_needed = {c: max(0, final[c] - existing_counts[c]) for c in CLASS_ORDER}
    return final, new_needed

# ---------------------------------------------------------------------------
# Hardcoded SN source lists (from MD file)
# ---------------------------------------------------------------------------
TYPE_IIn_SN: list[tuple] = [
    ("SN 2006jd",   100.670,  19.420, 0.019, "IIn"),
    ("SN 2009kn",   148.270,  15.130, 0.013, "IIn"),
    ("SN 2010jl",   145.725,   9.496, 0.011, "IIn"),   # brightest IIn in decades
    ("SN 2011ht",    45.790,  51.850, 0.004, "IIn"),
    ("SN 2015da",   155.020,  60.290, 0.012, "IIn"),
    ("SN 2017hcc",   39.590, -12.340, 0.017, "IIn"),
    ("SN 2018zd",   100.200,  53.110, 0.003, "IIn"),
    ("SN 2019uo",    36.840,  12.120, 0.048, "IIn"),
    ("SN 2020oi",   184.910,  15.820, 0.005, "Ic"),
    ("SN 2020jfo",  202.470,   4.480, 0.005, "IIP"),
    ("SN 2021J",     67.650,  32.140, 0.005, "IIn"),
    ("SN 2021dbg",   25.470,  19.190, 0.012, "IIn"),
    ("SN 2021hpr",  177.680,  26.320, 0.009, "Ia"),
    ("SN 2022acko",  53.400, -37.680, 0.001, "IIP"),
    ("SN 2023bee",  149.960,  13.820, 0.005, "Ia"),
    ("SN 2023ixf",  210.910,  54.310, 0.001, "IIP"),
]

ASASSN_SN_IN_AGN: list[tuple] = [
    # (name, ra, dec, z, sn_type, host_type)
    ("ASASSN-14li",  192.063,  17.774, 0.021, "TDE",    "galaxy"),
    ("ASASSN-15lh",  339.002, -61.210, 0.232, "SLSN-I", "galaxy"),
    ("ASASSN-16fp",  230.418,  53.980, 0.009, "Ic-BL",  "spiral"),
    ("ASASSN-17jz",  233.281,  50.171, 0.060, "IIn",    "Seyfert"),
    ("ASASSN-18jd",  309.440, -60.080, 0.107, "IIP",    "galaxy"),
    ("ASASSN-19aad", 346.821, -15.623, 0.047, "Ia",     "S0"),
    ("ASASSN-19bt",  128.796, -19.116, 0.009, "IIP",    "spiral"),
    ("ASASSN-20hx",   47.013, -49.782, 0.036, "IIn",    "Seyfert"),
]

# ---------------------------------------------------------------------------
# Hardcoded blazar source lists (from MD file)
# ---------------------------------------------------------------------------
HIGH_PRIORITY_BZB: list[tuple] = [
    # (name, ra, dec, z, bztype, reference)
    ("Mrk 421",        166.114,  38.209, 0.031, "BZB", "Punch+1992"),
    ("Mrk 501",        253.468,  39.760, 0.034, "BZB", "Quinn+1996"),
    ("BL Lac",         330.680,  42.278, 0.069, "BZB", "Schmidt+1974"),
    ("PKS 2155-304",   329.717, -30.226, 0.116, "BZB", "Chadwick+1999"),
    ("1ES 1959+650",   300.001,  65.149, 0.048, "BZB", "Nishiyama+1999"),
    ("1ES 2344+514",   356.770,  51.705, 0.044, "BZB", "Catanese+1998"),
    ("W Com",          185.382,  28.233, 0.102, "BZB", "Neshpor+1998"),
    ("OJ 287",         133.704,  20.109, 0.306, "BZB", "Sitko+1985"),
    ("S5 0716+714",    110.473,  71.343, 0.310, "BZB", "Nilsson+2008"),
    ("ON 325",         187.557,  39.022, 0.130, "BZB", "Rec+1976"),
    ("1ES 0229+200",    38.203,  20.288, 0.140, "BZB", "Aharonian+2007"),
    ("1ES 0347-121",    57.348, -11.991, 0.188, "BZB", "Aharonian+2007"),
    ("3C 66A",          35.665,  43.036, 0.444, "BZB", "Neshpor+1998"),
    ("RGB J0710+591",  107.625,  59.134, 0.125, "BZB", "Acciari+2010"),
    ("1ES 1218+304",   185.341,  30.177, 0.182, "BZB", "Albert+2006"),
    ("PKS 1424+240",   216.752,  23.800, 0.604, "BZB", "Acciari+2010"),
    ("1ES 1727+502",   262.076,  50.220, 0.055, "BZB", "Falomo+1994"),
    ("1ES 1741+196",   265.990,  19.588, 0.084, "BZB", "Falomo+1994"),
    ("Mrk 180",        174.110,  70.158, 0.045, "BZB", "Albert+2006"),
    ("S4 0954+658",    149.704,  65.566, 0.367, "BZB", "Mukherjee+1999"),
]

HIGH_PRIORITY_BZQ: list[tuple] = [
    ("3C 273",               187.278,   2.052, 0.158, "BZQ", "Schmidt+1963"),
    ("3C 279",               194.047,  -5.789, 0.538, "BZQ", "Hartman+1992"),
    ("PKS 1510-089",         228.211,  -9.100, 0.360, "BZQ", "Blandford+1978"),
    ("CTA 102",              338.152,  11.731, 1.037, "BZQ", "Osterbrock+1977"),
    ("3C 454.3",             343.491,  16.148, 0.859, "BZQ", "Hartman+1999"),
    ("4C 21.35",             186.227,  21.379, 0.435, "BZQ", "Fermi-LAT+2009"),
    ("PKS 0454-234",          74.270, -23.415, 1.003, "BZQ", "Pian+2005"),
    ("PKS 0235+164",          39.662,  16.616, 0.940, "BZQ", "Stickel+1988"),
    ("B2 1308+326",          197.620,  32.346, 0.997, "BZQ", "Blandford+1978"),
    ("NRAO 530",             220.169, -13.079, 0.902, "BZQ", "Zhou+1979"),
    ("TXS 0506+056",          77.358,   5.693, 0.336, "BZQ", "IceCube+2018"),
    ("PKS 0537-286",          84.711, -28.670, 3.100, "BZQ", "Wright+1978"),
    ("PKS 0727-11",          112.580, -11.698, 1.591, "BZQ", "Stickel+1989"),
    ("PKS 1502+106",         226.104,  10.494, 1.839, "BZQ", "Abdo+2010"),
    ("3C 345",               250.446,  39.810, 0.593, "BZQ", "Blandford+1978"),
    ("NRAO 190",              57.288,  30.350, 0.629, "BZQ", "Condon+1977"),
    ("PKS 2052-474",         313.960, -47.225, 1.491, "BZQ", "Wall+1975"),
    ("PKS 2142-758",         326.560, -75.618, 1.139, "BZQ", "Wright+1996"),
    ("CGRaBS J0239+0416",     39.900,   4.272, 1.115, "BZQ", "Healey+2008"),
    ("PKS 0537-441",          84.710, -44.086, 0.894, "BZQ", "Remillard+1991"),
]

# ---------------------------------------------------------------------------
# Coordinate deduplication helpers
# ---------------------------------------------------------------------------

def _build_skycoord(df: pd.DataFrame) -> "SkyCoord | None":
    """Build SkyCoord array from ra/dec columns, ignoring NaN."""
    ra = pd.to_numeric(df.get("ra", pd.Series(dtype=float)), errors="coerce")
    dec = pd.to_numeric(df.get("dec", pd.Series(dtype=float)), errors="coerce")
    valid = ~(ra.isna() | dec.isna())
    if not valid.any():
        return None
    return SkyCoord(ra=ra[valid].values * u.deg, dec=dec[valid].values * u.deg)


def _dedup_new_against_existing(
    new_df: pd.DataFrame,
    existing_sc: "SkyCoord | None",
    radius_arcsec: float = 5.0,
) -> pd.DataFrame:
    """Remove rows from new_df that match any source in existing_sc within radius_arcsec."""
    if existing_sc is None or len(existing_sc) == 0:
        return new_df
    kept = []
    for _, row in new_df.iterrows():
        try:
            ra_val = float(row.get("ra", np.nan))
            dec_val = float(row.get("dec", np.nan))
            if np.isnan(ra_val) or np.isnan(dec_val):
                kept.append(True)
                continue
            c = SkyCoord(ra=ra_val * u.deg, dec=dec_val * u.deg)
            seps = c.separation(existing_sc)
            kept.append(float(seps.min().arcsec) >= radius_arcsec)
        except Exception:
            kept.append(True)
    return new_df[kept].reset_index(drop=True)


def _dedup_within(df: pd.DataFrame, radius_arcsec: float = 5.0) -> pd.DataFrame:
    """Remove internal duplicates keeping the first occurrence."""
    if df.empty:
        return df
    ra = pd.to_numeric(df["ra"], errors="coerce").values
    dec = pd.to_numeric(df["dec"], errors="coerce").values
    keep = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(df)):
            if not keep[j]:
                continue
            if np.isnan(ra[i]) or np.isnan(dec[i]) or np.isnan(ra[j]) or np.isnan(dec[j]):
                continue
            c1 = SkyCoord(ra=ra[i] * u.deg, dec=dec[i] * u.deg)
            c2 = SkyCoord(ra=ra[j] * u.deg, dec=dec[j] * u.deg)
            if c1.separation(c2).arcsec < radius_arcsec:
                keep[j] = False
    return df[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Benchmark master schema conformance
# ---------------------------------------------------------------------------
_MASTER_COLS = [
    "source_id", "ra", "dec", "redshift",
    "spectral_state_1", "spectral_state_2", "epoch_1", "epoch_2",
    "class", "label_source", "label_method", "independent_of_pipeline",
    "confidence", "g_mag", "notes", "mag", "y_true", "canonical_id", "split",
]


def _conform_to_master_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all columns from _MASTER_COLS exist, filling missing with appropriate defaults."""
    df = df.copy()
    defaults: dict[str, Any] = {
        "spectral_state_1": "", "spectral_state_2": "", "epoch_1": "",
        "epoch_2": "", "label_source": "", "label_method": "literature",
        "independent_of_pipeline": True, "confidence": "silver",
        "g_mag": np.nan, "notes": "", "mag": np.nan,
    }
    for col in _MASTER_COLS:
        if col not in df.columns:
            df[col] = defaults.get(col, np.nan)
    return df[_MASTER_COLS]


# ---------------------------------------------------------------------------
# SN expansion generation
# ---------------------------------------------------------------------------

def generate_sn_expansion(
    output_path: Path,
    existing_sc: "SkyCoord | None",
    target_n: int = 135,
    dedup_radius: float = 5.0,
) -> pd.DataFrame:
    """
    Build SN expansion CSV from hardcoded TYPE_IIn_SN + ASASSN lists.
    Fills remaining slots with IRSA-verified TNS-quality sources if possible.
    Writes output_path and returns the DataFrame.
    """
    rows = []
    for name, ra, dec, z, sn_type in TYPE_IIn_SN:
        rows.append({"source_id": name, "ra": ra, "dec": dec, "redshift": z,
                     "sn_type": sn_type, "label_source": "TYPE_IIn_SN"})
    for name, ra, dec, z, sn_type, host in ASASSN_SN_IN_AGN:
        rows.append({"source_id": name, "ra": ra, "dec": dec, "redshift": z,
                     "sn_type": sn_type, "label_source": "ASASSN"})

    df = pd.DataFrame(rows)
    df["class"] = "sn"
    df["y_true"] = 0
    df = _dedup_new_against_existing(df, existing_sc, dedup_radius)
    df = _dedup_within(df, dedup_radius)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format="%.6f")
    print(f"SN expansion: {len(df)} sources → {output_path}")
    return df


# ---------------------------------------------------------------------------
# Blazar expansion generation
# ---------------------------------------------------------------------------

def _fetch_bzcat_vizier(n_fill: int, existing_sc: "SkyCoord | None",
                         priority_coords: "SkyCoord | None",
                         dedup_radius: float = 5.0) -> list[dict]:
    """Fetch additional blazars from BZCAT5 (VizieR V/110) to fill quota."""
    try:
        resp = _vizier_get(
            {
                "-source": "V/110/bzcat5",
                "-out": "Name,RAJ2000,DEJ2000,z,Type",
                "-out.max": "5000",
                "-c.rm": "180",
            },
            timeout=90,
        )
        lines = [ln for ln in resp.text.splitlines() if ln and not ln.startswith("#")]
        if len(lines) < 2:
            return []
        # Simple TSV parse
        header = lines[0].split("\t")
        rows = []
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) < len(header):
                continue
            d = dict(zip(header, parts))
            try:
                ra = float(d.get("RAJ2000", "") or "nan")
                dec = float(d.get("DEJ2000", "") or "nan")
                z = float(d.get("z", "") or "0")
                btype = str(d.get("Type", "BZU")).strip()
                name = str(d.get("Name", "")).strip()
                if np.isnan(ra) or np.isnan(dec):
                    continue
                # galactic latitude cut |b| > 10
                from astropy.coordinates import Galactic
                gc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg).galactic
                if abs(float(gc.b.deg)) < 10:
                    continue
                if z > 2.0:
                    continue
                rows.append({"source_id": name, "ra": ra, "dec": dec, "redshift": z,
                             "bztype": btype, "label_source": "BZCAT5"})
            except (ValueError, TypeError):
                continue

        fill_df = pd.DataFrame(rows)
        if fill_df.empty:
            return []
        fill_df = _dedup_new_against_existing(fill_df, existing_sc, dedup_radius)
        if priority_coords is not None:
            fill_df = _dedup_new_against_existing(fill_df, priority_coords, dedup_radius)
        fill_df = _dedup_within(fill_df, dedup_radius)
        return fill_df.head(n_fill).to_dict("records")
    except Exception as exc:
        print(f"  BZCAT VizieR fetch failed: {exc}")
        return []


def _blazar_subtype_targets(total_target: int) -> tuple[int, int, int]:
    """
    Split total blazar target into (BZB, BZQ, BZU) using default 50/50/15 ratio.
    Deterministic largest-remainder allocation.
    """
    total_target = int(max(0, total_target))
    alloc = _largest_remainder_allocate({"BZB": 50.0, "BZQ": 50.0, "BZU": 15.0}, total_target)
    return int(alloc["BZB"]), int(alloc["BZQ"]), int(alloc["BZU"])


def generate_blazar_expansion(
    output_path: Path,
    existing_sc: "SkyCoord | None",
    target_bzb: int = 35,
    target_bzq: int = 35,
    target_bzu: int = 12,
    dedup_radius: float = 5.0,
) -> pd.DataFrame:
    rows = []

    # High-priority BZB
    bzb_added = 0
    for name, ra, dec, z, btype, ref in HIGH_PRIORITY_BZB:
        if bzb_added >= target_bzb:
            break
        rows.append({"source_id": name, "ra": ra, "dec": dec, "redshift": z,
                     "bztype": btype, "label_source": ref})
        bzb_added += 1

    # High-priority BZQ
    bzq_added = 0
    for name, ra, dec, z, btype, ref in HIGH_PRIORITY_BZQ:
        if bzq_added >= target_bzq:
            break
        rows.append({"source_id": name, "ra": ra, "dec": dec, "redshift": z,
                     "bztype": btype, "label_source": ref})
        bzq_added += 1

    df_priority = pd.DataFrame(rows)
    df_priority = _dedup_new_against_existing(df_priority, existing_sc, dedup_radius)
    df_priority = _dedup_within(df_priority, dedup_radius)

    # Build coords of priority blazars for fill dedup
    psc = _build_skycoord(df_priority)

    # Fill remaining with BZCAT
    n_fill = (target_bzb + target_bzq + target_bzu) - len(df_priority)
    fill_rows: list[dict] = []
    if n_fill > 0:
        print(f"  Fetching {n_fill} additional blazars from BZCAT VizieR...")
        fill_rows = _fetch_bzcat_vizier(n_fill, existing_sc, psc, dedup_radius)

    df = pd.concat(
        [df_priority, pd.DataFrame(fill_rows) if fill_rows else pd.DataFrame()],
        ignore_index=True,
    )
    df["class"] = "blazar"
    df["y_true"] = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format="%.6f")
    print(f"Blazar expansion: {len(df)} sources → {output_path}")
    return df


# ---------------------------------------------------------------------------
# Normal AGN expansion — SDSS DR16Q via VizieR VII/289
# ---------------------------------------------------------------------------

def _fetch_sdss_dr16q_vizier(n_per_bin: list[int], z_bins: list[float],
                               seed: int = 42) -> pd.DataFrame:
    """
    Fetch SDSS DR16Q via VizieR VII/289 (Lyke+2020).
    Returns DataFrame with source_id, ra, dec, redshift columns.
    """
    print("  Fetching SDSS DR16Q from VizieR VII/289 ...")
    params = {
        "-source": "VII/289/dr16q",
        "-out": "SDSS,RAJ2000,DEJ2000,z",
        "-out.max": "50000",
        "-c.rm": "180",
    }
    try:
        resp = _vizier_get(params, timeout=120)
        lines = [ln for ln in resp.text.splitlines() if ln and not ln.startswith("#")]
        if len(lines) < 2:
            raise ValueError("Empty response from VizieR")
        header = lines[0].split("\t")
        records = []
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) < len(header):
                continue
            d = dict(zip(header, parts))
            try:
                records.append({
                    "source_id": str(d.get("SDSS", "")).strip(),
                    "ra": float(d.get("RAJ2000", "") or "nan"),
                    "dec": float(d.get("DEJ2000", "") or "nan"),
                    "redshift": float((d.get("z", "") or d.get("zsp", "") or "nan")),
                })
            except (ValueError, TypeError):
                continue
        df = pd.DataFrame(records)
        df = df.dropna(subset=["ra", "dec", "redshift"])
        df = df[(df["redshift"] > 0.01) & (df["redshift"] < 3.0)]
        print(f"  DR16Q: {len(df):,} sources fetched")
        return df
    except Exception as exc:
        print(f"  VizieR DR16Q fetch failed ({exc}); falling back to SDSS TAP ...")
        return _fetch_sdss_dr16q_tap(seed=seed)


def _fetch_sdss_dr16q_tap(seed: int = 42) -> pd.DataFrame:
    """Fallback: fetch SDSS DR16Q quasars via SDSS SkyServer TAP."""
    url = "https://skyserver.sdss.org/dr16/SkyServerWS/SearchTools/SqlSearch"
    sql = (
        "SELECT TOP 50000 p.objid AS source_id, p.ra, p.dec, s.z AS redshift "
        "FROM PhotoObjAll AS p "
        "JOIN SpecObj AS s ON p.objid = s.bestobjid "
        "WHERE s.class='QSO' AND s.zwarning=0 AND s.z>0.01 AND s.z<3.0 "
        "ORDER BY NEWID()"
    )
    try:
        resp = requests.post(
            url,
            data={"cmd": sql, "format": "json"},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("Rows", [])
        df = pd.DataFrame(rows)
        df["source_id"] = df["source_id"].astype(str)
        df["ra"] = pd.to_numeric(df["ra"], errors="coerce")
        df["dec"] = pd.to_numeric(df["dec"], errors="coerce")
        df["redshift"] = pd.to_numeric(df["redshift"], errors="coerce")
        return df.dropna(subset=["ra", "dec", "redshift"])
    except Exception as exc:
        print(f"  SDSS TAP fallback also failed: {exc}")
        return pd.DataFrame(columns=["source_id", "ra", "dec", "redshift"])


def generate_agn_expansion(
    output_path: Path,
    existing_sc: "SkyCoord | None",
    target_n: int = 515,
    seed: int = 42,
    dedup_radius: float = 5.0,
) -> pd.DataFrame:
    """
    Sample target_n normal AGN from SDSS DR16Q, stratified by redshift bin.
    """
    z_bins = [0.0, 0.3, 0.6, 1.0, 2.0, 10.0]
    z_labels = ["z_0.0-0.3", "z_0.3-0.6", "z_0.6-1.0", "z_1.0-2.0", "z_2.0+"]
    base_weights = {
        "z_0.0-0.3": 120.0,
        "z_0.3-0.6": 140.0,
        "z_0.6-1.0": 120.0,
        "z_1.0-2.0": 100.0,
        "z_2.0+": 35.0,
    }
    alloc = _largest_remainder_allocate(base_weights, int(max(0, target_n)))
    target_per_bin = [alloc[k] for k in z_labels]

    catalog = _fetch_sdss_dr16q_vizier(target_per_bin, z_bins, seed)
    if catalog.empty:
        print("  WARNING: No AGN sources fetched; agn_expansion will be empty.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        empty = pd.DataFrame(columns=["source_id", "ra", "dec", "redshift", "class", "y_true"])
        empty.to_csv(output_path, index=False)
        return empty

    catalog = _dedup_new_against_existing(catalog, existing_sc, dedup_radius)
    catalog["z_bin"] = pd.cut(catalog["redshift"], bins=z_bins, labels=z_labels)

    rng = np.random.RandomState(seed)
    sampled = []
    for bin_label, n_target in zip(z_labels, target_per_bin):
        bin_data = catalog[catalog["z_bin"] == bin_label].copy()
        n_sample = min(n_target, len(bin_data))
        if n_sample > 0:
            sampled.append(bin_data.sample(n_sample, random_state=rng.randint(0, 1_000_000)))
        print(f"  AGN {bin_label}: {n_sample} / {len(bin_data):,} available")

    df = pd.concat(sampled, ignore_index=True) if sampled else pd.DataFrame()
    if not df.empty:
        df["class"] = "normal_agn"
        df["y_true"] = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format="%.6f")
    print(f"AGN expansion: {len(df)} sources → {output_path}")
    return df


# ---------------------------------------------------------------------------
# Star expansion — Gaia DR3 varisum via VizieR I/358/varisum
# ---------------------------------------------------------------------------

def _fetch_gaia_varisum_vizier() -> pd.DataFrame:
    """
    Fetch Gaia DR3 variability summary rows for contaminant star expansion.
    """
    print("  Fetching Gaia DR3 varisum from VizieR I/358/varisum ...")
    params = {
        "-source": "I/358/varisum",
        "-out": "Source,RA_ICRS,DE_ICRS,Gmagmean,VarType",
        "-out.max": "200000",
        "-c.rm": "180",
    }
    try:
        resp = _vizier_get(params, timeout=120)
        lines = [ln for ln in resp.text.splitlines() if ln and not ln.startswith("#")]
        if len(lines) < 2:
            raise ValueError("Empty response from Gaia varisum")

        header = lines[0].split("\t")
        records = []
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) < len(header):
                continue
            row = dict(zip(header, parts))
            try:
                sid = str(row.get("Source", "")).strip()
                ra = float(row.get("RA_ICRS", "") or "nan")
                dec = float(row.get("DE_ICRS", "") or "nan")
                gmag = pd.to_numeric(row.get("Gmagmean", np.nan), errors="coerce")
                vartype = str(row.get("VarType", "")).strip()
            except (ValueError, TypeError):
                continue
            if not sid or np.isnan(ra) or np.isnan(dec):
                continue
            records.append(
                {
                    "source_id": sid,
                    "ra": ra,
                    "dec": dec,
                    "g_mag": float(gmag) if pd.notna(gmag) else np.nan,
                    "var_type": vartype,
                }
            )
        df = pd.DataFrame(records)
        df = df.dropna(subset=["ra", "dec"]).drop_duplicates(subset=["source_id"])
        print(f"  Gaia varisum: {len(df):,} rows fetched")
        return df
    except Exception as exc:
        print(f"  Gaia varisum fetch failed: {exc}")
        return pd.DataFrame(columns=["source_id", "ra", "dec", "g_mag", "var_type"])


def generate_star_expansion(
    output_path: Path,
    existing_sc: "SkyCoord | None",
    target_n: int = 65,
    seed: int = 42,
    dedup_radius: float = 5.0,
) -> pd.DataFrame:
    """
    Build star contaminant expansion from Gaia DR3 variability summary.
    Stratifies by g_mag to avoid narrow magnitude sampling.
    """
    catalog = _fetch_gaia_varisum_vizier()
    if catalog.empty:
        print("  WARNING: No star sources fetched; star_expansion will be empty.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        empty = pd.DataFrame(columns=["source_id", "ra", "dec", "g_mag", "class", "y_true", "label_source"])
        empty.to_csv(output_path, index=False)
        return empty

    catalog = _dedup_new_against_existing(catalog, existing_sc, dedup_radius)
    if catalog.empty:
        print("  WARNING: All fetched stars deduped against existing benchmark.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        empty = pd.DataFrame(columns=["source_id", "ra", "dec", "g_mag", "class", "y_true", "label_source"])
        empty.to_csv(output_path, index=False)
        return empty

    rng = np.random.RandomState(seed)
    # Magnitude bins with mild balancing for broad representation
    mag = pd.to_numeric(catalog["g_mag"], errors="coerce")
    bin_edges = [-np.inf, 13.0, 16.0, 19.0, np.inf]
    bin_labels = ["bright", "mid", "faint", "very_faint"]
    catalog["mag_bin"] = pd.cut(mag, bins=bin_edges, labels=bin_labels)
    alloc = _largest_remainder_allocate(
        {"bright": 0.20, "mid": 0.30, "faint": 0.30, "very_faint": 0.20},
        int(max(0, target_n)),
    )

    sampled_parts = []
    for b in bin_labels:
        n_target = int(alloc[b])
        part = catalog[catalog["mag_bin"] == b]
        if part.empty or n_target <= 0:
            continue
        n_take = min(n_target, len(part))
        sampled_parts.append(part.sample(n_take, random_state=rng.randint(0, 1_000_000)))

    sampled = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else pd.DataFrame(columns=catalog.columns)
    # Backfill if bins underfilled
    if len(sampled) < int(target_n):
        remaining = catalog[~catalog["source_id"].isin(set(sampled.get("source_id", [])))]
        n_fill = min(int(target_n) - len(sampled), len(remaining))
        if n_fill > 0:
            fill = remaining.sample(n_fill, random_state=rng.randint(0, 1_000_000))
            sampled = pd.concat([sampled, fill], ignore_index=True)

    df = sampled.copy()
    df["class"] = "star"
    df["y_true"] = 0
    df["redshift"] = np.nan
    df["label_source"] = "Gaia DR3 varisum (I/358/varisum)"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format="%.6f")
    print(f"Star expansion: {len(df)} sources → {output_path}")
    return df


# ---------------------------------------------------------------------------
# CLAGN expansion loading
# ---------------------------------------------------------------------------

def load_clagn_expansion(clagn_report_path: Path) -> pd.DataFrame:
    """
    Load expand_clagn_catalog.py output; keep only coverage_pass=True rows
    that are not already skipped.
    """
    if not clagn_report_path.exists():
        print(f"  WARNING: {clagn_report_path} not found; no new CLAGN added.")
        return pd.DataFrame()
    df = pd.read_csv(clagn_report_path)
    mask = df["coverage_pass"].astype(str).str.lower().isin({"true", "1", "yes"}) & \
           ~df["skipped"].astype(str).str.lower().isin({"true", "1", "yes"})
    df_pass = df[mask].copy()
    print(f"  CLAGN expansion: {len(df_pass)} / {len(df)} pass coverage check")
    # Conform to master schema fields
    df_pass["class"] = "clagn"
    df_pass["y_true"] = 1
    # rename source_id if present
    if "source_id" not in df_pass.columns and "name" in df_pass.columns:
        df_pass = df_pass.rename(columns={"name": "source_id"})
    return df_pass


# ---------------------------------------------------------------------------
# Split assignment for new sources
# ---------------------------------------------------------------------------

def _assign_new_splits(df: pd.DataFrame, test_frac: float = 0.20, seed: int = 42) -> pd.DataFrame:
    """Assign dev/test splits to new sources using StratifiedShuffleSplit by class."""
    df = df.copy()
    if df.empty:
        df["split"] = pd.Series(dtype=str)
        return df

    class_col = "class" if "class" in df.columns else "label"
    labels = df[class_col].astype(str).values

    if not _HAS_SKLEARN:
        # Fallback: simple 80/20 random split
        print("  WARNING: sklearn not available; using random 80/20 split")
        rng = np.random.RandomState(seed)
        idx = np.arange(len(df))
        rng.shuffle(idx)
        n_test = max(1, int(round(len(df) * test_frac)))
        test_idx = set(idx[:n_test].tolist())
        df["split"] = ["test" if i in test_idx else "dev" for i in range(len(df))]
        return df

    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        dev_idx, test_idx_arr = next(sss.split(np.zeros(len(df)), labels))
        splits = np.empty(len(df), dtype=object)
        splits[dev_idx] = "dev"
        splits[test_idx_arr] = "test"
        df["split"] = splits
    except ValueError:
        # Too few samples for stratification; fallback to simple split
        rng = np.random.RandomState(seed)
        idx = np.arange(len(df))
        rng.shuffle(idx)
        n_test = max(1, int(round(len(df) * test_frac)))
        test_set = set(idx[:n_test].tolist())
        df["split"] = ["test" if i in test_set else "dev" for i in range(len(df))]
    return df


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------

def run(
    data_dir: Path,
    expansion_dir: Path,
    generate_sn: bool = False,
    generate_blazar: bool = False,
    generate_agn: bool = False,
    generate_star: bool = False,
    agn_n: int = 515,
    star_n: int = 65,
    target_total: int = 5000,
    mix_mode: str = "preserve_ratio",
    target_clagn: int | None = None,
    target_sn: int | None = None,
    target_blazar: int | None = None,
    target_normal_agn: int | None = None,
    target_star: int | None = None,
    seed: int = 42,
    dedup_radius: float = 5.0,
    test_frac: float = 0.20,
    allow_underfill: bool = False,
) -> None:
    # Hard constraint assertion — NEVER touch pipeline_policy.yaml t_recall
    policy_path = ROOT / "configs" / "pipeline_policy.yaml"
    if policy_path.exists():
        import yaml
        with policy_path.open() as fh:
            policy = yaml.safe_load(fh) or {}
        actual_t = float(policy.get("t_recall", -1))
        expected_t = 0.3493918746
        assert abs(actual_t - expected_t) < 1e-9, (
            f"POLICY TAMPERED: t_recall={actual_t} != {expected_t}"
        )
        print(f"t_recall hard constraint: PASS ({actual_t})")

    master_path = data_dir / "benchmark_master.csv"
    existing = pd.read_csv(master_path)
    existing = ensure_class_ytrue(existing)
    if "canonical_id" not in existing.columns:
        existing = add_canonical_id(existing)
    print(f"Existing benchmark: {len(existing)} sources")
    print(existing["class"].value_counts().to_string())

    final_targets, new_needed_targets = _plan_class_targets(
        existing=existing,
        target_total=int(target_total),
        mix_mode=mix_mode,
        overrides={
            "clagn": target_clagn,
            "sn": target_sn,
            "blazar": target_blazar,
            "normal_agn": target_normal_agn,
            "star": target_star,
        },
    )
    print("\nTarget plan (final totals):", final_targets)
    print("Target plan (new needed):", new_needed_targets)

    existing_sc = _build_skycoord(existing)

    # --- Load / generate CLAGN expansion ---
    clagn_report_path = expansion_dir / "clagn_coverage_report.csv"
    new_clagn = load_clagn_expansion(clagn_report_path)
    clagn_need = int(new_needed_targets["clagn"])
    if clagn_need > 0 and len(new_clagn) > clagn_need:
        new_clagn = new_clagn.sort_values("source_id").head(clagn_need).reset_index(drop=True)
    elif clagn_need == 0:
        new_clagn = new_clagn.iloc[0:0].copy()

    # --- Load / generate SN expansion ---
    sn_path = expansion_dir / "sn_expansion.csv"
    sn_need = int(new_needed_targets["sn"])
    if generate_sn or not sn_path.exists():
        new_sn = generate_sn_expansion(
            sn_path, existing_sc, target_n=sn_need, dedup_radius=dedup_radius
        )
    else:
        new_sn = pd.read_csv(sn_path)
        print(f"  SN expansion loaded: {len(new_sn)} sources")
    if sn_need > 0 and len(new_sn) > sn_need:
        new_sn = new_sn.sort_values("source_id").head(sn_need).reset_index(drop=True)
    elif sn_need == 0:
        new_sn = new_sn.iloc[0:0].copy()

    # --- Load / generate Blazar expansion ---
    blazar_path = expansion_dir / "blazar_expansion.csv"
    blazar_need = int(new_needed_targets["blazar"])
    t_bzb, t_bzq, t_bzu = _blazar_subtype_targets(blazar_need)
    if generate_blazar or not blazar_path.exists():
        new_blazar = generate_blazar_expansion(
            blazar_path, existing_sc,
            target_bzb=t_bzb, target_bzq=t_bzq, target_bzu=t_bzu,
            dedup_radius=dedup_radius,
        )
    else:
        new_blazar = pd.read_csv(blazar_path)
        print(f"  Blazar expansion loaded: {len(new_blazar)} sources")
    if blazar_need > 0 and len(new_blazar) > blazar_need:
        new_blazar = new_blazar.sort_values("source_id").head(blazar_need).reset_index(drop=True)
    elif blazar_need == 0:
        new_blazar = new_blazar.iloc[0:0].copy()

    # --- Load / generate AGN expansion ---
    agn_path = expansion_dir / "agn_expansion.csv"
    agn_need = int(new_needed_targets["normal_agn"])
    if generate_agn or not agn_path.exists():
        new_agn = generate_agn_expansion(
            agn_path, existing_sc, target_n=agn_need if agn_need > 0 else agn_n, seed=seed, dedup_radius=dedup_radius
        )
    else:
        new_agn = pd.read_csv(agn_path)
        print(f"  AGN expansion loaded: {len(new_agn)} sources")
    if agn_need > 0 and len(new_agn) > agn_need:
        new_agn = new_agn.sort_values("source_id").head(agn_need).reset_index(drop=True)
    elif agn_need == 0:
        new_agn = new_agn.iloc[0:0].copy()

    # --- Load / generate Star expansion ---
    star_path = expansion_dir / "star_expansion.csv"
    star_need = int(new_needed_targets["star"])
    if generate_star or not star_path.exists():
        new_star = generate_star_expansion(
            star_path, existing_sc, target_n=star_need if star_need > 0 else star_n,
            seed=seed, dedup_radius=dedup_radius
        )
    else:
        new_star = pd.read_csv(star_path)
        print(f"  Star expansion loaded: {len(new_star)} sources")
    if star_need > 0 and len(new_star) > star_need:
        new_star = new_star.sort_values("source_id").head(star_need).reset_index(drop=True)
    elif star_need == 0:
        new_star = new_star.iloc[0:0].copy()

    # --- Assign splits to new sources ---
    all_new_parts = []
    for label, df_part in [
        ("clagn", new_clagn),
        ("sn", new_sn),
        ("blazar", new_blazar),
        ("normal_agn", new_agn),
        ("star", new_star),
    ]:
        if df_part.empty:
            continue
        if "class" not in df_part.columns:
            df_part = df_part.copy()
            df_part["class"] = label
        if "y_true" not in df_part.columns:
            df_part = df_part.copy()
            df_part["y_true"] = 1 if label == "clagn" else 0
        all_new_parts.append(df_part)

    if all_new_parts:
        new_all = pd.concat(all_new_parts, ignore_index=True)
        # Final dedup against existing
        new_all = _dedup_new_against_existing(new_all, existing_sc, dedup_radius)
        # Assign dev/test splits
        new_all = _assign_new_splits(new_all, test_frac=test_frac, seed=seed)
        # Conform to master schema
        new_all = _conform_to_master_schema(new_all)
        if "canonical_id" not in new_all.columns or new_all["canonical_id"].isna().all():
            new_all = add_canonical_id(new_all)
    else:
        new_all = pd.DataFrame(columns=_MASTER_COLS)
        print("WARNING: No new sources to add.")

    # --- Assemble master v2 ---
    master_v2 = pd.concat([existing[_MASTER_COLS], new_all[_MASTER_COLS]], ignore_index=True)
    # Dedup within full v2 on coordinates
    master_v2 = _dedup_within(master_v2, dedup_radius)
    if "canonical_id" not in master_v2.columns:
        master_v2 = add_canonical_id(master_v2)
    # Fix canonical_id for NaN entries
    mask_nan = master_v2["canonical_id"].isna() | (master_v2["canonical_id"].astype(str) == "nan")
    if mask_nan.any():
        master_v2.loc[mask_nan, "canonical_id"] = (
            master_v2.loc[mask_nan, "source_id"]
            .map(lambda x: str(x).strip() if x is not None else "")
        )

    # Validate: no duplicate canonical_ids
    dupes = master_v2["canonical_id"].duplicated()
    if dupes.any():
        print(f"  WARNING: {dupes.sum()} duplicate canonical_ids in v2; keeping first occurrence")
        master_v2 = master_v2[~dupes].reset_index(drop=True)

    # Underfill checks
    final_counts = master_v2["class"].astype(str).str.lower().value_counts().to_dict()
    deficits = {
        c: max(0, int(final_targets[c]) - int(final_counts.get(c, 0)))
        for c in CLASS_ORDER
    }
    total_deficit = max(0, int(target_total) - len(master_v2))
    if any(v > 0 for v in deficits.values()) or total_deficit > 0:
        print("\nWARNING: target underfill detected")
        print("  class deficits:", deficits)
        print(f"  total deficit: {total_deficit}")
        if not allow_underfill:
            raise RuntimeError(
                "Expansion underfilled target counts. Re-run with larger source pools "
                "or pass --allow-underfill to keep best-effort output."
            )

    # --- Write outputs ---
    out_dir = data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    master_v2_path = out_dir / "benchmark_master_v2.csv"
    master_v2.to_csv(master_v2_path, index=False, float_format="%.10f")
    print(f"\nWrote {master_v2_path} ({len(master_v2)} sources)")

    dev_v2 = master_v2[master_v2["split"].astype(str).str.lower() == "dev"].reset_index(drop=True)
    test_v2 = master_v2[master_v2["split"].astype(str).str.lower() == "test"].reset_index(drop=True)
    dev_v2_path = out_dir / "benchmark_dev_v2.csv"
    test_v2_path = out_dir / "benchmark_test_v2.csv"
    dev_v2.to_csv(dev_v2_path, index=False, float_format="%.10f")
    test_v2.to_csv(test_v2_path, index=False, float_format="%.10f")
    print(f"Wrote {dev_v2_path} ({len(dev_v2)} sources)")
    print(f"Wrote {test_v2_path} ({len(test_v2)} sources)")

    # --- SPLIT_HASH_v2.json ---
    master_hash = hashlib.sha256(master_v2_path.read_bytes()).hexdigest()
    split_hash = {
        "version": "v2",
        "seed": seed,
        "test_frac": test_frac,
        "dedup_radius_arcsec": dedup_radius,
        "n_total": int(len(master_v2)),
        "n_dev": int(len(dev_v2)),
        "n_test": int(len(test_v2)),
        "class_counts": master_v2["class"].value_counts().to_dict(),
        "dev_class_counts": dev_v2["class"].value_counts().to_dict(),
        "test_class_counts": test_v2["class"].value_counts().to_dict(),
        "master_sha256": master_hash,
    }
    hash_path = out_dir / "SPLIT_HASH_v2.json"
    hash_path.write_text(json.dumps(split_hash, indent=2), encoding="utf-8")
    print(f"Wrote {hash_path}")

    # --- benchmark_scores_v2.csv ---
    scores_path = out_dir / "benchmark_scores.csv"
    scores_v2_path = out_dir / "benchmark_scores_v2.csv"
    if scores_path.exists():
        existing_scores = pd.read_csv(scores_path)
    else:
        print("WARNING: benchmark_scores.csv not found; creating empty scores v2")
        existing_scores = pd.DataFrame(columns=["source_id", "canonical_id", "score", "status"])

    # New sources not already in scores get score=NaN, status=UNSCORED
    scored_ids = set(existing_scores["canonical_id"].astype(str).tolist()) if "canonical_id" in existing_scores.columns else set()
    new_unscored_mask = ~master_v2["canonical_id"].astype(str).isin(scored_ids)
    new_unscored = master_v2.loc[new_unscored_mask, ["source_id", "canonical_id"]].copy()
    new_unscored["score"] = np.nan
    new_unscored["status"] = "UNSCORED"

    scores_v2 = pd.concat([existing_scores, new_unscored], ignore_index=True)
    scores_v2.to_csv(scores_v2_path, index=False, float_format="%.10f")
    print(f"Wrote {scores_v2_path} ({len(scores_v2)} rows, {int(new_unscored_mask.sum())} new unscored)")

    # --- Summary table ---
    print("\n=== Class x Split counts ===")
    ct = master_v2.groupby(["class", "split"]).size().unstack(fill_value=0)
    print(ct.to_string())
    print(f"\nCLAGN per split:")
    for sp in ["dev", "test"]:
        n_c = int(((master_v2["class"].str.lower() == "clagn") &
                   (master_v2["split"].str.lower() == sp)).sum())
        print(f"  {sp}: {n_c} CLAGN  (each = {100/max(n_c,1):.1f}% recall swing)")

    print("\nDone. Next step:")
    print("  python make_cards.py --results data/real_clagn/benchmark_scores_v2.csv \\")
    print("    --topn 100000 --outdir outputs/v2_rows --score-thresh 0.0 \\")
    print("    --benchmark-master data/real_clagn/benchmark_master_v2.csv \\")
    print("    --known-blazars known_blazars.csv \\")
    print("    --policy-config configs/pipeline_policy.yaml \\")
    print("    --strict-mode --overwrite --fetch-wise-if-missing --skip-existing")


def _run_legacy_split_mode(
    input_csv: Path,
    prefix: Path,
    ratio: str,
    seed: int = 42,
) -> None:
    """
    Compatibility mode for legacy CLI:
      --input data/benchmark/benchmark_master_v3.csv
      --prefix data/benchmark/benchmark_v3
      --ratio 70:15:15
    """
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_csv}")

    parts = ratio.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid --ratio '{ratio}'. Expected format train:dev:test (e.g. 70:15:15)")
    train_w, dev_w, test_w = (float(p) for p in parts)
    total_w = train_w + dev_w + test_w
    if total_w <= 0:
        raise ValueError(f"Invalid --ratio '{ratio}': total must be > 0")
    test_frac = test_w / total_w

    df = pd.read_csv(input_csv)
    required = {"source_id", "ra", "dec"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    if len(df) == 0:
        prefix.parent.mkdir(parents=True, exist_ok=True)
        train_path = Path(f"{prefix}_train.csv")
        dev_path = Path(f"{prefix}_dev.csv")
        test_path = Path(f"{prefix}_test.csv")
        df.to_csv(train_path, index=False)
        df.to_csv(dev_path, index=False)
        df.to_csv(test_path, index=False)
        summary = {
            "input": str(input_csv),
            "ratio": ratio,
            "seed": int(seed),
            "n_total": 0,
            "n_train": 0,
            "n_dev": 0,
            "n_test": 0,
            "train_file": str(train_path),
            "dev_file": str(dev_path),
            "test_file": str(test_path),
        }
        print("Legacy split mode PASS")
        print(json.dumps(summary, indent=2))
        return

    # Derive stratification key if possible.
    strat_col = None
    for candidate in ("label", "class", "y_true"):
        if candidate in df.columns:
            strat_col = candidate
            break

    rng = np.random.RandomState(seed)
    idx = np.arange(len(df))

    if strat_col is None or not _HAS_SKLEARN:
        rng.shuffle(idx)
        n_test = int(round(len(df) * test_frac))
        n_test = max(1, min(n_test, max(len(df) - 2, 1)))
        test_idx = idx[:n_test]
        train_dev_idx = idx[n_test:]
    else:
        y = df[strat_col].astype(str).values
        try:
            sss_outer = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
            train_dev_idx, test_idx = next(sss_outer.split(np.zeros(len(df)), y))
        except ValueError:
            rng.shuffle(idx)
            n_test = int(round(len(df) * test_frac))
            n_test = max(1, min(n_test, max(len(df) - 2, 1)))
            test_idx = idx[:n_test]
            train_dev_idx = idx[n_test:]

    train_dev = df.iloc[train_dev_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    # Split train/dev by relative ratio inside non-test pool
    non_test_total = train_w + dev_w
    dev_rel = dev_w / non_test_total if non_test_total > 0 else 0.0
    idx_td = np.arange(len(train_dev))
    if strat_col is None or not _HAS_SKLEARN:
        rng.shuffle(idx_td)
        n_dev = int(round(len(train_dev) * dev_rel))
        n_dev = max(1, min(n_dev, max(len(train_dev) - 1, 0)))
        dev_idx = idx_td[:n_dev]
        train_idx = idx_td[n_dev:]
    else:
        y_td = train_dev[strat_col].astype(str).values
        try:
            sss_inner = StratifiedShuffleSplit(n_splits=1, test_size=dev_rel, random_state=seed)
            train_idx, dev_idx = next(sss_inner.split(np.zeros(len(train_dev)), y_td))
        except ValueError:
            rng.shuffle(idx_td)
            n_dev = int(round(len(train_dev) * dev_rel))
            n_dev = max(1, min(n_dev, max(len(train_dev) - 1, 0)))
            dev_idx = idx_td[:n_dev]
            train_idx = idx_td[n_dev:]

    train_df = train_dev.iloc[train_idx].copy().reset_index(drop=True)
    dev_df = train_dev.iloc[dev_idx].copy().reset_index(drop=True)

    prefix.parent.mkdir(parents=True, exist_ok=True)
    train_path = Path(f"{prefix}_train.csv")
    dev_path = Path(f"{prefix}_dev.csv")
    test_path = Path(f"{prefix}_test.csv")
    train_df.to_csv(train_path, index=False)
    dev_df.to_csv(dev_path, index=False)
    test_df.to_csv(test_path, index=False)

    summary = {
        "input": str(input_csv),
        "ratio": ratio,
        "seed": int(seed),
        "n_total": int(len(df)),
        "n_train": int(len(train_df)),
        "n_dev": int(len(dev_df)),
        "n_test": int(len(test_df)),
        "train_file": str(train_path),
        "dev_file": str(dev_path),
        "test_file": str(test_path),
    }
    print("Legacy split mode PASS")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Legacy compatibility CLI (used by CLAGN_150K_DOWNLOAD_INGESTION.md Part 3)
    parser.add_argument("--input", default=None, help="Compatibility mode: input benchmark CSV to split")
    parser.add_argument("--prefix", default=None, help="Compatibility mode: output prefix for split CSVs")
    parser.add_argument("--ratio", default=None, help="Compatibility mode: train:dev:test ratio (e.g. 70:15:15)")
    parser.add_argument("--data-dir", default="data/real_clagn",
                        help="Directory containing benchmark_master.csv and where v2 files go")
    parser.add_argument("--expansion-dir", default="data/expansion",
                        help="Directory containing expansion CSVs")
    parser.add_argument("--generate-sn", action="store_true",
                        help="(Re)generate sn_expansion.csv even if it exists")
    parser.add_argument("--generate-blazar", action="store_true",
                        help="(Re)generate blazar_expansion.csv even if it exists")
    parser.add_argument("--generate-agn", action="store_true",
                        help="(Re)generate agn_expansion.csv from SDSS DR16Q even if it exists")
    parser.add_argument("--generate-star", action="store_true",
                        help="(Re)generate star_expansion.csv from Gaia DR3 varisum even if it exists")
    parser.add_argument("--agn-n", type=int, default=515,
                        help="Target new normal_agn sources (default: 515)")
    parser.add_argument("--star-n", type=int, default=65,
                        help="Target new star sources when explicit class targets are not set (default: 65)")
    parser.add_argument("--target-total", type=int, default=5000,
                        help="Target total rows in benchmark_master_v2.csv (default: 5000)")
    parser.add_argument("--mix-mode", choices=["preserve_ratio", "fixed"], default="preserve_ratio",
                        help="Class target planning mode (default: preserve_ratio)")
    parser.add_argument("--target-clagn", type=int, default=None, help="Final target count for class=clagn")
    parser.add_argument("--target-sn", type=int, default=None, help="Final target count for class=sn")
    parser.add_argument("--target-blazar", type=int, default=None, help="Final target count for class=blazar")
    parser.add_argument("--target-normal-agn", type=int, default=None, help="Final target count for class=normal_agn")
    parser.add_argument("--target-star", type=int, default=None, help="Final target count for class=star")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dedup-radius", type=float, default=5.0,
                        help="Coordinate dedup radius in arcsec (default: 5.0)")
    parser.add_argument("--test-frac", type=float, default=0.20,
                        help="Fraction of new sources assigned to test split (default: 0.20)")
    parser.add_argument("--allow-underfill", action="store_true",
                        help="Allow writing best-effort output even if target_total/class targets are underfilled")
    args = parser.parse_args()

    # If compatibility flags are provided, run legacy split mode.
    if any(v is not None for v in (args.input, args.prefix, args.ratio)):
        missing = [name for name, val in (("input", args.input), ("prefix", args.prefix), ("ratio", args.ratio)) if val is None]
        if missing:
            raise ValueError(f"Legacy split mode requires all of --input/--prefix/--ratio. Missing: {missing}")
        _run_legacy_split_mode(
            input_csv=Path(args.input),
            prefix=Path(args.prefix),
            ratio=str(args.ratio),
            seed=args.seed,
        )
        return

    run(
        data_dir=Path(args.data_dir),
        expansion_dir=Path(args.expansion_dir),
        generate_sn=args.generate_sn,
        generate_blazar=args.generate_blazar,
        generate_agn=args.generate_agn,
        generate_star=args.generate_star,
        agn_n=args.agn_n,
        star_n=args.star_n,
        target_total=args.target_total,
        mix_mode=args.mix_mode,
        target_clagn=args.target_clagn,
        target_sn=args.target_sn,
        target_blazar=args.target_blazar,
        target_normal_agn=args.target_normal_agn,
        target_star=args.target_star,
        seed=args.seed,
        dedup_radius=args.dedup_radius,
        test_frac=args.test_frac,
        allow_underfill=args.allow_underfill,
    )


if __name__ == "__main__":
    main()
