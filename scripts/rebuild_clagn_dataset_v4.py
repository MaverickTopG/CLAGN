#!/usr/bin/env python3
"""
Build CLAGN benchmark v4 as a 2-tier dataset:
  1) Evaluation core (literature-only, class-balanced)
  2) Discovery pool (catalog-scale, weakly supervised)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

try:
    from sklearn.model_selection import StratifiedShuffleSplit

    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clagn.utils.benchmark_labels import ensure_class_ytrue
from clagn.utils.id_canonicalization import add_canonical_id


EVAL_CLASSES = ["clagn", "normal_agn", "blazar", "sn", "star"]
PROV_RANK = {
    "LIT_STRONG": 0,
    "LIT_MEDIUM": 1,
    "CATALOG_HIGHCONF": 2,
    "CATALOG_WEAK": 3,
}
CONF_RANK = {"gold": 0, "high": 1, "medium": 2, "low": 3, "unknown": 4}


@dataclass
class BuildContext:
    seed: int
    dedup_radius_arcsec: float
    eval_target_total: int
    discovery_target_total: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild CLAGN benchmark v4")
    p.add_argument("--eval-target-total", type=int, default=2000)
    p.add_argument("--discovery-target-total", type=int, default=150000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dedup-radius-arcsec", type=float, default=5.0)
    p.add_argument("--output-dir", default="data/benchmark")
    p.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Raw catalog directory used for discovery tier ingestion.",
    )
    p.add_argument(
        "--emit-deficit-only",
        action="store_true",
        help="Stop after writing deficit report if eval underfill is detected.",
    )
    p.add_argument(
        "--allow-underfill-rebalance",
        action="store_true",
        help=(
            "If literature floor (<250/class) is not met, continue by rebalancing "
            "eval to min available class count across classes."
        ),
    )
    p.add_argument(
        "--eval-mode",
        choices=["strict_gold", "gold_plus"],
        default="strict_gold",
        help=(
            "strict_gold: LIT_STRONG/LIT_MEDIUM only. "
            "gold_plus: adds CATALOG_HIGHCONF supplements for larger balanced eval."
        ),
    )
    return p.parse_args()


def _safe_float(v: Any, default: float = np.nan) -> float:
    try:
        if v is None or (isinstance(v, str) and not v.strip()):
            return default
        x = float(v)
        if math.isfinite(x):
            return float(x)
        return default
    except Exception:
        return default


def _read_tsv_loose(path: Path) -> pd.DataFrame:
    for kwargs in (
        {"sep": "\t", "comment": "#"},
        {"sep": "|", "comment": "#"},
        {"delim_whitespace": True, "comment": "#"},
    ):
        try:
            df = pd.read_csv(path, engine="python", **kwargs)
            if len(df.columns) > 1:
                return df
        except Exception:
            continue
    return pd.read_csv(path, sep="\t", comment="#", engine="python")


def _row_priority(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["provenance_group"] = out["provenance_group"].astype(str)
    out["confidence"] = out["confidence"].astype(str).str.lower().replace("", "unknown")
    out["_prov_rank"] = out["provenance_group"].map(PROV_RANK).fillna(9).astype(int)
    out["_tier_rank"] = pd.to_numeric(out["evidence_tier"], errors="coerce").fillna(9).astype(int)
    out["_conf_rank"] = out["confidence"].map(CONF_RANK).fillna(9).astype(int)
    out["_id_rank"] = out["source_id"].astype(str)
    return out


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=int)
        self.rank = np.zeros(n, dtype=int)

    def find(self, x: int) -> int:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def _standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "source_id",
        "ra",
        "dec",
        "redshift",
        "class",
        "reference",
        "evidence_tier",
        "provenance_group",
        "label_source",
        "label_method",
        "confidence",
        "catalog_name",
        "catalog_confidence",
        "qa_flags",
        "coverage_status",
        "transition_type",
    ]
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = ""
    out["source_id"] = out["source_id"].astype(str).str.strip()
    out["ra"] = pd.to_numeric(out["ra"], errors="coerce")
    out["dec"] = pd.to_numeric(out["dec"], errors="coerce")
    out["redshift"] = pd.to_numeric(out["redshift"], errors="coerce")
    out["class"] = out["class"].astype(str).str.strip().str.lower()
    out["reference"] = out["reference"].astype(str).str.strip()
    out["evidence_tier"] = pd.to_numeric(out["evidence_tier"], errors="coerce").fillna(3).astype(int)
    out["provenance_group"] = out["provenance_group"].astype(str).str.strip()
    out["label_source"] = out["label_source"].astype(str).str.strip()
    out["label_method"] = out["label_method"].astype(str).str.strip()
    out["confidence"] = (
        out["confidence"]
        .astype(str)
        .str.strip()
        .replace({"": "unknown", "nan": "unknown", "NaN": "unknown", "None": "unknown", "none": "unknown"})
    )
    out["catalog_name"] = out["catalog_name"].astype(str).str.strip()
    out["catalog_confidence"] = out["catalog_confidence"].astype(str).str.strip()
    out["qa_flags"] = out["qa_flags"].astype(str).str.strip()
    out["coverage_status"] = out["coverage_status"].astype(str).str.strip()
    out["transition_type"] = out["transition_type"].astype(str).str.strip()
    out = out[(out["source_id"] != "") & np.isfinite(out["ra"]) & np.isfinite(out["dec"])]
    out = out[(out["ra"] >= 0) & (out["ra"] <= 360) & (out["dec"] >= -90) & (out["dec"] <= 90)]
    out["redshift"] = out["redshift"].fillna(0.0)
    return out[cols]


def _ingest_real_benchmark(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame(
        {
            "source_id": df.get("source_id", ""),
            "ra": df.get("ra", np.nan),
            "dec": df.get("dec", np.nan),
            "redshift": df.get("redshift", df.get("z", 0.0)),
            "class": df.get("class", "").astype(str).str.lower(),
            "reference": df.get("label_source", "real_benchmark_v1"),
            "evidence_tier": np.where(
                df.get("confidence", "").astype(str).str.lower().eq("gold"), 1, 2
            ),
            "provenance_group": np.where(
                df.get("label_method", "").astype(str).str.lower().eq("spectroscopic"),
                "LIT_STRONG",
                "LIT_MEDIUM",
            ),
            "label_source": df.get("label_source", "real_benchmark_v1"),
            "label_method": df.get("label_method", "literature"),
            "confidence": df.get("confidence", "high"),
            "catalog_name": "real_clagn_benchmark",
            "catalog_confidence": "high",
            "qa_flags": "",
            "coverage_status": np.where(
                pd.to_numeric(df.get("independent_of_pipeline", False), errors="coerce").fillna(0).astype(bool),
                "independent",
                "unknown",
            ),
            "transition_type": "",
        }
    )
    return _standard_columns(out)


def _ingest_known_clagn(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame(
        {
            "source_id": df.get("name", ""),
            "ra": df.get("ra", np.nan),
            "dec": df.get("dec", np.nan),
            "redshift": df.get("redshift", 0.0),
            "class": "clagn",
            "reference": df.get("reference", "known_clagn"),
            "evidence_tier": pd.to_numeric(df.get("tier", 1), errors="coerce").fillna(1).astype(int),
            "provenance_group": np.where(
                pd.to_numeric(df.get("tier", 1), errors="coerce").fillna(1) <= 1,
                "LIT_STRONG",
                "LIT_MEDIUM",
            ),
            "label_source": df.get("reference", "known_clagn"),
            "label_method": "literature",
            "confidence": "high",
            "catalog_name": "known_clagn",
            "catalog_confidence": "high",
            "qa_flags": "",
            "coverage_status": "unknown",
            "transition_type": df.get("transition_type", ""),
        }
    )
    return _standard_columns(out)


def _map_contaminant(label: str) -> str:
    s = str(label).strip().lower()
    if s in {"normal_agn", "agn"}:
        return "normal_agn"
    if s in {"sn_in_host", "supernova", "sn"}:
        return "sn"
    if s in {"variable_star", "star"}:
        return "star"
    if s in {"blazar"}:
        return "blazar"
    return s


def _ingest_known_contaminants(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame(
        {
            "source_id": df.get("name", ""),
            "ra": df.get("ra", np.nan),
            "dec": df.get("dec", np.nan),
            "redshift": df.get("redshift", 0.0),
            "class": df.get("contaminant_type", "").map(_map_contaminant),
            "reference": df.get("reference", "known_contaminants"),
            "evidence_tier": 1,
            "provenance_group": "LIT_STRONG",
            "label_source": df.get("reference", "known_contaminants"),
            "label_method": "literature",
            "confidence": "high",
            "catalog_name": "known_contaminants",
            "catalog_confidence": "high",
            "qa_flags": df.get("notes", ""),
            "coverage_status": "unknown",
            "transition_type": "",
        }
    )
    return _standard_columns(out)


def _ingest_normal_agn_control(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame(
        {
            "source_id": df.get("name", ""),
            "ra": df.get("ra", np.nan),
            "dec": df.get("dec", np.nan),
            "redshift": df.get("redshift", 0.0),
            "class": "normal_agn",
            "reference": df.get("reference", "normal_agn_control"),
            "evidence_tier": 2,
            "provenance_group": "LIT_MEDIUM",
            "label_source": df.get("reference", "normal_agn_control"),
            "label_method": "literature",
            "confidence": "medium",
            "catalog_name": "normal_agn_control",
            "catalog_confidence": "medium",
            "qa_flags": "",
            "coverage_status": "unknown",
            "transition_type": "",
        }
    )
    return _standard_columns(out)


def _ingest_benchmark_v3(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame(
        {
            "source_id": df.get("source_id", ""),
            "ra": df.get("ra", np.nan),
            "dec": df.get("dec", np.nan),
            "redshift": df.get("z", 0.0),
            "class": df.get("label", "").astype(str).str.lower(),
            "reference": df.get("reference", "benchmark_v3"),
            "evidence_tier": pd.to_numeric(df.get("evidence_tier", 3), errors="coerce").fillna(3).astype(int),
            "provenance_group": "CATALOG_WEAK",
            "label_source": df.get("reference", "benchmark_v3"),
            "label_method": "catalog",
            "confidence": "low",
            "catalog_name": "benchmark_v3",
            "catalog_confidence": "low",
            "qa_flags": "",
            "coverage_status": "unknown",
            "transition_type": df.get("transition_type", ""),
        }
    )
    return _standard_columns(out)


def _ingest_dr16q(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = _read_tsv_loose(path)
    out = pd.DataFrame(
        {
            "source_id": df.get("SDSS", df.get("source_id", "")).astype(str),
            "ra": df.get("RAJ2000", np.nan),
            "dec": df.get("DEJ2000", np.nan),
            "redshift": df.get("z", 0.0),
            "class": "normal_agn",
            "reference": "Lyke+2020_DR16Q",
            "evidence_tier": 3,
            "provenance_group": "CATALOG_HIGHCONF",
            "label_source": "Lyke+2020_DR16Q",
            "label_method": "catalog",
            "confidence": "medium",
            "catalog_name": "dr16q_vizier",
            "catalog_confidence": "high",
            "qa_flags": "",
            "coverage_status": "unknown",
            "transition_type": "",
        }
    )
    return _standard_columns(out)


def _ingest_milliquas(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = _read_tsv_loose(path)
    out = pd.DataFrame(
        {
            "source_id": df.get("Name", "").astype(str),
            "ra": df.get("RAJ2000", np.nan),
            "dec": df.get("DEJ2000", np.nan),
            "redshift": df.get("z", 0.0),
            "class": "normal_agn",
            "reference": "Flesch+2023_Milliquas8",
            "evidence_tier": 3,
            "provenance_group": "CATALOG_HIGHCONF",
            "label_source": "Flesch+2023_Milliquas8",
            "label_method": "catalog",
            "confidence": "medium",
            "catalog_name": "milliquas_vizier",
            "catalog_confidence": "high",
            "qa_flags": "",
            "coverage_status": "unknown",
            "transition_type": "",
        }
    )
    return _standard_columns(out)


def _ingest_bzcat(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = _read_tsv_loose(path)
    out = pd.DataFrame(
        {
            "source_id": df.get("Name", "").astype(str),
            "ra": df.get("RAJ2000", np.nan),
            "dec": df.get("DEJ2000", np.nan),
            "redshift": df.get("z", 0.0),
            "class": "blazar",
            "reference": "Massaro+2015_BZCAT5",
            "evidence_tier": 2,
            "provenance_group": "CATALOG_HIGHCONF",
            "label_source": "Massaro+2015_BZCAT5",
            "label_method": "catalog",
            "confidence": "medium",
            "catalog_name": "bzcat5",
            "catalog_confidence": "high",
            "qa_flags": "",
            "coverage_status": "unknown",
            "transition_type": "",
        }
    )
    return _standard_columns(out)


def _ingest_fermi(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    from astropy.io import fits

    with fits.open(path) as hdu:
        d = hdu[1].data
        cls = np.array([str(x).strip().lower() for x in d["CLASS1"]])
        mask = np.isin(cls, ["bll", "fsrq", "bcu"])
        if "ASSOC_PROB_BAY" in d.columns.names:
            mask &= np.asarray(d["ASSOC_PROB_BAY"], dtype=float) >= 0.8
        d = d[mask]
        out = pd.DataFrame(
            {
                "source_id": [str(x).strip() for x in d["Source_Name"]],
                "ra": np.asarray(d["RAJ2000"], dtype=float),
                "dec": np.asarray(d["DEJ2000"], dtype=float),
                "redshift": np.asarray(d["Redshift"], dtype=float) if "Redshift" in d.columns.names else 0.0,
                "class": "blazar",
                "reference": "Abdollahi+2022_4FGL_DR4",
                "evidence_tier": 2,
                "provenance_group": "CATALOG_HIGHCONF",
                "label_source": "Abdollahi+2022_4FGL_DR4",
                "label_method": "catalog",
                "confidence": "medium",
                "catalog_name": "4fgl_dr4",
                "catalog_confidence": "high",
                "qa_flags": "",
                "coverage_status": "unknown",
                "transition_type": "",
            }
        )
    return _standard_columns(out)


def _ingest_osc_json(path: Path, sn_type: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return pd.DataFrame()
    if not isinstance(obj, dict) or not obj:
        return pd.DataFrame()

    rows = []
    for name, entry in obj.items():
        if not isinstance(entry, dict):
            continue

        def _extract(field: str) -> Any:
            v = entry.get(field)
            if isinstance(v, list) and v:
                f = v[0]
                if isinstance(f, dict):
                    return f.get("value")
                return f
            if isinstance(v, dict):
                return v.get("value")
            return v

        ra = _safe_float(_extract("ra"))
        dec = _safe_float(_extract("dec"))
        z = _safe_float(_extract("redshift"), default=0.0)
        if not (np.isfinite(ra) and np.isfinite(dec)):
            continue
        rows.append(
            {
                "source_id": str(name),
                "ra": ra,
                "dec": dec,
                "redshift": z,
                "class": "sn",
                "reference": "OSC_Guillochon+2017",
                "evidence_tier": 2,
                "provenance_group": "CATALOG_HIGHCONF",
                "label_source": "OSC_Guillochon+2017",
                "label_method": "catalog",
                "confidence": "medium",
                "catalog_name": f"osc_{sn_type}",
                "catalog_confidence": "medium",
                "qa_flags": f"sn_type={sn_type}",
                "coverage_status": "unknown",
                "transition_type": "",
            }
        )
    return _standard_columns(pd.DataFrame(rows))


def _ingest_expansion_sn(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame(
        {
            "source_id": df.get("source_id", "").astype(str),
            "ra": df.get("ra", np.nan),
            "dec": df.get("dec", np.nan),
            "redshift": df.get("redshift", 0.0),
            "class": "sn",
            "reference": df.get("label_source", "sn_expansion"),
            "evidence_tier": 2,
            "provenance_group": "LIT_MEDIUM",
            "label_source": df.get("label_source", "sn_expansion"),
            "label_method": "literature_compilation",
            "confidence": "medium",
            "catalog_name": "sn_expansion",
            "catalog_confidence": "medium",
            "qa_flags": df.get("sn_type", ""),
            "coverage_status": "unknown",
            "transition_type": "",
        }
    )
    return _standard_columns(out)


def _ingest_expansion_blazar(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame(
        {
            "source_id": df.get("source_id", "").astype(str),
            "ra": df.get("ra", np.nan),
            "dec": df.get("dec", np.nan),
            "redshift": df.get("redshift", 0.0),
            "class": "blazar",
            "reference": df.get("label_source", "blazar_expansion"),
            "evidence_tier": 2,
            "provenance_group": "LIT_MEDIUM",
            "label_source": df.get("label_source", "blazar_expansion"),
            "label_method": "literature_compilation",
            "confidence": "medium",
            "catalog_name": "blazar_expansion",
            "catalog_confidence": "medium",
            "qa_flags": df.get("bztype", ""),
            "coverage_status": "unknown",
            "transition_type": "",
        }
    )
    return _standard_columns(out)


def _ingest_expansion_star(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame(
        {
            "source_id": df.get("source_id", "").astype(str),
            "ra": df.get("ra", np.nan),
            "dec": df.get("dec", np.nan),
            "redshift": df.get("redshift", 0.0),
            "class": "star",
            "reference": df.get("label_source", "star_expansion"),
            "evidence_tier": 2,
            "provenance_group": "CATALOG_HIGHCONF",
            "label_source": df.get("label_source", "star_expansion"),
            "label_method": "catalog",
            "confidence": "medium",
            "catalog_name": "star_expansion",
            "catalog_confidence": "high",
            "qa_flags": df.get("var_type", ""),
            "coverage_status": "unknown",
            "transition_type": "",
        }
    )
    return _standard_columns(out)


def _ingest_expansion_agn(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame(
        {
            "source_id": df.get("source_id", "").astype(str),
            "ra": df.get("ra", np.nan),
            "dec": df.get("dec", np.nan),
            "redshift": df.get("redshift", 0.0),
            "class": "normal_agn",
            "reference": df.get("label_source", "agn_expansion"),
            "evidence_tier": 3,
            "provenance_group": "CATALOG_HIGHCONF",
            "label_source": df.get("label_source", "agn_expansion"),
            "label_method": "catalog",
            "confidence": "medium",
            "catalog_name": "agn_expansion",
            "catalog_confidence": "high",
            "qa_flags": df.get("z_bin", ""),
            "coverage_status": "unknown",
            "transition_type": "",
        }
    )
    return _standard_columns(out)


def _ingest_expansion_clagn(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if "coverage_pass" in df.columns:
        cov = df["coverage_pass"].astype(str).str.lower().isin({"1", "true", "yes"})
        df = df[cov].copy()
    elif {"n_clean_epochs", "n_season_bins_total"}.issubset(df.columns):
        n_clean = pd.to_numeric(df["n_clean_epochs"], errors="coerce").fillna(0)
        n_seas = pd.to_numeric(df["n_season_bins_total"], errors="coerce").fillna(0)
        df = df[(n_clean >= 10) & (n_seas >= 3)].copy()
    out = pd.DataFrame(
        {
            "source_id": df.get("source_id", "").astype(str),
            "ra": df.get("ra", np.nan),
            "dec": df.get("dec", np.nan),
            "redshift": df.get("redshift", 0.0),
            "class": "clagn",
            "reference": df.get("reference", "clagn_coverage_report"),
            "evidence_tier": pd.to_numeric(df.get("evidence_tier", 2), errors="coerce").fillna(2).astype(int),
            "provenance_group": "LIT_MEDIUM",
            "label_source": df.get("reference", "clagn_coverage_report"),
            "label_method": "literature_compilation",
            "confidence": "medium",
            "catalog_name": "clagn_coverage_report",
            "catalog_confidence": "medium",
            "qa_flags": "",
            "coverage_status": "coverage_pass",
            "transition_type": df.get("transition_type", ""),
        }
    )
    return _standard_columns(out)


def ingest_registry(raw_dir: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    expansion_dir = Path("data/expansion")
    sources: list[tuple[str, pd.DataFrame]] = [
        ("real_benchmark", _ingest_real_benchmark(Path("data/real_clagn/benchmark_master.csv"))),
        ("known_clagn", _ingest_known_clagn(Path("data/benchmark/known_clagn.csv"))),
        ("known_contaminants", _ingest_known_contaminants(Path("data/benchmark/known_contaminants.csv"))),
        ("normal_agn_control", _ingest_normal_agn_control(Path("data/benchmark/normal_agn_control.csv"))),
        ("benchmark_master_v3", _ingest_benchmark_v3(Path("data/benchmark/benchmark_master_v3.csv"))),
        ("dr16q_vizier", _ingest_dr16q(raw_dir / "dr16q_vizier.tsv")),
        ("milliquas_vizier", _ingest_milliquas(raw_dir / "milliquas_vizier.tsv")),
        ("bzcat5", _ingest_bzcat(raw_dir / "bzcat5.tsv")),
        ("4fgl_dr4", _ingest_fermi(raw_dir / "4fgl_dr4.fit")),
        ("osc_iin", _ingest_osc_json(raw_dir / "osc_IIn.json", "IIn")),
        ("osc_iip", _ingest_osc_json(raw_dir / "osc_IIP.json", "IIP")),
        ("osc_ia", _ingest_osc_json(raw_dir / "osc_Ia.json", "Ia")),
        ("osc_ib", _ingest_osc_json(raw_dir / "osc_Ib.json", "Ib")),
        ("osc_ic", _ingest_osc_json(raw_dir / "osc_Ic.json", "Ic")),
        ("osc_iib", _ingest_osc_json(raw_dir / "osc_IIb.json", "IIb")),
        ("osc_ibn", _ingest_osc_json(raw_dir / "osc_Ibn.json", "Ibn")),
        ("osc_slsn", _ingest_osc_json(raw_dir / "osc_SLSN.json", "SLSN")),
        ("osc_lbv", _ingest_osc_json(raw_dir / "osc_LBV.json", "LBV")),
        ("expansion_sn", _ingest_expansion_sn(expansion_dir / "sn_expansion.csv")),
        ("expansion_blazar", _ingest_expansion_blazar(expansion_dir / "blazar_expansion.csv")),
        ("expansion_star", _ingest_expansion_star(expansion_dir / "star_expansion.csv")),
        ("expansion_agn", _ingest_expansion_agn(expansion_dir / "agn_expansion.csv")),
        ("expansion_clagn", _ingest_expansion_clagn(expansion_dir / "clagn_coverage_report.csv")),
    ]
    counts = {name: int(len(df)) for name, df in sources}
    df = pd.concat([df for _, df in sources if not df.empty], ignore_index=True)
    df = _standard_columns(df)
    return df, counts


def deduplicate_registry(df: pd.DataFrame, radius_arcsec: float) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    work = df.copy()
    work["source_id"] = work["source_id"].astype(str).str.strip()
    work = add_canonical_id(work, source_col="source_id")
    missing_canon = work["canonical_id"].astype(str).eq("")
    if missing_canon.any():
        work.loc[missing_canon, "canonical_id"] = (
            "COORD_"
            + work.loc[missing_canon, "ra"].round(6).astype(str)
            + "_"
            + work.loc[missing_canon, "dec"].round(6).astype(str)
        )

    work = _row_priority(work)

    # pass 1: exact canonical_id dedup (keep best ranked row)
    work = work.sort_values(["canonical_id", "_prov_rank", "_tier_rank", "_conf_rank", "_id_rank"]).reset_index(drop=True)
    keep_idx = work.groupby("canonical_id", sort=False).head(1).index
    exact_out = work.loc[keep_idx].copy().reset_index(drop=True)
    exact_removed = int(len(work) - len(exact_out))

    # pass 2: coordinate dedup via union-find within radius
    coords = SkyCoord(ra=exact_out["ra"].to_numpy(dtype=float) * u.deg, dec=exact_out["dec"].to_numpy(dtype=float) * u.deg)
    i_idx, j_idx, _, _ = coords.search_around_sky(coords, radius_arcsec * u.arcsec)
    uf = UnionFind(len(exact_out))
    for a, b in zip(i_idx.tolist(), j_idx.tolist()):
        if a < b:
            uf.union(a, b)

    groups: dict[int, list[int]] = {}
    for i in range(len(exact_out)):
        groups.setdefault(uf.find(i), []).append(i)

    dedup_rows = []
    audit_rows = []
    for n, members in enumerate(groups.values(), start=1):
        grp = exact_out.iloc[members].copy()
        grp = grp.sort_values(["_prov_rank", "_tier_rank", "_conf_rank", "_id_rank"])
        kept = grp.iloc[0]
        dedup_group_id = f"DG{n:07d}"
        row = kept.to_dict()
        row["dedup_group_id"] = dedup_group_id
        dedup_rows.append(row)
        audit_rows.append(
            {
                "dedup_group_id": dedup_group_id,
                "kept_canonical_id": kept["canonical_id"],
                "kept_source_id": kept["source_id"],
                "n_members": int(len(grp)),
                "member_canonical_ids": "|".join(grp["canonical_id"].astype(str).tolist()),
                "member_source_ids": "|".join(grp["source_id"].astype(str).tolist()),
                "member_classes": "|".join(sorted(set(grp["class"].astype(str).tolist()))),
                "member_references": "|".join(sorted(set(grp["reference"].astype(str).tolist()))),
                "member_provenance_groups": "|".join(sorted(set(grp["provenance_group"].astype(str).tolist()))),
            }
        )

    dedup_df = pd.DataFrame(dedup_rows).reset_index(drop=True)
    audit_df = pd.DataFrame(audit_rows).sort_values("dedup_group_id").reset_index(drop=True)
    coord_removed = int(len(exact_out) - len(dedup_df))
    stats = {
        "rows_before_dedup": int(len(df)),
        "rows_after_exact_dedup": int(len(exact_out)),
        "rows_after_coord_dedup": int(len(dedup_df)),
        "exact_removed": exact_removed,
        "coord_removed": coord_removed,
    }
    drop_cols = ["_prov_rank", "_tier_rank", "_conf_rank", "_id_rank"]
    dedup_df = dedup_df.drop(columns=[c for c in drop_cols if c in dedup_df.columns])
    return dedup_df, audit_df, stats


def _stratified_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    if not _HAS_SKLEARN:
        raise RuntimeError("scikit-learn required for stratified split")
    y = df["class"].astype(str).to_numpy()
    idx_all = np.arange(len(df))
    s1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(s1.split(idx_all, y))
    y_temp = y[temp_idx]
    s2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
    dev_rel, test_rel = next(s2.split(temp_idx, y_temp))
    dev_idx = temp_idx[dev_rel]
    test_idx = temp_idx[test_rel]

    out = df.copy()
    out["split"] = "train"
    out.loc[dev_idx, "split"] = "dev"
    out.loc[test_idx, "split"] = "test"
    return out


def _sample_with_redshift_strata(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    rng = np.random.default_rng(seed)
    bins = [-np.inf, 0.1, 0.3, 0.6, 1.0, 2.0, np.inf]
    labels = np.digitize(df["redshift"].to_numpy(dtype=float), bins)
    work = df.copy()
    work["_zbin"] = labels
    groups = {b: g.index.to_numpy() for b, g in work.groupby("_zbin")}
    # Allocate by availability with largest-remainder.
    total = len(work)
    alloc = {}
    fracs = []
    used = 0
    for b, idxs in groups.items():
        raw = n * (len(idxs) / total)
        base = int(math.floor(raw))
        alloc[b] = min(base, len(idxs))
        used += alloc[b]
        fracs.append((raw - base, b))
    rem = n - used
    for _, b in sorted(fracs, reverse=True):
        if rem <= 0:
            break
        if alloc[b] < len(groups[b]):
            alloc[b] += 1
            rem -= 1
    picks = []
    for b, idxs in groups.items():
        k = alloc.get(b, 0)
        if k <= 0:
            continue
        if len(idxs) <= k:
            picks.extend(idxs.tolist())
        else:
            picks.extend(rng.choice(idxs, size=k, replace=False).tolist())
    # Fallback fill if rounding left short.
    if len(picks) < n:
        remaining = np.setdiff1d(work.index.to_numpy(), np.array(picks, dtype=int), assume_unique=False)
        extra_k = min(n - len(picks), len(remaining))
        if extra_k > 0:
            picks.extend(rng.choice(remaining, size=extra_k, replace=False).tolist())
    out = work.loc[picks].drop(columns=["_zbin"]).copy()
    return out


def build_eval_core(
    df: pd.DataFrame,
    ctx: BuildContext,
    allow_underfill_rebalance: bool = False,
    eval_mode: str = "strict_gold",
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, int] | None]:
    if eval_mode == "gold_plus":
        eligible_prov = ["LIT_STRONG", "LIT_MEDIUM", "CATALOG_HIGHCONF"]
    else:
        eligible_prov = ["LIT_STRONG", "LIT_MEDIUM"]

    lit = df[df["provenance_group"].isin(eligible_prov)].copy()
    lit = lit[lit["class"].isin(EVAL_CLASSES)].copy()
    avail = lit["class"].value_counts().to_dict()
    avail = {c: int(avail.get(c, 0)) for c in EVAL_CLASSES}

    floor = 250
    deficits = {c: max(0, floor - n) for c, n in avail.items()}
    if any(v > 0 for v in deficits.values()):
        if not allow_underfill_rebalance:
            return (
                pd.DataFrame(),
                {
                    "available_by_class": avail,
                    "floor": floor,
                    "deficits": deficits,
                    "eval_mode": eval_mode,
                    "eligible_provenance_groups": eligible_prov,
                },
                deficits,
            )

    initial_per_class = ctx.eval_target_total // len(EVAL_CLASSES)
    target_per_class = min(initial_per_class, min(avail.values()))
    if target_per_class <= 0:
        return (
            pd.DataFrame(),
            {
                "available_by_class": avail,
                "floor": floor,
                "deficits": deficits,
                "error": "At least one class has 0 available rows.",
            },
            deficits,
        )
    targets = {c: int(target_per_class) for c in EVAL_CLASSES}

    sampled_parts = []
    for i, c in enumerate(EVAL_CLASSES):
        cls_df = lit[lit["class"] == c].copy()
        cls_df = _row_priority(cls_df).sort_values(["_prov_rank", "_tier_rank", "_conf_rank", "_id_rank"])
        cls_df = cls_df.drop(columns=["_prov_rank", "_tier_rank", "_conf_rank", "_id_rank"])
        sample = _sample_with_redshift_strata(cls_df, targets[c], seed=ctx.seed + i * 17)
        sampled_parts.append(sample)
    eval_df = pd.concat(sampled_parts, ignore_index=True)
    eval_df = ensure_class_ytrue(eval_df)
    eval_df = _stratified_split(eval_df, seed=ctx.seed)

    required = [
        "source_id",
        "ra",
        "dec",
        "redshift",
        "class",
        "y_true",
        "label_source",
        "label_method",
        "confidence",
        "canonical_id",
        "split",
    ]
    extra = [
        "reference",
        "evidence_tier",
        "provenance_group",
        "dedup_group_id",
        "coverage_status",
        "transition_type",
    ]
    out_cols = required + extra
    for c in out_cols:
        if c not in eval_df.columns:
            eval_df[c] = ""
    eval_df["source_id"] = eval_df["source_id"].astype(str).fillna("")
    eval_df["label_source"] = eval_df["label_source"].astype(str).fillna("")
    eval_df["label_method"] = eval_df["label_method"].astype(str).fillna("literature")
    eval_df["confidence"] = eval_df["confidence"].fillna("unknown").astype(str)
    eval_df["canonical_id"] = eval_df["canonical_id"].fillna("").astype(str)
    eval_df["split"] = eval_df["split"].fillna("").astype(str)
    eval_df["class"] = eval_df["class"].fillna("").astype(str)
    eval_df["y_true"] = pd.to_numeric(eval_df["y_true"], errors="coerce").fillna(0).astype(int)
    eval_df = eval_df[out_cols].sort_values(["class", "split", "canonical_id"]).reset_index(drop=True)
    meta = {
        "available_by_class": avail,
        "target_per_class": targets,
        "final_per_class": eval_df["class"].value_counts().to_dict(),
        "underfill_rebalanced": bool(any(v > 0 for v in deficits.values())),
        "literature_floor": floor,
        "deficits": deficits,
        "eval_mode": eval_mode,
        "eligible_provenance_groups": eligible_prov,
    }
    return eval_df, meta, (deficits if any(v > 0 for v in deficits.values()) else None)


def build_discovery_pool(df: pd.DataFrame, eval_ids: set[str], ctx: BuildContext) -> tuple[pd.DataFrame, dict[str, Any]]:
    pool = df[~df["canonical_id"].isin(eval_ids)].copy()
    pool = pool[pool["class"].isin(EVAL_CLASSES)].copy()
    available = pool["class"].value_counts().to_dict()

    soft_ratio = {"normal_agn": 0.85, "blazar": 0.08, "sn": 0.03, "star": 0.03, "clagn": 0.01}
    desired = {k: int(round(ctx.discovery_target_total * v)) for k, v in soft_ratio.items()}
    desired["normal_agn"] += ctx.discovery_target_total - sum(desired.values())

    rng = np.random.default_rng(ctx.seed + 1000)
    parts = []
    selected = 0
    actual_targets = {}
    for c in ["clagn", "sn", "star", "blazar", "normal_agn"]:
        sub = pool[pool["class"] == c]
        k = min(desired.get(c, 0), len(sub))
        actual_targets[c] = int(k)
        if k > 0:
            idx = rng.choice(sub.index.to_numpy(), size=k, replace=False) if len(sub) > k else sub.index.to_numpy()
            parts.append(pool.loc[idx])
            selected += k
    if selected < ctx.discovery_target_total:
        remaining = pool.drop(index=pd.concat(parts).index if parts else []).copy()
        k = min(ctx.discovery_target_total - selected, len(remaining))
        if k > 0:
            idx = rng.choice(remaining.index.to_numpy(), size=k, replace=False)
            parts.append(remaining.loc[idx])
            selected += k

    disc = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=pool.columns)
    disc = disc.rename(columns={"class": "label"})
    out_cols = [
        "source_id",
        "ra",
        "dec",
        "redshift",
        "label",
        "reference",
        "evidence_tier",
        "canonical_id",
        "catalog_name",
        "catalog_confidence",
        "provenance_group",
        "qa_flags",
    ]
    for c in out_cols:
        if c not in disc.columns:
            disc[c] = ""
    disc = disc[out_cols].sort_values(["label", "canonical_id"]).reset_index(drop=True)
    meta = {
        "available_by_label": {k: int(v) for k, v in available.items()},
        "desired_targets": desired,
        "actual_targets": actual_targets,
        "final_total": int(len(disc)),
        "final_by_label": disc["label"].value_counts().to_dict(),
    }
    return disc, meta


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    if not _HAS_SKLEARN:
        raise SystemExit("scikit-learn is required (StratifiedShuffleSplit not available).")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = BuildContext(
        seed=args.seed,
        dedup_radius_arcsec=args.dedup_radius_arcsec,
        eval_target_total=args.eval_target_total,
        discovery_target_total=args.discovery_target_total,
    )

    registry, ingest_counts = ingest_registry(Path(args.raw_dir))
    registry_dedup, audit_df, dedup_stats = deduplicate_registry(registry, radius_arcsec=ctx.dedup_radius_arcsec)

    eval_df, eval_meta, deficits = build_eval_core(
        registry_dedup,
        ctx=ctx,
        allow_underfill_rebalance=args.allow_underfill_rebalance,
        eval_mode=args.eval_mode,
    )
    deficit_path = out_dir / "v4_deficit_report.json"
    if deficits:
        if args.allow_underfill_rebalance:
            deficit_report = {
                "status": "UNDERFILL_REBALANCED",
                "message": (
                    "Literature floor (<250/class) not met. Continued with balanced "
                    "downscaled eval core using min available class count."
                ),
                "eval_meta": eval_meta,
                "seed": args.seed,
                "dedup_radius_arcsec": args.dedup_radius_arcsec,
            }
            deficit_path.write_text(json.dumps(deficit_report, indent=2), encoding="utf-8")
            print(f"Wrote deficit report: {deficit_path}")
        else:
            deficit_report = {
                "status": "UNDERFILL_HARD_FAIL",
                "message": "At least one eval class has <250 literature rows.",
                "eval_meta": eval_meta,
                "seed": args.seed,
                "dedup_radius_arcsec": args.dedup_radius_arcsec,
            }
            deficit_path.write_text(json.dumps(deficit_report, indent=2), encoding="utf-8")
            print(f"Wrote deficit report: {deficit_path}")
            if args.emit_deficit_only:
                return
            raise SystemExit(2)

    eval_master = out_dir / "benchmark_eval_master_v4.csv"
    eval_train = out_dir / "benchmark_eval_train_v4.csv"
    eval_dev = out_dir / "benchmark_eval_dev_v4.csv"
    eval_test = out_dir / "benchmark_eval_test_v4.csv"
    audit_path = out_dir / "v4_provenance_audit.csv"
    split_hash_path = out_dir / "v4_split_hash.json"
    report_path = out_dir / "v4_build_report.json"
    discovery_path = out_dir / "benchmark_discovery_master_v4.csv"

    eval_df.to_csv(eval_master, index=False)
    eval_df[eval_df["split"] == "train"].to_csv(eval_train, index=False)
    eval_df[eval_df["split"] == "dev"].to_csv(eval_dev, index=False)
    eval_df[eval_df["split"] == "test"].to_csv(eval_test, index=False)
    audit_df.to_csv(audit_path, index=False)

    eval_ids = set(eval_df["canonical_id"].astype(str))
    discovery_df, discovery_meta = build_discovery_pool(registry_dedup, eval_ids=eval_ids, ctx=ctx)
    discovery_df.to_csv(discovery_path, index=False)

    split_hash = {
        "seed": args.seed,
        "train": {"rows": int((eval_df["split"] == "train").sum()), "sha256": _sha256(eval_train)},
        "dev": {"rows": int((eval_df["split"] == "dev").sum()), "sha256": _sha256(eval_dev)},
        "test": {"rows": int((eval_df["split"] == "test").sum()), "sha256": _sha256(eval_test)},
        "master": {"rows": int(len(eval_df)), "sha256": _sha256(eval_master)},
    }
    split_hash_path.write_text(json.dumps(split_hash, indent=2), encoding="utf-8")

    report = {
        "ingest_counts": ingest_counts,
        "dedup_stats": dedup_stats,
        "eval": eval_meta,
        "discovery": discovery_meta,
        "files": {
            "eval_master": str(eval_master),
            "eval_train": str(eval_train),
            "eval_dev": str(eval_dev),
            "eval_test": str(eval_test),
            "discovery_master": str(discovery_path),
            "provenance_audit": str(audit_path),
            "split_hash": str(split_hash_path),
        },
        "config": {
            "eval_target_total": args.eval_target_total,
            "discovery_target_total": args.discovery_target_total,
            "seed": args.seed,
            "dedup_radius_arcsec": args.dedup_radius_arcsec,
            "eval_mode": args.eval_mode,
            "policy_eval_provenance": eval_meta.get("eligible_provenance_groups", []),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote: {eval_master}")
    print(f"Wrote: {eval_train}")
    print(f"Wrote: {eval_dev}")
    print(f"Wrote: {eval_test}")
    print(f"Wrote: {discovery_path}")
    print(f"Wrote: {audit_path}")
    print(f"Wrote: {split_hash_path}")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
