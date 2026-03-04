#!/usr/bin/env python3
"""
Check WISE NEOWISE coverage for CLAGN literature candidates.

Coordinates are hardcoded in LITERATURE_SOURCES — no SIMBAD lookup needed.
For each source:
  - Dedup check against existing benchmark master (within dedup_radius_arcsec)
  - WISE NEOWISE coverage check via IRSA TAP
  - Pass criterion: n_clean >= 10 AND (season_bins_w1 + season_bins_w2) >= 3

Usage:
    python scripts/expand_clagn_catalog.py --output data/expansion/clagn_coverage_report.csv

Output columns:
    source_id, ra, dec, redshift, transition_type, reference, evidence_tier,
    skipped, skip_reason,
    n_clean_epochs, n_season_bins_w1, n_season_bins_w2, n_season_bins_total,
    coverage_pass
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.io import ascii as astro_ascii
import astropy.units as u

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Source list (85 candidates from literature — directly from MD file)
# Format: (name, ra_deg, dec_deg, z, transition_type, reference, evidence_tier)
# ---------------------------------------------------------------------------
LITERATURE_SOURCES: list[tuple] = [
    # MacLeod+2019 Table 1 — confirmed CLAGN from SDSS
    ("SDSS J015957.64+003310.5", 29.990,  0.553,  0.312, "turn_off", "MacLeod+2019", 1),
    ("SDSS J022556.07+003026.7", 36.484,  0.507,  0.504, "turn_on",  "MacLeod+2019", 1),
    ("SDSS J100220.17+450927.3", 150.584, 45.158, 0.179, "turn_off", "MacLeod+2019", 1),
    ("SDSS J102152.34+464515.6", 155.468, 46.754, 0.204, "turn_on",  "MacLeod+2019", 1),
    ("SDSS J141324.27+530527.0", 213.351, 53.091, 0.457, "turn_off", "MacLeod+2019", 1),
    ("SDSS J233602.98+001728.7", 354.012,  0.291, 0.243, "turn_on",  "MacLeod+2019", 1),
    ("SDSS J012648.08-083948.0",  21.700, -8.663, 0.192, "turn_off", "MacLeod+2019", 1),
    ("SDSS J075958.27+382422.4", 119.993, 38.406, 0.236, "turn_on",  "MacLeod+2019", 1),
    ("SDSS J094929.75+195559.0", 147.374, 19.933, 0.274, "turn_off", "MacLeod+2019", 1),
    ("SDSS J124634.65+023809.0", 191.644,  2.636, 0.374, "turn_on",  "MacLeod+2019", 1),
    ("SDSS J155440.25+362952.0", 238.668, 36.498, 0.194, "turn_off", "MacLeod+2019", 1),
    ("SDSS J161341.06+343237.9", 243.421, 34.544, 0.224, "turn_on",  "MacLeod+2019", 1),
    ("SDSS J214613.30+000930.0", 326.556,  0.158, 0.165, "turn_off", "MacLeod+2019", 1),

    # Sheng+2017 MIR confirmed CLAGN (all have WISE data)
    ("Mrk 1018",     32.100,  -0.293, 0.042, "turn_off", "Cohen+1986,McElroy+2016", 1),
    ("Mrk 590",      33.640,  -0.767, 0.026, "turn_off", "Denney+2014",             1),
    ("NGC 1566",     65.002, -54.938, 0.005, "turn_on",  "Oknyansky+2019",          1),
    ("NGC 2617",    129.568,  35.818, 0.014, "turn_on",  "Shappee+2014",            1),
    ("NGC 3516",    166.698,  72.568, 0.009, "unknown",  "Shapovalova+2019",        1),
    ("NGC 4151",    182.636,  39.406, 0.003, "turn_on",  "Shapovalova+2010",        1),
    ("NGC 7603",    349.722,   0.244, 0.030, "turn_on",  "Kollatschny+2000",        1),
    ("Mrk 1048",     37.490,  -0.035, 0.043, "turn_off", "Hutsemakers+2019",        1),
    ("1ES 1927+654", 292.030,  65.565, 0.019, "turn_on", "Trakhtenbrot+2019",       1),
    ("NGC 1365",     53.402, -36.140, 0.005, "unknown",  "Rivers+2015",             1),
    ("Fairall 9",    20.940, -58.806, 0.047, "turn_off", "Lohfink+2014",            1),
    ("NGC 4395",    186.454,  33.547, 0.001, "turn_on",  "Lira+1999",               1),

    # Sheng+2020 MIR CLAGN extension sample
    ("SDSS J022710.97-080332.0",  36.796,  -8.059, 0.289, "turn_off", "Sheng+2020", 1),
    ("SDSS J080101.41+184840.7", 120.256,  18.811, 0.256, "turn_on",  "Sheng+2020", 1),
    ("SDSS J082323.42+042048.1", 125.848,   4.347, 0.381, "turn_off", "Sheng+2020", 1),
    ("SDSS J091709.56+463821.0", 139.290,  46.639, 0.264, "turn_on",  "Sheng+2020", 1),
    ("SDSS J100428.43+321847.8", 151.118,  32.313, 0.411, "turn_off", "Sheng+2020", 1),
    ("SDSS J105053.69+525348.4", 162.724,  52.897, 0.307, "turn_on",  "Sheng+2020", 1),
    ("SDSS J112358.96+401948.7", 170.996,  40.330, 0.197, "turn_off", "Sheng+2020", 1),
    ("SDSS J115855.80+141402.0", 179.732,  14.234, 0.473, "turn_on",  "Sheng+2020", 1),
    ("SDSS J132753.78+565631.4", 201.974,  56.942, 0.288, "turn_off", "Sheng+2020", 1),
    ("SDSS J142536.79+344542.5", 216.403,  34.762, 0.359, "turn_on",  "Sheng+2020", 1),
    ("SDSS J150405.78+261305.3", 226.024,  26.218, 0.402, "turn_off", "Sheng+2020", 1),
    ("SDSS J155852.29+103530.3", 239.718,  10.592, 0.337, "turn_on",  "Sheng+2020", 1),
    ("SDSS J163350.55+373411.8", 248.461,  37.570, 0.271, "turn_off", "Sheng+2020", 1),
    ("SDSS J165732.15+281106.0", 254.384,  28.185, 0.318, "turn_on",  "Sheng+2020", 1),

    # Hon+2022 ZTF CLAGN (need WISE crosscheck)
    ("SDSS J002311.06+003517.5",   5.796,   0.588, 0.422, "turn_off", "Hon+2022", 2),
    ("SDSS J012217.73+052545.4",  20.574,   5.429, 0.348, "turn_on",  "Hon+2022", 2),
    ("SDSS J021225.23+010056.5",  33.105,   1.016, 0.534, "turn_off", "Hon+2022", 2),
    ("SDSS J032315.71+411250.0",  50.815,  41.214, 0.278, "turn_on",  "Hon+2022", 2),
    ("SDSS J073203.00+435617.4", 113.013,  43.938, 0.196, "turn_off", "Hon+2022", 2),
    ("SDSS J080957.97+181804.3", 122.492,  18.301, 0.347, "turn_on",  "Hon+2022", 2),
    ("SDSS J085039.94+241338.5", 132.666,  24.227, 0.509, "turn_off", "Hon+2022", 2),
    ("SDSS J090040.91+245719.2", 135.170,  24.955, 0.411, "turn_on",  "Hon+2022", 2),
    ("SDSS J091227.11+065258.5", 138.113,   6.883, 0.355, "turn_off", "Hon+2022", 2),
    ("SDSS J101220.49+632806.3", 153.085,  63.468, 0.284, "turn_on",  "Hon+2022", 2),
    ("SDSS J111322.76+212912.3", 168.345,  21.487, 0.468, "turn_off", "Hon+2022", 2),
    ("SDSS J113851.36+405245.7", 174.714,  40.879, 0.323, "turn_on",  "Hon+2022", 2),
    ("SDSS J121230.95+295849.2", 183.129,  29.980, 0.391, "turn_off", "Hon+2022", 2),
    ("SDSS J125343.60+122721.6", 193.432,  12.456, 0.267, "turn_on",  "Hon+2022", 2),
    ("SDSS J134246.25+284027.4", 205.693,  28.674, 0.444, "turn_off", "Hon+2022", 2),
    ("SDSS J143030.22+353808.4", 217.626,  35.636, 0.315, "turn_on",  "Hon+2022", 2),
    ("SDSS J151610.02+404333.3", 229.042,  40.726, 0.487, "turn_off", "Hon+2022", 2),
    ("SDSS J155419.60+391048.9", 238.582,  39.180, 0.298, "turn_on",  "Hon+2022", 2),
    ("SDSS J163049.96+281736.7", 247.708,  28.293, 0.362, "turn_off", "Hon+2022", 2),
    ("SDSS J171223.13+320757.5", 258.096,  32.133, 0.419, "turn_on",  "Hon+2022", 2),

    # Frederick+2019 LINER CLAGN
    ("SDSS J015716.82+010853.7",  29.320,   1.148, 0.083, "turn_on",  "Frederick+2019", 1),
    ("SDSS J101152.98+544206.4", 152.971,  54.702, 0.076, "turn_off", "Frederick+2019", 1),
    ("SDSS J120556.03+120900.5", 181.483,  12.150, 0.059, "turn_on",  "Frederick+2019", 1),
    ("SDSS J123359.12+084211.5", 188.496,   8.703, 0.089, "turn_off", "Frederick+2019", 1),
    ("SDSS J160152.77+331807.8", 240.470,  33.302, 0.073, "turn_on",  "Frederick+2019", 1),
    ("SDSS J163459.82+204936.4", 248.749,  20.827, 0.091, "turn_off", "Frederick+2019", 1),

    # Green+2022 PanSTARRS confirmed
    ("SDSS J030639.57+000343.2",  46.665,   0.062, 0.292, "turn_off", "Green+2022", 1),
    ("SDSS J043443.28-042048.3",  68.680,  -4.347, 0.468, "turn_on",  "Green+2022", 1),
    ("SDSS J084035.09+562419.4", 130.146,  56.405, 0.339, "turn_off", "Green+2022", 1),
    ("SDSS J104732.68+472532.0", 161.886,  47.425, 0.271, "turn_on",  "Green+2022", 1),
    ("SDSS J130621.19+291656.0", 196.588,  29.282, 0.388, "turn_off", "Green+2022", 1),
    ("SDSS J143450.62+033842.5", 218.711,   3.645, 0.325, "turn_on",  "Green+2022", 1),
    ("SDSS J160234.29+335931.5", 240.643,  33.992, 0.411, "turn_off", "Green+2022", 1),
    ("SDSS J172031.25+280501.6", 260.130,  28.084, 0.359, "turn_on",  "Green+2022", 1),

    # Additional nearby Seyferts with known variability
    ("NGC 5548",  214.498,  25.137, 0.017, "turn_on",  "Mathur+2017",       1),
    ("NGC 4051",  180.790,  44.531, 0.002, "turn_off", "Lamer+2003",        1),
    ("NGC 7469",  345.815,   8.874, 0.016, "turn_on",  "Kollatschny+2018",  1),
    ("NGC 3783",  174.757, -37.739, 0.010, "turn_off", "Mehdipour+2017",    1),
    ("NGC 4593",  189.914,  -5.344, 0.009, "turn_on",  "McHardy+2018",      1),
    ("Mrk 817",   217.272,  58.794, 0.031, "turn_on",  "Kaastra+2022",      1),
    ("Mrk 335",     1.580,  20.203, 0.026, "turn_off", "Parker+2019",       1),
]

# ---------------------------------------------------------------------------
# WISE NEOWISE coverage via IRSA TAP
# ---------------------------------------------------------------------------

def _irsa_wise_coverage(ra: float, dec: float, radius_arcsec: float = 3.0) -> dict:
    """
    Query IRSA NEOWISE for clean photometry around (ra, dec).
    Returns n_clean, season bins per band, and coverage_pass flag.
    """
    url = "https://irsa.ipac.caltech.edu/TAP/sync"
    table_name = "neowiser_p1bs_psd"
    debug_enabled = os.getenv("CLAGN_IRSA_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    timeout_s = int(os.getenv("CLAGN_IRSA_TIMEOUT_S", "180"))
    max_retries = int(os.getenv("CLAGN_IRSA_MAX_RETRIES", "3"))

    def _count_seasons(mjds: list) -> int:
        if not mjds:
            return 0
        arr = sorted(float(m) for m in mjds)
        gaps = np.diff(arr)
        return int((gaps > 100).sum()) + 1

    def _apply_local_qc(df: pd.DataFrame) -> pd.DataFrame:
        q = df.copy()
        if "qi_fact" in q.columns:
            q["qi_fact"] = pd.to_numeric(q["qi_fact"], errors="coerce")
            q = q[q["qi_fact"] >= 1]
        if "saa_sep" in q.columns:
            q["saa_sep"] = pd.to_numeric(q["saa_sep"], errors="coerce")
            q = q[q["saa_sep"] >= 5]
        if "moon_masked" in q.columns:
            q["moon_masked"] = pd.to_numeric(q["moon_masked"], errors="coerce")
            q = q[q["moon_masked"] == 0]
        return q

    def _from_df(df: pd.DataFrame) -> tuple[dict, int, int]:
        if df.empty:
            return ({
                "n_clean": 0, "n_seasons_w1": 0, "n_seasons_w2": 0,
                "n_seasons_total": 0, "coverage_pass": False,
            }, 0, 0)
        df = df.copy()
        df["mjd"] = pd.to_numeric(df["mjd"], errors="coerce")
        df = df.dropna(subset=["mjd"]).sort_values("mjd")
        if df.empty:
            return ({
                "n_clean": 0, "n_seasons_w1": 0, "n_seasons_w2": 0,
                "n_seasons_total": 0, "coverage_pass": False,
            }, 0, 0)

        parsed_rows = len(df)
        df_qc = _apply_local_qc(df)
        qa_rows = len(df_qc)
        n_clean = qa_rows

        w1_mjds = []
        if "w1sigmpro" in df_qc.columns:
            w1_sig = pd.to_numeric(df_qc["w1sigmpro"], errors="coerce")
            w1_mjds = df_qc.loc[w1_sig < 0.2, "mjd"].tolist()

        w2_mjds = []
        if "w2sigmpro" in df_qc.columns:
            w2_sig = pd.to_numeric(df_qc["w2sigmpro"], errors="coerce")
            w2_mjds = df_qc.loc[w2_sig < 0.2, "mjd"].tolist()

        ns_w1 = _count_seasons(w1_mjds)
        ns_w2 = _count_seasons(w2_mjds)
        ns_total = ns_w1 + ns_w2
        return ({
            "n_clean": int(n_clean),
            "n_seasons_w1": int(ns_w1),
            "n_seasons_w2": int(ns_w2),
            "n_seasons_total": int(ns_total),
            "coverage_pass": bool((n_clean >= 10) and (ns_total >= 3)),
        }, parsed_rows, qa_rows)

    def _print_zero_debug(*, adql_query: str, status: int | None, content_type: str | None,
                          response_text: str, parsed_rows: int, qa_rows: int, stage: str) -> None:
        print(
            "[IRSA TAP DEBUG] "
            f"{stage} at ra={ra:.6f}, dec={dec:.6f} | endpoint={url} | table={table_name} | "
            f"status={status} | content_type={content_type} | "
            f"parsed_rows={parsed_rows} | qa_rows={qa_rows}"
        )
        print(f"[IRSA TAP DEBUG] adql={adql_query}")
        print(f"[IRSA TAP DEBUG] response_head={response_text[:400]!r}")
        if debug_enabled:
            print(f"[IRSA TAP DEBUG] response_full={response_text!r}")

    def _parse_tap_votable_json(payload: dict) -> tuple[pd.DataFrame, int]:
        # Preferred IRSA structure: VOTABLE -> RESOURCE_ARRAY[0] -> TABLE -> FIELD_ARRAY + DATA.TABLEDATA
        try:
            table = payload["VOTABLE"]["RESOURCE_ARRAY"][0]["TABLE"]
            fields = table.get("FIELD_ARRAY", [])
            cols = [f.get("<xmlattr>", {}).get("name") for f in fields]
            cols = [c for c in cols if c]
            table_data = table.get("DATA", {}).get("TABLEDATA", [])
            if isinstance(table_data, list):
                rows = table_data
            elif isinstance(table_data, dict):
                tr = table_data.get("TR_ARRAY", [])
                rows = []
                for r in tr:
                    if isinstance(r, dict):
                        rows.append(r.get("TD_ARRAY", []))
            else:
                rows = []
            if cols and rows:
                return pd.DataFrame(rows, columns=cols), len(rows)
            return pd.DataFrame(columns=cols), 0
        except Exception:
            pass

        # Back-compat alternate shape
        rows = payload.get("data", [])
        cols = [c.get("name") for c in payload.get("metadata", [])]
        if rows and cols:
            return pd.DataFrame(rows, columns=cols), len(rows)
        return pd.DataFrame(), 0

    def _fallback_gator() -> dict:
        gator_url = "https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query"
        params = {
            "catalog": table_name,
            "spatial": "cone",
            "objstr": f"{ra:.6f} {dec:.6f}",
            "radius": f"{radius_arcsec:.6f}",
            "radunits": "arcsec",
            "outfmt": "1",
            "selcols": "mjd,w1mpro,w1sigmpro,w2mpro,w2sigmpro,qi_fact,saa_sep,moon_masked",
        }
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(gator_url, params=params, timeout=timeout_s)
                resp.raise_for_status()
                txt = resp.text
                if "ERROR" in txt.upper():
                    return {
                        "n_clean": 0, "n_seasons_w1": 0, "n_seasons_w2": 0,
                        "n_seasons_total": 0, "coverage_pass": False,
                        "irsa_error": f"Gator error: {txt[:300]}",
                    }
                # astropy parser behavior varies by version; try multiple parse paths.
                table = None
                parse_errors = []
                for parser in (
                    lambda t: astro_ascii.read(t, format="ipac"),
                    lambda t: Table.read(t, format="ascii.ipac"),
                    lambda t: Table.read(io.StringIO(t), format="ascii.ipac"),
                ):
                    try:
                        table = parser(txt)
                        break
                    except Exception as pe:  # pragma: no cover - debug path
                        parse_errors.append(str(pe))
                if table is None:
                    raise RuntimeError(
                        "Could not parse IRSA Gator IPAC response: "
                        + " | ".join(parse_errors[:3])
                    )
                df = table.to_pandas()
                out, _, _ = _from_df(df)
                return out
            except Exception as exc:
                last_exc = exc
                if debug_enabled:
                    print(f"[IRSA GATOR DEBUG] attempt {attempt}/{max_retries} failed: {exc}")
                if attempt < max_retries:
                    time.sleep(1.5 * attempt)
        return {
            "n_clean": 0, "n_seasons_w1": 0, "n_seasons_w2": 0,
            "n_seasons_total": 0, "coverage_pass": False,
            "irsa_error": f"Gator fallback failed: {last_exc}",
        }

    adql = (
        f"SELECT mjd, w1mpro, w1sigmpro, w2mpro, w2sigmpro, qi_fact, saa_sep, moon_masked "
        f"FROM {table_name} "
        f"WHERE CONTAINS("
        f"  POINT('ICRS', ra, dec),"
        f"  CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius_arcsec / 3600.0:.8f})"
        f") = 1 "
        f"ORDER BY mjd"
    )
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(
                url,
                params={"QUERY": adql, "FORMAT": "json", "LANG": "ADQL"},
                timeout=timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
            df, parsed_rows = _parse_tap_votable_json(data)
            out, _, qa_rows = _from_df(df)
            if out["n_clean"] == 0:
                _print_zero_debug(
                    adql_query=adql,
                    status=resp.status_code,
                    content_type=resp.headers.get("content-type"),
                    response_text=resp.text,
                    parsed_rows=parsed_rows,
                    qa_rows=qa_rows,
                    stage="0 usable rows",
                )
                return _fallback_gator()
            return out
        except Exception as exc:
            last_exc = exc
            if debug_enabled:
                print(f"[IRSA TAP DEBUG] retry {attempt}/{max_retries} failed at ra={ra:.6f}, dec={dec:.6f}: {exc}")
            if attempt < max_retries:
                time.sleep(1.5 * attempt)

    print(
        f"[IRSA TAP DEBUG] exception at ra={ra:.6f}, dec={dec:.6f} | "
        f"endpoint={url} | table={table_name} | error={last_exc}"
    )
    print(f"[IRSA TAP DEBUG] adql={adql}")
    return _fallback_gator()


# ---------------------------------------------------------------------------
# Deduplication against existing benchmark
# ---------------------------------------------------------------------------

def _load_existing_coords(benchmark_path: Path) -> list[SkyCoord] | None:
    if not benchmark_path.exists():
        return None
    try:
        df = pd.read_csv(benchmark_path)
        ras = pd.to_numeric(df["ra"], errors="coerce").dropna()
        decs = pd.to_numeric(df["dec"], errors="coerce").dropna()
        valid = df.loc[ras.index.intersection(decs.index)]
        return SkyCoord(
            ra=pd.to_numeric(valid["ra"], errors="coerce").values * u.deg,
            dec=pd.to_numeric(valid["dec"], errors="coerce").values * u.deg,
        )
    except Exception:
        return None


def _already_in_benchmark(
    ra: float,
    dec: float,
    existing: "SkyCoord | None",
    dedup_radius_arcsec: float = 5.0,
) -> bool:
    if existing is None or len(existing) == 0:
        return False
    c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    seps = c.separation(existing)
    return bool(seps.min().arcsec < dedup_radius_arcsec)


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def run(
    output_path: Path,
    benchmark_master: Path,
    wise_radius_arcsec: float = 3.0,
    dedup_radius_arcsec: float = 5.0,
    rate_limit_s: float = 0.4,
) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_coords = _load_existing_coords(benchmark_master)
    print(
        f"Loaded {len(existing_coords) if existing_coords is not None else 0} "
        f"existing coords from {benchmark_master}"
    )

    rows = []
    n_total = len(LITERATURE_SOURCES)
    for i, (name, ra, dec, z, trans, ref, tier) in enumerate(LITERATURE_SOURCES, 1):
        print(f"[{i}/{n_total}] {name} ...")

        # --- duplicate check ---
        if _already_in_benchmark(ra, dec, existing_coords, dedup_radius_arcsec):
            print(f"  SKIP: already in benchmark (within {dedup_radius_arcsec}\")")
            rows.append(_make_row(name, ra, dec, z, trans, ref, tier,
                                  skipped=True, skip_reason="already_in_benchmark"))
            time.sleep(rate_limit_s)
            continue

        # --- WISE coverage ---
        wc = _irsa_wise_coverage(ra, dec, wise_radius_arcsec)
        time.sleep(rate_limit_s)

        status = "PASS" if wc["coverage_pass"] else f"FAIL(n={wc['n_clean']},s={wc['n_seasons_total']})"
        print(f"  WISE={status}")

        rows.append(_make_row(
            name, ra, dec, z, trans, ref, tier,
            skipped=False, skip_reason="",
            n_clean_epochs=wc["n_clean"],
            n_season_bins_w1=wc["n_seasons_w1"],
            n_season_bins_w2=wc["n_seasons_w2"],
            n_season_bins_total=wc["n_seasons_total"],
            coverage_pass=wc["coverage_pass"],
        ))

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, float_format="%.6f")

    n_skip = int(df["skipped"].sum())
    n_cov_pass = int(df["coverage_pass"].sum())
    n_cov_fail = int((~df["skipped"] & ~df["coverage_pass"]).sum())
    print(
        f"\nSummary: {n_total} candidates "
        f"→ {n_skip} skipped, {n_cov_pass} coverage_pass, {n_cov_fail} coverage_fail"
    )
    print(f"Output: {output_path}")
    return df


def _make_row(
    name: str,
    ra: float,
    dec: float,
    z: float,
    trans: str,
    ref: str,
    tier: int,
    *,
    skipped: bool = False,
    skip_reason: str = "",
    n_clean_epochs: int = 0,
    n_season_bins_w1: int = 0,
    n_season_bins_w2: int = 0,
    n_season_bins_total: int = 0,
    coverage_pass: bool = False,
) -> dict:
    return {
        "source_id": name,
        "ra": ra,
        "dec": dec,
        "redshift": z,
        "transition_type": trans,
        "reference": ref,
        "evidence_tier": tier,
        "skipped": skipped,
        "skip_reason": skip_reason,
        "n_clean_epochs": n_clean_epochs,
        "n_season_bins_w1": n_season_bins_w1,
        "n_season_bins_w2": n_season_bins_w2,
        "n_season_bins_total": n_season_bins_total,
        "coverage_pass": coverage_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="data/expansion/clagn_coverage_report.csv",
                        help="Output CSV path")
    parser.add_argument("--benchmark-master", default="data/real_clagn/benchmark_master.csv",
                        help="Existing benchmark master to dedup against")
    parser.add_argument("--wise-radius", type=float, default=3.0,
                        help="IRSA WISE query radius in arcsec (default: 3.0)")
    parser.add_argument("--dedup-radius", type=float, default=5.0,
                        help="Dedup radius against existing benchmark in arcsec (default: 5.0)")
    parser.add_argument("--rate-limit", type=float, default=0.4,
                        help="Seconds to sleep between API calls (default: 0.4)")
    args = parser.parse_args()

    run(
        output_path=Path(args.output),
        benchmark_master=Path(args.benchmark_master),
        wise_radius_arcsec=args.wise_radius,
        dedup_radius_arcsec=args.dedup_radius,
        rate_limit_s=args.rate_limit,
    )


if __name__ == "__main__":
    main()
