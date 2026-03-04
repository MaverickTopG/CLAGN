"""
Fetch real CLAGN positives and contaminants from public catalogs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BIB_JANA_2025 = "2025A&A...693A..35J"
BIB_GREEN_2022 = "2022ApJ...933..180G"
BIB_MACLEOD_2019 = "2019ApJ...874....8M"
BIB_COMP_2024 = "2024ApJS..272...13S"
BIB_APJ_887_15 = "2019ApJ...887...15G"
BIB_GAIA_DR3 = "2022A&A...667A.148G"
BIB_SDSS_DR16Q = "2020ApJS..250....9L"
BIB_BLAZAR = "III/157"
BIB_SN = "B/sn"

ROW_LIMIT = 2000


def _safe_str(x) -> str:
    if x is None:
        return ''
    return str(x).strip()


def _vizier() -> Vizier:
    v = Vizier(columns=['*'])
    v.ROW_LIMIT = ROW_LIMIT
    return v


def _build_jana_clagn() -> list[dict]:
    v = _vizier()
    t1 = v.get_catalogs('J/A+A/693/A35/table1')[0].to_pandas()
    tc = v.get_catalogs('J/A+A/693/A35/tablec')[0].to_pandas()
    refs = v.get_catalogs('J/A+A/693/A35/refs')[0].to_pandas()

    ref_map = {}
    for _, row in refs.iterrows():
        ref_map[_safe_str(row.get('Ref'))] = _safe_str(row.get('BibCode'))

    t1['Name'] = t1['Name'].astype(str)
    tc['Name'] = tc['Name'].astype(str)

    t1_idx = t1.set_index('Name')

    rows = []
    for name, group in tc.groupby('Name'):
        if name not in t1_idx.index:
            continue
        group = group.copy()
        group['Obs.date'] = group['Obs.date'].astype(str)
        group = group[group['Obs.date'].astype(str).str.strip() != '']
        if len(group) < 2:
            continue
        # sort by date string
        group = group.sort_values('Obs.date')
        first = group.iloc[0]
        last = group.iloc[-1]
        type1 = _safe_str(first.get('Type'))
        type2 = _safe_str(last.get('Type'))
        if not type1 or not type2:
            continue
        ra = float(t1_idx.loc[name]['RAJ2000'])
        dec = float(t1_idx.loc[name]['DEJ2000'])
        z = float(t1_idx.loc[name]['z'])
        ref_code = _safe_str(first.get('Refs')) or _safe_str(last.get('Refs'))
        bib = ref_map.get(ref_code, BIB_JANA_2025)

        rows.append({
            'source_id': f"{name}",
            'ra': ra,
            'dec': dec,
            'redshift': z,
            'spectral_state_1': type1,
            'spectral_state_2': type2,
            'epoch_1': _safe_str(first.get('Obs.date')),
            'epoch_2': _safe_str(last.get('Obs.date')),
            'spectroscopic_reference': bib,
            'label_method': 'spectroscopic',
            'independent_of_pipeline': True,
        })
    return rows


def _build_green_clagn() -> list[dict]:
    v = _vizier()
    t = v.get_catalogs('J/ApJ/933/180/table2')[0].to_pandas()
    t['State'] = t['State'].astype(str)
    t = t[t['State'].str.strip() != '']

    rows = []
    for sdss, group in t.groupby('SDSS'):
        states = set(group['State'].astype(str).str.strip())
        if not ('B' in states and 'D' in states):
            continue
        group = group.sort_values('MJD')
        first = group.iloc[0]
        last = group.iloc[-1]
        if _safe_str(first.get('State')) == _safe_str(last.get('State')):
            # choose first with different state
            alt = group[group['State'].astype(str).str.strip() != _safe_str(first.get('State'))]
            if alt.empty:
                continue
            last = alt.iloc[-1]
        rows.append({
            'source_id': f"SDSSJ{sdss}",
            'ra': float(first['_RA']),
            'dec': float(first['_DE']),
            'redshift': float(first['zspec']),
            'spectral_state_1': _safe_str(first.get('State')),
            'spectral_state_2': _safe_str(last.get('State')),
            'epoch_1': str(first.get('MJD')),
            'epoch_2': str(last.get('MJD')),
            'spectroscopic_reference': BIB_GREEN_2022,
            'label_method': 'spectroscopic',
            'independent_of_pipeline': True,
        })
    return rows


def _build_macleod_clagn() -> list[dict]:
    v = _vizier()
    t = v.get_catalogs('J/ApJ/874/8/table2')[0].to_pandas()
    t = t[t['CLQ?'] == 1]

    rows = []
    for _, row in t.iterrows():
        mjd1 = row.get('MJD1s')
        mjd2 = row.get('MJD2s')
        if pd.isna(mjd1) or pd.isna(mjd2):
            continue
        rows.append({
            'source_id': f"SDSSJ{row['SDSS']}",
            'ra': float(row['_RA']),
            'dec': float(row['_DE']),
            'redshift': float(row['z']),
            'g_mag': float(row['gmag1']) if 'gmag1' in row and pd.notna(row['gmag1']) else None,
            'spectral_state_1': 'spec_epoch1',
            'spectral_state_2': 'spec_epoch2',
            'epoch_1': str(mjd1),
            'epoch_2': str(mjd2),
            'spectroscopic_reference': BIB_MACLEOD_2019,
            'label_method': 'spectroscopic',
            'independent_of_pipeline': True,
        })
    return rows


def _build_compilation_clagn() -> list[dict]:
    v = _vizier()
    t = v.get_catalogs('J/ApJS/272/13/table2')[0].to_pandas()
    rows = []
    for sdss, group in t.groupby('SDSS'):
        group = group.sort_values('MJD')
        first = group.iloc[0]
        last = group.iloc[-1]
        rows.append({
            'source_id': f"SDSSJ{sdss}",
            'ra': float(first['RAJ2000']),
            'dec': float(first['DEJ2000']),
            'redshift': float(first['zspec']),
            'g_mag': float(first['m_SDSS']) if 'm_SDSS' in first and pd.notna(first['m_SDSS']) else None,
            'spectral_state_1': 'spec_epoch1',
            'spectral_state_2': 'spec_epoch2',
            'epoch_1': str(first.get('MJD')),
            'epoch_2': str(last.get('MJD')),
            'spectroscopic_reference': BIB_COMP_2024,
            'label_method': 'spectroscopic',
            'independent_of_pipeline': True,
        })
    return rows


def _build_apj887_clagn() -> list[dict]:
    v = _vizier()
    t = v.get_catalogs('J/ApJ/887/15/table1')[0].to_pandas()
    rows = []
    for sdss, group in t.groupby('SDSS'):
        group = group.sort_values('Date')
        if len(group) < 2:
            continue
        first = group.iloc[0]
        last = group.iloc[-1]
        rows.append({
            'source_id': f"SDSSJ{sdss}",
            'ra': float(first['_RA']),
            'dec': float(first['_DE']),
            'redshift': float(first['z']),
            'g_mag': float(first['gmag']) if 'gmag' in first and pd.notna(first['gmag']) else None,
            'spectral_state_1': 'spec_epoch1',
            'spectral_state_2': 'spec_epoch2',
            'epoch_1': str(first.get('Date')),
            'epoch_2': str(last.get('Date')),
            'spectroscopic_reference': BIB_APJ_887_15,
            'label_method': 'spectroscopic',
            'independent_of_pipeline': True,
        })
    return rows


def _sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


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


def _dedupe_arcsec(df: pd.DataFrame, max_arcsec: float = 1.0) -> pd.DataFrame:
    kept = []
    for _, row in df.iterrows():
        ra = float(row['ra'])
        dec = float(row['dec'])
        duplicate = False
        for k in kept:
            if _angular_sep_arcsec(ra, dec, k['ra'], k['dec']) < max_arcsec:
                duplicate = True
                break
        if not duplicate:
            kept.append(row)
    if not kept:
        return df.iloc[0:0]
    return pd.DataFrame(kept)


def _coords_to_deg(ra_series: pd.Series, dec_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    if ra_series.dtype == object:
        ra_vals = ra_series.astype(str).str.strip().values
        dec_vals = dec_series.astype(str).str.strip().values
        mask = (ra_vals != '') & (dec_vals != '')
        ra_deg = np.full(len(ra_vals), np.nan, dtype=float)
        dec_deg = np.full(len(dec_vals), np.nan, dtype=float)
        if mask.any():
            coords = SkyCoord(ra_vals[mask], dec_vals[mask], unit=(u.hourangle, u.deg))
            ra_deg[mask] = coords.ra.deg
            dec_deg[mask] = coords.dec.deg
        return pd.Series(ra_deg), pd.Series(dec_deg)
    return pd.to_numeric(ra_series, errors='coerce'), pd.to_numeric(dec_series, errors='coerce')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/real_clagn')
    parser.add_argument('--n_positives', type=int, default=50)
    parser.add_argument('--n_contaminants', type=int, default=150)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    base = Path(args.data_dir)
    raw_dir = base / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Build positives
    clagn_rows = []
    clagn_rows.extend(_build_jana_clagn())
    clagn_rows.extend(_build_green_clagn())
    clagn_rows.extend(_build_macleod_clagn())
    clagn_rows.extend(_build_compilation_clagn())
    clagn_rows.extend(_build_apj887_clagn())

    clagn_df = pd.DataFrame(clagn_rows)
    if clagn_df.empty:
        print("No CLAGN rows fetched.")
        sys.exit(1)

    clagn_df = clagn_df.drop_duplicates(subset=['source_id'])
    clagn_df = _dedupe_arcsec(clagn_df, max_arcsec=1.0)
    clagn_df = _sample(clagn_df, args.n_positives, args.seed)

    # Contaminants
    v = _vizier()

    # SN catalog
    sn = v.get_catalogs('B/sn/sncat')[0].to_pandas()
    sn_ra, sn_dec = _coords_to_deg(sn['RAJ2000'], sn['DEJ2000'])
    sn_rows = pd.DataFrame({
        'source_id': sn['SN'].astype(str),
        'ra': sn_ra,
        'dec': sn_dec,
        'contaminant_type': 'sn',
        'citation': BIB_SN,
        'classification_method': 'Asiago Supernova Catalog',
        'independent_of_pipeline': True,
        'mag': pd.to_numeric(sn.get('MaxMag'), errors='coerce') if 'MaxMag' in sn.columns else np.nan,
    })
    sn_rows = sn_rows.dropna(subset=['ra', 'dec'])

    # Blazars
    bl = v.get_catalogs('III/157/objects')[0].to_pandas()
    bl_rows = pd.DataFrame({
        'source_id': bl['Name'].astype(str),
        'ra': bl['_RA'].astype(float),
        'dec': bl['_DE'].astype(float),
        'contaminant_type': 'blazar',
        'citation': BIB_BLAZAR,
        'classification_method': 'Blazar catalog (III/157)',
        'independent_of_pipeline': True,
        'mag': pd.to_numeric(bl.get('Vmag'), errors='coerce') if 'Vmag' in bl.columns else np.nan,
    })
    bl_rows = bl_rows.dropna(subset=['ra', 'dec'])

    # Variable stars
    vs = v.get_catalogs('I/358/varisum')[0].to_pandas()
    vs_rows = pd.DataFrame({
        'source_id': vs['Source'].astype(str),
        'ra': vs['RA_ICRS'].astype(float),
        'dec': vs['DE_ICRS'].astype(float),
        'contaminant_type': 'star',
        'citation': BIB_GAIA_DR3,
        'classification_method': 'Gaia DR3 variability',
        'independent_of_pipeline': True,
        'mag': pd.to_numeric(vs.get('Gmagmean'), errors='coerce') if 'Gmagmean' in vs.columns else np.nan,
    })
    vs_rows = vs_rows.dropna(subset=['ra', 'dec'])

    # Normal AGN controls
    agn = v.get_catalogs('VII/289/dr16q')[0].to_pandas()
    agn_rows = pd.DataFrame({
        'source_id': agn['SDSS'].astype(str),
        'ra': agn['RAJ2000'].astype(float),
        'dec': agn['DEJ2000'].astype(float),
        'contaminant_type': 'normal_agn',
        'citation': BIB_SDSS_DR16Q,
        'classification_method': 'SDSS DR16Q',
        'independent_of_pipeline': True,
        'redshift': agn['z'].astype(float),
        'mag': pd.to_numeric(agn.get('r_z'), errors='coerce') if 'r_z' in agn.columns else np.nan,
    })
    agn_rows = agn_rows.dropna(subset=['ra', 'dec'])

    # Sample contaminants
    n_each = max(1, args.n_contaminants // 4)
    sn_rows = _sample(sn_rows, n_each, args.seed)
    bl_rows = _sample(bl_rows, n_each, args.seed + 1)
    vs_rows = _sample(vs_rows, n_each, args.seed + 2)
    agn_rows = _sample(agn_rows, args.n_contaminants - 3 * n_each, args.seed + 3)

    cont_df = pd.concat([sn_rows, bl_rows, vs_rows, agn_rows], ignore_index=True)
    cont_df['source_id'] = cont_df['source_id'].astype(str)
    cont_df = cont_df[cont_df['source_id'].str.strip() != '']
    cont_df = cont_df.dropna(subset=['ra', 'dec'])
    cont_df = cont_df.drop_duplicates(subset=['source_id'])
    cont_df = _dedupe_arcsec(cont_df, max_arcsec=1.0)

    # Write JSON
    clagn_path = raw_dir / 'literature_sources.json'
    cont_path = raw_dir / 'contaminants_sources.json'
    clagn_path.write_text(json.dumps(clagn_df.to_dict(orient='records'), indent=2))
    cont_path.write_text(json.dumps(cont_df.to_dict(orient='records'), indent=2))

    print(f"Wrote {clagn_path} and {cont_path}")


if __name__ == '__main__':
    main()
