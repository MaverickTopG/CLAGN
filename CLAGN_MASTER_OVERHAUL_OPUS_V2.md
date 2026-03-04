# CLAGN Pipeline Master Overhaul — Opus Implementation Plan

# Version 2.0 — Complete, Robust, Publication-Grade

### Target: >92% overall accuracy, >88% balanced accuracy

### Model: Claude Opus (see Section 0 for toggle instructions)

### Journal target: ApJ or MNRAS methods paper + ApJL discovery letter

### DO NOT run evaluate_benchmark.py until Step 8

### DO NOT modify pipeline_policy.yaml until Step 6

### Read ALL existing MD files in the project root before writing any code

---

## SECTION 1 — PRE-FLIGHT CHECKLIST

Run this entire block first. Fix every failure before proceeding.
Do not skip any check. Do not assume dependencies are installed.

```bash
#!/bin/bash
# scripts/preflight.sh

set -e
echo "=== CLAGN Pipeline Preflight Check ==="
echo "Date: $(date)"
echo "Python: $(python3 --version)"
echo "Working directory: $(pwd)"

# Create all required directories
mkdir -p data/raw data/benchmark data/features \
         cards/benchmark_v3 logs results figures \
         scripts models data/spectra

# Install all Python dependencies
pip install lightgbm optuna pyarrow fastparquet healpy PyWavelets \
    scikit-learn scipy astropy pandas numpy matplotlib seaborn \
    ruptures requests tqdm joblib shap statsmodels \
    --break-system-packages --quiet

# Verify each critical import
python3 -c "
import healpy, pywt, lightgbm, optuna, ruptures, shap
import astropy, sklearn, scipy, pandas, numpy, matplotlib
print('All imports OK')
print(f'  LightGBM: {lightgbm.__version__}')
print(f'  Optuna:   {optuna.__version__}')
print(f'  Healpy:   {healpy.__version__}')
print(f'  SHAP:     {shap.__version__}')
"

# Verify IRSA TAP connectivity
echo "Testing IRSA connectivity..."
curl -s --max-time 30 \
  "https://irsa.ipac.caltech.edu/TAP/sync?QUERY=SELECT+TOP+1+ra,dec+FROM+neowiser_p1bs_psd&FORMAT=json&LANG=ADQL" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('IRSA: OK' if d.get('data') else 'IRSA: FAIL — check connectivity')"

# Verify SDSS SkyServer connectivity
echo "Testing SDSS connectivity..."
curl -s --max-time 20 "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/RadialSearch?ra=0.497&dec=3.731&radius=0.05&limit=1&format=json&whichquery=spectro" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'SDSS: OK ({len(d.get(\"Rows\",[]))} rows)')" 2>/dev/null || echo "SDSS: connectivity issue"

# Print current benchmark state
python3 -c "
import pandas as pd, os, json

print('\n=== CURRENT STATE ===')

# Benchmark
for f in ['data/benchmark/benchmark_master_v3.csv',
          'data/benchmark/benchmark_master_v2.csv']:
    if os.path.exists(f):
        df = pd.read_csv(f)
        print(f'Benchmark ({f}): {len(df):,} sources')
        print(df[\"label\"].value_counts().to_string())
        break
else:
    print('No benchmark found')

# Cards
for d in ['cards/benchmark_v3','cards/benchmark_v2']:
    if os.path.exists(d):
        n = len(list(__import__('pathlib').Path(d).glob('*.json')))
        print(f'Cards ({d}): {n:,}')
        break

# Features
if os.path.exists('data/features/all_features.csv'):
    feat = pd.read_csv('data/features/all_features.csv')
    print(f'Features: {len(feat):,} rows, {len(feat.columns)} columns')

# Models
if os.path.exists('models/lgbm_classifier.pkl'):
    print('Classifier v2: EXISTS')
else:
    print('Classifier v2: NOT YET TRAINED')

# Existing pipeline threshold
for cfg in ['pipeline_policy.yaml', 'config.yaml']:
    if os.path.exists(cfg):
        import yaml
        with open(cfg) as f: c = yaml.safe_load(f)
        print(f'Existing threshold: {c}')
        break
"

echo ""
echo "=== Preflight complete. Review above before proceeding. ==="
```

Run:

```bash
bash scripts/preflight.sh 2>&1 | tee logs/preflight_$(date +%Y%m%d_%H%M%S).log
```

---

## SECTION 2 — BULK WISE DOWNLOAD

The per-source IRSA TAP approach (original implementation) takes 37 days
for 156k sources. This section replaces it entirely with HEALPix bulk
cone searches. Expected time: 4–12 hours total.

### Why HEALPix

Instead of 156,000 individual 3-arcsec cone queries, we:

1. Divide the sky into ~1.8-degree HEALPix cells (nside=32 → 12,288 cells)
2. Find which cells contain at least one benchmark source (~800–1200 cells)
3. Download ALL NEOWISE-R photometry per occupied cell in one query
4. Crossmatch against benchmark sources locally (no IRSA needed after download)

800 queries × ~10 seconds each = ~2 hours minimum vs 156,000 × ~8 seconds = 14+ days.

### 2A. Bulk Downloader

```python
# scripts/bulk_wise_download.py
"""
Bulk NEOWISE-R download via HEALPix cell queries.
Fully resumable — skips completed cells.
16 parallel workers with rate limit handling and exponential backoff.
Output: data/raw/neowise_bulk/healpix_{cell_id:05d}.parquet
"""
import healpy as hp
import numpy as np
import pandas as pd
import requests
import os, time, json, logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Event
from tqdm import tqdm
from datetime import datetime, timedelta

# ── Config ────────────────────────────────────────────────────────
BENCHMARK_CSV  = "data/benchmark/benchmark_master_v3.csv"
OUTPUT_DIR     = "data/raw/neowise_bulk"
PROGRESS_LOG   = "logs/bulk_download_progress.json"
FAILED_LOG     = "logs/bulk_download_failed.csv"
IRSA_URL       = "https://irsa.ipac.caltech.edu/TAP/sync"
NSIDE          = 32        # ~1.83 degree cells
N_WORKERS      = 16
TIMEOUT        = 360       # 6 min per bulk query (large cells can be slow)
MAX_RETRIES    = 5
BACKOFF_BASE   = 15        # seconds, doubles each retry
RATE_LIMIT_PAUSE = 90      # pause all workers on 429
CHECKPOINT_N   = 25        # save progress every N cells
MIN_DISK_GB    = 5.0       # stop if less than this free space
MAX_DISK_GB    = 35.0      # stop if cards+bulk exceeds this

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

logging.basicConfig(filename='logs/bulk_download.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
print_lock   = Lock()
pause_event  = Event()
pause_event.set()  # not paused initially
stats = {'success':0,'failed':0,'skipped':0,'empty':0,'total_rows':0}
start_time = time.time()

def check_disk():
    import shutil
    free_gb = shutil.disk_usage(OUTPUT_DIR).free / 1e9
    used_gb = sum(f.stat().st_size for f in Path(OUTPUT_DIR).glob('*.parquet')) / 1e9
    if free_gb < MIN_DISK_GB:
        raise SystemExit(f"DISK FULL: only {free_gb:.1f}GB free — stopping safely")
    if used_gb > MAX_DISK_GB:
        raise SystemExit(f"DISK LIMIT: bulk dir is {used_gb:.1f}GB > {MAX_DISK_GB}GB cap")

def get_cell_path(cell_id):
    return os.path.join(OUTPUT_DIR, f"healpix_{cell_id:05d}.parquet")

def query_irsa_cell(cell_id, ra_center, dec_center, radius_deg):
    """Query IRSA for all NEOWISE-R sources in a HEALPix cell."""
    adql = f"""
    SELECT ra, dec, mjd,
           w1mpro, w1sigmpro,
           w2mpro, w2sigmpro,
           qi_fact, saa_sep, moon_masked,
           cc_flags, ext_flg
    FROM neowiser_p1bs_psd
    WHERE CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra_center:.6f}, {dec_center:.6f}, {radius_deg:.6f})
    ) = 1
    AND qi_fact   >= 1
    AND saa_sep   >= 5
    AND moon_masked = 0
    AND w1sigmpro IS NOT NULL
    AND w1sigmpro  < 0.2
    AND w2sigmpro IS NOT NULL
    AND w2sigmpro  < 0.3
    AND (cc_flags  IS NULL OR SUBSTRING(cc_flags,1,1) NOT IN ('H','O','P','S'))
    AND (ext_flg   IS NULL OR ext_flg = 0)
    ORDER BY mjd
    """
    for attempt in range(MAX_RETRIES):
        pause_event.wait()  # block if rate limited
        try:
            resp = requests.get(
                IRSA_URL,
                params={'QUERY': adql, 'FORMAT': 'json', 'LANG': 'ADQL'},
                timeout=TIMEOUT
            )
            if resp.status_code == 429:
                with print_lock:
                    print(f"\n  [429 RATE LIMITED] Pausing all workers {RATE_LIMIT_PAUSE}s")
                pause_event.clear()
                time.sleep(RATE_LIMIT_PAUSE)
                pause_event.set()
                continue
            resp.raise_for_status()
            data = resp.json()
            if not data.get('data'):
                return pd.DataFrame()
            cols = [c['name'] for c in data['metadata']]
            df   = pd.DataFrame(data['data'], columns=cols)
            numeric_cols = ['ra','dec','mjd','w1mpro','w1sigmpro',
                            'w2mpro','w2sigmpro','qi_fact','saa_sep']
            for c in numeric_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
            df = df.dropna(subset=['ra','dec','mjd','w1mpro','w2mpro'])
            logging.info(f"Cell {cell_id}: {len(df)} rows")
            return df
        except requests.exceptions.Timeout:
            wait = BACKOFF_BASE * (2 ** attempt)
            with print_lock:
                print(f"  Cell {cell_id} timeout (attempt {attempt+1}/{MAX_RETRIES}) — wait {wait}s")
            logging.warning(f"Cell {cell_id} timeout attempt {attempt+1}")
            time.sleep(wait)
        except Exception as e:
            wait = BACKOFF_BASE * (2 ** attempt)
            with print_lock:
                print(f"  Cell {cell_id} error: {e} — wait {wait}s")
            logging.error(f"Cell {cell_id} error attempt {attempt+1}: {e}")
            time.sleep(wait)
    logging.error(f"Cell {cell_id} FAILED all {MAX_RETRIES} retries")
    return None

def process_cell(args):
    cell_id, ra_c, dec_c, radius = args
    out_path = get_cell_path(cell_id)
    if os.path.exists(out_path):
        return 'skipped', cell_id, 0

    df = query_irsa_cell(cell_id, ra_c, dec_c, radius)

    if df is None:
        # Write failure marker
        with open(FAILED_LOG, 'a') as f:
            f.write(f"{cell_id},{ra_c:.4f},{dec_c:.4f},{datetime.utcnow().isoformat()}\n")
        return 'failed', cell_id, 0

    if len(df) == 0:
        # Write empty parquet (don't re-query)
        empty = pd.DataFrame(columns=['ra','dec','mjd','w1mpro','w1sigmpro',
                                       'w2mpro','w2sigmpro'])
        empty.to_parquet(out_path, index=False)
        return 'empty', cell_id, 0

    df.to_parquet(out_path, index=False, compression='snappy')
    return 'success', cell_id, len(df)

def save_progress(cells_done, cells_total):
    elapsed = time.time() - start_time
    rate    = cells_done / max(elapsed, 1)
    eta_sec = (cells_total - cells_done) / max(rate, 1e-6)
    with open(PROGRESS_LOG, 'w') as f:
        json.dump({
            **stats,
            'cells_done':    cells_done,
            'cells_total':   cells_total,
            'elapsed_h':     round(elapsed/3600, 2),
            'eta_h':         round(eta_sec/3600, 2),
            'eta_str':       str(timedelta(seconds=int(eta_sec))),
            'timestamp':     datetime.utcnow().isoformat()
        }, f, indent=2)

def main():
    df_bench = pd.read_csv(BENCHMARK_CSV)
    print(f"Benchmark: {len(df_bench):,} sources")

    # ── Find occupied HEALPix cells ────────────────────────────────
    ra  = df_bench['ra'].dropna().values
    dec = df_bench['dec'].dropna().values
    theta = np.radians(90 - dec)
    phi   = np.radians(ra % 360)
    pixel_ids = hp.ang2pix(NSIDE, theta, phi, nest=False)
    occupied  = np.unique(pixel_ids)
    print(f"Occupied cells (nside={NSIDE}): {len(occupied):,}")

    # Include neighbors for sources near cell edges
    all_cells = set(occupied.tolist())
    for pix in occupied:
        neighbors = hp.get_all_neighbours(NSIDE, pix, nest=False)
        all_cells.update(int(n) for n in neighbors if n >= 0)
    all_cells = sorted(all_cells)
    print(f"Cells including neighbors: {len(all_cells):,}")

    # ── Build task list ────────────────────────────────────────────
    cell_radius_deg = np.degrees(hp.max_pixrad(NSIDE)) * 1.15
    print(f"Cell query radius: {cell_radius_deg:.3f} deg")
    tasks = []
    for cell_id in all_cells:
        theta_c, phi_c = hp.pix2ang(NSIDE, cell_id, nest=False)
        ra_c  = np.degrees(phi_c)
        dec_c = 90 - np.degrees(theta_c)
        tasks.append((int(cell_id), ra_c, dec_c, cell_radius_deg))

    already_done = sum(1 for _, cid, _, _ in tasks
                       if os.path.exists(get_cell_path(cid)))
    print(f"Already completed: {already_done:,} / {len(tasks):,}")
    print(f"Remaining: {len(tasks)-already_done:,}")
    print(f"\nStarting {N_WORKERS}-worker download at {datetime.utcnow().isoformat()}")

    with open(FAILED_LOG, 'w') as f:
        f.write("cell_id,ra,dec,timestamp\n")

    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(process_cell, t): t for t in tasks}
        for i, future in enumerate(tqdm(as_completed(futures),
                                        total=len(tasks), desc="Cells")):
            status, cell_id, n_rows = future.result()
            stats[status] = stats.get(status, 0) + 1
            stats['total_rows'] += n_rows

            if (i + 1) % CHECKPOINT_N == 0:
                save_progress(i + 1, len(tasks))
                check_disk()

            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                h, m = divmod(int(elapsed), 3600); m //= 60
                rate = (i+1) / elapsed
                eta  = (len(tasks)-i-1) / rate
                eh, em = divmod(int(eta), 3600); em //= 60
                with print_lock:
                    print(f"\n[{i+1}/{len(tasks)}] "
                          f"ok={stats['success']} skip={stats['skipped']} "
                          f"fail={stats['failed']} rows={stats['total_rows']:,} "
                          f"elapsed={h}h{m:02d}m ETA={eh}h{em:02d}m")

    save_progress(len(tasks), len(tasks))
    elapsed = time.time() - start_time
    total_gb = sum(f.stat().st_size for f in Path(OUTPUT_DIR).glob('*.parquet'))/1e9

    print(f"\n{'='*60}")
    print(f"BULK DOWNLOAD COMPLETE in {elapsed/3600:.1f}h")
    print(f"  Succeeded:   {stats['success']:,}")
    print(f"  Skipped:     {stats['skipped']:,}")
    print(f"  Failed:      {stats['failed']:,}")
    print(f"  Empty cells: {stats['empty']:,}")
    print(f"  Total rows:  {stats['total_rows']:,}")
    print(f"  Disk used:   {total_gb:.1f} GB")
    if stats['failed'] > 0:
        print(f"  Failed cells logged: {FAILED_LOG}")
        print(f"  Re-run this script to retry failed cells (they are not checkpointed)")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
```

### 2B. Retry Failed Cells

If any cells failed in the first pass, re-run the same script.
Failed cells have no parquet file so they will be retried automatically.
Run up to 3 times until failed count reaches 0 or stabilizes.

```bash
# Run in tmux for persistence
tmux new -s bulk_fetch
python scripts/bulk_wise_download.py 2>&1 | tee logs/bulk_run1.log

# If still failures after first run:
python scripts/bulk_wise_download.py 2>&1 | tee logs/bulk_run2.log

# Monitor from another terminal:
watch -n 60 "cat logs/bulk_download_progress.json | python3 -m json.tool"
```

### 2C. Local Card Builder

```python
# scripts/build_cards_from_bulk.py
"""
Build WISE light curve cards by crossmatching benchmark sources against
locally downloaded HEALPix parquet files. Zero IRSA queries.
Expected runtime: 15-30 minutes for 156k sources.
"""
import healpy as hp
import numpy as np
import pandas as pd
import json, os
from pathlib import Path
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

BENCHMARK_CSV     = "data/benchmark/benchmark_master_v3.csv"
WISE_DIR          = "data/raw/neowise_bulk"
CARDS_DIR         = "cards/benchmark_v3"
MATCH_RADIUS_ARCS = 3.0
NSIDE             = 32
N_JOBS            = 8
MIN_EPOCHS        = 10
MIN_SEASONS       = 3

os.makedirs(CARDS_DIR, exist_ok=True)

# Cache loaded cells in memory (avoid re-reading same parquet repeatedly)
_cell_cache = {}

def get_cell_data(cell_id):
    if cell_id in _cell_cache:
        return _cell_cache[cell_id]
    path = os.path.join(WISE_DIR, f"healpix_{cell_id:05d}.parquet")
    if not os.path.exists(path):
        _cell_cache[cell_id] = pd.DataFrame()
        return pd.DataFrame()
    df = pd.read_parquet(path)
    _cell_cache[cell_id] = df
    return df

def season_count(mjd_arr):
    if len(mjd_arr) < 2:
        return 1
    gaps = np.diff(np.sort(mjd_arr))
    return int((gaps > 100).sum()) + 1

def build_single_card(row_dict):
    source_id = str(row_dict['source_id'])
    safe_id   = source_id.replace('/','_').replace(' ','_').replace('+','p')
    card_path = os.path.join(CARDS_DIR, f"{safe_id}.json")

    if os.path.exists(card_path):
        return 'skipped', source_id

    ra, dec = float(row_dict['ra']), float(row_dict['dec'])

    # Get relevant HEALPix cells
    theta = np.radians(90 - dec)
    phi   = np.radians(ra % 360)
    pix   = hp.ang2pix(NSIDE, theta, phi, nest=False)
    neighbors = hp.get_all_neighbours(NSIDE, theta, phi, nest=False)
    cell_ids  = list(set([int(pix)] + [int(n) for n in neighbors if n >= 0]))

    # Concatenate relevant cell data
    frames = [get_cell_data(c) for c in cell_ids]
    frames = [f for f in frames if len(f) > 0]
    if not frames:
        return 'no_data', source_id
    cell_data = pd.concat(frames, ignore_index=True)

    # Crossmatch
    if len(cell_data) == 0:
        return 'no_data', source_id
    src_coord   = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    wise_coords = SkyCoord(ra=cell_data['ra'].values*u.deg,
                           dec=cell_data['dec'].values*u.deg)
    seps = src_coord.separation(wise_coords).arcsec
    matched = cell_data[seps < MATCH_RADIUS_ARCS].copy()

    if len(matched) == 0:
        return 'no_match', source_id

    matched = matched.sort_values('mjd').reset_index(drop=True)
    mjd_arr = matched['mjd'].values

    n_seasons = season_count(mjd_arr)
    baseline  = float(mjd_arr[-1] - mjd_arr[0]) if len(mjd_arr) > 1 else 0

    epochs = matched[['mjd','w1mpro','w1sigmpro','w2mpro','w2sigmpro']
                    ].to_dict('records')

    card = {
        'source_id':     source_id,
        'ra':            ra,
        'dec':           dec,
        'z':             float(row_dict.get('z', 0) or 0),
        'label':         str(row_dict.get('label', '')),
        'reference':     str(row_dict.get('reference', '')),
        'transition_type': str(row_dict.get('transition_type', 'unknown')),
        'wise_epochs':   epochs,
        'n_epochs':      len(epochs),
        'n_seasons':     n_seasons,
        'baseline_days': baseline,
        'has_coverage':  len(epochs) >= MIN_EPOCHS and n_seasons >= MIN_SEASONS,
        'built_at':      pd.Timestamp.utcnow().isoformat()
    }
    with open(card_path, 'w') as f:
        json.dump(card, f, default=float)

    return 'success', source_id

def main():
    df = pd.read_csv(BENCHMARK_CSV)
    print(f"Building cards for {len(df):,} sources with {N_JOBS} workers...")
    rows = df.to_dict('records')

    results = Parallel(n_jobs=N_JOBS, verbose=0, prefer='threads')(
        delayed(build_single_card)(r) for r in tqdm(rows, desc="Cards")
    )
    stats = {}
    for status, _ in results:
        stats[status] = stats.get(status, 0) + 1

    total = len(df)
    print(f"\nCard build complete:")
    for k, v in sorted(stats.items()):
        print(f"  {k:15s}: {v:,} ({v/total:.1%})")
    coverage = stats.get('success',0) / max(total - stats.get('skipped',0), 1)
    print(f"\nWISE coverage rate: {coverage:.1%}")
    if coverage < 0.70:
        print("WARNING: Coverage below 70% — check bulk download completeness")

if __name__ == "__main__":
    main()
```

Run:

```bash
python scripts/build_cards_from_bulk.py 2>&1 | tee logs/card_build.log
```

---

## SECTION 3 — FEATURE ENGINEERING (7 Novel Feature Groups)

All features computed from card JSON. Output: `data/features/all_features.csv`.
Every feature function must handle edge cases gracefully and return NaN
rather than raising exceptions. No feature computation should ever crash
the entire run.

### Feature Group A — W1−W2 Color Trajectory Asymmetry

**Novel contribution. From CLAGN_NOVEL_METHOD_IMPLEMENTATION.md.**
Physical basis: dust echo responds asymmetrically to turn-on vs turn-off.
Import from `scripts/color_asymmetry_features.py` — already implemented.
Features: `color_slope_early`, `color_slope_late`, `color_asymmetry`,
`color_range`, `color_flux_corr`, `dust_lag_proxy`,
`is_turnon_color_signal`, `is_turnoff_color_signal`

### Feature Group B — Wavelet Decomposition Variability

**Novel contribution.**
Physical basis: CLAGN transitions operate on year timescales → power
concentrates in coarse wavelet scales. Blazars: fine scale. SN: impulsive.

```python
import pywt
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import entropy as scipy_entropy

def wavelet_features(mjd, w1, min_points=20):
    nan_out = {f'wavelet_d{i}': np.nan for i in range(1,4)}
    nan_out.update({'wavelet_approx_power': np.nan,
                    'wavelet_power_ratio_3_1': np.nan,
                    'wavelet_power_ratio_3_2': np.nan,
                    'wavelet_entropy': np.nan,
                    'wavelet_hurst': np.nan})
    if len(mjd) < min_points:
        return nan_out
    try:
        n_interp = min(512, 2 ** int(np.floor(np.log2(len(mjd) * 2))))
        t_uniform = np.linspace(mjd[0], mjd[-1], n_interp)
        f_interp  = interp1d(mjd, w1, kind='linear', fill_value='extrapolate')
        w1u = f_interp(t_uniform)
        # Detrend before wavelet transform
        trend = np.polyval(np.polyfit(t_uniform, w1u, 1), t_uniform)
        w1u_detrended = w1u - trend

        coeffs = pywt.wavedec(w1u_detrended, 'db4', level=3)
        cA3, cD3, cD2, cD1 = coeffs
        p1 = float(np.var(cD1))
        p2 = float(np.var(cD2))
        p3 = float(np.var(cD3))
        pA = float(np.var(cA3))
        total = p1 + p2 + p3 + pA + 1e-12

        # Spectral entropy across wavelet scales
        probs = np.array([p1,p2,p3,pA]) / total
        probs = np.clip(probs, 1e-10, 1)
        wentropy = float(-np.sum(probs * np.log(probs)))

        # Hurst exponent proxy via wavelet variance scaling
        log_scales = np.log([1, 2, 4])
        log_powers = np.log([p1 + 1e-12, p2 + 1e-12, p3 + 1e-12])
        hurst_proxy = float(np.polyfit(log_scales, log_powers, 1)[0] / 2 + 0.5)

        return {
            'wavelet_d1':            p1,
            'wavelet_d2':            p2,
            'wavelet_d3':            p3,
            'wavelet_approx_power':  pA,
            'wavelet_power_ratio_3_1': p3 / (p1 + 1e-12),
            'wavelet_power_ratio_3_2': p3 / (p2 + 1e-12),
            'wavelet_entropy':       wentropy,
            'wavelet_hurst':         hurst_proxy,
        }
    except Exception:
        return nan_out
```

### Feature Group C — Structural Break Detection (Bai-Perron)

**Novel contribution.**
Physical basis: CLAGN have a genuine breakpoint in their light curve.
Normal AGN flicker without breaks. SN: fast break then slow decay.

```python
import ruptures as rpt
from scipy import stats as scipy_stats

def structural_break_features(mjd, w1, min_points=15):
    keys = ['break_mjd','break_magnitude','break_significance',
            'pre_break_slope','post_break_slope','slope_change',
            'break_fractional_position','break_duration_yr',
            'n_detected_breaks']
    nan_out = {k: np.nan for k in keys}

    if len(mjd) < min_points:
        return nan_out
    try:
        w1_flux = 10 ** (-0.4 * np.array(w1, dtype=float))
        t_yr    = (np.array(mjd, dtype=float) - mjd[0]) / 365.25
        signal  = w1_flux.reshape(-1, 1)

        # Detect up to 3 breakpoints
        algo = rpt.Pelt(model="rbf", min_size=4, jump=1).fit(signal)
        bps  = algo.predict(pen=np.var(w1_flux) * np.log(len(w1_flux)))
        # bps includes len(signal) as last element
        interior_bps = [b for b in bps if 0 < b < len(w1_flux)]
        n_breaks = len(interior_bps)

        if n_breaks == 0:
            nan_out['n_detected_breaks'] = 0
            return nan_out

        # Use first (most significant) breakpoint
        bp_idx = interior_bps[0]
        pre    = slice(0, bp_idx)
        post   = slice(bp_idx, len(w1_flux))

        pre_mean  = np.mean(w1_flux[pre])
        post_mean = np.mean(w1_flux[post])

        # F-test for break significance
        ss0 = np.sum((w1_flux - np.mean(w1_flux))**2)
        ss1 = (np.sum((w1_flux[pre]  - pre_mean)**2) +
               np.sum((w1_flux[post] - post_mean)**2))
        df1 = 1
        df2 = max(len(w1_flux) - 3, 1)
        f_stat = ((ss0 - ss1) / df1) / (ss1 / df2 + 1e-12)

        pre_slope  = float(scipy_stats.linregress(t_yr[pre],  w1_flux[pre])[0])  if bp_idx >= 3 else 0.0
        post_slope = float(scipy_stats.linregress(t_yr[post], w1_flux[post])[0]) if len(w1_flux)-bp_idx >= 3 else 0.0
        break_mag  = abs(-2.5 * np.log10(post_mean / (pre_mean + 1e-12)))

        return {
            'break_mjd':                  float(mjd[bp_idx]),
            'break_magnitude':            float(break_mag),
            'break_significance':         float(f_stat),
            'pre_break_slope':            pre_slope,
            'post_break_slope':           post_slope,
            'slope_change':               float(abs(post_slope - pre_slope)),
            'break_fractional_position':  float(bp_idx / len(w1_flux)),
            'break_duration_yr':          float(t_yr[bp_idx] - t_yr[0]),
            'n_detected_breaks':          n_breaks,
        }
    except Exception:
        return nan_out
```

### Feature Group D — Seasonal Coherence Score

**Novel contribution.**
Physical basis: DRW process is memory-preserving → normal AGN have high
season-to-season coherence. CLAGN: coherence drops during state change.
Blazars: chaotic low coherence. SN: one-season anomaly then recovery.

```python
def seasonal_coherence_features(mjd, w1, min_seasons=3):
    nan_out = {'mean_season_coherence': np.nan, 'coherence_trend': np.nan,
               'min_season_coherence': np.nan, 'max_season_coherence': np.nan,
               'coherence_drop_amplitude': np.nan, 'coherence_recovery': np.nan,
               'n_coherent_seasons': np.nan}

    mjd = np.array(mjd, dtype=float)
    w1  = np.array(w1,  dtype=float)
    gaps = np.diff(mjd)
    boundaries = np.where(gaps > 100)[0] + 1
    seasons = np.split(np.arange(len(mjd)), boundaries)
    seasons = [s for s in seasons if len(s) >= 4]

    if len(seasons) < min_seasons:
        return nan_out

    corrs = []
    for i in range(len(seasons) - 1):
        s1 = w1[seasons[i]]
        s2 = w1[seasons[i+1]]
        n  = min(len(s1), len(s2), 20)
        if n < 4: continue
        s1i = np.interp(np.linspace(0,1,n), np.linspace(0,1,len(s1)), s1)
        s2i = np.interp(np.linspace(0,1,n), np.linspace(0,1,len(s2)), s2)
        try:
            r, _ = scipy_stats.pearsonr(s1i, s2i)
            corrs.append(float(r))
        except Exception:
            continue

    if len(corrs) < 2:
        return nan_out

    c = np.array(corrs)
    trend, *_ = scipy_stats.linregress(np.arange(len(c)), c) if len(c) >= 3 else (np.nan,)
    min_idx = int(np.argmin(c))

    # Coherence recovery: correlation after the minimum vs minimum
    recovery = float(np.mean(c[min_idx+1:])) - float(c[min_idx]) if min_idx < len(c)-1 else np.nan

    return {
        'mean_season_coherence':    float(np.mean(c)),
        'coherence_trend':          float(trend),
        'min_season_coherence':     float(np.min(c)),
        'max_season_coherence':     float(np.max(c)),
        'coherence_drop_amplitude': float(np.max(c) - np.min(c)),
        'coherence_recovery':       recovery,
        'n_coherent_seasons':       int((c > 0.5).sum()),
    }
```

### Feature Group E — W1/W2 Flux Ratio Evolution (Linear Space)

**Novel contribution.**
Physical basis: W1/W2 flux ratio in linear space directly traces dust
temperature. Increasing ratio = dust heating = increasing accretion.
More physically motivated than magnitude-space color asymmetry.

```python
W1_ZP = 309.540  # Jansky zero point
W2_ZP = 171.787

def flux_ratio_features(mjd, w1, w2, min_points=10):
    nan_out = {k: np.nan for k in ['flux_ratio_mean','flux_ratio_std',
               'flux_ratio_trend','flux_ratio_skewness',
               'flux_ratio_peak_to_trough','flux_ratio_percentile_90_10',
               'flux_ratio_late_minus_early']}
    if len(mjd) < min_points or len(w2) < min_points:
        return nan_out
    try:
        f1 = W1_ZP * 10**(-0.4 * np.array(w1, dtype=float))
        f2 = W2_ZP * 10**(-0.4 * np.array(w2, dtype=float))
        ratio = f1 / (f2 + 1e-12)
        ratio = ratio[np.isfinite(ratio)]
        if len(ratio) < 5:
            return nan_out
        t_yr  = (np.array(mjd, dtype=float) - mjd[0]) / 365.25
        t_yr  = t_yr[:len(ratio)]
        trend, *_ = scipy_stats.linregress(t_yr, ratio)

        n_half = len(ratio) // 2
        late_minus_early = float(np.mean(ratio[n_half:])) - float(np.mean(ratio[:n_half]))

        return {
            'flux_ratio_mean':            float(np.mean(ratio)),
            'flux_ratio_std':             float(np.std(ratio)),
            'flux_ratio_trend':           float(trend),
            'flux_ratio_skewness':        float(scipy_stats.skew(ratio)),
            'flux_ratio_peak_to_trough':  float(np.max(ratio) - np.min(ratio)),
            'flux_ratio_percentile_90_10':float(np.percentile(ratio,90) - np.percentile(ratio,10)),
            'flux_ratio_late_minus_early':late_minus_early,
        }
    except Exception:
        return nan_out
```

### Feature Group F — DRW Residual Analysis

**Novel contribution.**
Physical basis: A DRW model describes ordinary AGN variability. CLAGN
violate DRW assumptions → large structured residuals. The quality of
the DRW fit is itself a detection feature.

```python
from scipy.optimize import minimize

def drw_residual_features(mjd, w1, w1_err, min_points=15):
    nan_out = {k: np.nan for k in ['drw_sigma','drw_tau',
               'drw_residual_rms','drw_residual_skewness',
               'drw_residual_autocorr','drw_fit_quality',
               'drw_residual_trend','drw_chi2_per_dof']}
    if len(mjd) < min_points:
        return nan_out
    mjd  = np.array(mjd,   dtype=float)
    w1   = np.array(w1,    dtype=float)
    err  = np.array(w1_err,dtype=float)
    err  = np.clip(err, 0.01, 1.0)  # bound errors

    def neg_log_likelihood(params):
        try:
            sigma = np.exp(np.clip(params[0], -5, 5))
            tau   = np.exp(np.clip(params[1], 2, 9))  # 7 to 8000 days
            dt    = np.abs(mjd[:,None] - mjd[None,:])
            C     = sigma**2 * np.exp(-dt/tau) + np.diag(err**2 + 1e-6)
            L     = np.linalg.cholesky(C)
            mu    = np.mean(w1)
            resid = w1 - mu
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, resid))
            logdet = 2 * np.sum(np.log(np.diag(L)))
            return 0.5 * (np.dot(resid, alpha) + logdet + len(w1)*np.log(2*np.pi))
        except (np.linalg.LinAlgError, OverflowError):
            return 1e10

    try:
        # Multiple starting points to avoid local minima
        best_nll, best_x = 1e10, [np.log(0.1), np.log(200)]
        for s0, t0 in [(0.05,100),(0.1,300),(0.2,600),(0.3,1000)]:
            res = minimize(neg_log_likelihood, [np.log(s0), np.log(t0)],
                           method='Nelder-Mead',
                           options={'maxiter':800,'xatol':0.01,'fatol':0.01})
            if res.fun < best_nll:
                best_nll, best_x = res.fun, res.x

        sigma = float(np.exp(np.clip(best_x[0], -5, 5)))
        tau   = float(np.exp(np.clip(best_x[1],  2, 9)))

        # Compute posterior mean (DRW prediction at observed times)
        dt = np.abs(mjd[:,None] - mjd[None,:])
        C  = sigma**2 * np.exp(-dt/tau) + np.diag(err**2 + 1e-6)
        try:
            C_inv   = np.linalg.inv(C)
            drw_pred = np.mean(w1) + C @ C_inv @ (w1 - np.mean(w1))
        except np.linalg.LinAlgError:
            drw_pred = np.full_like(w1, np.mean(w1))

        residuals = w1 - drw_pred
        t_yr = (mjd - mjd[0]) / 365.25
        r_trend, *_ = scipy_stats.linregress(t_yr, residuals)

        ac = float(np.corrcoef(residuals[:-1], residuals[1:])[0,1]) if len(residuals) > 2 else np.nan
        chi2_dof = float(np.sum((residuals/err)**2) / max(len(w1)-2,1))

        return {
            'drw_sigma':             sigma,
            'drw_tau':               tau,
            'drw_residual_rms':      float(np.sqrt(np.mean(residuals**2))),
            'drw_residual_skewness': float(scipy_stats.skew(residuals)),
            'drw_residual_autocorr': ac,
            'drw_fit_quality':       float(best_nll / max(len(w1),1)),
            'drw_residual_trend':    float(r_trend),
            'drw_chi2_per_dof':      chi2_dof,
        }
    except Exception:
        return nan_out
```

### Feature Group G — Redshift-Corrected Amplitude Features

**Novel contribution.**
Physical basis: at higher redshift, the same intrinsic luminosity change
produces a smaller observed flux change. Sources with large amplitude
changes at high-z are MORE physically anomalous per unit flux change.
Correcting for this reduces redshift-driven false negatives at z > 0.5.

```python
def redshift_features(z, break_magnitude, color_slope_early,
                      flux_ratio_trend, drw_sigma):
    nan_out = {k: np.nan for k in ['z_corrected_amplitude',
               'z_corrected_color_slope','luminosity_change_proxy',
               'z_corrected_drw_sigma','intrinsic_variability_proxy']}
    if np.isnan(z) or z <= 0 or z > 5:
        return nan_out
    zf = 1 + z
    try:
        return {
            'z_corrected_amplitude':
                float(break_magnitude * np.sqrt(zf)) if not np.isnan(break_magnitude) else np.nan,
            'z_corrected_color_slope':
                float(color_slope_early * zf)        if not np.isnan(color_slope_early) else np.nan,
            'luminosity_change_proxy':
                float(break_magnitude * np.log10(zf + 1)) if not np.isnan(break_magnitude) else np.nan,
            'z_corrected_drw_sigma':
                float(drw_sigma * np.sqrt(zf))       if not np.isnan(drw_sigma) else np.nan,
            'intrinsic_variability_proxy':
                float(drw_sigma / np.sqrt(zf))       if not np.isnan(drw_sigma) else np.nan,
        }
    except Exception:
        return nan_out
```

### Master Feature Runner

```python
# scripts/feature_engineering.py
"""
Compute all 7 feature groups for all cards.
Parallel execution with joblib.
Output: data/features/all_features.csv
Expected runtime: 30-90 minutes for 150k cards.
"""
import json, os, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

CARDS_DIR  = "cards/benchmark_v3"
OUTPUT_CSV = "data/features/all_features.csv"
N_JOBS     = 8

# Import all feature functions (assumed defined in same file or imported)

def process_card(card_path):
    try:
        with open(card_path) as f:
            card = json.load(f)

        epochs = card.get('wise_epochs', [])
        if len(epochs) < 10:
            return None

        df = pd.DataFrame(epochs)
        required = ['mjd','w1mpro','w2mpro']
        if not all(c in df.columns for c in required):
            return None
        df = df.dropna(subset=required).astype(float).sort_values('mjd')
        if 'w1sigmpro' not in df.columns:
            df['w1sigmpro'] = 0.05

        mjd  = df['mjd'].values
        w1   = df['w1mpro'].values
        w2   = df['w2mpro'].values
        w1e  = df['w1sigmpro'].values

        row = {
            'source_id': card['source_id'],
            'label':     card.get('label', ''),
            'z':         float(card.get('z', np.nan) or np.nan),
            'n_epochs':  len(df),
            'n_seasons': card.get('n_seasons', 1),
            'baseline_days': card.get('baseline_days', 0),
        }

        # Group A — Color Asymmetry
        try:
            from scripts.color_asymmetry_features import compute_color_features
            ca = compute_color_features(card_path)
            if ca:
                row.update({k:v for k,v in ca.items()
                            if k not in ('source_id','label','n_epochs')})
        except Exception:
            pass

        # Groups B-G
        row.update(wavelet_features(mjd, w1))
        row.update(structural_break_features(mjd, w1))
        row.update(seasonal_coherence_features(mjd, w1))
        row.update(flux_ratio_features(mjd, w1, w2))
        row.update(drw_residual_features(mjd, w1, w1e))
        row.update(redshift_features(
            row.get('z', np.nan),
            row.get('break_magnitude', np.nan),
            row.get('color_slope_early', np.nan),
            row.get('flux_ratio_trend', np.nan),
            row.get('drw_sigma', np.nan)
        ))
        return row
    except Exception:
        return None

def main():
    cards = sorted(Path(CARDS_DIR).glob("*.json"))
    print(f"Computing features for {len(cards):,} cards ({N_JOBS} workers)...")

    # Skip already-computed sources if output exists
    done_ids = set()
    if os.path.exists(OUTPUT_CSV):
        done_ids = set(pd.read_csv(OUTPUT_CSV)['source_id'].astype(str))
        print(f"Skipping {len(done_ids):,} already computed")
        cards = [c for c in cards if c.stem not in done_ids]

    results = Parallel(n_jobs=N_JOBS, verbose=0, prefer='threads')(
        delayed(process_card)(c) for c in tqdm(cards, desc="Features")
    )
    new_rows = [r for r in results if r is not None]
    df_new   = pd.DataFrame(new_rows)

    if os.path.exists(OUTPUT_CSV) and len(done_ids) > 0:
        df_existing = pd.read_csv(OUTPUT_CSV)
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_all = df_new

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_all.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved {len(df_all):,} rows to {OUTPUT_CSV}")
    print(f"Feature columns: {len(df_all.columns)}")
    print(f"Failed/skipped cards: {len(cards) - len(new_rows):,}")
    missing = (df_all.isna().mean() * 100).sort_values(ascending=False)
    print(f"\nTop 10 features by missing rate:")
    print(missing.head(10).to_string())

if __name__ == "__main__":
    main()
```

Run:

```bash
python scripts/feature_engineering.py 2>&1 | tee logs/feature_engineering.log
```

---

## SECTION 4 — MERGE EXISTING PIPELINE SCORES

```bash
# Run existing pipeline on all cards to get composite_score
python main.py \
    --input  data/benchmark/benchmark_master_v3.csv \
    --topn   999999 \
    --score-thresh 0.0 \
    --output results/benchmark_v3_scores.csv \
    --skip-existing \
    2>&1 | tee logs/main_pipeline_run.log

# Merge with feature table
python3 << 'EOF'
import pandas as pd

feat   = pd.read_csv("data/features/all_features.csv")
scores = pd.read_csv("results/benchmark_v3_scores.csv")

# Normalize source_id join key
feat['source_id']   = feat['source_id'].astype(str).str.strip()
scores['source_id'] = scores['source_id'].astype(str).str.strip()

merged = feat.merge(scores[['source_id','composite_score']],
                    on='source_id', how='left')
merged.to_csv("data/features/all_features_with_scores.csv", index=False)
print(f"Merged: {len(merged):,} rows, {len(merged.columns)} features")
print(f"composite_score coverage: {merged.composite_score.notna().mean():.1%}")
EOF
```

---

## SECTION 5 — TWO-LAYER CLASSIFIER

```python
# scripts/classifier_v2.py
"""
Two-layer classifier:
Layer 1: Isolation Forest pre-filter (removes obvious negatives)
Layer 2: LightGBM with Optuna tuning (50 trials, maximize balanced accuracy)
SHAP values computed for interpretability (required for paper).
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import shap
import joblib, os, json
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (balanced_accuracy_score, accuracy_score,
                              recall_score, precision_score,
                              confusion_matrix, roc_auc_score,
                              classification_report, matthews_corrcoef)
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

optuna.logging.set_verbosity(optuna.logging.WARNING)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

FEATURES_CSV = "data/features/all_features_with_scores.csv"
TRAIN_CSV    = "data/benchmark/benchmark_v3_train.csv"
DEV_CSV      = "data/benchmark/benchmark_v3_dev.csv"

ALL_FEATURE_COLS = [
    'composite_score',
    # A
    'color_slope_early','color_slope_late','color_asymmetry',
    'color_range','color_flux_corr','dust_lag_proxy',
    # B
    'wavelet_d1','wavelet_d2','wavelet_d3','wavelet_approx_power',
    'wavelet_power_ratio_3_1','wavelet_power_ratio_3_2',
    'wavelet_entropy','wavelet_hurst',
    # C
    'break_magnitude','break_significance','pre_break_slope',
    'post_break_slope','slope_change','break_fractional_position',
    'break_duration_yr','n_detected_breaks',
    # D
    'mean_season_coherence','coherence_trend','min_season_coherence',
    'max_season_coherence','coherence_drop_amplitude',
    'coherence_recovery','n_coherent_seasons',
    # E
    'flux_ratio_mean','flux_ratio_std','flux_ratio_trend',
    'flux_ratio_skewness','flux_ratio_peak_to_trough',
    'flux_ratio_percentile_90_10','flux_ratio_late_minus_early',
    # F
    'drw_sigma','drw_tau','drw_residual_rms','drw_residual_skewness',
    'drw_residual_autocorr','drw_fit_quality',
    'drw_residual_trend','drw_chi2_per_dof',
    # G
    'z_corrected_amplitude','z_corrected_color_slope',
    'luminosity_change_proxy','z_corrected_drw_sigma',
    'intrinsic_variability_proxy',
    # Metadata features
    'n_epochs','n_seasons','baseline_days','z',
]

def load_split(feat_csv, split_csv, label='split'):
    feat  = pd.read_csv(feat_csv)
    split = pd.read_csv(split_csv)[['source_id','label']]
    df    = split.merge(feat, on='source_id', how='left')
    df['y'] = (df['label'] == 'CLAGN').astype(int)
    avail   = [f for f in ALL_FEATURE_COLS if f in df.columns]
    missing = [f for f in ALL_FEATURE_COLS if f not in df.columns]
    if missing:
        print(f"  [{label}] Missing features ({len(missing)}): {missing[:5]}...")
    X = df[avail].copy()
    # Fill NaN with column medians (computed on this split only)
    medians = X.median()
    X = X.fillna(medians)
    y = df['y'].values
    print(f"  [{label}] {len(df):,} sources | {len(avail)} features | "
          f"{y.sum()} CLAGN ({y.mean():.2%})")
    return X, y, df, avail

print("Loading data...")
X_train, y_train, df_train, feat_cols = load_split(FEATURES_CSV, TRAIN_CSV, 'train')
X_dev,   y_dev,   df_dev,   _         = load_split(FEATURES_CSV, DEV_CSV,   'dev')

# ── Layer 1: Isolation Forest ─────────────────────────────────────
print("\nTraining Isolation Forest (Layer 1)...")
scaler = StandardScaler()
Xts = scaler.fit_transform(X_train)
Xds = scaler.transform(X_dev)

iso = IsolationForest(
    n_estimators=500,
    contamination=0.12,
    max_samples='auto',
    random_state=42,
    n_jobs=-1
)
iso.fit(Xts)

train_scores = iso.score_samples(Xts)
dev_scores   = iso.score_samples(Xds)

# Threshold: all CLAGN pass Layer 1 (set threshold below min CLAGN score)
clagn_scores = train_scores[y_train == 1]
iso_threshold = np.percentile(clagn_scores, 5) - 0.01  # 5th percentile of CLAGN scores
print(f"Isolation Forest threshold: {iso_threshold:.4f}")
print(f"Train pass rate: {(train_scores > iso_threshold).mean():.1%} "
      f"(CLAGN pass: {(train_scores[y_train==1] > iso_threshold).mean():.1%})")
print(f"Dev pass rate:   {(dev_scores > iso_threshold).mean():.1%}")

joblib.dump(iso,    "models/isolation_forest.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(iso_threshold, "models/iso_threshold.pkl")

# ── Layer 2: LightGBM + Optuna ────────────────────────────────────
l2_mask_train = train_scores > iso_threshold
X_l2  = X_train[l2_mask_train]
y_l2  = y_train[l2_mask_train]
n_neg = (y_l2 == 0).sum()
n_pos = (y_l2 == 1).sum()
spw   = n_neg / max(n_pos, 1)
print(f"\nLayer 2 training data: {len(X_l2):,} sources "
      f"({n_pos} CLAGN, base scale_pos_weight={spw:.1f})")

def objective(trial):
    p = {
        'objective':         'binary',
        'metric':            'binary_logloss',
        'verbosity':         -1,
        'boosting_type':     'gbdt',
        'num_leaves':        trial.suggest_int('num_leaves', 15, 255),
        'max_depth':         trial.suggest_int('max_depth', 3, 12),
        'learning_rate':     trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample':         trial.suggest_float('subsample', 0.4, 1.0),
        'subsample_freq':    1,
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha',  1e-5, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-5, 10.0, log=True),
        'scale_pos_weight':  trial.suggest_float('scale_pos_weight', spw*0.5, spw*3.0),
        'min_split_gain':    trial.suggest_float('min_split_gain', 0.0, 1.0),
        'n_estimators':      1000,
        'random_state':      42,
    }
    cv_bal, cv_auc = [], []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, val_idx in skf.split(X_l2, y_l2):
        clf = lgb.LGBMClassifier(**p)
        clf.fit(X_l2.iloc[tr_idx], y_l2[tr_idx],
                eval_set=[(X_l2.iloc[val_idx], y_l2[val_idx])],
                callbacks=[lgb.early_stopping(40, verbose=False),
                           lgb.log_evaluation(period=-1)])
        proba = clf.predict_proba(X_l2.iloc[val_idx])[:,1]
        pred  = (proba >= 0.5).astype(int)
        cv_bal.append(balanced_accuracy_score(y_l2[val_idx], pred))
        cv_auc.append(roc_auc_score(y_l2[val_idx], proba))
    # Weighted objective: 70% balanced accuracy, 30% AUC
    return 0.7 * np.mean(cv_bal) + 0.3 * np.mean(cv_auc)

print("\nRunning Optuna (50 trials)...")
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
)
study.optimize(objective, n_trials=50, show_progress_bar=True, n_jobs=1)

print(f"\nBest trial value: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# Save Optuna study
pd.DataFrame([{**t.params, 'value': t.value}
              for t in study.trials
              if t.value is not None]).to_csv(
    "results/optuna_trials.csv", index=False)

# Train final model
best_p = {**study.best_params,
          'objective': 'binary', 'metric': 'binary_logloss',
          'verbosity': -1, 'n_estimators': 2000, 'random_state': 42}
print("\nTraining final model on all Layer 2 training data...")
final_clf = lgb.LGBMClassifier(**best_p)
final_clf.fit(X_l2, y_l2,
              eval_set=[(X_dev[dev_scores > iso_threshold],
                         y_dev[dev_scores > iso_threshold])],
              callbacks=[lgb.early_stopping(50, verbose=True),
                         lgb.log_evaluation(period=100)])
joblib.dump(final_clf, "models/lgbm_classifier.pkl")
joblib.dump(feat_cols,  "models/feature_columns.pkl")

# ── SHAP values (for paper) ───────────────────────────────────────
print("\nComputing SHAP values...")
explainer = shap.TreeExplainer(final_clf)
shap_sample = X_l2.sample(min(2000, len(X_l2)), random_state=42)
shap_vals   = explainer.shap_values(shap_sample)
if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]

fig, ax = plt.subplots(figsize=(10,8))
shap.summary_plot(shap_vals, shap_sample, show=False,
                  max_display=20, plot_type='bar')
plt.title("Feature Importance (SHAP)", fontsize=13)
plt.tight_layout()
plt.savefig("figures/fig_shap_importance.pdf", dpi=300, bbox_inches='tight')
plt.savefig("figures/fig_shap_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved SHAP figure")

# Save SHAP mean values
shap_df = pd.DataFrame({
    'feature': X_l2.columns,
    'mean_abs_shap': np.abs(shap_vals).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)
shap_df.to_csv("results/shap_values.csv", index=False)

# ── Dev evaluation ────────────────────────────────────────────────
print("\n=== DEV EVALUATION (not the final test) ===")
l2_mask_dev = dev_scores > iso_threshold
y_dev_pred  = np.zeros(len(y_dev), dtype=int)
proba_dev   = np.zeros(len(y_dev))

if l2_mask_dev.sum() > 0:
    proba_l2 = final_clf.predict_proba(X_dev[l2_mask_dev])[:,1]
    proba_dev[l2_mask_dev] = proba_l2
    y_dev_pred[l2_mask_dev] = (proba_l2 >= 0.5).astype(int)

print(classification_report(y_dev, y_dev_pred, target_names=['non-CLAGN','CLAGN']))
metrics = {
    'overall_accuracy':   round(float(accuracy_score(y_dev, y_dev_pred)), 4),
    'balanced_accuracy':  round(float(balanced_accuracy_score(y_dev, y_dev_pred)), 4),
    'clagn_recall':       round(float(recall_score(y_dev, y_dev_pred, zero_division=0)), 4),
    'clagn_precision':    round(float(precision_score(y_dev, y_dev_pred, zero_division=0)), 4),
    'roc_auc':            round(float(roc_auc_score(y_dev, proba_dev)), 4),
    'mcc':                round(float(matthews_corrcoef(y_dev, y_dev_pred)), 4),
}
print(f"\nKey metrics:")
for k, v in metrics.items():
    print(f"  {k:25s}: {v}")
with open("results/dev_metrics_v2.json", 'w') as f:
    json.dump(metrics, f, indent=2)

# Fail fast if dev performance is poor
if metrics['balanced_accuracy'] < 0.80:
    print(f"\nWARNING: Dev balanced accuracy {metrics['balanced_accuracy']:.3f} < 0.80")
    print("Do NOT proceed to Step 6 until this is diagnosed and fixed.")
    print("Check: feature completeness, class balance, WISE coverage rate")
else:
    print(f"\nDev balanced accuracy {metrics['balanced_accuracy']:.3f} >= 0.80")
    print("Proceed to Step 6 (threshold calibration).")

print("\nDO NOT run evaluate_benchmark.py test mode yet.")
```

Run:

```bash
python scripts/classifier_v2.py 2>&1 | tee logs/classifier_v2_training.log
```

---

## SECTION 6 — NOVEL METHOD VALIDATION + CANDIDATE CONFIRMATION

Run scripts from CLAGN_NOVEL_METHOD_IMPLEMENTATION.md:

```bash
# Statistical validation of color asymmetry features
python scripts/validate_color_asymmetry.py 2>&1 | tee logs/validation.log

# Spectroscopic confirmation of SDSS J000159.27+034352.9
python scripts/confirm_candidate.py 2>&1 | tee logs/candidate_confirmation.log

# Candidate color trajectory analysis
python scripts/analyze_candidate_color.py

# Interpret results:
# - If confirm_candidate.py finds >= 2 spectra AND significance >= 3σ:
#     CONFIRMED — state in abstract as confirmed new CLAGN
# - If significance 2-3σ:
#     MARGINAL — state as "strong candidate requiring further confirmation"
# - If < 2 spectra or < 2σ:
#     UNCONFIRMED — remove from abstract claim, keep as appendix
```

---

## SECTION 7 — ABLATION STUDY

```python
# scripts/ablation_study.py
"""
Train classifier with each feature group added incrementally.
Produces the ablation table for the paper.
Shows contribution of each novel feature group.
"""
import pandas as pd, numpy as np, lightgbm as lgb, joblib, json
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, recall_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

feat_cols_loaded = joblib.load("models/feature_columns.pkl")
best_params      = json.load(open("results/optuna_trials.csv"))  # load best params

GROUP_SETS = {
    'A_existing_only':        ['composite_score'],
    'B_+color_asymmetry':     ['composite_score','color_slope_early','color_slope_late',
                               'color_asymmetry','color_range','color_flux_corr','dust_lag_proxy'],
    'C_+wavelets':            None,  # previous + wavelet features
    'D_+structural_break':    None,
    'E_+seasonal_coherence':  None,
    'F_+flux_ratio':          None,
    'G_+drw_residuals':       None,
    'H_+redshift_correction': None,  # all features
    'I_novel_only_no_composite': None,  # all except composite_score
}

# Build cumulative feature sets
FEATURE_GROUPS_ORDERED = [
    ('color_asymmetry',   ['color_slope_early','color_slope_late','color_asymmetry',
                           'color_range','color_flux_corr','dust_lag_proxy']),
    ('wavelets',          ['wavelet_d1','wavelet_d2','wavelet_d3','wavelet_entropy',
                           'wavelet_hurst','wavelet_power_ratio_3_1','wavelet_power_ratio_3_2']),
    ('structural_break',  ['break_magnitude','break_significance','pre_break_slope',
                           'post_break_slope','slope_change','n_detected_breaks']),
    ('coherence',         ['mean_season_coherence','coherence_trend','min_season_coherence',
                           'coherence_drop_amplitude','coherence_recovery']),
    ('flux_ratio',        ['flux_ratio_mean','flux_ratio_std','flux_ratio_trend',
                           'flux_ratio_skewness','flux_ratio_peak_to_trough']),
    ('drw_residuals',     ['drw_sigma','drw_tau','drw_residual_rms','drw_residual_autocorr',
                           'drw_fit_quality','drw_residual_trend','drw_chi2_per_dof']),
    ('redshift_corr',     ['z_corrected_amplitude','z_corrected_color_slope',
                           'luminosity_change_proxy','intrinsic_variability_proxy']),
]

cumulative = ['composite_score']
results = []

for group_name, group_feats in FEATURE_GROUPS_ORDERED:
    cumulative = cumulative + group_feats
    avail = [f for f in cumulative if f in feat_cols_loaded]
    # Load data, train, evaluate dev
    # ... (same load_split logic as classifier_v2.py)
    clf = lgb.LGBMClassifier(**{**best_params, 'n_estimators':500, 'verbosity':-1})
    clf.fit(X_train[avail], y_train)
    proba = clf.predict_proba(X_dev[avail])[:,1]
    pred  = (proba >= 0.5).astype(int)
    results.append({
        'feature_set':    f"+{group_name}",
        'n_features':     len(avail),
        'balanced_acc':   round(balanced_accuracy_score(y_dev, pred), 4),
        'recall':         round(recall_score(y_dev, pred, zero_division=0), 4),
        'roc_auc':        round(roc_auc_score(y_dev, proba), 4),
    })
    print(f"+{group_name:20s}: bal_acc={results[-1]['balanced_acc']:.4f} "
          f"recall={results[-1]['recall']:.4f} auc={results[-1]['roc_auc']:.4f}")

df_abl = pd.DataFrame(results)
df_abl.to_csv("results/ablation_study.csv", index=False)

# Plot
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(df_abl['feature_set'], df_abl['balanced_acc'],
        'o-', color='#2D6BE4', label='Balanced Accuracy', linewidth=2, markersize=8)
ax.plot(df_abl['feature_set'], df_abl['recall'],
        's--', color='#E44B3F', label='CLAGN Recall', linewidth=1.5, markersize=6)
ax.axhline(0.85, color='gray', linestyle=':', linewidth=0.8, label='Target (85%)')
ax.set_xlabel('Feature set', fontsize=11)
ax.set_ylabel('Score', fontsize=11)
ax.set_title('Ablation Study: Feature Group Contributions', fontsize=12)
plt.xticks(rotation=30, ha='right')
ax.legend()
ax.set_ylim(0.5, 1.0)
plt.tight_layout()
plt.savefig("figures/fig_ablation.pdf", dpi=300, bbox_inches='tight')
plt.savefig("figures/fig_ablation.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved ablation figure and CSV")
```

Run:

```bash
python scripts/ablation_study.py 2>&1 | tee logs/ablation.log
```

---

## SECTION 8 — THRESHOLD CALIBRATION

Only proceed if: dev balanced accuracy >= 0.82 AND dev recall >= 0.78.
If either is not met, stop and diagnose before proceeding.

```python
# scripts/calibrate_threshold_v2.py
"""
Find optimal Layer 2 probability threshold on dev set.
Maximizes balanced accuracy subject to recall >= 0.80 and blazar FPR = 0.
Threshold is frozen after this step and never adjusted again.
"""
import numpy as np, pandas as pd, json, joblib
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score

# Load models and dev predictions
iso       = joblib.load("models/isolation_forest.pkl")
scaler    = joblib.load("models/scaler.pkl")
clf       = joblib.load("models/lgbm_classifier.pkl")
iso_thresh= joblib.load("models/iso_threshold.pkl")
feat_cols = joblib.load("models/feature_columns.pkl")

# ... (load X_dev, y_dev, df_dev using same logic)

dev_scores   = iso.score_samples(scaler.transform(X_dev))
l2_mask      = dev_scores > iso_thresh
proba_dev_l2 = clf.predict_proba(X_dev[l2_mask])[:,1]

# Also check blazar FPR at each threshold
blazar_mask = df_dev['label'] == 'blazar'

thresholds = np.arange(0.05, 0.95, 0.005)
records = []
for t in thresholds:
    y_pred_full = np.zeros(len(y_dev), dtype=int)
    y_pred_full[l2_mask] = (proba_dev_l2 >= t).astype(int)
    bal_acc    = balanced_accuracy_score(y_dev, y_pred_full)
    recall     = recall_score(y_dev, y_pred_full, zero_division=0)
    precision  = precision_score(y_dev, y_pred_full, zero_division=0)
    blazar_fpr = y_pred_full[blazar_mask].mean() if blazar_mask.sum() > 0 else 0.0
    records.append({'t': t, 'bal_acc': bal_acc, 'recall': recall,
                    'precision': precision, 'blazar_fpr': blazar_fpr})

df_t = pd.DataFrame(records)

# Constraints: recall >= 0.80, blazar_fpr == 0
valid = df_t[(df_t['recall'] >= 0.80) & (df_t['blazar_fpr'] == 0.0)]
if len(valid) == 0:
    print("WARNING: No threshold satisfies recall>=0.80 AND blazar_fpr=0")
    print("Relaxing to recall>=0.75...")
    valid = df_t[df_t['recall'] >= 0.75]

best_row = valid.loc[valid['bal_acc'].idxmax()]
best_t   = float(best_row['t'])

print(f"\nOptimal threshold: {best_t:.4f}")
print(f"  Balanced accuracy: {best_row['bal_acc']:.4f}")
print(f"  Recall:            {best_row['recall']:.4f}")
print(f"  Precision:         {best_row['precision']:.4f}")
print(f"  Blazar FPR:        {best_row['blazar_fpr']:.4f}")

config = {
    'threshold_v2':    best_t,
    'dev_balanced_acc':float(best_row['bal_acc']),
    'dev_recall':      float(best_row['recall']),
    'dev_precision':   float(best_row['precision']),
    'dev_blazar_fpr':  float(best_row['blazar_fpr']),
    'frozen':          True,
    'note': 'This threshold was set on dev set ONCE and must not be changed.'
}
with open("models/threshold_v2.json", 'w') as f:
    json.dump(config, f, indent=2)
print("\nThreshold frozen. DO NOT modify models/threshold_v2.json")
df_t.to_csv("results/threshold_sweep_dev.csv", index=False)
```

Run only if dev balanced accuracy >= 0.82:

```bash
python scripts/calibrate_threshold_v2.py
```

---

## SECTION 9 — FINAL TEST EVALUATION (run ONCE)

Only run after ALL of the following are satisfied:

- [ ] Dev balanced accuracy >= 0.82
- [ ] Ablation study complete and saved
- [ ] Threshold frozen in models/threshold_v2.json
- [ ] SHAP values computed and saved
- [ ] Candidate confirmation result known
- [ ] All figures for paper generated
- [ ] Paper draft outline written

```bash
python evaluate_benchmark.py \
    --cards      cards/benchmark_v3/ \
    --benchmark  data/benchmark/benchmark_v3_test.csv \
    --features   data/features/all_features_with_scores.csv \
    --model_dir  models/ \
    --mode       test \
    --report     results/FINAL_TEST_RESULTS.json \
    2>&1 | tee logs/FINAL_TEST_EVALUATION.log

echo "=== FINAL RESULTS ==="
cat results/FINAL_TEST_RESULTS.json

# This number goes in the paper abstract. NEVER re-run this script.
```

---

## SECTION 10 — PAPER FIGURES CHECKLIST

After Section 9 completes, generate all remaining figures:

```bash
# Already generated:
# figures/fig1_color_asymmetry_violin.pdf
# figures/fig2_color_flux_corr.pdf
# figures/fig3_dust_lag.pdf
# figures/fig4_2d_scatter.pdf
# figures/fig5_roc_curves.pdf
# figures/fig6_spectra_comparison.pdf
# figures/fig7_candidate_color_evolution.pdf
# figures/fig_shap_importance.pdf
# figures/fig_ablation.pdf

# Still needed — generate with:
python scripts/plot_final_confusion_matrix.py    # Figure: test confusion matrix
python scripts/plot_light_curve_examples.py      # Figure: 4-panel example light curves
                                                 # (1 confirmed CLAGN, 1 blazar, 1 SN IIn,
                                                 #  1 normal AGN — show why each is right/wrong)

# Verify all figures
echo "Figure count: $(ls figures/*.pdf | wc -l)"
echo "Results files: $(ls results/*.csv results/*.json | wc -l)"
```

---

## SECTION 11 — ROBUSTNESS CHECKS (required for peer review)

Referees at ApJ will ask for these. Implement before submission.

```bash
# Check 1: Does performance degrade at high ecliptic latitude?
# (Tests Option 2 cadence bias — even though we're not publishing it as novel,
#  we need to show it doesn't affect our results)
python scripts/check_ecliptic_latitude_bias.py

# Check 2: Does performance degrade at high redshift?
python scripts/check_redshift_dependence.py

# Check 3: Is the result stable across random seeds?
# (Re-train with 5 different random seeds, report mean ± std)
python scripts/check_seed_stability.py --n_seeds 5

# Check 4: Does the Isolation Forest Layer 1 introduce any CLAGN bias?
# (Verify that CLAGN pass Layer 1 at >=99% rate)
python scripts/check_layer1_clagn_passthrough.py

# Check 5: Confusion matrix stratified by transition type
# (Does the classifier perform equally well on turn-on vs turn-off CLAGN?)
python scripts/check_transition_type_performance.py
```

---

## SECTION 12 — PAPER OUTLINE

**Title (working):**
"W1−W2 Color Trajectory Asymmetry: A Novel Infrared Discriminator of
Changing-Look AGN Transition Direction with Discovery of a New z=0.85 CLAGN"

**Abstract (key numbers to fill in after Section 9):**
We present [X] novel photometric features derived from 14 years of WISE/NEOWISE-R
infrared photometry for the automated detection of Changing-Look Active Galactic
Nuclei (CLAGN). The primary novel contribution is W1−W2 color trajectory asymmetry,
which exploits the directional response of the AGN dust torus to accretion rate changes.
Applied to a [150k]-source benchmark including [N] confirmed CLAGN, our two-layer
classifier achieves [X]% overall accuracy and [Y]% balanced accuracy, improvements
of [ΔX]% and [ΔY]% over a baseline amplitude-only method. We report the [spectroscopic
confirmation / strong candidate status] of SDSS J000159.27+034352.9 at z=0.850
as a new turn-on CLAGN identified by our pipeline.

**Sections:**

1. Introduction (CLAGN physics, amplitude-only limitation, this paper's contribution)
2. Data (WISE/NEOWISE-R photometry, benchmark construction)
3. Novel Features (Groups A-G with physical motivation for each)
4. Classification Method (two-layer classifier, Optuna tuning)
5. Results (ablation table, ROC curves, confusion matrix)
6. The New Candidate (WISE light curve, spectroscopic confirmation, Mg II measurement)
7. Systematic Checks (ecliptic bias, redshift dependence, seed stability)
8. Discussion (why it works, limitations, future LSST application)
9. Conclusion

---

## SECTION 13 — EXECUTION ORDER SUMMARY

```
STEP                              SCRIPT                          ETA
─────────────────────────────────────────────────────────────────────────
0. Preflight check                scripts/preflight.sh            5 min
1a. Bulk WISE download            bulk_wise_download.py           4-12 hr
1b. Build cards                   build_cards_from_bulk.py        20 min
2.  Feature engineering           feature_engineering.py          30-90 min
3.  Merge pipeline scores         main.py + merge script          30-60 min
4.  Train classifier              classifier_v2.py                2-4 hr
5.  Validate novel method         validate_color_asymmetry.py     10 min
    Confirm candidate             confirm_candidate.py            5 min
    Analyze candidate             analyze_candidate_color.py      2 min
6.  Calibrate threshold           calibrate_threshold_v2.py       2 min
7.  Ablation study                ablation_study.py               30-60 min
8.  Robustness checks             scripts/check_*.py              30 min
9.  FINAL test evaluation         evaluate_benchmark.py           10 min
10. Paper figures                 scripts/plot_*.py               20 min
─────────────────────────────────────────────────────────────────────────
TOTAL COMPUTE TIME: ~12-20 hours across 2-3 days
TOTAL DISK USAGE: ~20-35 GB
```

---

## SECTION 14 — PERFORMANCE TARGETS

| Metric            | Current | Target | What Gets You There            |
| ----------------- | ------- | ------ | ------------------------------ |
| Overall accuracy  | 87.8%   | >92%   | DRW residuals + Layer 1 filter |
| Balanced accuracy | 85.2%   | >88%   | Color asymmetry + coherence    |
| CLAGN recall      | 80.0%   | >85%   | Structural break + wavelets    |
| CLAGN precision   | 72.7%   | >80%   | Layer 1 Isolation Forest       |
| Blazar FPR        | 0.0%    | 0.0%   | Maintain (hard constraint)     |
| MCC               | 0.682   | >0.75  | All novel features combined    |
| ROC-AUC           | —       | >0.95  | LightGBM + all features        |

If any target is not met after Section 9, do NOT change the threshold.
Instead diagnose which feature group is underperforming using the ablation
study and SHAP values, then improve that specific feature's implementation.
