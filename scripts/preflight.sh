#!/bin/bash
# scripts/preflight.sh — CLAGN Pipeline Preflight Check
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
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('IRSA: OK' if d.get('data') else 'IRSA: FAIL')"

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
        print(df['label'].value_counts().to_string())
        break
else:
    print('No benchmark found')

# Cards
for d in ['cards/benchmark_v3','cards/benchmark_v2']:
    if os.path.exists(d):
        import pathlib
        n = len(list(pathlib.Path(d).glob('*.json')))
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
"

echo ""
echo "=== Preflight complete. Review above before proceeding. ==="
