#!/usr/bin/env bash
set -u

LOG_DIR="logs"
PART1_LOG="${LOG_DIR}/part1_downloads.log"
PART1_FAIL="${LOG_DIR}/part1_failures.csv"
PART3_SUMMARY="${LOG_DIR}/part3_split_summary.json"
INGEST_COUNTS="${LOG_DIR}/ingest_catalog_counts.json"

mkdir -p data/raw data/benchmark cards/benchmark_v3 "${LOG_DIR}"

log() {
  printf "[%s] %s\n" "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "${PART1_LOG}"
}

run_step() {
  local name="$1"
  local cmd="$2"
  log "START ${name}"
  {
    echo "----- ${name} -----"
    echo "${cmd}"
  } >> "${PART1_LOG}" 2>&1
  eval "${cmd}" >> "${PART1_LOG}" 2>&1
  local code=$?
  {
    echo "exit_code=${code}"
    echo
  } >> "${PART1_LOG}" 2>&1
  if [[ ${code} -ne 0 ]]; then
    echo "\"${name}\",\"${code}\",\"$(date -u +'%Y-%m-%dT%H:%M:%SZ')\"" >> "${PART1_FAIL}"
    log "FAIL  ${name} (code=${code}) -- continuing"
  else
    log "OK    ${name}"
  fi
  return 0
}

fetch_vizier_tsv() {
  local source="$1"
  local outcols="$2"
  local outmax="$3"
  local outfile="$4"
  python3 - "$source" "$outcols" "$outmax" "$outfile" <<'PY'
import sys
from pathlib import Path
import requests

source, outcols, outmax, outfile = sys.argv[1:5]
params = {
    "-source": source,
    "-out": outcols,
    "-out.max": outmax,
}
r = requests.get("https://vizier.u-strasbg.fr/viz-bin/asu-tsv", params=params, timeout=300)
r.raise_for_status()
Path(outfile).write_text(r.text, encoding="utf-8")
print(f"Wrote {outfile} ({len(r.text)} bytes)")
PY
}

fetch_osc_type() {
  local claimedtype="$1"
  local limit="$2"
  local outfile="$3"
  python3 - "$claimedtype" "$limit" "$outfile" <<'PY'
import json
import sys
from pathlib import Path
import requests

claimedtype, limit, outfile = sys.argv[1:4]
urls = [
    "https://api.sne.space/catalog",
    "https://api.astrocats.space/catalog",
]
params = {
    "claimedtype": claimedtype,
    "format": "json",
    "limit": int(limit),
    "quantity": "name,ra,dec,redshift,claimedtype,discoverdate,maxdate",
}
last_err = None
for url in urls:
    try:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        txt = r.text.strip()
        # keep valid JSON only; "{}" is allowed but noted by caller
        json.loads(txt if txt else "{}")
        Path(outfile).write_text(txt if txt else "{}", encoding="utf-8")
        print(f"Wrote {outfile} via {url}")
        raise SystemExit(0)
    except Exception as exc:  # noqa: BLE001
        last_err = str(exc)
Path(outfile).write_text("{}", encoding="utf-8")
raise SystemExit(f"OSC fetch failed for {claimedtype}: {last_err}")
PY
}

echo "step,exit_code,timestamp_utc" > "${PART1_FAIL}"
: > "${PART1_LOG}"

echo "=== PART 0: PREFLIGHT ==="
for bin in wget curl python3; do
  if ! command -v "${bin}" >/dev/null 2>&1; then
    echo "Missing required binary: ${bin}" >&2
    exit 1
  fi
done

python3 - <<'PY'
mods=["pandas","numpy","requests","astropy"]
missing=[]
for m in mods:
    try:
        __import__(m)
    except Exception:
        missing.append(m)
if missing:
    raise SystemExit(f"Missing Python modules: {missing}")
print("Python module check PASS")
PY

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found; installing via Homebrew..."
  if command -v brew >/dev/null 2>&1; then
    brew install tmux
  else
    echo "Homebrew not found; cannot install tmux." >&2
    exit 1
  fi
fi

echo "Verifying IRSA connectivity..."
curl --max-time 30 -s "https://irsa.ipac.caltech.edu/TAP/sync?QUERY=SELECT+TOP+1+ra,dec+FROM+neowiser_p1bs_psd&FORMAT=json&LANG=ADQL" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('IRSA OK, got', len(d.get('data',[])), 'rows')"

echo "=== PART 1: DOWNLOADS (continue on single-step failure) ==="
run_step "1.1_milliquas_vizier" "fetch_vizier_tsv 'VII/290/catalog' 'Name,RAJ2000,DEJ2000,z,Type,Rmag' '200000' 'data/raw/milliquas_vizier.tsv' && wc -l data/raw/milliquas_vizier.tsv && ls -lh data/raw/milliquas_vizier.tsv"
run_step "1.2_bzcat5" "fetch_vizier_tsv 'VII/274/bzcat5' 'Name,RAJ2000,DEJ2000,z,Type' '10000' 'data/raw/bzcat5.tsv' && wc -l data/raw/bzcat5.tsv && ls -lh data/raw/bzcat5.tsv"
run_step "1.3_4fgl_dr4" "wget -O data/raw/4fgl_dr4.fit https://fermi.gsfc.nasa.gov/ssc/data/access/lat/14yr_catalog/gll_psc_v35.fit && python3 -c \"from astropy.io import fits; h=fits.open('data/raw/4fgl_dr4.fit'); print('4FGL rows:', len(h[1].data)); h.close()\" && ls -lh data/raw/4fgl_dr4.fit"
run_step "1.4_macleod2019" "fetch_vizier_tsv 'J/ApJ/874/8/table2' '*' '5000' 'data/raw/macleod2019.tsv' && wc -l data/raw/macleod2019.tsv && ls -lh data/raw/macleod2019.tsv"
run_step "1.5_sheng2020" "fetch_vizier_tsv 'J/ApJ/889/46/table1' '*' '5000' 'data/raw/sheng2020.tsv' && wc -l data/raw/sheng2020.tsv && ls -lh data/raw/sheng2020.tsv"
run_step "1.6_hon2022" "fetch_vizier_tsv 'J/MNRAS/511/54/table1' '*' '5000' 'data/raw/hon2022.tsv' && wc -l data/raw/hon2022.tsv && ls -lh data/raw/hon2022.tsv"

run_step "1.7_osc_IIn" "fetch_osc_type 'IIn' '5000' 'data/raw/osc_IIn.json'"
run_step "1.7_osc_IIP" "fetch_osc_type 'IIP' '5000' 'data/raw/osc_IIP.json'"
run_step "1.7_osc_Ia" "fetch_osc_type 'Ia' '5000' 'data/raw/osc_Ia.json'"
run_step "1.7_osc_Ib" "fetch_osc_type 'Ib' '5000' 'data/raw/osc_Ib.json'"
run_step "1.7_osc_Ic" "fetch_osc_type 'Ic' '5000' 'data/raw/osc_Ic.json'"
run_step "1.7_osc_IIb" "fetch_osc_type 'IIb' '5000' 'data/raw/osc_IIb.json'"
run_step "1.7_osc_SLSN" "fetch_osc_type 'SLSN-I' '5000' 'data/raw/osc_SLSN.json'"
run_step "1.7_osc_LBV" "fetch_osc_type 'LBV' '1000' 'data/raw/osc_LBV.json'"
run_step "1.7_osc_Ibn" "fetch_osc_type 'Ibn' '1000' 'data/raw/osc_Ibn.json'"
run_step "1.7_osc_verify" "for f in data/raw/osc_*.json; do count=\$(python3 -c \"import json,sys; d=json.load(open(sys.argv[1])); print(len(d))\" \"\$f\" 2>/dev/null || echo 0); size=\$(ls -lh \"\$f\" 2>/dev/null | awk '{print \$5}'); echo \"\$f: \$count sources, \$size\"; done"

run_step "1.8_dr16q_vizier" "fetch_vizier_tsv 'VII/289/dr16q' 'SDSS,RAJ2000,DEJ2000,z' '120000' 'data/raw/dr16q_vizier.tsv' && wc -l data/raw/dr16q_vizier.tsv && ls -lh data/raw/dr16q_vizier.tsv"
run_step "1.9_crts_agn" "fetch_vizier_tsv 'J/MNRAS/470/4112/table1' '*' '20000' 'data/raw/crts_agn.tsv' && wc -l data/raw/crts_agn.tsv && ls -lh data/raw/crts_agn.tsv"
run_step "1.10_ward2024" "fetch_vizier_tsv 'J/ApJ/962/L3/table1' '*' '5000' 'data/raw/ward2024.tsv' && wc -l data/raw/ward2024.tsv"

echo "=== PART 2: INGESTION ==="
python3 scripts/ingest_all_catalogs.py
if [[ ! -f data/benchmark/benchmark_master_v3.csv ]]; then
  echo "Part 2 failed: data/benchmark/benchmark_master_v3.csv missing" >&2
  exit 1
fi

echo "=== PART 3: REBUILD SPLITS ==="
python3 scripts/rebuild_benchmark_splits.py \
  --input data/benchmark/benchmark_master_v3.csv \
  --prefix data/benchmark/benchmark_v3 \
  --ratio 70:15:15 \
  --seed 42

python3 - <<'PY'
import json
from pathlib import Path
import pandas as pd
master = Path("data/benchmark/benchmark_master_v3.csv")
train = Path("data/benchmark/benchmark_v3_train.csv")
dev = Path("data/benchmark/benchmark_v3_dev.csv")
test = Path("data/benchmark/benchmark_v3_test.csv")
for p in [master, train, dev, test]:
    if not p.exists():
        raise SystemExit(f"Missing split file: {p}")
m = len(pd.read_csv(master))
t = len(pd.read_csv(train))
d = len(pd.read_csv(dev))
e = len(pd.read_csv(test))
if t + d + e != m:
    raise SystemExit(f"Split mismatch: train+dev+test={t+d+e} != master={m}")
summary = {"master": m, "train": t, "dev": d, "test": e}
Path("logs/part3_split_summary.json").write_text(json.dumps(summary, indent=2))
print("Part 3 summary:", summary)
PY

echo "=== PRE-PART-4 GATE ==="
python3 - <<'PY'
import json
from pathlib import Path
import pandas as pd
part1_fail = Path("logs/part1_failures.csv")
ing = Path("logs/ingest_catalog_counts.json")
split = Path("logs/part3_split_summary.json")
if not ing.exists():
    raise SystemExit("Part 2 summary missing")
if not split.exists():
    raise SystemExit("Part 3 summary missing")
fail_rows = []
if part1_fail.exists():
    fail_rows = [x for x in part1_fail.read_text().splitlines()[1:] if x.strip()]
counts = json.loads(ing.read_text())
split_s = json.loads(split.read_text())
print("Parts 1-3 completion check: PASS")
print(f"Part 1 failures tolerated: {len(fail_rows)}")
print("Ingestion catalog row counts:")
for k in sorted(counts):
    print(f"  {k}: {counts[k]}")
print("Split counts:")
print(f"  master={split_s['master']} train={split_s['train']} dev={split_s['dev']} test={split_s['test']}")
if int(counts.get("total_after_dedup", 0)) <= 0:
    raise SystemExit("Part 2/3 not successful: total_after_dedup is 0; refusing to start Part 4")
if int(counts.get("total_after_dedup", 0)) < 150000:
    raise SystemExit(
        f"Under target: total_after_dedup={counts.get('total_after_dedup', 0)} < 150000; "
        "refusing to start Part 4"
    )
PY

echo "=== PART 4: START WISE FETCH IN TMUX (wise_fetch) ==="
if tmux has-session -t wise_fetch 2>/dev/null; then
  tmux kill-session -t wise_fetch
fi
tmux new-session -d -s wise_fetch "cd '$(pwd)' && python3 scripts/fetch_wise_cards_parallel.py 2>&1 | tee logs/wise_fetch.log"
echo "Started tmux session 'wise_fetch'."
echo "Attach: tmux attach -t wise_fetch"
echo "Monitor: watch -n 30 \"cat logs/fetch_progress.json\""
echo "Cards count: ls cards/benchmark_v3/ | wc -l"
echo "Failures: wc -l logs/failed_sources.csv"

echo "=== PART 5 NOTE ==="
echo "Run verification snippet after Part 4 completes."
