# scripts/confirm_candidate.py
"""
Pull all available SDSS spectra for SDSS J000159.27+034352.9
and measure Mg II broad line flux in each epoch.
Requires: sdss_access or direct SDSS SAS download
Produces: Figure 6 (two-epoch spectra comparison) and Table 3 (line measurements)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import requests
import os
from scipy.ndimage import uniform_filter1d

os.makedirs("data/spectra", exist_ok=True)
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

TARGET_RA   = 0.496966    # SDSS J000159.27+034352.9
TARGET_DEC  = 3.731361
TARGET_Z    = 0.850
TARGET_NAME = "SDSS J000159.27+034352.9"

# ── Step 1: Query SDSS for all spectra of this object ─────────────
def query_sdss_spectra(ra, dec, radius_arcsec=3.0):
    """
    Query SDSS CasJobs-style via the SDSS SkyServer API.
    Returns list of (plate, mjd, fiberid) tuples.
    """
    url = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/RadialSearch"
    params = {
        'ra': ra, 'dec': dec,
        'radius': radius_arcsec / 60.0,  # arcmin
        'limit': 20,
        'format': 'json',
        'whichquery': 'spectro'
    }
    resp = requests.get(url, params=params, timeout=30)
    try:
        data = resp.json()
    except Exception:
        print(f"  WARNING: SDSS API returned non-JSON response (status {resp.status_code})")
        return []
    spectra = []
    rows = data if isinstance(data, list) else data.get('Rows', [])
    for row in rows:
        spectra.append({
            'plate':   int(row.get('plate', 0)),
            'mjd':     int(row.get('mjd', 0)),
            'fiberid': int(row.get('fiberid', 0)),
            'z':       float(row.get('z', 0)),
            'zwarning': int(row.get('zWarning', 0)),
        })
    return spectra

print(f"Querying SDSS for spectra of {TARGET_NAME}...")
spectra_list = query_sdss_spectra(TARGET_RA, TARGET_DEC)
print(f"Found {len(spectra_list)} SDSS spectra:")
for s in spectra_list:
    print(f"  plate={s['plate']} mjd={s['mjd']} fiber={s['fiberid']} z={s['z']:.4f}")

if len(spectra_list) < 2:
    print("WARNING: Fewer than 2 spectra found in SDSS DR17.")
    print("Try SDSS DR18 or BOSS spectrograph data.")
    print("Manual fallback: download from https://dr17.sdss.org/optical/spectrum/search")
    print(f"  Search RA={TARGET_RA}, Dec={TARGET_DEC}, radius=3 arcsec")
    print("VERDICT: INSUFFICIENT SPECTRA")
else:
    print(f"SUCCESS: {len(spectra_list)} spectra found — sufficient for epoch comparison")

# ── Step 2: Download spectral FITS files ──────────────────────────
def download_sdss_spectrum(plate, mjd, fiberid, output_dir="data/spectra"):
    """Download spectrum FITS from SDSS SAS."""
    fname = f"spec-{plate:04d}-{mjd}-{fiberid:04d}.fits"
    fpath = os.path.join(output_dir, fname)
    if os.path.exists(fpath):
        print(f"  Already downloaded: {fname}")
        return fpath
    url = (f"https://data.sdss.org/sas/dr17/sdss/spectro/redux/26/spectra/lite/"
           f"{plate:04d}/{fname}")
    resp = requests.get(url, timeout=60, stream=True)
    if resp.status_code == 200:
        with open(fpath, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  Downloaded: {fname}")
        return fpath
    else:
        # Try BOSS redux path
        url2 = (f"https://data.sdss.org/sas/dr17/eboss/spectro/redux/v5_13_2/"
                f"spectra/lite/{plate:04d}/{fname}")
        resp2 = requests.get(url2, timeout=60, stream=True)
        if resp2.status_code == 200:
            with open(fpath, 'wb') as f:
                for chunk in resp2.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"  Downloaded (BOSS): {fname}")
            return fpath
    print(f"  FAILED to download {fname} (HTTP {resp.status_code})")
    return None

spec_paths = []
for s in spectra_list:
    path = download_sdss_spectrum(s['plate'], s['mjd'], s['fiberid'])
    if path:
        spec_paths.append((s['mjd'], path))

spec_paths.sort(key=lambda x: x[0])  # sort by MJD (chronological)

# ── Step 3: Load spectra and measure Mg II broad line ─────────────
def load_sdss_spectrum(fits_path):
    """
    Load SDSS spectrum from FITS file.
    Returns (wavelength_angstrom, flux_1e-17_erg_s_cm2_A, ivar).
    """
    with fits.open(fits_path) as hdul:
        # SDSS spectra: extension 1 = COADD (best), or extension 0
        if len(hdul) > 1:
            data = hdul[1].data
        else:
            data = hdul[0].data

        log_wave = data['loglam']  # log10(wavelength in Angstrom)
        flux     = data['flux']    # 10^-17 erg/s/cm^2/A
        ivar     = data['ivar']    # inverse variance
        wave     = 10 ** log_wave

    return wave, flux, ivar


def measure_mgii_flux(wave, flux, ivar, z, line_center_rest=2798.0,
                       continuum_windows=[(2650,2720),(2850,2920)],
                       line_window=(2720,2860)):
    """
    Measure Mg II broad line flux using local continuum subtraction.
    Returns (line_flux, line_flux_err, continuum_level, ew_rest).
    """
    wave_rest = wave / (1 + z)

    # Fit local linear continuum
    cont_mask = np.zeros(len(wave_rest), dtype=bool)
    for w1, w2 in continuum_windows:
        cont_mask |= (wave_rest >= w1) & (wave_rest <= w2)

    if cont_mask.sum() < 5:
        return np.nan, np.nan, np.nan, np.nan

    cont_wave = wave_rest[cont_mask]
    cont_flux = flux[cont_mask]
    cont_coef = np.polyfit(cont_wave, cont_flux, 1)
    continuum  = np.polyval(cont_coef, wave_rest)

    # Measure line
    line_mask = (wave_rest >= line_window[0]) & (wave_rest <= line_window[1])
    if line_mask.sum() < 5:
        return np.nan, np.nan, np.nan, np.nan

    flux_above_cont = flux[line_mask] - continuum[line_mask]
    d_wave = np.diff(wave[line_mask])
    d_wave = np.append(d_wave, d_wave[-1])

    line_flux = np.sum(flux_above_cont * d_wave)  # in 1e-17 erg/s/cm^2

    # Error from inverse variance
    var_arr = np.where(ivar[line_mask] > 0, 1.0/ivar[line_mask], 0)
    line_flux_err = np.sqrt(np.sum(var_arr * d_wave**2))

    # Continuum level at line center
    cont_at_center = np.polyval(cont_coef, line_center_rest)

    # Rest-frame equivalent width
    ew_rest = line_flux / cont_at_center if cont_at_center > 0 else np.nan

    return float(line_flux), float(line_flux_err), float(cont_at_center), float(ew_rest)


# Process each spectrum
epoch_results = []
spectra_data  = []

for mjd, fpath in spec_paths:
    wave, flux, ivar = load_sdss_spectrum(fpath)
    lf, lf_err, cont, ew = measure_mgii_flux(wave, flux, ivar, TARGET_Z)
    print(f"\nMJD {mjd}:")
    print(f"  Mg II flux:  {lf:.3f} ± {lf_err:.3f}  (×10⁻¹⁷ erg/s/cm²)")
    print(f"  Continuum:   {cont:.3f}")
    print(f"  Rest EW:     {ew:.1f} Å")
    epoch_results.append({'mjd': mjd, 'mgii_flux': lf, 'mgii_flux_err': lf_err,
                          'continuum_2800': cont, 'rest_ew_mgii': ew})
    spectra_data.append((mjd, wave, flux, ivar))

# ── Step 4: Variability assessment ────────────────────────────────
# Always save table3 with whatever was collected (0 or more rows)
df_epochs = pd.DataFrame(epoch_results)
df_epochs.to_csv("results/table3_spectral_epochs.csv", index=False)

if len(epoch_results) >= 2:
    # Is the Mg II flux change significant?
    flux_vals = df_epochs['mgii_flux'].dropna().values
    flux_errs = df_epochs['mgii_flux_err'].dropna().values

    if len(flux_vals) >= 2:
        delta_flux = flux_vals[-1] - flux_vals[0]
        delta_err  = np.sqrt(flux_errs[0]**2 + flux_errs[-1]**2)
        significance = abs(delta_flux) / delta_err

        print(f"\n{'='*50}")
        print(f"Mg II VARIABILITY ASSESSMENT")
        print(f"  Epoch 1 (MJD {df_epochs.iloc[0]['mjd']}): {flux_vals[0]:.3f} ± {flux_errs[0]:.3f}")
        print(f"  Epoch 2 (MJD {df_epochs.iloc[-1]['mjd']}): {flux_vals[-1]:.3f} ± {flux_errs[-1]:.3f}")
        print(f"  Delta flux: {delta_flux:+.3f} ± {delta_err:.3f}")
        print(f"  Significance: {significance:.1f}σ")

        if significance >= 3.0 and delta_flux > 0:
            verdict = "CONFIRMED TURN-ON CLAGN (≥3σ Mg II brightening)"
        elif significance >= 3.0 and delta_flux < 0:
            verdict = "CONFIRMED TURN-OFF CLAGN (≥3σ Mg II dimming)"
        elif significance >= 2.0:
            verdict = "MARGINAL CLAGN CANDIDATE (2-3σ Mg II change)"
        else:
            verdict = "NOT CONFIRMED (<2σ — inconclusive)"

        print(f"  VERDICT: {verdict}")
        print(f"{'='*50}")

# ── Step 5: Figure 6 — Two-epoch spectral comparison ──────────────
if len(spectra_data) >= 2:
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False)

    colors_epoch = ['#2D6BE4','#E44B3F','#1DB87A','#F5A623']

    # Top panel: full spectra comparison
    ax = axes[0]
    for i, (mjd, wave, flux, ivar) in enumerate(spectra_data[:4]):
        wave_rest = wave / (1 + TARGET_Z)
        # Smooth for display
        flux_smooth = uniform_filter1d(flux, size=5)
        # Only plot rest-frame 2400-5000 Å (covers Mg II through Hβ)
        mask = (wave_rest >= 2400) & (wave_rest <= 5000)
        ax.plot(wave_rest[mask], flux_smooth[mask],
                color=colors_epoch[i % len(colors_epoch)], alpha=0.8,
                linewidth=0.8, label=f"MJD {mjd} (Epoch {i+1})")
    ax.axvspan(2720, 2860, alpha=0.1, color='orange', label='Mg II window')
    ax.axvline(2798, color='orange', linestyle='--', linewidth=0.8,
               label='Mg II λ2798')
    ax.set_ylabel('Flux (×10⁻¹⁷ erg s⁻¹ cm⁻² Å⁻¹)', fontsize=10)
    ax.set_title(f'{TARGET_NAME} — Two-Epoch Spectral Comparison (z={TARGET_Z})',
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(2400, 5000)

    # Bottom panel: zoom on Mg II region
    ax2 = axes[1]
    for i, (mjd, wave, flux, ivar) in enumerate(spectra_data[:4]):
        wave_rest = wave / (1 + TARGET_Z)
        mask = (wave_rest >= 2600) & (wave_rest <= 3000)
        if mask.sum() < 5:
            continue
        flux_smooth = uniform_filter1d(flux, size=3)
        ax2.plot(wave_rest[mask], flux_smooth[mask],
                 color=colors_epoch[i % len(colors_epoch)], alpha=0.9,
                 linewidth=1.2, label=f"MJD {mjd}")
    ax2.axvspan(2720, 2860, alpha=0.15, color='orange')
    ax2.axvline(2798, color='orange', linestyle='--', linewidth=1.2,
                label='Mg II λ2798 (broad)')
    ax2.axvline(2803, color='orange', linestyle=':', linewidth=0.8,
                label='Mg II λ2803 (broad)')
    ax2.set_xlabel('Rest-frame wavelength (Å)', fontsize=10)
    ax2.set_ylabel('Flux (×10⁻¹⁷ erg s⁻¹ cm⁻² Å⁻¹)', fontsize=10)
    ax2.set_title('Zoom: Mg II Broad Line Region', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_xlim(2600, 3000)

    plt.tight_layout()
    plt.savefig("figures/fig6_spectra_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.savefig("figures/fig6_spectra_comparison.png", dpi=150, bbox_inches='tight')
    print("\nSaved Figure 6")
    plt.close()
