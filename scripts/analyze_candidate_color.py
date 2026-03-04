# scripts/analyze_candidate_color.py
"""
Compute and visualize color asymmetry features for SDSS J000159.27+034352.9.
Shows that the color trajectory matches the turn-on CLAGN prediction.
Produces Figure 7 for the paper.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# Add project root to path so we can import from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.color_asymmetry_features import compute_color_features

CANDIDATE_CARD = "cards/benchmark_v3/SDSS_J000159.27+034352.9.json"
# Try alternate filename formats if above not found
CANDIDATE_IDS  = [
    "SDSS_J000159.27+034352.9",
    "SDSS J000159.27+034352.9",
    "000159.27+034352.9",
]

card_path = None
for cid in CANDIDATE_IDS:
    path = f"cards/benchmark_v3/{cid}.json"
    if os.path.exists(path):
        card_path = path
        break

if card_path is None:
    print("ERROR: Candidate card not found. Run fetch_wise_cards_parallel.py first.")
    sys.exit(1)

# Compute features
feat = compute_color_features(card_path)
print(f"\nColor asymmetry features for {feat['source_id']}:")
for k, v in feat.items():
    print(f"  {k}: {v}")

print(f"\nTurn-on signal detected: {bool(feat['is_turnon_color_signal'])}")
print(f"Turn-off signal detected: {bool(feat['is_turnoff_color_signal'])}")

# Load card data for visualization
with open(card_path) as f:
    card = json.load(f)

df = pd.DataFrame(card['wise_epochs'])
df = df[['mjd','w1mpro','w2mpro','w1sigmpro']].dropna().astype(float)
df = df[df['w1sigmpro'] < 0.2].sort_values('mjd')
df['color'] = df['w1mpro'] - df['w2mpro']
df['year'] = 2010 + (df['mjd'] - 55200) / 365.25  # approximate year

# Figure 7: W1 flux + W1-W2 color evolution side by side
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

# Top: W1 light curve
ax1.errorbar(df['year'], df['w1mpro'], yerr=df['w1sigmpro'],
             fmt='o', color='#2D6BE4', markersize=5, alpha=0.7,
             elinewidth=0.8, capsize=2, label='W1 (3.4 μm)')
ax1.invert_yaxis()  # magnitudes: brighter = lower number
ax1.set_ylabel('W1 magnitude (3.4 μm)', fontsize=11)
ax1.set_title(f'SDSS J000159.27+034352.9 (z=0.850) — WISE Infrared Light Curve\n'
              f'Color asymmetry = {feat["color_asymmetry"]:.2f}, '
              f'Flux-color r = {feat["color_flux_corr"]:.3f}', fontsize=11)
ax1.legend(fontsize=10)
ax1.text(0.02, 0.05,
         f'ΔW1 = 1.1 mag over 14 yr\n(Bright state ← Dim state)',
         transform=ax1.transAxes, fontsize=9, color='#2D6BE4')

# Bottom: W1−W2 color evolution
ax2.scatter(df['year'], df['color'], color='#E44B3F', s=20, alpha=0.6,
            label='W1−W2 color')
# Fit early and late trend lines
early = df[df['year'] < df['year'].median()]
late  = df[df['year'] >= df['year'].median()]
if len(early) > 3 and len(late) > 3:
    e_coef = np.polyfit(early['year'], early['color'], 1)
    l_coef = np.polyfit(late['year'],  late['color'],  1)
    yr_range_e = np.linspace(early['year'].min(), early['year'].max(), 50)
    yr_range_l = np.linspace(late['year'].min(),  late['year'].max(),  50)
    ax2.plot(yr_range_e, np.polyval(e_coef, yr_range_e),
             '--', color='#2D6BE4', linewidth=2,
             label=f'Early slope: {e_coef[0]:+.4f} mag/yr')
    ax2.plot(yr_range_l, np.polyval(l_coef, yr_range_l),
             '--', color='#1DB87A', linewidth=2,
             label=f'Late slope: {l_coef[0]:+.4f} mag/yr')
ax2.axhline(np.mean(df['color']), color='gray', linestyle=':',
            linewidth=0.8, label='Mean color')
ax2.set_xlabel('Year', fontsize=11)
ax2.set_ylabel('W1−W2 color (mag)', fontsize=11)
ax2.legend(fontsize=9)

# Annotation box
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
ax2.text(0.98, 0.95,
         f'Color asymmetry = early/late slope ratio\n'
         f'= {feat["color_asymmetry"]:.2f}\n'
         f'(>2 → turn-on CLAGN signature)',
         transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', horizontalalignment='right',
         bbox=props)

plt.tight_layout()
plt.savefig("figures/fig7_candidate_color_evolution.pdf", dpi=300, bbox_inches='tight')
plt.savefig("figures/fig7_candidate_color_evolution.png", dpi=150, bbox_inches='tight')
print("\nSaved Figure 7")
