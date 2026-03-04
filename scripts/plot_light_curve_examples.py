#!/usr/bin/env python3
"""
4-panel example light curves for the paper:
  Panel 1: Confirmed CLAGN (correctly classified)
  Panel 2: Blazar (correctly rejected)
  Panel 3: SN IIn or contaminant (correctly rejected)
  Panel 4: Normal AGN (correctly classified as negative)

Reads cards from cards/benchmark_v3/ and predictions from
results/FINAL_TEST_RESULTS_predictions.csv (or dev predictions).

Usage:
  python scripts/plot_light_curve_examples.py
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.makedirs("figures", exist_ok=True)

CARDS_DIR = "cards/benchmark_v3"

# Load predictions
for pred_csv in ["results/FINAL_TEST_RESULTS_predictions.csv",
                 "results/dev_predictions_predictions.csv"]:
    if os.path.exists(pred_csv):
        preds = pd.read_csv(pred_csv)
        break
else:
    preds = None
    print("WARNING: No prediction CSV found. Using label-only selection.")


def load_card(source_id):
    """Load WISE card JSON by source_id."""
    safe_id = (str(source_id).replace('/', '_')
               .replace(' ', '_').replace('+', 'p'))
    card_path = os.path.join(CARDS_DIR, f"{safe_id}.json")
    if not os.path.exists(card_path):
        return None
    with open(card_path) as f:
        return json.load(f)


def find_example(label_val, y_pred_val=None, preds_df=None):
    """Find a source matching label and optionally prediction."""
    if preds_df is not None and y_pred_val is not None:
        candidates = preds_df[
            (preds_df['label'] == label_val) &
            (preds_df['y_pred'] == y_pred_val)
        ]['source_id'].tolist()
    else:
        # Fall back to cards directory
        candidates = []
        for card_path in Path(CARDS_DIR).glob("*.json"):
            with open(card_path) as f:
                card = json.load(f)
            if card.get('label', '') == label_val:
                candidates.append(card['source_id'])

    # Try to find one with enough data
    for sid in candidates[:20]:
        card = load_card(sid)
        if card and len(card.get('wise_epochs', [])) >= 15:
            return card
    return None


def plot_card(ax, card, title, color='#2D6BE4'):
    """Plot W1 and W2 light curves for a card on given axes."""
    epochs = card.get('wise_epochs', [])
    df = pd.DataFrame(epochs)
    if len(df) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha='center', va='center')
        ax.set_title(title, fontsize=10)
        return

    df = df.dropna(subset=['mjd', 'w1mpro', 'w2mpro']).astype(float)
    df = df.sort_values('mjd')
    mjd = df['mjd'].values
    w1 = df['w1mpro'].values
    w2 = df['w2mpro'].values

    # Convert MJD to year
    year = 2000 + (mjd - 51544.5) / 365.25

    w1e = df.get('w1sigmpro', pd.Series(np.full(len(df), 0.05))).values
    w2e = df.get('w2sigmpro', pd.Series(np.full(len(df), 0.05))).values

    ax.errorbar(year, w1, yerr=w1e, fmt='o', ms=2, color='#2D6BE4',
                elinewidth=0.5, alpha=0.7, label='W1')
    ax.errorbar(year, w2, yerr=w2e, fmt='s', ms=2, color='#E44B3F',
                elinewidth=0.5, alpha=0.7, label='W2')
    ax.invert_yaxis()
    ax.set_xlabel('Year', fontsize=9)
    ax.set_ylabel('Magnitude', fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc='upper right')
    ax.tick_params(labelsize=8)


# ── Select example sources ────────────────────────────────────────
examples = [
    ('CLAGN', 1, 'Confirmed CLAGN (TP)', '#2D6BE4'),
    ('blazar', 0, 'Blazar (TN)', '#E44B3F'),
    ('SN_IIn', 0, 'SN IIn (TN)', '#4CAF50'),
    ('normal_agn', 0, 'Normal AGN (TN)', '#9C27B0'),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.ravel()

for i, (label, pred, title, color) in enumerate(examples):
    card = find_example(label, pred, preds)
    if card is None and label == 'SN_IIn':
        # SN IIn may not exist; try SN
        for alt_label in ['SN', 'sn_ia', 'sn_iin', 'contaminant']:
            card = find_example(alt_label, pred, preds)
            if card:
                break
    if card is None:
        # Last resort: just pick any card
        card = find_example(label, None, None)

    if card:
        plot_card(axes[i], card,
                  f"{title}\n{card['source_id']}", color)
    else:
        axes[i].text(0.5, 0.5, f"No {label} card found",
                     transform=axes[i].transAxes,
                     ha='center', va='center')
        axes[i].set_title(title, fontsize=10)

plt.suptitle('Example WISE Light Curves by Source Type', fontsize=13)
plt.tight_layout()
plt.savefig("figures/fig_light_curve_examples.pdf",
            dpi=300, bbox_inches='tight')
plt.savefig("figures/fig_light_curve_examples.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/fig_light_curve_examples.pdf")
