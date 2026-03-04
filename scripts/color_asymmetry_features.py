# scripts/color_asymmetry_features.py
"""
Compute W1-W2 color trajectory features for WISE light curve cards.
These are the novel features introduced in this paper.
Call this on all cards AFTER fetch_wise_cards_parallel.py completes.
DO NOT modify existing feature computation in make_cards.py.
Add these as new columns: color_slope_early, color_slope_late,
color_asymmetry, color_range, color_flux_corr, dust_lag_proxy.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats, signal
from pathlib import Path
import os

def season_split(mjds, values, n_seasons_per_half=3):
    """
    Split light curve into first N seasons and last N seasons.
    Season boundary = gap > 100 days.
    Returns (early_mjd, early_val, late_mjd, late_val).
    """
    mjds = np.array(mjds, dtype=float)
    values = np.array(values, dtype=float)
    sort_idx = np.argsort(mjds)
    mjds, values = mjds[sort_idx], values[sort_idx]

    # Find season boundaries
    gaps = np.diff(mjds)
    boundaries = np.where(gaps > 100)[0] + 1
    seasons = np.split(np.arange(len(mjds)), boundaries)

    if len(seasons) < 2:
        return None, None, None, None

    n_half = min(n_seasons_per_half, len(seasons) // 2)
    if n_half < 1:
        return None, None, None, None

    early_idx = np.concatenate([s for s in seasons[:n_half]])
    late_idx  = np.concatenate([s for s in seasons[-n_half:]])

    return (mjds[early_idx], values[early_idx],
            mjds[late_idx],  values[late_idx])


def compute_color_features(card_path):
    """
    Load a WISE card JSON and compute color asymmetry features.
    Returns dict of features or None if insufficient data.
    """
    with open(card_path) as f:
        card = json.load(f)

    epochs = card.get('wise_epochs', [])
    if len(epochs) < 20:
        return None

    # Parse photometry
    try:
        df = pd.DataFrame(epochs)
        df = df[['mjd','w1mpro','w2mpro','w1sigmpro']].dropna()
        df = df.astype(float)
        df = df[df['w1sigmpro'] < 0.2]  # quality cut
        df = df.sort_values('mjd').reset_index(drop=True)
    except Exception:
        return None

    if len(df) < 15:
        return None

    mjd   = df['mjd'].values
    w1    = df['w1mpro'].values
    w2    = df['w2mpro'].values
    color = w1 - w2  # W1−W2 color index

    # ── Feature 1 & 2: Early and late color slopes ─────────────────
    early_mjd, early_color, late_mjd, late_color = season_split(mjd, color)

    if early_mjd is None or len(early_mjd) < 4 or len(late_mjd) < 4:
        color_slope_early = np.nan
        color_slope_late  = np.nan
        color_asymmetry   = np.nan
    else:
        # Normalize MJD to years
        t0 = mjd[0]
        early_t = (early_mjd - t0) / 365.25
        late_t  = (late_mjd  - t0) / 365.25

        slope_e, _, _, _, _ = stats.linregress(early_t, early_color)
        slope_l, _, _, _, _ = stats.linregress(late_t,  late_color)

        color_slope_early = float(slope_e)
        color_slope_late  = float(slope_l)

        # Asymmetry: ratio of early to late slope
        # Large positive = turn-on signal (early reddening then plateau)
        # Large negative = turn-off signal (early bluing then plateau)
        # Near ±1 = symmetric = normal AGN or blazar
        eps = 1e-6
        if abs(slope_l) < eps:
            color_asymmetry = float(np.sign(slope_e) * 10.0)  # cap at ±10
        else:
            color_asymmetry = float(np.clip(slope_e / slope_l, -10, 10))

    # ── Feature 3: Color range ─────────────────────────────────────
    color_range = float(np.max(color) - np.min(color))

    # ── Feature 4: Color-flux correlation ──────────────────────────
    # Convert magnitude to flux proxy (brighter = lower mag = higher flux)
    w1_flux_proxy = 10 ** (-0.4 * w1)  # proportional to flux
    if len(w1_flux_proxy) >= 5:
        r, p = stats.pearsonr(w1_flux_proxy, color)
        color_flux_corr = float(r)
        color_flux_corr_pval = float(p)
    else:
        color_flux_corr = np.nan
        color_flux_corr_pval = np.nan

    # ── Feature 5: Dust lag proxy ───────────────────────────────────
    # Cross-correlate W1 flux with W1-W2 color to find lag
    # Positive lag = color change follows flux change (physically expected)
    try:
        if len(w1) >= 20:
            # Interpolate to uniform grid for cross-correlation
            t_uniform = np.linspace(mjd[0], mjd[-1], 200)
            w1_interp    = np.interp(t_uniform, mjd, w1_flux_proxy)
            color_interp = np.interp(t_uniform, mjd, color)

            # Normalize
            w1_norm    = (w1_interp - w1_interp.mean()) / (w1_interp.std() + 1e-8)
            color_norm = (color_interp - color_interp.mean()) / (color_interp.std() + 1e-8)

            xcorr = np.correlate(color_norm, w1_norm, mode='full')
            lags  = np.arange(-(len(w1_norm)-1), len(w1_norm))
            dt_days = (mjd[-1] - mjd[0]) / 200  # days per interpolation step
            lag_at_peak = float(lags[np.argmax(xcorr)] * dt_days)
            # Convert to years
            dust_lag_proxy = lag_at_peak / 365.25
        else:
            dust_lag_proxy = np.nan
    except Exception:
        dust_lag_proxy = np.nan

    # ── Physical plausibility flags ─────────────────────────────────
    # CLAGN turn-on signature: asymmetry > 2, flux-color corr > 0.3,
    #                           dust lag 0.5-3yr
    # CLAGN turn-off signature: asymmetry < -2, flux-color corr < -0.3,
    #                            dust lag 0.5-3yr
    is_turnon_color_signal = (
        color_asymmetry > 2.0 and
        color_flux_corr > 0.25 and
        0.3 < dust_lag_proxy < 3.5
    ) if not np.isnan(color_asymmetry) else False

    is_turnoff_color_signal = (
        color_asymmetry < -2.0 and
        color_flux_corr < -0.25 and
        0.3 < dust_lag_proxy < 3.5
    ) if not np.isnan(color_asymmetry) else False

    return {
        'source_id':             card['source_id'],
        'label':                 card.get('label', ''),
        'color_slope_early':     color_slope_early,
        'color_slope_late':      color_slope_late,
        'color_asymmetry':       color_asymmetry,
        'color_range':           color_range,
        'color_flux_corr':       color_flux_corr,
        'color_flux_corr_pval':  color_flux_corr_pval,
        'dust_lag_proxy':        dust_lag_proxy,
        'is_turnon_color_signal':  int(is_turnon_color_signal),
        'is_turnoff_color_signal': int(is_turnoff_color_signal),
        'n_epochs':              len(df),
    }


def run_all(cards_dir="cards/benchmark_v3",
            output_csv="data/features/color_asymmetry_features.csv",
            existing_features_csv=None):
    """
    Compute color asymmetry features for all cards.
    Skips cards already in existing_features_csv if provided.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    cards = list(Path(cards_dir).glob("*.json"))
    print(f"Computing color asymmetry features for {len(cards):,} cards...")

    # Load existing to skip
    done_ids = set()
    if existing_features_csv and os.path.exists(existing_features_csv):
        done_ids = set(pd.read_csv(existing_features_csv)['source_id'].astype(str))
        print(f"  Skipping {len(done_ids):,} already computed")

    rows = []
    failed = 0
    for i, card_path in enumerate(cards):
        source_id = card_path.stem
        if source_id in done_ids:
            continue
        try:
            feat = compute_color_features(card_path)
            if feat:
                rows.append(feat)
        except Exception as e:
            failed += 1
            if failed < 10:
                print(f"  Failed {card_path.name}: {e}")

        if (i + 1) % 5000 == 0:
            print(f"  [{i+1:,}/{len(cards):,}] computed={len(rows):,} failed={failed}")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df):,} feature rows to {output_csv}")
    print(f"Failed: {failed}")
    print(f"\nClass breakdown of color signal flags:")
    for label in df['label'].unique():
        sub = df[df['label'] == label]
        ton  = sub['is_turnon_color_signal'].mean()
        toff = sub['is_turnoff_color_signal'].mean()
        print(f"  {label:25s}: turn-on signal {ton:.1%}, turn-off signal {toff:.1%}")

    return df


if __name__ == "__main__":
    run_all()
