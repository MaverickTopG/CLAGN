#!/usr/bin/env python3
"""
Compute all 7 feature groups for all WISE light curve cards.
Parallel execution with joblib.
Output: data/features/all_features.csv
Expected runtime: 30-90 minutes for 150k cards.

Feature Groups:
  A - W1-W2 Color Trajectory Asymmetry (from color_asymmetry_features.py)
  B - Wavelet Decomposition Variability
  C - Structural Break Detection (Bai-Perron via ruptures)
  D - Seasonal Coherence Score
  E - W1/W2 Flux Ratio Evolution (linear space)
  F - DRW Residual Analysis
  G - Redshift-Corrected Amplitude Features
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy import stats as scipy_stats
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Import Group A from existing script
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from color_asymmetry_features import compute_color_features

CARDS_DIR  = "cards/benchmark_v3"
OUTPUT_CSV = "data/features/all_features.csv"
N_JOBS     = 8

# ═══════════════════════════════════════════════════════════════════
# Group B — Wavelet Decomposition Variability
# ═══════════════════════════════════════════════════════════════════

def wavelet_features(mjd, w1, min_points=20):
    import pywt
    nan_out = {f'wavelet_d{i}': np.nan for i in range(1, 4)}
    nan_out.update({
        'wavelet_approx_power': np.nan,
        'wavelet_power_ratio_3_1': np.nan,
        'wavelet_power_ratio_3_2': np.nan,
        'wavelet_entropy': np.nan,
        'wavelet_hurst': np.nan,
    })
    if len(mjd) < min_points:
        return nan_out
    try:
        n_interp = min(512, 2 ** int(np.floor(np.log2(len(mjd) * 2))))
        t_uniform = np.linspace(mjd[0], mjd[-1], n_interp)
        f_interp = interp1d(mjd, w1, kind='linear', fill_value='extrapolate')
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
        probs = np.array([p1, p2, p3, pA]) / total
        probs = np.clip(probs, 1e-10, 1)
        wentropy = float(-np.sum(probs * np.log(probs)))

        # Hurst exponent proxy via wavelet variance scaling
        log_scales = np.log([1, 2, 4])
        log_powers = np.log([p1 + 1e-12, p2 + 1e-12, p3 + 1e-12])
        hurst_proxy = float(np.polyfit(log_scales, log_powers, 1)[0] / 2 + 0.5)

        return {
            'wavelet_d1': p1,
            'wavelet_d2': p2,
            'wavelet_d3': p3,
            'wavelet_approx_power': pA,
            'wavelet_power_ratio_3_1': p3 / (p1 + 1e-12),
            'wavelet_power_ratio_3_2': p3 / (p2 + 1e-12),
            'wavelet_entropy': wentropy,
            'wavelet_hurst': hurst_proxy,
        }
    except Exception:
        return nan_out


# ═══════════════════════════════════════════════════════════════════
# Group C — Structural Break Detection (Bai-Perron)
# ═══════════════════════════════════════════════════════════════════

def structural_break_features(mjd, w1, min_points=15):
    import ruptures as rpt

    keys = [
        'break_mjd', 'break_magnitude', 'break_significance',
        'pre_break_slope', 'post_break_slope', 'slope_change',
        'break_fractional_position', 'break_duration_yr',
        'n_detected_breaks',
    ]
    nan_out = {k: np.nan for k in keys}

    if len(mjd) < min_points:
        return nan_out
    try:
        w1_flux = 10 ** (-0.4 * np.array(w1, dtype=float))
        t_yr = (np.array(mjd, dtype=float) - mjd[0]) / 365.25
        signal = w1_flux.reshape(-1, 1)

        # Detect up to 3 breakpoints
        algo = rpt.Pelt(model="rbf", min_size=4, jump=1).fit(signal)
        bps = algo.predict(pen=np.var(w1_flux) * np.log(len(w1_flux)))
        # bps includes len(signal) as last element
        interior_bps = [b for b in bps if 0 < b < len(w1_flux)]
        n_breaks = len(interior_bps)

        if n_breaks == 0:
            nan_out['n_detected_breaks'] = 0
            return nan_out

        # Use first (most significant) breakpoint
        bp_idx = interior_bps[0]
        pre = slice(0, bp_idx)
        post = slice(bp_idx, len(w1_flux))

        pre_mean = np.mean(w1_flux[pre])
        post_mean = np.mean(w1_flux[post])

        # F-test for break significance
        ss0 = np.sum((w1_flux - np.mean(w1_flux)) ** 2)
        ss1 = (np.sum((w1_flux[pre] - pre_mean) ** 2) +
               np.sum((w1_flux[post] - post_mean) ** 2))
        df1 = 1
        df2 = max(len(w1_flux) - 3, 1)
        f_stat = ((ss0 - ss1) / df1) / (ss1 / df2 + 1e-12)

        pre_slope = (float(scipy_stats.linregress(t_yr[pre], w1_flux[pre])[0])
                     if bp_idx >= 3 else 0.0)
        post_slope = (float(scipy_stats.linregress(t_yr[post], w1_flux[post])[0])
                      if len(w1_flux) - bp_idx >= 3 else 0.0)
        break_mag = abs(-2.5 * np.log10(post_mean / (pre_mean + 1e-12)))

        return {
            'break_mjd': float(mjd[bp_idx]),
            'break_magnitude': float(break_mag),
            'break_significance': float(f_stat),
            'pre_break_slope': pre_slope,
            'post_break_slope': post_slope,
            'slope_change': float(abs(post_slope - pre_slope)),
            'break_fractional_position': float(bp_idx / len(w1_flux)),
            'break_duration_yr': float(t_yr[bp_idx] - t_yr[0]),
            'n_detected_breaks': n_breaks,
        }
    except Exception:
        return nan_out


# ═══════════════════════════════════════════════════════════════════
# Group D — Seasonal Coherence Score
# ═══════════════════════════════════════════════════════════════════

def seasonal_coherence_features(mjd, w1, min_seasons=3):
    nan_out = {
        'mean_season_coherence': np.nan,
        'coherence_trend': np.nan,
        'min_season_coherence': np.nan,
        'max_season_coherence': np.nan,
        'coherence_drop_amplitude': np.nan,
        'coherence_recovery': np.nan,
        'n_coherent_seasons': np.nan,
    }

    mjd = np.array(mjd, dtype=float)
    w1 = np.array(w1, dtype=float)
    gaps = np.diff(mjd)
    boundaries = np.where(gaps > 100)[0] + 1
    seasons = np.split(np.arange(len(mjd)), boundaries)
    seasons = [s for s in seasons if len(s) >= 4]

    if len(seasons) < min_seasons:
        return nan_out

    corrs = []
    for i in range(len(seasons) - 1):
        s1 = w1[seasons[i]]
        s2 = w1[seasons[i + 1]]
        n = min(len(s1), len(s2), 20)
        if n < 4:
            continue
        s1i = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(s1)), s1)
        s2i = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(s2)), s2)
        try:
            r, _ = scipy_stats.pearsonr(s1i, s2i)
            corrs.append(float(r))
        except Exception:
            continue

    if len(corrs) < 2:
        return nan_out

    c = np.array(corrs)
    trend = (scipy_stats.linregress(np.arange(len(c)), c)[0]
             if len(c) >= 3 else np.nan)
    min_idx = int(np.argmin(c))

    # Coherence recovery: correlation after the minimum vs minimum
    recovery = (float(np.mean(c[min_idx + 1:])) - float(c[min_idx])
                if min_idx < len(c) - 1 else np.nan)

    return {
        'mean_season_coherence': float(np.mean(c)),
        'coherence_trend': float(trend),
        'min_season_coherence': float(np.min(c)),
        'max_season_coherence': float(np.max(c)),
        'coherence_drop_amplitude': float(np.max(c) - np.min(c)),
        'coherence_recovery': recovery,
        'n_coherent_seasons': int((c > 0.5).sum()),
    }


# ═══════════════════════════════════════════════════════════════════
# Group E — W1/W2 Flux Ratio Evolution (Linear Space)
# ═══════════════════════════════════════════════════════════════════

W1_ZP = 309.540   # Jansky zero point
W2_ZP = 171.787

def flux_ratio_features(mjd, w1, w2, min_points=10):
    nan_out = {k: np.nan for k in [
        'flux_ratio_mean', 'flux_ratio_std',
        'flux_ratio_trend', 'flux_ratio_skewness',
        'flux_ratio_peak_to_trough', 'flux_ratio_percentile_90_10',
        'flux_ratio_late_minus_early',
    ]}
    if len(mjd) < min_points or len(w2) < min_points:
        return nan_out
    try:
        f1 = W1_ZP * 10 ** (-0.4 * np.array(w1, dtype=float))
        f2 = W2_ZP * 10 ** (-0.4 * np.array(w2, dtype=float))
        ratio = f1 / (f2 + 1e-12)
        ratio = ratio[np.isfinite(ratio)]
        if len(ratio) < 5:
            return nan_out
        t_yr = (np.array(mjd, dtype=float) - mjd[0]) / 365.25
        t_yr = t_yr[:len(ratio)]
        trend = scipy_stats.linregress(t_yr, ratio)[0]

        n_half = len(ratio) // 2
        late_minus_early = (float(np.mean(ratio[n_half:])) -
                            float(np.mean(ratio[:n_half])))

        return {
            'flux_ratio_mean': float(np.mean(ratio)),
            'flux_ratio_std': float(np.std(ratio)),
            'flux_ratio_trend': float(trend),
            'flux_ratio_skewness': float(scipy_stats.skew(ratio)),
            'flux_ratio_peak_to_trough': float(np.max(ratio) - np.min(ratio)),
            'flux_ratio_percentile_90_10': float(
                np.percentile(ratio, 90) - np.percentile(ratio, 10)),
            'flux_ratio_late_minus_early': late_minus_early,
        }
    except Exception:
        return nan_out


# ═══════════════════════════════════════════════════════════════════
# Group F — DRW Residual Analysis
# ═══════════════════════════════════════════════════════════════════

def drw_residual_features(mjd, w1, w1_err, min_points=15):
    nan_out = {k: np.nan for k in [
        'drw_sigma', 'drw_tau',
        'drw_residual_rms', 'drw_residual_skewness',
        'drw_residual_autocorr', 'drw_fit_quality',
        'drw_residual_trend', 'drw_chi2_per_dof',
    ]}
    if len(mjd) < min_points:
        return nan_out
    mjd = np.array(mjd, dtype=float)
    w1 = np.array(w1, dtype=float)
    err = np.array(w1_err, dtype=float)
    err = np.clip(err, 0.01, 1.0)

    # Subsample if too many points (DRW NxN matrix gets expensive)
    if len(mjd) > 200:
        idx = np.sort(np.random.RandomState(42).choice(
            len(mjd), 200, replace=False))
        mjd, w1, err = mjd[idx], w1[idx], err[idx]

    def neg_log_likelihood(params):
        try:
            sigma = np.exp(np.clip(params[0], -5, 5))
            tau = np.exp(np.clip(params[1], 2, 9))  # 7 to 8000 days
            dt = np.abs(mjd[:, None] - mjd[None, :])
            C = sigma ** 2 * np.exp(-dt / tau) + np.diag(err ** 2 + 1e-6)
            L = np.linalg.cholesky(C)
            mu = np.mean(w1)
            resid = w1 - mu
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, resid))
            logdet = 2 * np.sum(np.log(np.diag(L)))
            return 0.5 * (np.dot(resid, alpha) + logdet +
                          len(w1) * np.log(2 * np.pi))
        except (np.linalg.LinAlgError, OverflowError):
            return 1e10

    try:
        best_nll, best_x = 1e10, [np.log(0.1), np.log(200)]
        for s0, t0 in [(0.05, 100), (0.1, 300), (0.2, 600), (0.3, 1000)]:
            res = minimize(neg_log_likelihood, [np.log(s0), np.log(t0)],
                           method='Nelder-Mead',
                           options={'maxiter': 800, 'xatol': 0.01,
                                    'fatol': 0.01})
            if res.fun < best_nll:
                best_nll, best_x = res.fun, res.x

        sigma = float(np.exp(np.clip(best_x[0], -5, 5)))
        tau = float(np.exp(np.clip(best_x[1], 2, 9)))

        # Compute posterior mean (DRW prediction at observed times)
        dt = np.abs(mjd[:, None] - mjd[None, :])
        C = sigma ** 2 * np.exp(-dt / tau) + np.diag(err ** 2 + 1e-6)
        try:
            C_inv = np.linalg.inv(C)
            drw_pred = np.mean(w1) + C @ C_inv @ (w1 - np.mean(w1))
        except np.linalg.LinAlgError:
            drw_pred = np.full_like(w1, np.mean(w1))

        residuals = w1 - drw_pred
        t_yr = (mjd - mjd[0]) / 365.25
        r_trend = scipy_stats.linregress(t_yr, residuals)[0]

        ac = (float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1])
              if len(residuals) > 2 else np.nan)
        chi2_dof = float(np.sum((residuals / err) ** 2) /
                         max(len(w1) - 2, 1))

        return {
            'drw_sigma': sigma,
            'drw_tau': tau,
            'drw_residual_rms': float(np.sqrt(np.mean(residuals ** 2))),
            'drw_residual_skewness': float(scipy_stats.skew(residuals)),
            'drw_residual_autocorr': ac,
            'drw_fit_quality': float(best_nll / max(len(w1), 1)),
            'drw_residual_trend': float(r_trend),
            'drw_chi2_per_dof': chi2_dof,
        }
    except Exception:
        return nan_out


# ═══════════════════════════════════════════════════════════════════
# Group G — Redshift-Corrected Amplitude Features
# ═══════════════════════════════════════════════════════════════════

def redshift_features(z, break_magnitude, color_slope_early,
                      flux_ratio_trend, drw_sigma):
    nan_out = {k: np.nan for k in [
        'z_corrected_amplitude', 'z_corrected_color_slope',
        'luminosity_change_proxy', 'z_corrected_drw_sigma',
        'intrinsic_variability_proxy',
    ]}
    if np.isnan(z) or z <= 0 or z > 5:
        return nan_out
    zf = 1 + z
    try:
        return {
            'z_corrected_amplitude': (
                float(break_magnitude * np.sqrt(zf))
                if not np.isnan(break_magnitude) else np.nan),
            'z_corrected_color_slope': (
                float(color_slope_early * zf)
                if not np.isnan(color_slope_early) else np.nan),
            'luminosity_change_proxy': (
                float(break_magnitude * np.log10(zf + 1))
                if not np.isnan(break_magnitude) else np.nan),
            'z_corrected_drw_sigma': (
                float(drw_sigma * np.sqrt(zf))
                if not np.isnan(drw_sigma) else np.nan),
            'intrinsic_variability_proxy': (
                float(drw_sigma / np.sqrt(zf))
                if not np.isnan(drw_sigma) else np.nan),
        }
    except Exception:
        return nan_out


# ═══════════════════════════════════════════════════════════════════
# Master card processor
# ═══════════════════════════════════════════════════════════════════

def _get_seasons(mjd):
    """Compute n_seasons and baseline_days from epoch MJDs."""
    if len(mjd) < 2:
        return 1, 0.0
    gaps = np.diff(np.sort(mjd))
    n_seasons = int((gaps > 100).sum()) + 1
    baseline = float(mjd[-1] - mjd[0])
    return n_seasons, baseline


def process_card(card_path):
    try:
        with open(card_path) as f:
            card = json.load(f)

        epochs = card.get('wise_epochs', [])
        if len(epochs) < 10:
            return None

        df = pd.DataFrame(epochs)
        required = ['mjd', 'w1mpro', 'w2mpro']
        if not all(c in df.columns for c in required):
            return None
        df = df.dropna(subset=required).astype(float).sort_values('mjd')
        if len(df) < 10:
            return None
        if 'w1sigmpro' not in df.columns:
            df['w1sigmpro'] = 0.05
        else:
            df['w1sigmpro'] = df['w1sigmpro'].fillna(0.05)

        mjd = df['mjd'].values
        w1 = df['w1mpro'].values
        w2 = df['w2mpro'].values
        w1e = df['w1sigmpro'].values

        # Compute n_seasons/baseline from epoch data as fallback
        n_seasons_card = card.get('n_seasons')
        baseline_card = card.get('baseline_days')
        if n_seasons_card is None or baseline_card is None:
            n_seasons_card, baseline_card = _get_seasons(mjd)

        row = {
            'source_id': card['source_id'],
            'label': card.get('label', ''),
            'z': float(card.get('z', np.nan) or np.nan),
            'n_epochs': len(df),
            'n_seasons': n_seasons_card,
            'baseline_days': baseline_card,
        }

        # Group A — Color Asymmetry (from existing script)
        try:
            ca = compute_color_features(str(card_path))
            if ca:
                row.update({k: v for k, v in ca.items()
                            if k not in ('source_id', 'label', 'n_epochs')})
        except Exception:
            pass

        # Group B — Wavelet Decomposition
        row.update(wavelet_features(mjd, w1))

        # Group C — Structural Break Detection
        row.update(structural_break_features(mjd, w1))

        # Group D — Seasonal Coherence
        row.update(seasonal_coherence_features(mjd, w1))

        # Group E — Flux Ratio Evolution
        row.update(flux_ratio_features(mjd, w1, w2))

        # Group F — DRW Residual Analysis
        row.update(drw_residual_features(mjd, w1, w1e))

        # Group G — Redshift-Corrected Amplitude
        row.update(redshift_features(
            row.get('z', np.nan),
            row.get('break_magnitude', np.nan),
            row.get('color_slope_early', np.nan),
            row.get('flux_ratio_trend', np.nan),
            row.get('drw_sigma', np.nan),
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

    if len(cards) == 0:
        print("No new cards to process.")
        return

    results = Parallel(n_jobs=N_JOBS, verbose=0, prefer='threads')(
        delayed(process_card)(c) for c in tqdm(cards, desc="Features")
    )
    new_rows = [r for r in results if r is not None]
    df_new = pd.DataFrame(new_rows)

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
