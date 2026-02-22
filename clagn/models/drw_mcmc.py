"""
drw_mcmc.py — Full MCMC posterior sampling and advanced DRW variants.

Implements three increasingly flexible models for AGN variability:
    1. Stationary DRW (MCMC posterior, 64 walkers × 3000 steps)
    2. Broken DRW (structural break scan across 15–85% of time series)
    3. Non-stationary GP (DRW + step component, MAP via L-BFGS-B)

The broken DRW and non-stationary GP models explicitly capture the
state transitions that define changing-look AGN.

References:
    Kelly et al. 2009, ApJ 698, 895 (DRW for AGN)
    Gelman & Rubin 1992, Stat. Sci. 7, 457 (convergence diagnostic)
    Kozlowski et al. 2010, ApJ 708, 927 (optical DRW)
"""
import logging
import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve

from ..config import (
    DRW_MCMC_N_WALKERS,
    DRW_MCMC_N_STEPS,
    DRW_MCMC_N_BURN,
    DRW_GR_THRESHOLD,
    DRW_LOG_TAU_MIN,
    DRW_LOG_TAU_MAX,
    DRW_LOG_SIGMA_MIN,
    DRW_LOG_SIGMA_MAX,
)
from .drw import fit_drw_map, _drw_nll, _maybe_subsample

logger = logging.getLogger(__name__)

# Physical prior on log_tau: Gaussian centred on ln(~300 days), σ=1.0
# Based on Kelly+2009 and Kozlowski+2010 empirical DRW timescale distributions
_LOG_TAU_PRIOR_MU = 5.7       # ln(days); e^5.7 ≈ 299 days
_LOG_TAU_PRIOR_SIGMA = 1.0    # dex uncertainty in characteristic timescale


# ---------------------------------------------------------------------------
# Log-probability with physical prior
# ---------------------------------------------------------------------------

def _log_prob_drw(params: np.ndarray,
                  times: np.ndarray,
                  fluxes: np.ndarray,
                  flux_errors: np.ndarray) -> float:
    """
    Log-posterior for the DRW model.

    Prior:
        log_sigma ~ Uniform(DRW_LOG_SIGMA_MIN, DRW_LOG_SIGMA_MAX)
        log_tau   ~ Gaussian(_LOG_TAU_PRIOR_MU, _LOG_TAU_PRIOR_SIGMA²)
                    truncated to (DRW_LOG_TAU_MIN, DRW_LOG_TAU_MAX)

    Likelihood: Gaussian process with DRW covariance kernel.
    """
    log_sigma, log_tau = params

    # Hard bounds
    if not (DRW_LOG_SIGMA_MIN < log_sigma < DRW_LOG_SIGMA_MAX):
        return -np.inf
    if not (DRW_LOG_TAU_MIN < log_tau < DRW_LOG_TAU_MAX):
        return -np.inf

    # Physical prior on log_tau
    log_prior_tau = -0.5 * ((log_tau - _LOG_TAU_PRIOR_MU) / _LOG_TAU_PRIOR_SIGMA) ** 2

    # Likelihood
    nll = _drw_nll(log_sigma, log_tau, times, fluxes, flux_errors)
    if nll >= 1e9:
        return -np.inf

    return -nll + log_prior_tau


# ---------------------------------------------------------------------------
# Gelman-Rubin convergence diagnostic
# ---------------------------------------------------------------------------

def _gelman_rubin(chains: np.ndarray) -> np.ndarray:
    """
    Compute the Gelman-Rubin R-hat statistic.

    Parameters
    ----------
    chains : array of shape (n_steps, n_walkers, n_params)

    Returns
    -------
    r_hat : array of length n_params
    """
    n_steps, n_walkers, n_params = chains.shape
    r_hats = np.zeros(n_params)

    for p in range(n_params):
        chain_p = chains[:, :, p]  # (n_steps, n_walkers)
        # Within-chain variance
        within_var = np.mean(np.var(chain_p, axis=0, ddof=1))
        # Between-chain variance (variance of chain means)
        chain_means = np.mean(chain_p, axis=0)
        between_var = n_steps * np.var(chain_means, ddof=1)
        # Pooled variance estimate
        var_plus = ((n_steps - 1) * within_var + between_var) / n_steps
        r_hats[p] = np.sqrt(var_plus / within_var) if within_var > 0 else np.nan

    return r_hats


# ---------------------------------------------------------------------------
# Full MCMC DRW fit
# ---------------------------------------------------------------------------

def fit_drw_full_mcmc(times: np.ndarray, fluxes: np.ndarray, errors: np.ndarray,
                      z: float, source_luminosity: float = None,
                      results_dir: str = None, source_id: str = None) -> dict:
    """
    Fit DRW model with full MCMC posterior sampling.

    Uses emcee EnsembleSampler with DRW_MCMC_N_WALKERS walkers,
    DRW_MCMC_N_STEPS steps, and DRW_MCMC_N_BURN burn-in steps.
    Falls back to MAP (fit_drw_map) if emcee is not installed.

    Physical prior:
        log_tau ~ Gaussian(5.7, 1.0) (Kelly+2009 empirical distribution)
        log_sigma ~ Uniform (non-informative)

    Parameters
    ----------
    times    : array, MJD (observer frame)
    fluxes   : array, flux densities in mJy
    errors   : array, flux uncertainties in mJy
    z        : float, source redshift
    source_luminosity : float or None, L_bol in erg/s (for sigma_excess)
    results_dir : str or None, directory to save chains
    source_id   : str or None, used for chain filename

    Returns
    -------
    result : dict with keys:
        tau_rest_days, tau_lo, tau_hi
        sigma_drw, sigma_lo, sigma_hi
        log_tau_median, log_sigma_median
        acceptance_fraction
        r_hat_tau, r_hat_sigma (Gelman-Rubin)
        converged : bool
        method : 'mcmc' or 'map_fallback'
        samples : array (n_flat, 2) or None
        chain_path : str or None
    """
    times = np.asarray(times, dtype=float)
    fluxes = np.asarray(fluxes, dtype=float)
    errors = np.asarray(errors, dtype=float)

    # Convert to rest-frame times
    times_rest = times / (1.0 + max(float(z), 0.0))

    # Subsample for MCMC tractability
    t_s, f_s, e_s, _ = _maybe_subsample(times_rest, fluxes, errors)

    # Get MAP solution for walker initialization
    try:
        map_result = fit_drw_map(times, fluxes, errors, z=z)
        log_sigma_map = float(map_result.get('log_sigma_map', 0.0))
        log_tau_map = float(map_result.get('log_tau_map', _LOG_TAU_PRIOR_MU))
    except Exception as exc:
        logger.warning(f"MAP fit failed for initialization: {exc}")
        log_sigma_map = 0.0
        log_tau_map = _LOG_TAU_PRIOR_MU

    # Try emcee
    try:
        import emcee

        p0_center = np.array([log_sigma_map, log_tau_map])
        rng = np.random.default_rng(seed=42)
        p0 = p0_center + 1e-3 * rng.standard_normal((DRW_MCMC_N_WALKERS, 2))

        # Validate all walkers
        for i in range(DRW_MCMC_N_WALKERS):
            lp = _log_prob_drw(p0[i], t_s, f_s, e_s)
            if not np.isfinite(lp):
                p0[i] = np.array([
                    np.clip(log_sigma_map, DRW_LOG_SIGMA_MIN + 0.5, DRW_LOG_SIGMA_MAX - 0.5),
                    np.clip(log_tau_map, DRW_LOG_TAU_MIN + 0.5, DRW_LOG_TAU_MAX - 0.5),
                ]) + 0.01 * rng.standard_normal(2)

        sampler = emcee.EnsembleSampler(
            DRW_MCMC_N_WALKERS, 2, _log_prob_drw,
            args=(t_s, f_s, e_s)
        )

        logger.info(
            f"DRW MCMC: {DRW_MCMC_N_WALKERS} walkers × "
            f"{DRW_MCMC_N_STEPS} steps ({DRW_MCMC_N_BURN} burn)"
        )
        sampler.run_mcmc(p0, DRW_MCMC_N_STEPS, progress=False)

        # Extract full chain for GR diagnostic: shape (steps, walkers, ndim)
        full_chain = sampler.get_chain()  # (n_steps, n_walkers, 2)
        post_burn_chain = full_chain[DRW_MCMC_N_BURN:, :, :]

        r_hats = _gelman_rubin(post_burn_chain)
        r_hat_sigma = float(r_hats[0])
        r_hat_tau = float(r_hats[1])

        converged = bool(
            np.all(r_hats < DRW_GR_THRESHOLD) and
            np.all(np.isfinite(r_hats))
        )

        if not converged:
            logger.warning(
                f"DRW MCMC did not converge: R-hat = {r_hats} "
                f"(threshold={DRW_GR_THRESHOLD})"
            )

        # Flatten post-burn chain
        flat_chain = sampler.get_chain(discard=DRW_MCMC_N_BURN, flat=True)
        log_sigma_samples = flat_chain[:, 0]
        log_tau_samples = flat_chain[:, 1]

        sigma_samples = np.exp(log_sigma_samples)
        tau_samples = np.exp(log_tau_samples)

        acceptance_fraction = float(np.mean(sampler.acceptance_fraction))

        # Save chains if requested
        chain_path = None
        if results_dir is not None and source_id is not None:
            chains_dir = Path(results_dir) / 'chains'
            chains_dir.mkdir(parents=True, exist_ok=True)
            chain_path = str(chains_dir / f'{source_id}_drw_chain.npy')
            np.save(chain_path, flat_chain)
            logger.debug(f"DRW chain saved to {chain_path}")

        logger.info(
            f"DRW MCMC: τ_rest={np.median(tau_samples):.1f} d "
            f"[{np.percentile(tau_samples, 16):.1f}–"
            f"{np.percentile(tau_samples, 84):.1f}] | "
            f"σ={np.median(sigma_samples):.4f} mJy | "
            f"acceptance={acceptance_fraction:.2f} | "
            f"R-hat=[{r_hat_sigma:.3f}, {r_hat_tau:.3f}]"
        )

        return {
            'tau_rest_days': float(np.median(tau_samples)),
            'tau_lo': float(np.percentile(tau_samples, 16)),
            'tau_hi': float(np.percentile(tau_samples, 84)),
            'sigma_drw': float(np.median(sigma_samples)),
            'sigma_lo': float(np.percentile(sigma_samples, 16)),
            'sigma_hi': float(np.percentile(sigma_samples, 84)),
            'log_tau_median': float(np.median(log_tau_samples)),
            'log_sigma_median': float(np.median(log_sigma_samples)),
            'acceptance_fraction': acceptance_fraction,
            'r_hat_tau': r_hat_tau,
            'r_hat_sigma': r_hat_sigma,
            'converged': converged,
            'method': 'mcmc',
            'samples': flat_chain,
            'chain_path': chain_path,
        }

    except ImportError:
        logger.warning("emcee not installed — falling back to MAP estimate")
        map_result = fit_drw_map(times, fluxes, errors, z=z)
        tau = float(map_result.get('tau_rest_days', np.exp(_LOG_TAU_PRIOR_MU)))
        sigma = float(map_result.get('sigma_drw', 0.1))

        return {
            'tau_rest_days': tau,
            'tau_lo': tau * 0.7,
            'tau_hi': tau * 1.4,
            'sigma_drw': sigma,
            'sigma_lo': sigma * 0.7,
            'sigma_hi': sigma * 1.4,
            'log_tau_median': np.log(tau),
            'log_sigma_median': np.log(sigma),
            'acceptance_fraction': np.nan,
            'r_hat_tau': np.nan,
            'r_hat_sigma': np.nan,
            'converged': bool(map_result.get('converged', False)),
            'method': 'map_fallback',
            'samples': None,
            'chain_path': None,
        }

    except Exception as exc:
        logger.error(f"MCMC fit failed: {exc}", exc_info=True)
        return {
            'tau_rest_days': np.nan, 'tau_lo': np.nan, 'tau_hi': np.nan,
            'sigma_drw': np.nan, 'sigma_lo': np.nan, 'sigma_hi': np.nan,
            'log_tau_median': np.nan, 'log_sigma_median': np.nan,
            'acceptance_fraction': np.nan,
            'r_hat_tau': np.nan, 'r_hat_sigma': np.nan,
            'converged': False, 'method': 'failed',
            'samples': None, 'chain_path': None,
        }


# ---------------------------------------------------------------------------
# Broken DRW (structural break)
# ---------------------------------------------------------------------------

def _drw_nll_segment(log_sigma: float, log_tau: float,
                     times: np.ndarray, fluxes: np.ndarray,
                     flux_errors: np.ndarray) -> float:
    """NLL for a single DRW segment (wrapper for _drw_nll)."""
    return _drw_nll(log_sigma, log_tau, times, fluxes, flux_errors)


def _fit_single_drw_map(times: np.ndarray, fluxes: np.ndarray,
                         errors: np.ndarray) -> tuple:
    """
    Fast MAP DRW fit for a single segment. Returns (log_sigma, log_tau, nll).
    """
    best_nll = np.inf
    best_p0 = [0.0, _LOG_TAU_PRIOR_MU]

    for ls0 in np.linspace(-2.0, 2.0, 3):
        for lt0 in np.linspace(3.5, 7.5, 4):
            nll = _drw_nll(ls0, lt0, times, fluxes, errors)
            if nll < best_nll:
                best_nll = nll
                best_p0 = [ls0, lt0]

    result = minimize(
        lambda p: _drw_nll(p[0], p[1], times, fluxes, errors),
        best_p0,
        method='L-BFGS-B',
        bounds=[(DRW_LOG_SIGMA_MIN, DRW_LOG_SIGMA_MAX),
                (DRW_LOG_TAU_MIN, DRW_LOG_TAU_MAX)],
        options={'ftol': 1e-8, 'maxiter': 500},
    )

    return result.x[0], result.x[1], float(result.fun)


def fit_broken_drw(times: np.ndarray, fluxes: np.ndarray,
                   errors: np.ndarray, z: float) -> dict:
    """
    Fit a broken DRW model: two independent DRW segments separated by a break.

    Scans break positions from 15% to 85% of the time series.
    Compares single-DRW BIC vs broken-DRW BIC.

    BIC = -2 ln(L_max) + k ln(N)
        single DRW: k = 2
        broken DRW: k = 5 (2 per segment + t_break)

    Parameters
    ----------
    times, fluxes, errors : arrays (observer frame)
    z                     : float, redshift

    Returns
    -------
    result : dict with keys:
        t_break_mjd         : float, best-fit break epoch
        t_break_mjd_lo, t_break_mjd_hi : 68% confidence interval
        tau_pre, sigma_pre  : DRW parameters before break
        tau_post, sigma_post: DRW parameters after break
        delta_tau           : tau_post / tau_pre
        delta_sigma         : sigma_post / sigma_pre
        bic_single          : float
        bic_broken          : float
        delta_bic_broken    : float (BIC_single - BIC_broken; >0 favours broken)
        log_bayes_approx    : float (-0.5 * delta_bic)
        n_break_points_tried: int
    """
    times = np.asarray(times, dtype=float)
    fluxes = np.asarray(fluxes, dtype=float)
    errors = np.asarray(errors, dtype=float)

    times_rest = times / (1.0 + max(float(z), 0.0))

    t_s, f_s, e_s, _ = _maybe_subsample(times_rest, fluxes, errors)

    n = len(t_s)
    if n < 10:
        return {
            'delta_bic_broken': np.nan, 'log_bayes_approx': np.nan,
            'bic_single': np.nan, 'bic_broken': np.nan,
            't_break_mjd': np.nan, 't_break_mjd_lo': np.nan, 't_break_mjd_hi': np.nan,
            'tau_pre': np.nan, 'sigma_pre': np.nan,
            'tau_post': np.nan, 'sigma_post': np.nan,
            'delta_tau': np.nan, 'delta_sigma': np.nan,
            'n_break_points_tried': 0,
        }

    # ---- Fit single DRW (null hypothesis) -----------------------------------
    ls_single, lt_single, nll_single = _fit_single_drw_map(t_s, f_s, e_s)
    bic_single = 2.0 * nll_single + 2.0 * np.log(n)

    # ---- Scan break positions -----------------------------------------------
    idx_lo = max(1, int(0.15 * n))
    idx_hi = min(n - 1, int(0.85 * n))
    break_indices = np.arange(idx_lo, idx_hi)

    if len(break_indices) == 0:
        break_indices = np.array([n // 2])

    best_bic_broken = np.inf
    best_break_idx = n // 2
    best_ls_pre = ls_single
    best_lt_pre = lt_single
    best_ls_post = ls_single
    best_lt_post = lt_single

    bic_broken_vals = []
    break_times = []

    for b_idx in break_indices:
        t_pre = t_s[:b_idx]
        f_pre = f_s[:b_idx]
        e_pre = e_s[:b_idx]
        t_post = t_s[b_idx:]
        f_post = f_s[b_idx:]
        e_post = e_s[b_idx:]

        if len(t_pre) < 3 or len(t_post) < 3:
            continue

        try:
            ls_pre, lt_pre, nll_pre = _fit_single_drw_map(t_pre, f_pre, e_pre)
            ls_post, lt_post, nll_post = _fit_single_drw_map(t_post, f_post, e_post)
        except Exception:
            continue

        nll_broken = nll_pre + nll_post
        # 5 parameters: log_sigma_pre, log_tau_pre, log_sigma_post, log_tau_post, t_break
        bic_broken = 2.0 * nll_broken + 5.0 * np.log(n)

        bic_broken_vals.append(bic_broken)
        break_times.append(float(t_s[b_idx]))

        if bic_broken < best_bic_broken:
            best_bic_broken = bic_broken
            best_break_idx = b_idx
            best_ls_pre, best_lt_pre = ls_pre, lt_pre
            best_ls_post, best_lt_post = ls_post, lt_post

    delta_bic = bic_single - best_bic_broken
    log_bayes = -0.5 * (bic_single - best_bic_broken)

    # 68% confidence interval: break positions with ΔBIC < 1
    if len(bic_broken_vals) > 1:
        bic_arr = np.array(bic_broken_vals)
        bt_arr = np.array(break_times)
        within_1bic = bt_arr[bic_arr < best_bic_broken + 1.0]
        t_break_lo = float(within_1bic.min()) if len(within_1bic) > 0 else np.nan
        t_break_hi = float(within_1bic.max()) if len(within_1bic) > 0 else np.nan
    else:
        t_break_lo = t_break_hi = np.nan

    t_break_mjd = float(t_s[best_break_idx]) * (1.0 + float(z))  # Back to observer frame
    tau_pre = float(np.exp(best_lt_pre))
    sigma_pre = float(np.exp(best_ls_pre))
    tau_post = float(np.exp(best_lt_post))
    sigma_post = float(np.exp(best_ls_post))

    delta_tau = float(tau_post / tau_pre) if tau_pre > 0 else np.nan
    delta_sigma = float(sigma_post / sigma_pre) if sigma_pre > 0 else np.nan

    logger.info(
        f"Broken DRW: ΔBIC={delta_bic:.2f} | "
        f"t_break={t_break_mjd:.1f} MJD | "
        f"τ_pre={tau_pre:.1f}→τ_post={tau_post:.1f} d | "
        f"σ_pre={sigma_pre:.4f}→σ_post={sigma_post:.4f}"
    )

    return {
        't_break_mjd': t_break_mjd,
        't_break_mjd_lo': t_break_lo * (1.0 + float(z)) if np.isfinite(t_break_lo) else np.nan,
        't_break_mjd_hi': t_break_hi * (1.0 + float(z)) if np.isfinite(t_break_hi) else np.nan,
        'tau_pre': tau_pre,
        'sigma_pre': sigma_pre,
        'tau_post': tau_post,
        'sigma_post': sigma_post,
        'delta_tau': delta_tau,
        'delta_sigma': delta_sigma,
        'bic_single': float(bic_single),
        'bic_broken': float(best_bic_broken),
        'delta_bic_broken': float(delta_bic),
        'log_bayes_approx': float(log_bayes),
        'n_break_points_tried': len(bic_broken_vals),
    }


# ---------------------------------------------------------------------------
# Non-stationary GP
# ---------------------------------------------------------------------------

def _ns_gp_kernel(t1: np.ndarray, t2: np.ndarray,
                  log_sigma: float, log_tau: float,
                  log_delta: float, t_break: float) -> np.ndarray:
    """
    Non-stationary DRW kernel: DRW + step transition at t_break.

    K(t_i, t_j) = σ² exp(-|t_i - t_j| / τ)
                + δ² × H(t_i - t_break) × H(t_j - t_break)

    where H(t) = 1 if t > 0 else 0 (Heaviside step function).
    The δ² term models an abrupt change in the mean flux level.
    """
    sigma = np.exp(log_sigma)
    tau = np.exp(log_tau)
    delta = np.exp(log_delta)

    dt = np.abs(t1[:, None] - t2[None, :])
    K_drw = sigma ** 2 * np.exp(-dt / tau)

    # Step component: active only for pairs where both t_i, t_j > t_break
    H_i = (t1 > t_break).astype(float)
    H_j = (t2 > t_break).astype(float)
    K_step = delta ** 2 * np.outer(H_i, H_j)

    return K_drw + K_step


def _ns_gp_nll(params: np.ndarray, times: np.ndarray, fluxes: np.ndarray,
               flux_errors: np.ndarray) -> float:
    """Negative log-likelihood for the non-stationary GP."""
    log_sigma, log_tau, log_delta, t_break = params

    # Bounds
    if not (DRW_LOG_SIGMA_MIN < log_sigma < DRW_LOG_SIGMA_MAX):
        return 1e10
    if not (DRW_LOG_TAU_MIN < log_tau < DRW_LOG_TAU_MAX):
        return 1e10
    if not (-5.0 < log_delta < 5.0):
        return 1e10
    if not (times.min() < t_break < times.max()):
        return 1e10

    try:
        K = _ns_gp_kernel(times, times, log_sigma, log_tau, log_delta, t_break)
        jitter = max(1e-6, 0.001 * np.exp(2 * log_sigma))
        K += np.diag(flux_errors ** 2 + jitter)

        c, low = cho_factor(K, lower=True, check_finite=False)
        alpha = cho_solve((c, low), fluxes, check_finite=False)
        logdet = 2.0 * np.sum(np.log(np.diag(c)))
        n = len(fluxes)
        nll = 0.5 * (np.dot(fluxes, alpha) + logdet + n * np.log(2.0 * np.pi))
        return float(nll)
    except Exception:
        return 1e10


def fit_nonstationary_gp(times: np.ndarray, fluxes: np.ndarray,
                          errors: np.ndarray, z: float) -> dict:
    """
    Fit a non-stationary GP model (DRW + step component) via MAP.

    The step component explicitly models the mean-flux transition
    characteristic of changing-look AGN.

    Parameters
    ----------
    times, fluxes, errors : arrays (observer frame)
    z                     : float, redshift

    Returns
    -------
    result : dict with keys:
        log_sigma, log_tau, log_delta, t_break_mjd
        sigma_ns, tau_ns_days, delta_ns
        nll_ns, bic_ns
        nll_stat (stationary DRW NLL), bic_stat
        log_bayes_factor    : -0.5 * (bic_stat - bic_ns)
        converged           : bool
    """
    times = np.asarray(times, dtype=float)
    fluxes = np.asarray(fluxes, dtype=float)
    errors = np.asarray(errors, dtype=float)

    times_rest = times / (1.0 + max(float(z), 0.0))
    t_s, f_s, e_s, _ = _maybe_subsample(times_rest, fluxes, errors)
    n = len(t_s)

    if n < 10:
        return {
            'log_bayes_factor': np.nan, 'bic_ns': np.nan, 'bic_stat': np.nan,
            't_break_mjd': np.nan, 'sigma_ns': np.nan, 'tau_ns_days': np.nan,
            'delta_ns': np.nan, 'converged': False,
        }

    # ---- Stationary DRW baseline -------------------------------------------
    from .drw import _drw_nll as _stat_nll

    best_stat_nll = np.inf
    best_p0_stat = [0.0, _LOG_TAU_PRIOR_MU]
    for ls0 in np.linspace(-1.5, 2.0, 3):
        for lt0 in np.linspace(4.0, 7.5, 4):
            nll = _drw_nll(ls0, lt0, t_s, f_s, e_s)
            if nll < best_stat_nll:
                best_stat_nll = nll
                best_p0_stat = [ls0, lt0]

    stat_result = minimize(
        lambda p: _drw_nll(p[0], p[1], t_s, f_s, e_s),
        best_p0_stat,
        method='L-BFGS-B',
        bounds=[(DRW_LOG_SIGMA_MIN, DRW_LOG_SIGMA_MAX),
                (DRW_LOG_TAU_MIN, DRW_LOG_TAU_MAX)],
        options={'ftol': 1e-8, 'maxiter': 500},
    )
    nll_stat = float(stat_result.fun)
    bic_stat = 2.0 * nll_stat + 2.0 * np.log(n)  # k=2

    # ---- Non-stationary GP MAP optimization ---------------------------------
    t_mid = float(0.5 * (t_s.min() + t_s.max()))
    p0_ns = [stat_result.x[0], stat_result.x[1], -0.5, t_mid]

    # Try multiple break positions
    best_nll_ns = np.inf
    best_result_ns = None

    for frac in [0.3, 0.5, 0.7]:
        t_break0 = float(t_s[int(frac * len(t_s))])
        p0 = [stat_result.x[0], stat_result.x[1], -0.5, t_break0]

        try:
            res = minimize(
                lambda p: _ns_gp_nll(p, t_s, f_s, e_s),
                p0,
                method='L-BFGS-B',
                bounds=[
                    (DRW_LOG_SIGMA_MIN, DRW_LOG_SIGMA_MAX),
                    (DRW_LOG_TAU_MIN, DRW_LOG_TAU_MAX),
                    (-5.0, 5.0),
                    (t_s.min() + 1.0, t_s.max() - 1.0),
                ],
                options={'ftol': 1e-8, 'maxiter': 800},
            )

            if res.fun < best_nll_ns:
                best_nll_ns = res.fun
                best_result_ns = res

        except Exception as exc:
            logger.debug(f"NS-GP optimization attempt failed: {exc}")
            continue

    if best_result_ns is None or best_nll_ns >= 1e9:
        return {
            'log_bayes_factor': np.nan, 'bic_ns': np.nan,
            'bic_stat': float(bic_stat),
            't_break_mjd': np.nan, 'sigma_ns': np.nan,
            'tau_ns_days': np.nan, 'delta_ns': np.nan,
            'converged': False,
        }

    nll_ns = float(best_nll_ns)
    bic_ns = 2.0 * nll_ns + 4.0 * np.log(n)  # k=4 (sigma, tau, delta, t_break)
    log_bayes_factor = -0.5 * (bic_stat - bic_ns)

    ls_ns, lt_ns, ld_ns, t_break_ns = best_result_ns.x
    # Convert break time back to observer frame
    t_break_mjd = float(t_break_ns) * (1.0 + float(z))

    logger.info(
        f"NS-GP: log_B={log_bayes_factor:.2f} | "
        f"ΔBIC={bic_stat - bic_ns:.2f} | "
        f"t_break={t_break_mjd:.1f} MJD | "
        f"δ={np.exp(ld_ns):.4f}"
    )

    return {
        'log_sigma': float(ls_ns),
        'log_tau': float(lt_ns),
        'log_delta': float(ld_ns),
        't_break_mjd': float(t_break_mjd),
        'sigma_ns': float(np.exp(ls_ns)),
        'tau_ns_days': float(np.exp(lt_ns)),
        'delta_ns': float(np.exp(ld_ns)),
        'nll_ns': float(nll_ns),
        'bic_ns': float(bic_ns),
        'nll_stat': float(nll_stat),
        'bic_stat': float(bic_stat),
        'log_bayes_factor': float(log_bayes_factor),
        'converged': bool(best_result_ns.success),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_advanced_drw(results_dir: str = './results/') -> None:
    """
    Run advanced DRW analysis (MCMC + broken DRW + NS-GP) for top candidates.

    Loads top_candidates.csv, fits all three models for each source,
    and saves advanced_drw_{source_id}.pkl + advanced_drw_summary.csv.

    Parameters
    ----------
    results_dir : str, path to results directory
    """
    import pickle
    import pandas as pd
    from pathlib import Path

    results_path = Path(results_dir)
    candidates_file = results_path / 'top_candidates.csv'

    if not candidates_file.exists():
        logger.error(f"Candidates file not found: {candidates_file}")
        return

    try:
        candidates = pd.read_csv(candidates_file)
    except Exception as exc:
        logger.error(f"Failed to load candidates: {exc}")
        return

    adv_drw_dir = results_path / 'advanced_drw'
    adv_drw_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for idx, row in candidates.iterrows():
        source_id = str(row.get('source_id', f'source_{idx}'))
        z = float(row.get('z', row.get('redshift', 0.1)))

        # Try loading WISE light curve
        wise_pkl = results_path / 'wise_enhanced' / f'enhanced_wise_{source_id}.pkl'
        if not wise_pkl.exists():
            # Fallback: try standard results
            wise_pkl = results_path / f'wise_{source_id}.pkl'

        lc_df = None
        if wise_pkl.exists():
            with open(wise_pkl, 'rb') as f:
                wise_data = pickle.load(f)
            lc_df = wise_data.get('lc', None)

        if lc_df is None or lc_df.empty:
            logger.warning(f"{source_id}: No WISE light curve found, skipping")
            continue

        # Extract flux arrays
        times = lc_df['mjd'].values if 'mjd' in lc_df.columns else np.array([])
        fluxes = (
            lc_df['w1_flux_mjy'].values
            if 'w1_flux_mjy' in lc_df.columns
            else np.array([])
        )
        errors = (
            lc_df['w1_flux_err_mjy'].values
            if 'w1_flux_err_mjy' in lc_df.columns
            else np.ones_like(fluxes) * 0.01
        )

        if len(times) < 10:
            logger.warning(f"{source_id}: Too few epochs ({len(times)}), skipping")
            continue

        valid = np.isfinite(times) & np.isfinite(fluxes) & np.isfinite(errors)
        times = times[valid]
        fluxes = fluxes[valid]
        errors = errors[valid]

        logger.info(
            f"Advanced DRW for {source_id} ({idx + 1}/{len(candidates)}): "
            f"{len(times)} epochs, z={z:.4f}"
        )

        result = {'source_id': source_id, 'z': z, 'n_epochs': len(times)}

        try:
            mcmc_result = fit_drw_full_mcmc(
                times, fluxes, errors, z,
                results_dir=results_dir,
                source_id=source_id,
            )
            result['mcmc'] = mcmc_result
        except Exception as exc:
            logger.error(f"{source_id}: MCMC failed: {exc}")
            result['mcmc'] = {}

        try:
            broken_result = fit_broken_drw(times, fluxes, errors, z)
            result['broken_drw'] = broken_result
        except Exception as exc:
            logger.error(f"{source_id}: Broken DRW failed: {exc}")
            result['broken_drw'] = {}

        try:
            ns_result = fit_nonstationary_gp(times, fluxes, errors, z)
            result['nonstationary_gp'] = ns_result
        except Exception as exc:
            logger.error(f"{source_id}: NS-GP failed: {exc}")
            result['nonstationary_gp'] = {}

        # Save per-source result
        pkl_path = adv_drw_dir / f'advanced_drw_{source_id}.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(result, f, protocol=4)

        # Summary row
        mcmc = result.get('mcmc', {})
        broken = result.get('broken_drw', {})
        ns = result.get('nonstationary_gp', {})

        summary_rows.append({
            'source_id': source_id,
            'z': z,
            'n_epochs': len(times),
            'tau_rest_days': mcmc.get('tau_rest_days', np.nan),
            'tau_lo': mcmc.get('tau_lo', np.nan),
            'tau_hi': mcmc.get('tau_hi', np.nan),
            'sigma_drw': mcmc.get('sigma_drw', np.nan),
            'mcmc_converged': mcmc.get('converged', False),
            'mcmc_r_hat_tau': mcmc.get('r_hat_tau', np.nan),
            'broken_delta_bic': broken.get('delta_bic_broken', np.nan),
            'broken_log_bayes': broken.get('log_bayes_approx', np.nan),
            'broken_t_break_mjd': broken.get('t_break_mjd', np.nan),
            'ns_log_bayes_factor': ns.get('log_bayes_factor', np.nan),
            'ns_t_break_mjd': ns.get('t_break_mjd', np.nan),
        })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = results_path / 'advanced_drw_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Advanced DRW summary saved to {summary_path}")
    else:
        logger.warning("No advanced DRW results to save.")
