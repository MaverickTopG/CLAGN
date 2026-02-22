"""
drw.py — Damped Random Walk GP fitting with MAP optimization and MCMC posterior.

The DRW (Ornstein-Uhlenbeck process) is the standard model for AGN stochastic
variability (Kelly et al. 2009, Kozłowski et al. 2010, MacLeod et al. 2010).

Covariance kernel: k(dt) = σ_DRW² × exp(-|dt| / τ)

All fitting is done in FLUX space (mJy). Never in magnitude space.
All timescales are returned in the source REST FRAME: τ_rest = τ_obs / (1+z).

Uses numpy fallback if celerite2 is unavailable (O(N³), subsampled at N>300).
"""
import logging
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve

from ..config import (
    DRW_LOG_TAU_MIN, DRW_LOG_TAU_MAX,
    DRW_LOG_SIGMA_MIN, DRW_LOG_SIGMA_MAX,
    DRW_MAX_EPOCHS_FOR_FULL_FIT,
)

# FLAW A5: Bounds constants for celerite2 optimization (must match numpy fallback)
LOG_TAU_MIN = np.log(2.0)       # tau_min = 2 days
LOG_TAU_MAX = np.log(5000.0)    # tau_max = 5000 days
LOG_SIG_MIN = np.log(1e-4)      # sigma_min = 0.0001 mJy
LOG_SIG_MAX = np.log(1e4)       # sigma_max = 10000 mJy
# Convert to rho-space: rho ≈ 2π*tau for SHOTerm Q=0.5
LOG_RHO_MIN = LOG_TAU_MIN + np.log(2 * np.pi)
LOG_RHO_MAX = LOG_TAU_MAX + np.log(2 * np.pi)

logger = logging.getLogger(__name__)


def _maybe_subsample(times, fluxes, flux_errors, max_n=DRW_MAX_EPOCHS_FOR_FULL_FIT,
                     seed=42):
    """
    Subsample light curve to at most `max_n` epochs for O(N³) matrix operations.
    Uses a uniform random subsample preserving temporal coverage.

    Returns (t_sub, f_sub, e_sub, original_indices)
    """
    if len(times) <= max_n:
        return times, fluxes, flux_errors, np.arange(len(times))

    rng = np.random.default_rng(seed=seed)
    idx = np.sort(rng.choice(len(times), size=max_n, replace=False))
    logger.debug(f"DRW: subsampling {len(times)} → {max_n} epochs for matrix fit")
    return times[idx], fluxes[idx], flux_errors[idx], idx


def _drw_nll(log_sigma, log_tau, times, fluxes, flux_errors):
    """
    Negative log-likelihood of the DRW GP model (to be minimized).

    Covariance: K_ij = σ² × exp(-|t_i - t_j| / τ) + diag(err² + jitter)
    NLL = 0.5 × [y^T K^{-1} y + ln|K| + N×ln(2π)]

    Uses Cholesky decomposition for stability.
    Log-determinant computed from the Cholesky diagonal: log|K| = 2 Σ log(L_ii).

    Returns large positive value (1e10) on numerical failure — treated as
    rejected parameter combination.
    """
    # Prior bounds
    if not (DRW_LOG_SIGMA_MIN < log_sigma < DRW_LOG_SIGMA_MAX):
        return 1e10
    if not (DRW_LOG_TAU_MIN < log_tau < DRW_LOG_TAU_MAX):
        return 1e10

    sigma = np.exp(log_sigma)
    tau = np.exp(log_tau)

    dt = np.abs(times[:, None] - times[None, :])
    K = sigma**2 * np.exp(-dt / tau)

    # Jitter: larger of fixed floor or fraction of process variance
    jitter = max(1e-6, 0.001 * sigma**2)
    K += np.diag(flux_errors**2 + jitter)

    try:
        c, low = cho_factor(K, lower=True, overwrite_a=False, check_finite=False)
        alpha = cho_solve((c, low), fluxes, overwrite_b=False, check_finite=False)
        # log|K| = 2 × sum(log(diagonal of L))
        logdet = 2.0 * np.sum(np.log(np.diag(c)))
        nll = 0.5 * (np.dot(fluxes, alpha) + logdet + len(fluxes) * np.log(2.0 * np.pi))
        return float(nll)
    except (np.linalg.LinAlgError, Exception):
        return 1e10


def _drw_predict(times_train, fluxes_train, flux_errors_train,
                 times_pred, sigma, tau):
    """
    Compute GP posterior mean and standard deviation at prediction times.

    Uses the exact GP posterior formula:
        μ_pred = K_cross @ K_train^{-1} @ y
        σ²_pred = diag(K_pred) - diag(K_cross @ K_train^{-1} @ K_cross^T)

    Parameters
    ----------
    times_train, fluxes_train, flux_errors_train : training data
    times_pred  : array, prediction MJDs
    sigma, tau  : MAP DRW parameters

    Returns
    -------
    pred_mean, pred_std : arrays
    """
    jitter = max(1e-6, 0.001 * sigma**2)

    dt_train = np.abs(times_train[:, None] - times_train[None, :])
    K_train = sigma**2 * np.exp(-dt_train / tau)
    K_train += np.diag(flux_errors_train**2 + jitter)

    dt_cross = np.abs(times_pred[:, None] - times_train[None, :])
    K_cross = sigma**2 * np.exp(-dt_cross / tau)   # shape (N_pred, N_train)

    try:
        c, low = cho_factor(K_train, lower=True, check_finite=False)
        alpha = cho_solve((c, low), fluxes_train, check_finite=False)
        pred_mean = K_cross @ alpha

        # Predictive variance: σ²(x*) = k(x*,x*) - k(x*,X) K^{-1} k(X,x*)
        v = cho_solve((c, low), K_cross.T, check_finite=False)   # (N_train, N_pred)
        K_pred_diag = np.full(len(times_pred), sigma**2)
        pred_var = K_pred_diag - np.einsum('ij,ji->i', K_cross, v)
        pred_std = np.sqrt(np.maximum(pred_var, 0.0))
    except Exception:
        pred_mean = np.full(len(times_pred), np.nanmean(fluxes_train))
        pred_std = np.full(len(times_pred), sigma)

    return pred_mean, pred_std


def _fit_drw_numpy(times, fluxes, flux_errors):
    """
    Manual DRW GP implementation using numpy/scipy.

    MAP optimization via L-BFGS-B with grid initialization to avoid local minima.
    Grid: 4 log-sigma × 4 log-tau = 16 starting points.
    Subsampled to DRW_MAX_EPOCHS_FOR_FULL_FIT epochs for O(N³) tractability.

    Returns dict with: tau_rest_days, sigma_drw, log_like, converged
    """
    t_s, f_s, e_s, _ = _maybe_subsample(times, fluxes, flux_errors)

    # ---- Grid initialization ------------------------------------------------
    best_nll = np.inf
    best_p0 = None
    for log_sigma0 in np.linspace(-2.0, 2.0, 4):
        for log_tau0 in np.linspace(4.0, 7.0, 4):
            nll = _drw_nll(log_sigma0, log_tau0, t_s, f_s, e_s)
            if nll < best_nll:
                best_nll = nll
                best_p0 = [log_sigma0, log_tau0]

    if best_p0 is None:
        best_p0 = [0.0, 6.0]   # Fallback: σ=1, τ=400 days

    # ---- MAP optimization ---------------------------------------------------
    result = minimize(
        lambda p: _drw_nll(p[0], p[1], t_s, f_s, e_s),
        best_p0,
        method='L-BFGS-B',
        bounds=[(DRW_LOG_SIGMA_MIN, DRW_LOG_SIGMA_MAX),
                (DRW_LOG_TAU_MIN, DRW_LOG_TAU_MAX)],
        options={'ftol': 1e-10, 'maxiter': 1000},
    )

    log_sigma_map = result.x[0]
    log_tau_map = result.x[1]
    sigma_map = float(np.exp(log_sigma_map))
    tau_map = float(np.exp(log_tau_map))   # Already rest-frame (caller applied z-correction)

    return {
        'tau_rest_days': tau_map,
        'sigma_drw': sigma_map,
        'log_like': float(-result.fun),
        'converged': result.success,
        'log_sigma_map': log_sigma_map,
        'log_tau_map': log_tau_map,
    }


def fit_drw_numpy_fallback(times_rest, fluxes_mjy, flux_errors_mjy):
    """
    NumPy/scipy fallback DRW fit with CRITICAL mean subtraction.

    FLAW A3 FIX: Must subtract weighted mean before computing likelihood.
    If the mean is not subtracted, the optimizer partly explains the large
    DC offset as stochastic variability, biasing both tau and sigma.

    Parameters
    ----------
    times_rest : array, rest-frame MJD
    fluxes_mjy : array, flux density in mJy
    flux_errors_mjy : array, flux uncertainty in mJy

    Returns
    -------
    result : dict with all _fit_drw_numpy keys plus:
        mu_flux_mjy, drw_fit_on_centered_flux, sigma_drw_mjy
    """
    weights = 1.0 / np.maximum(flux_errors_mjy ** 2, 1e-30)
    mu_flux = float(np.average(fluxes_mjy, weights=weights))
    f_centered = fluxes_mjy - mu_flux

    result = _fit_drw_numpy(times_rest, f_centered, flux_errors_mjy)

    result['mu_flux_mjy'] = mu_flux
    result['drw_fit_on_centered_flux'] = True
    result['sigma_drw_mjy'] = result['sigma_drw']   # alias for test verification

    return result


def check_drw_reliability(tau_rest_days, baseline_obs_days, z):
    """
    Apply the Kozlowski+2017 reliability criterion for DRW timescales.

    The baseline (in rest frame) must be >= 10 * tau_rest for tau to be
    reliably measured. Sources failing this test can still be CLAGN candidates
    based on other metrics, but their DRW tau should not be used for
    physics (M_BH estimation, disk timescales).

    Also checks: tau must not be hitting prior bounds (LOG_TAU_MIN=1.0,
    LOG_TAU_MAX=8.5). If within 0.3 of the prior bounds, tau is unconstrained.

    Parameters
    ----------
    tau_rest_days : float
        Fitted DRW rest-frame timescale
    baseline_obs_days : float
        Observer-frame light curve baseline
    z : float
        Redshift

    Returns
    -------
    reliable : bool
    reason : str
    """
    if not np.isfinite(tau_rest_days) or tau_rest_days <= 0:
        return False, "tau_rest is not finite or non-positive"

    baseline_rest_days = baseline_obs_days / max(1.0 + z, 1.0)

    if tau_rest_days > baseline_rest_days / 10.0:
        return False, (
            f"tau_rest={tau_rest_days:.0f}d > baseline_rest/10="
            f"{baseline_rest_days/10:.0f}d — Kozlowski+2017 criterion fails"
        )

    # Check: tau should not be hitting prior bounds
    log_tau = np.log(tau_rest_days)
    if log_tau < DRW_LOG_TAU_MIN + 0.3:
        return False, "tau at lower prior boundary — unconstrained"
    if log_tau > DRW_LOG_TAU_MAX - 0.3:
        return False, "tau at upper prior boundary — unconstrained"

    return True, "OK"


def fit_drw_map(times, fluxes, flux_errors, z=0.0):
    """
    Fit DRW model via Maximum A Posteriori (MAP) optimization.

    Fast first-pass; results feed MCMC for final candidates.

    IMPORTANT: Fits in FLUX SPACE (mJy). NEVER in magnitude space.
    DRW GP likelihood assumes Gaussian residuals — satisfied in flux space,
    NOT in magnitude space (MacLeod+2010, Kelly+2009).

    Converts to rest-frame times before fitting:
        times_rest = times_obs / (1 + z)
    All returned timescales are rest-frame.

    Parameters
    ----------
    times       : array, MJD (observer frame)
    fluxes      : array, flux density in mJy (NOT magnitudes)
    flux_errors : array, flux uncertainty in mJy
    z           : float, source redshift

    Returns
    -------
    result : dict with keys:
        tau_rest_days, sigma_drw, log_like, converged,
        residuals, pred_mean, pred_std,
        sigma_drw_mag_equiv, tau_reliable, unreliable_reason
    """
    times = np.asarray(times, dtype=float)
    fluxes = np.asarray(fluxes, dtype=float)
    flux_errors = np.asarray(flux_errors, dtype=float)

    # FIX 3: Sanity check — input must be in flux space (mJy range)
    median_val = float(np.nanmedian(fluxes))
    assert 0.001 < median_val < 100000, (
        f"Median flux {median_val:.3f} is outside mJy range [0.001, 100000]. "
        f"Did you accidentally pass magnitudes instead of fluxes?"
    )

    # ---- Rest-frame time correction (Ricci+2022; mandatory) -----------------
    times_rest = times / (1.0 + max(z, 0.0))
    baseline_obs_days = float(times.max() - times.min()) if len(times) > 1 else 1.0

    # ---- Try celerite2 first; fall back to numpy ----------------------------
    try:
        import celerite2
        map_result = _fit_drw_celerite2(times_rest, fluxes, flux_errors)
    except ImportError:
        # FLAW A3: use fallback with mean subtraction
        map_result = fit_drw_numpy_fallback(times_rest, fluxes, flux_errors)

    tau = map_result['tau_rest_days']
    sigma = map_result['sigma_drw']

    # ---- GP posterior prediction on full light curve ------------------------
    # Subsample training set for prediction (avoids huge matrices)
    t_s, f_s, e_s, _ = _maybe_subsample(times_rest, fluxes, flux_errors)
    pred_mean, pred_std = _drw_predict(t_s, f_s, e_s, times_rest, sigma, tau)

    # ---- Standardized residuals (should be ~N(0,1) for stationary DRW) -----
    total_std = np.sqrt(pred_std**2 + flux_errors**2)
    residuals = np.where(total_std > 0, (fluxes - pred_mean) / total_std, 0.0)

    map_result['residuals'] = residuals
    map_result['pred_mean'] = pred_mean
    map_result['pred_std'] = pred_std
    map_result['times_rest'] = times_rest

    # FIX 3: DRW sigma in magnitude-equivalent units
    # sigma_drw_mag_equiv = 2.5 * log10(1 + sigma_mJy / median_flux_mJy)
    median_flux = float(np.nanmedian(fluxes))
    if median_flux > 0 and sigma > 0:
        sigma_drw_mag_equiv = float(2.5 * np.log10(1.0 + sigma / median_flux))
    else:
        sigma_drw_mag_equiv = np.nan
    map_result['sigma_drw_mag_equiv'] = sigma_drw_mag_equiv

    # FIX 5: DRW reliability check (Kozlowski+2017)
    tau_reliable, unreliable_reason = check_drw_reliability(
        tau, baseline_obs_days, z
    )
    map_result['tau_reliable'] = tau_reliable
    map_result['unreliable_reason'] = unreliable_reason if not tau_reliable else None

    logger.debug(
        f"DRW MAP: τ_rest={tau:.1f} d, σ={sigma:.4f} mJy, "
        f"σ_mag_equiv={sigma_drw_mag_equiv:.4f}, "
        f"log_like={map_result['log_like']:.1f}, "
        f"tau_reliable={tau_reliable}"
    )

    return map_result


def _fit_drw_celerite2(times_rest, fluxes, flux_errors):
    """celerite2-based DRW fit (optional fast path).

    FLAW A5 FIX: Add parameter bounds matching the numpy fallback.
    Post-fit: flag if solution is at boundary (tau_reliable=False).
    """
    import celerite2
    import celerite2.terms as terms

    # celerite2 DRW term: SHOTerm with Q=0.5 approximates Ornstein-Uhlenbeck
    t_s, f_s, e_s, _ = _maybe_subsample(times_rest, fluxes, flux_errors)

    # Subtract weighted mean before fitting (FLAW A3 consistency)
    weights = 1.0 / np.maximum(e_s ** 2, 1e-30)
    mu_f = float(np.average(f_s, weights=weights))
    f_centered = f_s - mu_f

    def nll(params):
        log_sigma, log_rho = params
        try:
            term = terms.SHOTerm(sigma=np.exp(log_sigma), rho=np.exp(log_rho), Q=0.5)
            gp = celerite2.GaussianProcess(term, mean=0.0)
            gp.compute(t_s, yerr=e_s)
            return -gp.log_likelihood(f_centered)
        except Exception:
            return 1e10

    # FLAW A5: Add bounds matching the numpy fallback path
    bounds = [(LOG_SIG_MIN, LOG_SIG_MAX), (LOG_RHO_MIN, LOG_RHO_MAX)]
    x0 = [0.0, np.log(200.0 * 2.0 * np.pi)]  # sigma=1, tau=200d in rho-space

    result = minimize(
        nll,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-9},
    )
    log_sigma, log_rho = result.x
    # rho ≈ 2πτ for SHO → τ ≈ rho/(2π)
    tau = float(np.exp(log_rho) / (2.0 * np.pi))
    sigma = float(np.exp(log_sigma))

    # FLAW A5: Check if solution is at a parameter boundary
    at_boundary = (
        log_rho < LOG_RHO_MIN + 0.1 or
        log_rho > LOG_RHO_MAX - 0.1 or
        log_sigma < LOG_SIG_MIN + 0.1 or
        log_sigma > LOG_SIG_MAX - 0.1
    )

    drw_result = {
        'tau_rest_days': tau,
        'sigma_drw': sigma,
        'log_like': float(-result.fun),
        'converged': result.success,
        'log_sigma_map': log_sigma,
        'log_tau_map': np.log(max(tau, 1e-10)),
        'mu_flux_mjy': mu_f,
        'drw_fit_on_centered_flux': True,
        'sigma_drw_mjy': sigma,
    }

    if at_boundary:
        drw_result['tau_reliable'] = False
        drw_result['unreliable_reason'] = 'solution_at_parameter_boundary'

    return drw_result


def fit_drw_mcmc(times, fluxes, flux_errors, z=0.0,
                  n_walkers=32, n_steps=2000, n_burn=500):
    """
    Full MCMC posterior sampling of DRW parameters using emcee.

    Run this ONLY on top candidates identified by MAP fitting.

    Priors (log-uniform, physically motivated):
        ln(τ) ~ Uniform(DRW_LOG_TAU_MIN, DRW_LOG_TAU_MAX)
        ln(σ) ~ Uniform(DRW_LOG_SIGMA_MIN, DRW_LOG_SIGMA_MAX)

    Parameters
    ----------
    times, fluxes, flux_errors : arrays
    z          : float, redshift
    n_walkers  : int (default 32)
    n_steps    : int (default 2000)
    n_burn     : int (default 500)

    Returns
    -------
    result : dict with:
        tau_rest_days, tau_lo, tau_hi (16th/84th percentile)
        sigma_drw, sigma_lo, sigma_hi
        samples : array (n_samples, 2)
        acceptance_fraction : float (should be 0.2–0.5)
    """
    import emcee

    times_rest = np.asarray(times, dtype=float) / (1.0 + z)
    fluxes = np.asarray(fluxes, dtype=float)
    flux_errors = np.asarray(flux_errors, dtype=float)

    # Subsample for MCMC speed
    t_s, f_s, e_s, _ = _maybe_subsample(times_rest, fluxes, flux_errors)

    def log_prob(params):
        log_sigma, log_tau = params
        nll = _drw_nll(log_sigma, log_tau, t_s, f_s, e_s)
        if nll >= 1e9:
            return -np.inf
        return -nll  # log-likelihood (uniform prior in log-space)

    # ---- Initialize walkers around MAP solution ----------------------------
    map_result = fit_drw_map(times, fluxes, flux_errors, z=z)
    p0_center = np.array([
        map_result['log_sigma_map'],
        map_result['log_tau_map'],
    ])

    rng = np.random.default_rng(seed=42)
    p0 = p0_center + 1e-3 * rng.standard_normal((n_walkers, 2))

    # Verify all walkers have finite log_prob; if not, perturb toward interior
    for i, p in enumerate(p0):
        lp = log_prob(p)
        if not np.isfinite(lp):
            p0[i] = np.array([
                np.clip(p0_center[0], DRW_LOG_SIGMA_MIN + 0.5, DRW_LOG_SIGMA_MAX - 0.5),
                np.clip(p0_center[1], DRW_LOG_TAU_MIN + 0.5, DRW_LOG_TAU_MAX - 0.5),
            ]) + 0.01 * rng.standard_normal(2)

    # ---- Run MCMC -----------------------------------------------------------
    sampler = emcee.EnsembleSampler(n_walkers, 2, log_prob)
    sampler.run_mcmc(p0, n_steps, progress=False)

    # Discard burn-in, flatten chain
    chain = sampler.get_chain(discard=n_burn, flat=True)   # shape (n_flat, 2)

    log_sigma_samples = chain[:, 0]
    log_tau_samples   = chain[:, 1]

    sigma_samples = np.exp(log_sigma_samples)
    tau_samples   = np.exp(log_tau_samples)

    acceptance_fraction = float(np.mean(sampler.acceptance_fraction))
    logger.info(
        f"DRW MCMC: τ_rest={np.median(tau_samples):.1f} d "
        f"[{np.percentile(tau_samples,16):.1f}–{np.percentile(tau_samples,84):.1f}] | "
        f"acceptance={acceptance_fraction:.2f}"
    )

    tau_median = float(np.median(tau_samples))
    sigma_median = float(np.median(sigma_samples))
    baseline_obs_days = float(np.asarray(times, dtype=float).ptp())

    # FIX 5: Reliability check
    tau_reliable, unreliable_reason = check_drw_reliability(
        tau_median, baseline_obs_days, z
    )

    # FIX 3: sigma_drw_mag_equiv
    median_flux = float(np.nanmedian(fluxes))
    sigma_drw_mag_equiv = (
        float(2.5 * np.log10(1.0 + sigma_median / median_flux))
        if median_flux > 0 and sigma_median > 0 else np.nan
    )

    return {
        'tau_rest_days':  tau_median,
        'tau_lo':         float(np.percentile(tau_samples, 16)),
        'tau_hi':         float(np.percentile(tau_samples, 84)),
        'sigma_drw':      sigma_median,
        'sigma_lo':       float(np.percentile(sigma_samples, 16)),
        'sigma_hi':       float(np.percentile(sigma_samples, 84)),
        'sigma_drw_mag_equiv': sigma_drw_mag_equiv,
        'samples':        chain,
        'acceptance_fraction': acceptance_fraction,
        'tau_reliable':   tau_reliable,
        'unreliable_reason': unreliable_reason if not tau_reliable else None,
    }


def compute_drw_nonstationarity(times, fluxes, flux_errors, z, tau, sigma_drw):
    """
    Test whether the light curve is consistent with a STATIONARY DRW.

    CLAGN are by definition NON-STATIONARY — their mean flux level changes.
    A stationary DRW has constant mean and variance.

    Method:
    - Split at temporal midpoint (median MJD)
    - Compute 90-day rolling medians in each half to suppress short-term flares
    - Test if the difference between half-means exceeds DRW expectation

    DRW variance at lag T: Var(T) = σ² × (1 - exp(-T/τ))
    Expected scatter between half-means under stationary DRW:
        σ_expected = σ_DRW × sqrt(2/N_half) × f_corr(T_half, τ)
    where f_corr accounts for correlation between epochs.

    Parameters
    ----------
    times       : array, MJD (observer frame)
    fluxes      : array, mJy
    flux_errors : array, mJy
    z           : float, redshift
    tau         : float, DRW rest-frame timescale (days)
    sigma_drw   : float, DRW amplitude (mJy)

    Returns
    -------
    nonstationarity_sigma : float (>3σ: strong, >5σ: very strong CLAGN evidence)
    delta_mean_normalized : float ((mean_late - mean_early) / sigma_expected)
    mean_early            : float, mJy
    mean_late             : float, mJy
    """
    times = np.asarray(times, dtype=float)
    fluxes = np.asarray(fluxes, dtype=float)
    flux_errors = np.asarray(flux_errors, dtype=float)

    # Sort by time
    idx = np.argsort(times)
    times = times[idx]
    fluxes = fluxes[idx]
    flux_errors = flux_errors[idx]

    # ---- Split at temporal midpoint ----------------------------------------
    mjd_mid = np.median(times)
    early_mask = times <= mjd_mid
    late_mask  = times > mjd_mid

    if early_mask.sum() < 3 or late_mask.sum() < 3:
        return np.nan, np.nan, np.nan, np.nan

    # ---- Use rolling 90-day medians for robustness -------------------------
    from ..utils.photometry import rolling_median_flux
    rolling = rolling_median_flux(times, fluxes, window_years=90.0 / 365.25)

    early_median = np.nanmedian(rolling[early_mask])
    late_median  = np.nanmedian(rolling[late_mask])

    if not np.isfinite(early_median) or not np.isfinite(late_median):
        # Fall back to direct medians
        early_median = float(np.nanmedian(fluxes[early_mask]))
        late_median  = float(np.nanmedian(fluxes[late_mask]))

    delta_mean = late_median - early_median

    # ---- Expected scatter under stationary DRW -----------------------------
    N_half = max(early_mask.sum(), late_mask.sum())
    T_half = (times[late_mask].mean() - times[early_mask].mean()) / (1.0 + z)

    # Correlation between epoch means: DRW variance at lag T_half
    drw_var_at_lag = sigma_drw**2 * (1.0 - np.exp(-abs(T_half) / max(tau, 1.0)))
    # Standard error of the half-mean (assuming tau >> cadence for conservative bound)
    sigma_expected = sigma_drw * np.sqrt(2.0 / N_half)
    # Correction for within-segment correlations: multiply by sqrt(var_at_lag/sigma²)
    corr_factor = np.sqrt(max(drw_var_at_lag / sigma_drw**2, 0.01))
    sigma_expected *= (1.0 + corr_factor)

    if sigma_expected <= 0:
        return np.nan, np.nan, early_median, late_median

    nonstationarity_sigma = abs(delta_mean) / sigma_expected
    delta_mean_normalized = delta_mean / sigma_expected

    logger.debug(
        f"DRW nonstationarity: Δμ={delta_mean:.4f} mJy, "
        f"σ_expected={sigma_expected:.4f} mJy, "
        f"nonstationarity={nonstationarity_sigma:.2f}σ"
    )

    return (float(nonstationarity_sigma), float(delta_mean_normalized),
            float(early_median), float(late_median))
