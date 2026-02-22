"""
physics.py — Physical parameter estimation for CLAGN candidates.

Derives black hole masses, bolometric luminosities, Eddington ratios,
dust sublimation radii, BLR sizes, and accretion disk timescales from
WISE photometry and DRW variability parameters.

All luminosities in erg/s. All masses in solar masses. All radii in pc.
All timescales in days.

References:
    Richards et al. 2006, ApJS 166, 470 (IR bolometric correction)
    Kelly et al. 2009, ApJ 698, 895 (DRW mass scaling)
    Barvainis 1987, ApJ 320, 537 (dust sublimation radius)
    Bentz et al. 2013, ApJ 767, 149 (BLR radius-luminosity)
    Shakura & Sunyaev 1973, A&A 24, 337 (disk timescales)
"""
import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FLAW B5: Hard-block unreliable tau from all physics calculations
# ---------------------------------------------------------------------------

def _require_reliable_tau(drw_result, context='unknown'):
    """
    Call this before any physical computation that uses DRW tau.
    Raises ValueError if tau is not reliable.

    Parameters
    ----------
    drw_result : dict with 'tau_reliable' and 'unreliable_reason' keys
    context    : str, name of the calling computation (for error message)

    Raises
    ------
    ValueError if tau is not reliable
    """
    if not drw_result.get('tau_reliable', False):
        reason = drw_result.get('unreliable_reason', 'unknown')
        raise ValueError(
            f"Attempted to use unreliable DRW tau in {context}. "
            f"Reason: {reason}. "
            f"This computation must be skipped for this source."
        )

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_C_CGS       = 2.997924e10     # cm/s
_H_CGS       = 6.626070e-27    # erg·s
_K_B_CGS     = 1.380649e-16    # erg/K
_PC_CM       = 3.085678e18     # cm per parsec
_MSUN_G      = 1.989000e33     # grams per solar mass
_G_CGS       = 6.674300e-8     # cm^3 g^-1 s^-2
_L_SUN       = 3.828000e33     # erg/s
_SIGMA_T     = 6.652458e-25    # cm^2 (Thomson cross-section)
_M_P         = 1.672623e-24    # g (proton mass)

# W1 and W2 effective frequencies (Hz)
_NU_W1 = 3.0e14 / 3.4          # Hz (W1 = 3.4 micron)
_NU_W2 = 3.0e14 / 4.6          # Hz (W2 = 4.6 micron)

# W1 and W2 Vega zero points (Jy)
_ZP_W1 = 309.540
_ZP_W2 = 171.787

# Eddington luminosity constant: L_Edd = 4πGMc / (κ_es) erg/s
# = 1.26e38 × (M/M_sun) erg/s
_L_EDD_PER_MSUN = 1.26e38  # erg/s per solar mass

# Kelly+2009 DRW mass scaling coefficients (fiducial values)
_KELLY_A = 2.4    # log(tau_days) = A + B*log(L44) + C*log(M8)
_KELLY_B = 0.17
_KELLY_C = 0.038


# ---------------------------------------------------------------------------
# Luminosity distance helper
# ---------------------------------------------------------------------------

def _luminosity_distance_mpc(z: float) -> float:
    """
    Compute luminosity distance in Mpc using Planck18 cosmology.
    Falls back to a simple flat ΛCDM approximation if astropy unavailable.
    """
    try:
        from astropy.cosmology import Planck18
        return float(Planck18.luminosity_distance(z).to('Mpc').value)
    except Exception:
        # Flat ΛCDM approximation: H0=67.4, Omega_m=0.315
        H0 = 67.4  # km/s/Mpc
        Omega_m = 0.315
        Omega_L = 1.0 - Omega_m

        if z < 1e-6:
            return 1e-10  # Avoid divide-by-zero

        # Comoving distance via Simpson's rule
        n_steps = 1000
        z_arr = np.linspace(0.0, z, n_steps + 1)
        integrand = 1.0 / np.sqrt(
            Omega_m * (1.0 + z_arr) ** 3 + Omega_L
        )
        dz = z / n_steps
        d_c_mpc = (_C_CGS / 1e5 / H0) * np.trapz(integrand, z_arr)

        return float((1.0 + z) * d_c_mpc)


# ---------------------------------------------------------------------------
# Bolometric luminosity from WISE
# ---------------------------------------------------------------------------

def estimate_lbol_from_wise(w1_mag: float, w2_mag: float, z: float,
                             kappa_ir: float = 8.0) -> float:
    """
    Estimate bolometric luminosity from WISE W1 photometry.

    Method:
        1. Convert W1 magnitude to flux density: F_nu = F_nu0 × 10^(-0.4 × m)
        2. Compute monochromatic luminosity: nu*L_nu = 4π d_L² × nu_W1 × F_nu
        3. Apply K-correction: multiply by (1+z)^(alpha - 1)
           where alpha = 0.5 (typical AGN IR spectral index)
        4. Apply IR bolometric correction: L_bol = kappa_ir × nu*L_nu

    Parameters
    ----------
    w1_mag  : float, W1 Vega magnitude
    w2_mag  : float, W2 Vega magnitude (used for spectral index check)
    z       : float, source redshift
    kappa_ir: float, IR bolometric correction factor (default 8.0)

    Returns
    -------
    L_bol : float, bolometric luminosity in erg/s (NaN on failure)
    """
    if not (np.isfinite(w1_mag) and np.isfinite(z) and z > 0):
        return np.nan

    # Flux density in Jy, then erg/s/cm^2/Hz
    F_nu_jy = _ZP_W1 * 10.0 ** (-0.4 * w1_mag)
    F_nu_cgs = F_nu_jy * 1e-23  # 1 Jy = 1e-23 erg/s/cm^2/Hz

    # Luminosity distance in cm
    d_L_mpc = _luminosity_distance_mpc(z)
    d_L_cm = d_L_mpc * 1e6 * _PC_CM

    # Spectral index from W1-W2 color if available
    if np.isfinite(w2_mag):
        # F_nu ∝ nu^alpha → alpha = -0.4 * (w2_mag - w1_mag) / log10(nu_W2/nu_W1)
        dnu = np.log10(_NU_W2 / _NU_W1)
        if abs(dnu) > 0.01:
            alpha = -0.4 * (w2_mag - w1_mag) / dnu
        else:
            alpha = 0.5
    else:
        alpha = 0.5

    # K-correction: (1+z)^(alpha - 1) for a power-law SED
    k_corr = (1.0 + z) ** (alpha - 1.0)

    # Monochromatic luminosity nu_W1 × L_nu
    nu_L_nu = 4.0 * np.pi * d_L_cm ** 2 * _NU_W1 * F_nu_cgs * k_corr

    L_bol = kappa_ir * nu_L_nu

    return float(L_bol) if np.isfinite(L_bol) else np.nan


# ---------------------------------------------------------------------------
# Black hole mass from DRW
# ---------------------------------------------------------------------------

def estimate_mbh_from_drw(tau_rest_days: float, L_bol: float,
                           tau_reliable: bool = True,
                           A: float = _KELLY_A, B: float = _KELLY_B,
                           C: float = _KELLY_C,
                           kappa_5100: float = 10.3):
    """
    Estimate black hole mass from DRW timescale using Kelly+2009 scaling.

    FIX 11: If tau_reliable is False (Kozlowski+2017 criterion fails),
    return NaN with a flag. Using unreliable tau produces physically
    meaningless M_BH estimates.

    Scaling relation:
        log(tau_days) = A + B * log(L44) + C * log(M8)

    where L44 = L_bol / (10^44 erg/s), M8 = M_BH / (10^8 M_sun).

    Rearranging:
        log(M8) = [log(tau_days) - A - B * log(L44)] / C

    Uses iterative solve (converges within 5 iterations).

    Parameters
    ----------
    tau_rest_days : float, DRW rest-frame timescale in days
    L_bol         : float, bolometric luminosity in erg/s
    tau_reliable  : bool, False if Kozlowski+2017 criterion fails
    A, B, C       : float, Kelly+2009 scaling coefficients
    kappa_5100    : float, 5100A bolometric correction (reserved)

    Returns
    -------
    M_BH_solar : float, black hole mass in solar masses (NaN if unreliable/failed)
    """
    # FIX 11: Gate on reliability
    if not tau_reliable:
        logger.warning(
            "estimate_mbh_from_drw: tau_reliable=False — "
            "M_BH not computed (Kozlowski+2017: tau > baseline/10)"
        )
        return np.nan

    if not (np.isfinite(tau_rest_days) and tau_rest_days > 0 and
            np.isfinite(L_bol) and L_bol > 0):
        return np.nan

    log_tau = np.log10(max(tau_rest_days, 1.0))
    log_L44 = np.log10(max(L_bol / 1e44, 1e-10))

    # Iterative solution
    log_M8 = 0.0  # Initial guess: M_BH = 10^8 Msun

    for _ in range(5):
        log_tau_pred = A + B * log_L44 + C * log_M8
        residual = log_tau - log_tau_pred
        log_M8 = log_M8 + residual / C

    M_BH_solar = (10.0 ** log_M8) * 1e8

    if not np.isfinite(M_BH_solar) or M_BH_solar <= 0:
        return np.nan

    return float(M_BH_solar)


# ---------------------------------------------------------------------------
# Eddington ratio
# ---------------------------------------------------------------------------

def estimate_eddington_ratio(L_bol: float, M_BH_solar: float) -> float:
    """
    Compute Eddington ratio: lambda_Edd = L_bol / L_Eddington.

    L_Edd = 1.26e38 × (M_BH / M_sun) erg/s

    Parameters
    ----------
    L_bol      : float, bolometric luminosity (erg/s)
    M_BH_solar : float, black hole mass (M_sun)

    Returns
    -------
    lambda_Edd : float (NaN on failure)
    """
    if not (np.isfinite(L_bol) and np.isfinite(M_BH_solar) and M_BH_solar > 0):
        return np.nan

    L_edd = _L_EDD_PER_MSUN * M_BH_solar
    lambda_edd = L_bol / L_edd

    return float(lambda_edd) if np.isfinite(lambda_edd) else np.nan


# ---------------------------------------------------------------------------
# Dust sublimation radii
# ---------------------------------------------------------------------------

def estimate_sublimation_radii(L_bol: float) -> dict:
    """
    Estimate dust sublimation radii for graphite and silicate grains.

    From Barvainis 1987 and Kishimoto et al. 2007:
        R_sub_graphite = 0.5 * (L_bol / 1e46)^0.5  pc
        R_sub_silicate = 1.3 * (L_bol / 1e46)^0.5  pc

    The graphite sublimation temperature is ~1500 K.
    The silicate sublimation temperature is ~1000 K.

    Parameters
    ----------
    L_bol : float, bolometric luminosity in erg/s

    Returns
    -------
    result : dict with keys:
        graphite_pc, silicate_pc, graphite_ld (light-days), silicate_ld
    """
    if not (np.isfinite(L_bol) and L_bol > 0):
        return {
            'graphite_pc': np.nan, 'silicate_pc': np.nan,
            'graphite_ld': np.nan, 'silicate_ld': np.nan,
        }

    L_norm = L_bol / 1e46

    r_graphite_pc = 0.5 * np.sqrt(L_norm)
    r_silicate_pc = 1.3 * np.sqrt(L_norm)

    # Convert parsec to light-days: 1 pc = 3.26 ly = 1190 ld
    pc_to_ld = 1190.0

    return {
        'graphite_pc': float(r_graphite_pc),
        'silicate_pc': float(r_silicate_pc),
        'graphite_ld': float(r_graphite_pc * pc_to_ld),
        'silicate_ld': float(r_silicate_pc * pc_to_ld),
    }


# ---------------------------------------------------------------------------
# BLR radius
# ---------------------------------------------------------------------------

def estimate_blr_radius(L_bol: float, kappa_x: float = 20.0) -> float:
    """
    Estimate BLR radius from the radius-luminosity relation.

    Uses the X-ray luminosity proxy:
        L_X = L_bol / kappa_x
        R_BLR = 7.2e-3 * (L_X / 1e43)^0.532  pc   (Bentz+2013 extrapolation)

    Parameters
    ----------
    L_bol   : float, bolometric luminosity in erg/s
    kappa_x : float, X-ray bolometric correction (default 20.0)

    Returns
    -------
    R_BLR : float, BLR radius in parsecs (NaN on failure)
    """
    if not (np.isfinite(L_bol) and L_bol > 0 and kappa_x > 0):
        return np.nan

    L_X = L_bol / kappa_x
    R_BLR_pc = 7.2e-3 * (L_X / 1e43) ** 0.532

    return float(R_BLR_pc) if np.isfinite(R_BLR_pc) else np.nan


# ---------------------------------------------------------------------------
# Disk timescales
# ---------------------------------------------------------------------------

def estimate_disk_timescales(M_BH_solar: float, r_rg: float = 150.0) -> dict:
    """
    Compute characteristic accretion disk timescales at r_rg gravitational radii.

    Timescales (Shakura & Sunyaev 1973; Frank, King & Raine 2002):
        r_g  = GM/c^2 (gravitational radius in cm)
        R    = r_rg * r_g  (physical radius)

        t_light_crossing  = R / c
        t_dynamical       = (R^3 / GM)^0.5  [orbital period / 2π]
        t_thermal         = t_dynamical / alpha  (viscosity parameter alpha=0.1)
        t_viscous         = t_thermal * (R/H)^2  [H/R = 0.05 assumed]
        t_propagation_front = t_viscous  (same order)

    All timescales returned in days.

    Parameters
    ----------
    M_BH_solar : float, black hole mass in solar masses
    r_rg       : float, radius in units of r_g (default 150)

    Returns
    -------
    result : dict with timescale names (days)
    """
    if not (np.isfinite(M_BH_solar) and M_BH_solar > 0):
        return {
            't_light_crossing': np.nan, 't_dynamical': np.nan,
            't_thermal': np.nan, 't_propagation_front': np.nan,
            't_viscous': np.nan,
        }

    M_g = M_BH_solar * _MSUN_G
    r_g_cm = _G_CGS * M_g / (_C_CGS ** 2)
    R_cm = r_rg * r_g_cm

    # Light crossing time
    t_light_s = R_cm / _C_CGS

    # Dynamical (Keplerian) timescale
    t_dyn_s = np.sqrt(R_cm ** 3 / (_G_CGS * M_g))

    # Thermal timescale (alpha = 0.1 viscosity parameter)
    alpha_visc = 0.1
    t_therm_s = t_dyn_s / alpha_visc

    # Viscous timescale: t_visc = t_thermal × (R/H)^2
    h_r_ratio = 0.05  # H/R for thin disk
    r_over_h = 1.0 / h_r_ratio
    t_visc_s = t_therm_s * r_over_h ** 2

    # Propagation front timescale (same order as viscous for standard disk)
    t_prop_s = t_visc_s

    def _s_to_days(t_s):
        return float(t_s / 86400.0)

    return {
        't_light_crossing': _s_to_days(t_light_s),
        't_dynamical': _s_to_days(t_dyn_s),
        't_thermal': _s_to_days(t_therm_s),
        't_propagation_front': _s_to_days(t_prop_s),
        't_viscous': _s_to_days(t_visc_s),
    }


# ---------------------------------------------------------------------------
# Mechanism identification
# ---------------------------------------------------------------------------

def identify_mechanism(t_obs_days: float, timescales: dict) -> str:
    """
    Identify the most likely physical mechanism driving the variability
    by comparing the observed transition timescale to disk timescales.

    Rules:
        t_obs < 2 × t_light_crossing   → 'disk_instability' (very fast)
        t_obs < 5 × t_thermal          → 'disk_instability'
        t_obs < 2 × t_viscous          → 'accretion_rate_change'
        t_obs < 5 × t_viscous          → 'viscous_transition'
        otherwise                       → 'external_trigger'

    Parameters
    ----------
    t_obs_days : float, observed transition duration in days
    timescales : dict, output of estimate_disk_timescales

    Returns
    -------
    mechanism : str
    """
    t_lc = timescales.get('t_light_crossing', np.nan)
    t_th = timescales.get('t_thermal', np.nan)
    t_vi = timescales.get('t_viscous', np.nan)

    if not np.isfinite(t_obs_days):
        return 'unknown'

    if np.isfinite(t_lc) and t_obs_days < 2.0 * t_lc:
        return 'disk_instability'

    if np.isfinite(t_th) and t_obs_days < 5.0 * t_th:
        return 'disk_instability'

    if np.isfinite(t_vi):
        if t_obs_days < 2.0 * t_vi:
            return 'accretion_rate_change'
        if t_obs_days < 5.0 * t_vi:
            return 'viscous_transition'

    return 'external_trigger'


# ---------------------------------------------------------------------------
# Dust temperature
# ---------------------------------------------------------------------------

def estimate_dust_temperature(w1_mag: float, w2_mag: float) -> float:
    """
    Estimate dust temperature from the W1-W2 color.

    Assuming Rayleigh-Jeans tail of a blackbody:
        F_nu ∝ T × nu^2   →   F_W1/F_W2 = (nu_W1/nu_W2)^2
    More precisely, for a Wien blackbody:
        B_nu ∝ nu^3 × exp(-hν/kT)  →  T_dust from color ratio

    Approximate analytical formula (valid 500 < T < 2000 K):
        T_dust = 2000 × (F_W1/F_W2)^0.42  K

    Parameters
    ----------
    w1_mag : float, W1 Vega magnitude
    w2_mag : float, W2 Vega magnitude

    Returns
    -------
    T_dust : float, dust temperature in Kelvin (NaN on failure)
    """
    if not (np.isfinite(w1_mag) and np.isfinite(w2_mag)):
        return np.nan

    F_W1 = _ZP_W1 * 10.0 ** (-0.4 * w1_mag)
    F_W2 = _ZP_W2 * 10.0 ** (-0.4 * w2_mag)

    if F_W2 <= 0:
        return np.nan

    T_dust = 2000.0 * (F_W1 / F_W2) ** 0.42

    if not (100.0 < T_dust < 5000.0):
        logger.debug(f"T_dust={T_dust:.1f} K outside expected range [100, 5000]")

    return float(T_dust) if np.isfinite(T_dust) else np.nan


# ---------------------------------------------------------------------------
# Comprehensive physical parameter estimation
# ---------------------------------------------------------------------------

def estimate_all_physical_parameters(source: dict, drw_results: dict,
                                      cp_results: dict,
                                      wise_lc: dict) -> dict:
    """
    Compute all physical parameters for a single CLAGN candidate.

    Parameters
    ----------
    source      : dict with at minimum keys: ra, dec, z, w1_mag, w2_mag
    drw_results : dict, output of fit_drw_full_mcmc or fit_drw_map
    cp_results  : dict, output of changepoint detection
    wise_lc     : dict, WISE light curve result

    Returns
    -------
    params : dict with all derived physical parameters
    """
    z = float(source.get('z', source.get('redshift', 0.1)))
    w1_mag = float(source.get('w1_mag', source.get('w1_mag_median', np.nan)))
    w2_mag = float(source.get('w2_mag', source.get('w2_mag_median', np.nan)))

    # Use light curve median magnitudes if available
    lc_df = wise_lc.get('lc', pd.DataFrame()) if isinstance(wise_lc, dict) else pd.DataFrame()
    if not lc_df.empty:
        if 'w1_mag' in lc_df.columns:
            w1_vals = pd.to_numeric(lc_df['w1_mag'], errors='coerce').dropna()
            if len(w1_vals) > 0:
                w1_mag = float(np.nanmedian(w1_vals))
        if 'w2_mag' in lc_df.columns:
            w2_vals = pd.to_numeric(lc_df['w2_mag'], errors='coerce').dropna()
            if len(w2_vals) > 0:
                w2_mag = float(np.nanmedian(w2_vals))

    # ---- Bolometric luminosity -----------------------------------------------
    L_bol = estimate_lbol_from_wise(w1_mag, w2_mag, z)

    # ---- Black hole mass -------------------------------------------------------
    tau_rest = float(drw_results.get('tau_rest_days', np.nan))
    # FIX 11: Thread tau_reliable from DRW result to gate M_BH estimation
    tau_reliable = bool(drw_results.get('tau_reliable', True))
    M_BH_solar = estimate_mbh_from_drw(tau_rest, L_bol, tau_reliable=tau_reliable)

    # ---- Eddington ratio -------------------------------------------------------
    lambda_edd = estimate_eddington_ratio(L_bol, M_BH_solar)

    # ---- Sublimation radii ----------------------------------------------------
    sub_radii = estimate_sublimation_radii(L_bol)

    # ---- BLR radius -----------------------------------------------------------
    R_BLR_pc = estimate_blr_radius(L_bol)

    # ---- Disk timescales -------------------------------------------------------
    timescales = estimate_disk_timescales(M_BH_solar)

    # ---- Mechanism ------------------------------------------------------------
    t_break_duration = float(cp_results.get('segment_duration_days', np.nan)
                             if cp_results else np.nan)
    mechanism = identify_mechanism(t_break_duration, timescales)

    # ---- Dust temperature -------------------------------------------------------
    T_dust = estimate_dust_temperature(w1_mag, w2_mag)

    # ---- Dust temperature evolution --------------------------------------------
    dust_evol = compute_dust_temperature_evolution(wise_lc, z)

    logger.info(
        f"Physics: L_bol={L_bol:.2e} erg/s | "
        f"M_BH={M_BH_solar:.2e} Msun | "
        f"lambda_Edd={lambda_edd:.4f} | "
        f"T_dust={T_dust:.1f} K | "
        f"mechanism={mechanism}"
    )

    return {
        'z': z,
        'w1_mag_median': float(w1_mag),
        'w2_mag_median': float(w2_mag),
        'L_bol_erg_s': float(L_bol) if np.isfinite(L_bol) else np.nan,
        'log_L_bol': float(np.log10(L_bol)) if (np.isfinite(L_bol) and L_bol > 0) else np.nan,
        'M_BH_solar': float(M_BH_solar) if np.isfinite(M_BH_solar) else np.nan,
        'log_M_BH': float(np.log10(M_BH_solar)) if (np.isfinite(M_BH_solar) and M_BH_solar > 0) else np.nan,
        'lambda_Edd': float(lambda_edd) if np.isfinite(lambda_edd) else np.nan,
        'log_lambda_Edd': float(np.log10(lambda_edd)) if (np.isfinite(lambda_edd) and lambda_edd > 0) else np.nan,
        'R_sub_graphite_pc': sub_radii['graphite_pc'],
        'R_sub_silicate_pc': sub_radii['silicate_pc'],
        'R_sub_graphite_ld': sub_radii['graphite_ld'],
        'R_sub_silicate_ld': sub_radii['silicate_ld'],
        'R_BLR_pc': float(R_BLR_pc) if np.isfinite(R_BLR_pc) else np.nan,
        't_light_crossing_days': timescales['t_light_crossing'],
        't_dynamical_days': timescales['t_dynamical'],
        't_thermal_days': timescales['t_thermal'],
        't_propagation_front_days': timescales['t_propagation_front'],
        't_viscous_days': timescales['t_viscous'],
        'tau_rest_days': float(tau_rest),
        'tau_reliable': tau_reliable,
        'mechanism': mechanism,
        'T_dust_K': float(T_dust) if np.isfinite(T_dust) else np.nan,
        'T_dust_pre_K': dust_evol.get('T_dust_pre', np.nan),
        'T_dust_post_K': dust_evol.get('T_dust_post', np.nan),
        'delta_T_dust_K': dust_evol.get('delta_T_dust', np.nan),
    }


# ---------------------------------------------------------------------------
# Dust temperature evolution
# ---------------------------------------------------------------------------

def compute_dust_temperature_evolution(wise_lc: dict, z: float) -> dict:
    """
    Compute how the dust temperature evolved before, during, and after
    the CLAGN transition.

    Splits the light curve into three epochs:
        pre       : first 30% of baseline
        transition: middle 40%
        post      : last 30%

    Computes median W1-W2 color → T_dust in each epoch.

    Parameters
    ----------
    wise_lc : dict, output of query_wise_lightcurve or query_all_wise_epochs
    z       : float, source redshift

    Returns
    -------
    result : dict with keys:
        T_dust_pre, T_dust_transition, T_dust_post, delta_T_dust
        w1w2_pre, w1w2_transition, w1w2_post
    """
    lc_df = wise_lc.get('lc', pd.DataFrame()) if isinstance(wise_lc, dict) else pd.DataFrame()

    _empty = {
        'T_dust_pre': np.nan, 'T_dust_transition': np.nan,
        'T_dust_post': np.nan, 'delta_T_dust': np.nan,
        'w1w2_pre': np.nan, 'w1w2_transition': np.nan, 'w1w2_post': np.nan,
    }

    if lc_df.empty:
        return _empty

    # Ensure we have color data
    if 'w1_mag' not in lc_df.columns or 'w2_mag' not in lc_df.columns:
        return _empty

    mjd = pd.to_numeric(lc_df['mjd'], errors='coerce').values
    w1 = pd.to_numeric(lc_df['w1_mag'], errors='coerce').values
    w2 = pd.to_numeric(lc_df['w2_mag'], errors='coerce').values

    valid = np.isfinite(mjd) & np.isfinite(w1) & np.isfinite(w2)
    if valid.sum() < 6:
        return _empty

    mjd = mjd[valid]
    w1 = w1[valid]
    w2 = w2[valid]

    sort_idx = np.argsort(mjd)
    mjd = mjd[sort_idx]
    w1 = w1[sort_idx]
    w2 = w2[sort_idx]

    n = len(mjd)
    n_pre = max(1, int(0.30 * n))
    n_post = max(1, int(0.30 * n))
    n_trans_start = n_pre
    n_trans_end = n - n_post

    w1_pre = w1[:n_pre]
    w2_pre = w2[:n_pre]
    w1_trans = w1[n_trans_start:n_trans_end]
    w2_trans = w2[n_trans_start:n_trans_end]
    w1_post = w1[-n_post:]
    w2_post = w2[-n_post:]

    def _epoch_T_dust(w1e, w2e):
        med_w1 = float(np.nanmedian(w1e))
        med_w2 = float(np.nanmedian(w2e))
        return estimate_dust_temperature(med_w1, med_w2), float(med_w1 - med_w2)

    T_pre, c_pre = _epoch_T_dust(w1_pre, w2_pre)
    T_trans, c_trans = _epoch_T_dust(w1_trans, w2_trans) if len(w1_trans) > 0 else (np.nan, np.nan)
    T_post, c_post = _epoch_T_dust(w1_post, w2_post)

    delta_T = float(T_post - T_pre) if (np.isfinite(T_pre) and np.isfinite(T_post)) else np.nan

    return {
        'T_dust_pre': float(T_pre) if np.isfinite(T_pre) else np.nan,
        'T_dust_transition': float(T_trans) if np.isfinite(T_trans) else np.nan,
        'T_dust_post': float(T_post) if np.isfinite(T_post) else np.nan,
        'delta_T_dust': float(delta_T),
        'w1w2_pre': float(c_pre),
        'w1w2_transition': float(c_trans) if np.isfinite(c_trans) else np.nan,
        'w1w2_post': float(c_post),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_physics_analysis(results_dir: str = './results/') -> None:
    """
    Run full physical parameter analysis for all top candidates.

    Loads:
        - top_candidates.csv
        - advanced_drw_{source_id}.pkl (or drw fallback)
        - enhanced_wise_{source_id}.pkl (or wise fallback)
        - changepoint results (if available)

    Saves:
        - physics_{source_id}.pkl
        - physics_summary.csv
        - physics_stats.json

    Parameters
    ----------
    results_dir : str, path to results directory
    """
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

    physics_dir = results_path / 'physics'
    physics_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for idx, row in candidates.iterrows():
        source_id = str(row.get('source_id', f'source_{idx}'))

        logger.info(f"Physics analysis for {source_id} ({idx + 1}/{len(candidates)})")

        # Load DRW results
        drw_results = {}
        for drw_path in [
            results_path / 'advanced_drw' / f'advanced_drw_{source_id}.pkl',
            results_path / f'drw_{source_id}.pkl',
        ]:
            if drw_path.exists():
                with open(drw_path, 'rb') as f:
                    drw_data = pickle.load(f)
                drw_results = drw_data.get('mcmc', drw_data)
                break

        # Load WISE light curve
        wise_lc = {}
        for wise_path in [
            results_path / 'wise_enhanced' / f'enhanced_wise_{source_id}.pkl',
            results_path / f'wise_{source_id}.pkl',
        ]:
            if wise_path.exists():
                with open(wise_path, 'rb') as f:
                    wise_lc = pickle.load(f)
                break

        # Load changepoint results
        cp_results = {}
        cp_path = results_path / f'changepoint_{source_id}.pkl'
        if cp_path.exists():
            with open(cp_path, 'rb') as f:
                cp_results = pickle.load(f)

        try:
            params = estimate_all_physical_parameters(
                source=row.to_dict(),
                drw_results=drw_results,
                cp_results=cp_results,
                wise_lc=wise_lc,
            )
            params['source_id'] = source_id

            # Save per-source
            pkl_path = physics_dir / f'physics_{source_id}.pkl'
            with open(pkl_path, 'wb') as f:
                pickle.dump(params, f, protocol=4)

            summary_rows.append(params)

        except Exception as exc:
            logger.error(f"{source_id}: Physics analysis failed: {exc}", exc_info=True)
            summary_rows.append({'source_id': source_id, 'error': str(exc)})

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = results_path / 'physics_summary.csv'
        summary_df.to_csv(summary_path, index=False)

        # JSON stats
        numeric_cols = summary_df.select_dtypes(include=np.number).columns
        stats = {
            col: {
                'median': float(summary_df[col].median()),
                'std': float(summary_df[col].std()),
                'min': float(summary_df[col].min()),
                'max': float(summary_df[col].max()),
            }
            for col in numeric_cols
            if summary_df[col].notna().sum() > 0
        }
        json_path = results_path / 'physics_stats.json'
        with open(json_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(
            f"Physics analysis complete. "
            f"{len(summary_rows)} sources processed. "
            f"Summary: {summary_path}"
        )
    else:
        logger.warning("No physics results to save.")
