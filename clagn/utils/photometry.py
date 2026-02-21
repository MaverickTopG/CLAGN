"""
photometry.py — Magnitude/flux conversion, sigma clipping, epoch weighting.

All DRW fitting must be done in flux space (mJy). Magnitude errors are
asymmetric and heteroscedastic; flux errors are approximately Gaussian.
"""
import logging
import numpy as np
from astropy.stats import sigma_clip as astropy_sigma_clip

from ..config import (
    WISE_ZERO_POINTS,
    SIGMA_CLIP_SIGMA, SIGMA_CLIP_ITERS,
    HOST_CONTAMINATION_Z_THRESHOLD,
)

logger = logging.getLogger(__name__)


def mag_to_flux_mjy(mag, mag_err, band):
    """
    Convert Vega magnitudes to flux density in mJy.

    Uses WISE zero points from config (Jarrett et al. 2011).
    Propagates errors correctly.

    flux_mjy     = F0_jy * 10^(-mag/2.5) * 1000
    flux_err_mjy = flux_mjy * ln(10)/2.5 * mag_err

    Parameters
    ----------
    mag     : float or array
    mag_err : float or array
    band    : str, one of 'W1','W2','W3','W4'

    Returns
    -------
    flux_mjy, flux_err_mjy : float or array
    """
    mag = np.asarray(mag, dtype=float)
    mag_err = np.asarray(mag_err, dtype=float)

    F0 = WISE_ZERO_POINTS[band]   # Jy
    flux_jy = F0 * 10.0 ** (-mag / 2.5)
    flux_mjy = flux_jy * 1000.0
    flux_err_mjy = flux_mjy * (np.log(10.0) / 2.5) * np.abs(mag_err)

    # Preserve NaN positions from the input
    bad = ~np.isfinite(mag) | ~np.isfinite(mag_err)
    flux_mjy[bad] = np.nan
    flux_err_mjy[bad] = np.nan

    return flux_mjy, flux_err_mjy


def flux_to_mag(flux_mjy, flux_err_mjy, band):
    """
    Convert flux density (mJy) back to Vega magnitudes.

    Used by structure_function.py which operates in magnitude space.

    Parameters
    ----------
    flux_mjy     : float or array
    flux_err_mjy : float or array
    band         : str

    Returns
    -------
    mag, mag_err : float or array
    """
    flux_mjy = np.asarray(flux_mjy, dtype=float)
    flux_err_mjy = np.asarray(flux_err_mjy, dtype=float)

    F0 = WISE_ZERO_POINTS[band]   # Jy
    flux_jy = flux_mjy / 1000.0
    # Guard against non-positive flux
    with np.errstate(divide='ignore', invalid='ignore'):
        mag = -2.5 * np.log10(flux_jy / F0)
        mag_err = (2.5 / np.log(10.0)) * np.abs(flux_err_mjy / flux_mjy)

    bad = (flux_mjy <= 0) | ~np.isfinite(flux_mjy)
    mag[bad] = np.nan
    mag_err[bad] = np.nan

    return mag, mag_err


def compute_epoch_weights(snr_w1, snr_w2, moon_sep, ecl_lat=None):
    """
    Compute inverse-variance weights for each WISE epoch.

    Accounts for:
    - SNR (higher SNR → higher weight)
    - Moon proximity (closer to moon → downweight)
    - Ecliptic latitude (informational; nearby ecliptic = more visits but not
      higher per-visit quality — not used in the weight formula)

    Parameters
    ----------
    snr_w1   : array
    snr_w2   : array
    moon_sep : array, degrees from moon
    ecl_lat  : array or None (unused but kept for API completeness)

    Returns
    -------
    weights : array, shape (N,), non-negative (not normalized to sum=1)
    """
    snr_w1 = np.asarray(snr_w1, dtype=float)
    snr_w2 = np.asarray(snr_w2, dtype=float)
    moon_sep = np.asarray(moon_sep, dtype=float)

    snr_weight = 0.5 * (snr_w1 + snr_w2)
    moon_weight = np.clip((moon_sep - 10.0) / 90.0, 0.1, 1.0)

    return snr_weight * moon_weight


def rolling_median_flux(times, fluxes, window_years=1.0):
    """
    Compute a rolling median of fluxes over a window of `window_years` years.

    Used to measure long-term amplitude change robustly (suppresses short flares).

    Parameters
    ----------
    times        : array, MJD
    fluxes       : array, flux in mJy
    window_years : float

    Returns
    -------
    rolling_med : array, same length as input (edges padded with NaN)
    """
    times = np.asarray(times, dtype=float)
    fluxes = np.asarray(fluxes, dtype=float)
    window_days = window_years * 365.25
    rolling_med = np.full_like(fluxes, np.nan)

    for i, t in enumerate(times):
        mask = np.abs(times - t) <= window_days / 2.0
        if mask.sum() >= 3:
            rolling_med[i] = np.nanmedian(fluxes[mask])

    return rolling_med


def check_host_contamination(z, w1_flux_mjy=None):
    """
    At z < 0.05, the WISE PSF (6 arcsec FWHM) can blend AGN and host galaxy light.
    Flag sources where host contamination may inflate apparent variability.

    Parameters
    ----------
    z            : float, source redshift
    w1_flux_mjy  : float or None (unused, kept for future host subtraction)

    Returns
    -------
    risk : bool
        True if z < HOST_CONTAMINATION_Z_THRESHOLD (0.05)
    """
    return float(z) < HOST_CONTAMINATION_Z_THRESHOLD


def sigma_clip_lightcurve(times, fluxes, flux_errors,
                           sigma=SIGMA_CLIP_SIGMA, maxiters=SIGMA_CLIP_ITERS):
    """
    Apply sigma clipping to remove outlier epochs before DRW fitting.

    Uses astropy.stats.sigma_clip (sigma=4, maxiters=3 by default).
    Clipped epochs are removed entirely — do NOT replace with median.

    Parameters
    ----------
    times       : array
    fluxes      : array
    flux_errors : array
    sigma       : float (default 4.0 from config)
    maxiters    : int   (default 3 from config)

    Returns
    -------
    t_clean, f_clean, e_clean : arrays with outliers removed
    clip_mask : bool array, True where epochs were KEPT
    """
    fluxes = np.asarray(fluxes, dtype=float)
    clipped = astropy_sigma_clip(fluxes, sigma=sigma, maxiters=maxiters, masked=True)
    keep = ~clipped.mask

    n_removed = np.sum(~keep)
    if n_removed > 0:
        logger.debug(f"Sigma clipping removed {n_removed}/{len(fluxes)} epochs")

    return (np.asarray(times)[keep],
            np.asarray(fluxes)[keep],
            np.asarray(flux_errors)[keep],
            keep)
