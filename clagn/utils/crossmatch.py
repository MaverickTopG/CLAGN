"""
crossmatch.py — Positional cross-matching utilities.
"""
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u


def angular_separation(ra1, dec1, ra2, dec2):
    """
    Compute angular separation between two sky positions.

    Parameters
    ----------
    ra1, dec1 : float, degrees
    ra2, dec2 : float or array, degrees

    Returns
    -------
    sep_arcsec : float or array, arcseconds
    """
    c1 = SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg, frame='icrs')
    c2 = SkyCoord(ra=np.atleast_1d(ra2) * u.deg,
                  dec=np.atleast_1d(dec2) * u.deg, frame='icrs')
    sep = c1.separation(c2).to(u.arcsec).value
    return sep[0] if np.ndim(ra2) == 0 else sep


def find_best_match(source_ra, source_dec, catalog_ra, catalog_dec,
                    max_sep_arcsec):
    """
    Find the best positional match within a search radius.

    Parameters
    ----------
    source_ra, source_dec : float, degrees (the source to match)
    catalog_ra, catalog_dec : array, degrees (the catalog to search)
    max_sep_arcsec : float

    Returns
    -------
    best_idx : int or None
        Index into catalog arrays of best match within radius.
        None if no match found.
    best_sep_arcsec : float or None
    """
    catalog_ra = np.asarray(catalog_ra)
    catalog_dec = np.asarray(catalog_dec)

    if len(catalog_ra) == 0:
        return None, None

    seps = angular_separation(source_ra, source_dec, catalog_ra, catalog_dec)
    min_idx = np.argmin(seps)
    min_sep = seps[min_idx]

    if min_sep <= max_sep_arcsec:
        return int(min_idx), float(min_sep)
    return None, None


def galactic_latitude(ra, dec):
    """
    Compute Galactic latitude b for a sky position.

    Parameters
    ----------
    ra, dec : float, degrees (ICRS)

    Returns
    -------
    b : float, Galactic latitude in degrees
    """
    c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
    return float(c.galactic.b.deg)
