"""
test_photometry_units.py — Unit tests for WISE photometry conversions.

All values derived from official WISE formula: F = F0 * 10^(-0.4*mag)
W1 zero point: 309.540 Jy (Wright+2010 Table 1)
W2 zero point: 171.787 Jy (Wright+2010 Table 1)

At mag=13:
  W1: F = 309.540e3 mJy * 10^(-0.4*13) = 1.953 mJy
  W2: F = 171.787e3 mJy * 10^(-0.4*13) = 1.083 mJy
"""
import numpy as np
from clagn.ingestion.catalog import mag_to_flux_mjy, flux_mjy_to_mag


def test_w1_flux_conversion():
    """Values computed from official WISE formula: F = F0 * 10^(-0.4*mag)"""
    # W1 mag=13 → F = 309.540e3 mJy * 10^(-0.4*13) = 1.953 mJy
    F, dF = mag_to_flux_mjy(13.0, 0.02, 'W1')
    assert 1.90 < F < 2.00, f"W1 mag=13 flux wrong: {F:.4f} mJy (expected ~1.953)"
    assert 0.034 < dF < 0.040, f"W1 mag=13 flux error wrong: {dF:.4f} mJy"


def test_w2_flux_conversion():
    # W2 mag=13 → F = 171.787e3 mJy * 10^(-0.4*13) = 1.083 mJy
    F, dF = mag_to_flux_mjy(13.0, 0.02, 'W2')
    assert 1.05 < F < 1.12, f"W2 mag=13 flux wrong: {F:.4f} mJy (expected ~1.083)"


def test_roundtrip():
    """Convert mag→flux→mag and get back original value"""
    mag_in = 14.5
    F, dF = mag_to_flux_mjy(mag_in, 0.05, 'W1')
    mag_out, dmag_out = flux_mjy_to_mag(F, dF, 'W1')
    assert abs(mag_out - mag_in) < 1e-6, f"Roundtrip failed: {mag_in} → {mag_out}"


def test_brighter_magnitude_larger_flux():
    """Brighter (smaller mag number) must give larger flux"""
    F_bright, _ = mag_to_flux_mjy(12.0, 0.02, 'W1')
    F_faint,  _ = mag_to_flux_mjy(14.0, 0.02, 'W1')
    assert F_bright > F_faint, "Brighter source must have larger flux"


if __name__ == '__main__':
    test_w1_flux_conversion()
    test_w2_flux_conversion()
    test_roundtrip()
    test_brighter_magnitude_larger_flux()
    print("ALL PHOTOMETRY UNIT TESTS PASSED")
    print(f"  W1 mag=13 → {mag_to_flux_mjy(13.0, 0.02, 'W1')[0]:.4f} mJy")
    print(f"  W2 mag=13 → {mag_to_flux_mjy(13.0, 0.02, 'W2')[0]:.4f} mJy")
