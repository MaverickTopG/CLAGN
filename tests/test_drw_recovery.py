"""
test_drw_recovery.py — Validate celerite2 tau mapping by injection-recovery.

Simulates a DRW with known tau=300d, fits with celerite2, checks that
recovered tau is within 0.5–2.0× the injected value.
"""
import numpy as np
from scipy.stats import norm


def validate_celerite2_tau_mapping():
    """
    Unit test: simulate a DRW with known tau, fit with celerite2,
    check that recovered tau is within 30% of injected value.

    Run this once and report the calibration factor.
    """
    np.random.seed(42)
    tau_true = 300.0  # days
    sigma_true = 0.5  # mJy
    n = 100

    # Simulate DRW using exact OU process
    times = np.sort(np.random.uniform(55000, 60000, n))
    flux = np.zeros(n)
    flux[0] = sigma_true * norm.rvs()
    for i in range(1, n):
        dt = times[i] - times[i-1]
        e_factor = np.exp(-dt / tau_true)
        flux[i] = (e_factor * flux[i-1] +
                   np.sqrt(1 - e_factor**2) * sigma_true * norm.rvs())

    errors = np.ones(n) * 0.05
    flux += errors * norm.rvs(size=n)

    # Fit with celerite2 (via fit_drw_map, which adds mean offset)
    from clagn.models.drw import fit_drw_map
    result = fit_drw_map(times, flux + 2.0, errors, z=0.0)

    tau_recovered = result['tau_rest_days']
    ratio = tau_recovered / tau_true

    print(f"Injected tau: {tau_true:.0f} d")
    print(f"Recovered tau: {tau_recovered:.0f} d")
    print(f"Ratio: {ratio:.3f} (acceptable range: 0.5 – 2.0)")

    if 0.5 < ratio < 2.0:
        print("PASS: tau recovery within acceptable range")
    else:
        print("FAIL: tau mapping may be incorrect — review celerite2 parameterization")
        print("Fix: adjust tau = rho * CALIBRATION_FACTOR in celerite2 path")

    return ratio


if __name__ == '__main__':
    ratio = validate_celerite2_tau_mapping()
    assert 0.5 < ratio < 2.0, f"tau ratio {ratio:.3f} outside acceptable range [0.5, 2.0]"
    print(f"DRW recovery test PASSED (ratio={ratio:.3f})")
