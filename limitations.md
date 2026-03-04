11.1 Selection Function
  - We are sensitive only to transitions of |Δmag| > {DETECTION_THRESHOLD_MAG} (from injection-recovery)
  - We are insensitive to transitions shorter than ~1 WISE season (6 months)
  - We are insensitive to transitions longer than our baseline (~12 years)
  - Coverage gaps: WISE hibernation 2011-2013 creates a blind period

11.2 Contamination
  - Nuclear SNe are our dominant photometric contaminant
  - We estimate residual SN contamination rate of {SN_CONTAMINATION_RATE}% (from control sample)
  - Blazars: estimated contamination {BLAZAR_CONTAMINATION_RATE}% after radio crossmatch filter
  - Dust obscuration events: cannot be fully distinguished photometrically

11.3 Redshift Dependence
  - Host galaxy contamination increases at low z (z < 0.05)
  - Completeness decreases at high z (z > 0.5) due to flux limit
  - No K-correction applied to WISE photometry

11.4 WISE-Specific Limitations
  - Spatial resolution ~6 arcsec — cannot resolve nuclear vs. off-nuclear events
  - Only two IR bands — limited SED information
  - WISE calibration uncertainty ~2.8% limits amplitude precision below ~0.03 mag

11.5 DRW Model Limitations
  - DRW is a stationary GP — misspecified for true CLAGN transitions
  - τ measurement unreliable when baseline < 10τ
  - DRW nonstationarity statistic is heuristically calibrated
    (simulation calibration provided for top candidates only)
