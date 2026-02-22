"""
clagn.analysis — Physical characterization and statistical analysis of CLAGN candidates.

Submodules
----------
physics
    Derives physical parameters (M_BH, L_bol, lambda_Edd, dust temperature,
    disk timescales) from the WISE+Gaia photometric data and DRW fits.

validation
    Injection-recovery tests, false positive rate estimation, jackknife
    stability analysis, and coordinate-scramble null tests.

population
    Population statistics comparing CLAGN candidates to the parent AGN
    sample: occurrence rates, Spearman correlations, KS tests.

source_narrative
    Generates human-readable, science-fair-grade narratives for each candidate
    by auto-filling templated paragraphs from the pipeline results dict.
"""
