# Score Definition (Template)

## 1. What the Current Score Is

The current `score` is treated as a **ranking statistic**, not a calibrated probability.
It is used to rank CLAGN candidates for triage, inspection, and follow-up prioritization.

## 2. What the Score Is Not

- It is **not** a posterior probability that a source is a CLAGN.
- It is **not** a direct estimate of a physical parameter (e.g., accretion rate change).
- It should **not** be interpreted probabilistically unless calibrated against independent labels.

## 3. Current Usage

- Rank candidates for manual review.
- Apply quality flags and coverage filters.
- Select subsets for crossmatch, light-curve inspection, and spectroscopy prioritization.

## 4. Calibration Roadmap (Future Work)

- Injection-recovery characterization across cadence, amplitude, and baseline.
- Reliability calibration using independent labeled benchmarks.
- Probability calibration (e.g., isotonic regression or Platt scaling) on a dev set only.
- Locked threshold evaluation on a held-out test set with frozen benchmark hashes.

## 5. Reporting Rules

- Do not describe the current score as a probability.
- Report thresholds together with validation context and selection effects.
- Report uncertainty / confidence intervals for performance metrics when labels are available.

## 6. Versioning Note

Record the scoring pipeline version, configuration hash, and benchmark hash in all derived results.
