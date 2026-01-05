# AGN Pulse - Enterprise Light-Curve Lab

This frontend is a judge-ready console for IRSA-compatible AGN light-curve datasets. It preserves the BrandPulse visual style while presenting AGN candidate scoring, multi-band light curves, and cross-survey metadata.

## Features
- Upload `scored.json` outputs or single light-curve CSVs from IRSA.
- Auto-mapped metrics: Delta m, rest-frame delta t, intrinsic/obscuration probabilities, QA badges, DRW excess.
- Custom pulse graph + per-candidate light-curve chart (g/r/i filters).
- Full parameter dump for every candidate to ensure transparency.

## Run locally
```bash
npm install
npm run dev
```

## Data formats
- `scored.json` should contain a `candidates` array (or `objects`/`targets`) with fields like `object_id`, `delta_mag`, `score`, `intrinsic_probability`.
- Light-curve CSVs should include `mjd` + `mag` columns (filter/err optional).

## Tips
- Use the "Attach CSV" button to bind a light curve to the currently selected candidate.
- If your JSON includes `light_curves` keyed by object_id, the chart will load automatically.
