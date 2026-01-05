export interface LightCurvePoint {
  mjd: number;
  mag: number;
  magerr?: number;
  filter?: string;
}

export interface AgnCandidate {
  object_id: string;
  score?: number | null;
  score_uncertainty?: number | null;
  delta_mag?: number | null;
  restframe_timescale_days?: number | null;
  intrinsic_probability?: number | null;
  obscuration_probability?: number | null;
  intrinsic_obscuration_flag?: string | null;
  intrinsic_obscuration_evidence?: string | null;
  qa_badge?: string | null;
  qa_badge_reason?: string | null;
  drw_peak_sigma?: number | null;
  ir_lag_days?: number | null;
  redshift?: number | null;
  ra?: number | null;
  dec?: number | null;
  clean_light_curve_path?: string | null;
  light_curve_path?: string | null;
  light_curve_csv?: string | null;
  [key: string]: unknown;
}

export interface AgnDataset {
  metadata?: Record<string, unknown>;
  candidates: AgnCandidate[];
  light_curves?: Record<string, LightCurvePoint[]>;
}
