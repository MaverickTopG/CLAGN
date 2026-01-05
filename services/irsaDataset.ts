import { AgnCandidate, AgnDataset, LightCurvePoint } from '../types';

const FILTER_MAP: Record<string, string> = {
  zg: 'g',
  zr: 'r',
  zi: 'i',
  g: 'g',
  r: 'r',
  i: 'i',
};

const NUMBER_KEYS = [
  'score',
  'score_uncertainty',
  'delta_mag',
  'restframe_timescale_days',
  'intrinsic_probability',
  'obscuration_probability',
  'drw_peak_sigma',
  'ir_lag_days',
  'redshift',
  'ra',
  'dec',
];

const FALLBACK_KEYS: Record<string, string[]> = {
  object_id: ['object_id', 'objectid', 'objid', 'name', 'id', 'source_id'],
  delta_mag: ['delta_mag', 'delta_m', 'dmag', 'delta_mag_g', 'delta_mag_r'],
  restframe_timescale_days: ['restframe_timescale_days', 'rest_delta_t', 'rest_dt', 'restframe_dt', 'rest_frame_dt'],
  intrinsic_probability: ['intrinsic_probability', 'intrinsic_prob', 'p_intrinsic', 'p_intr'],
  obscuration_probability: ['obscuration_probability', 'obscuration_prob', 'p_obscuration', 'p_obsc'],
  drw_peak_sigma: ['drw_peak_sigma', 'drw_sigma', 'drw_excess'],
  ir_lag_days: ['ir_lag_days', 'ir_lag', 'wise_lag', 'ir_lag_d'],
  qa_badge: ['qa_badge', 'qa', 'quality_badge', 'qa_flag'],
  qa_badge_reason: ['qa_badge_reason', 'qa_reason', 'qa_notes', 'qa_flags'],
  intrinsic_obscuration_flag: ['intrinsic_obscuration_flag', 'classification', 'class', 'scenario'],
  intrinsic_obscuration_evidence: ['intrinsic_obscuration_evidence', 'evidence', 'notes', 'intrinsic_notes'],
  score_uncertainty: ['score_uncertainty', 'score_sigma', 'score_err'],
  redshift: ['redshift', 'z'],
  ra: ['ra', 'ra_deg'],
  dec: ['dec', 'dec_deg'],
  clean_light_curve_path: ['clean_light_curve_path', 'light_curve_path', 'lc_path', 'light_curve_csv'],
};

const normalizeFilter = (value: string | number | null | undefined) => {
  if (value === null || value === undefined) return 'unknown';
  const normalized = String(value).trim().toLowerCase();
  if (!normalized) return 'unknown';
  if (FILTER_MAP[normalized]) return FILTER_MAP[normalized];
  if (/^[0-9]+$/.test(normalized)) {
    if (normalized === '1') return 'g';
    if (normalized === '2') return 'r';
    if (normalized === '3') return 'i';
  }
  if (normalized.length === 2 && normalized.startsWith('z')) {
    return normalized[1];
  }
  return normalized;
};

const toNumber = (value: unknown): number | null => {
  if (value === null || value === undefined || value === '') return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const pickFirst = (raw: Record<string, unknown>, keys: string[]) => {
  for (const key of keys) {
    if (raw[key] !== undefined && raw[key] !== null && raw[key] !== '') {
      return raw[key];
    }
  }
  return null;
};

const normalizeCandidate = (raw: Record<string, unknown>, fallbackId: string): AgnCandidate => {
  const objectId = pickFirst(raw, FALLBACK_KEYS.object_id) ?? fallbackId;
  const candidate: AgnCandidate = {
    ...raw,
    object_id: String(objectId ?? fallbackId ?? 'UNKNOWN'),
  };

  for (const key of NUMBER_KEYS) {
    const fallback = pickFirst(raw, FALLBACK_KEYS[key] ?? [key]);
    const number = toNumber(fallback);
    if (number !== null) {
      candidate[key] = number;
    }
  }

  for (const [target, keys] of Object.entries(FALLBACK_KEYS)) {
    if (candidate[target] === undefined || candidate[target] === null || candidate[target] === '') {
      const picked = pickFirst(raw, keys);
      if (picked !== null) {
        candidate[target] = picked as string;
      }
    }
  }

  if (!candidate.light_curve_path && candidate.clean_light_curve_path) {
    candidate.light_curve_path = String(candidate.clean_light_curve_path);
  }

  return candidate;
};

const extractCandidates = (data: Record<string, unknown>): Record<string, unknown>[] => {
  const candidateSources = [
    data.candidates,
    data.objects,
    data.targets,
    data.results,
    data.items,
    data.data,
  ];
  for (const source of candidateSources) {
    if (Array.isArray(source)) return source as Record<string, unknown>[];
  }
  if (data.data && typeof data.data === 'object') {
    const nested = data.data as Record<string, unknown>;
    for (const source of [nested.candidates, nested.objects, nested.targets, nested.results]) {
      if (Array.isArray(source)) return source as Record<string, unknown>[];
    }
  }
  return [];
};

const extractLightCurves = (data: Record<string, unknown>): Record<string, LightCurvePoint[]> => {
  const lightCurveSources = [
    data.light_curves,
    data.lightcurves,
    data.light_curve_map,
    data.lightCurveMap,
  ];
  if (data.data && typeof data.data === 'object') {
    const nested = data.data as Record<string, unknown>;
    lightCurveSources.push(
      nested.light_curves,
      nested.lightcurves,
      nested.light_curve_map,
      nested.lightCurveMap
    );
  }
  for (const source of lightCurveSources) {
    if (source && typeof source === 'object' && !Array.isArray(source)) {
      const map: Record<string, LightCurvePoint[]> = {};
      for (const [key, value] of Object.entries(source as Record<string, unknown>)) {
        if (Array.isArray(value)) {
          map[key] = value
            .map((entry) => normalizeLightCurvePoint(entry as Record<string, unknown>))
            .filter((point): point is LightCurvePoint => point !== null);
        }
      }
      return map;
    }
  }
  return {};
};

const normalizeLightCurvePoint = (raw: Record<string, unknown>): LightCurvePoint | null => {
  const mjdValue = raw.mjd ?? raw.jd ?? raw.hjd ?? raw.time ?? raw.t;
  const magValue = raw.mag ?? raw.magnitude ?? raw.magpsf ?? raw.mag_psf ?? raw.flux;
  const magerrValue = raw.magerr ?? raw.mag_err ?? raw.sigmag ?? raw.magerr_psf ?? raw.magerrpsf;
  const mjd = toNumber(mjdValue);
  const mag = toNumber(magValue);
  if (mjd === null || mag === null) return null;
  return {
    mjd,
    mag,
    magerr: toNumber(magerrValue) ?? undefined,
    filter: normalizeFilter(raw.filter ?? raw.band ?? raw.fid ?? raw.filtercode),
  };
};

export const parseLightCurveCsv = (text: string) => {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (!lines.length) return { objectId: null, points: [] };

  const useComma = lines[0].includes(',');
  const header = lines[0].split(useComma ? ',' : /\s+/).map((col) => col.trim().toLowerCase());
  const points: LightCurvePoint[] = [];
  let objectId: string | null = null;

  for (let i = 1; i < lines.length; i += 1) {
    const cols = lines[i].split(useComma ? ',' : /\s+/).map((col) => col.trim());
    const row: Record<string, unknown> = {};
    header.forEach((key, idx) => {
      row[key] = cols[idx];
    });
    const point = normalizeLightCurvePoint(row);
    if (point) points.push(point);
    const rowObjectId = row.object_id ?? row.objectid ?? row.objid ?? row.oid ?? row.source_id;
    if (rowObjectId && !objectId) {
      objectId = String(rowObjectId);
    }
  }

  return { objectId, points };
};

export const buildCandidateFromLightCurve = (points: LightCurvePoint[], fallbackId: string): AgnCandidate => {
  const mags = points.map((point) => point.mag).filter((value) => Number.isFinite(value));
  const mjds = points.map((point) => point.mjd).filter((value) => Number.isFinite(value));
  const baselineSample = mags.slice(0, Math.max(3, Math.floor(mags.length * 0.2)));
  const baseline = baselineSample.length ? median(baselineSample) : null;
  const minMag = mags.length ? Math.min(...mags) : null;
  const deltaMag = baseline !== null && minMag !== null ? baseline - minMag : null;
  const restDuration = mjds.length ? Math.max(...mjds) - Math.min(...mjds) : null;
  const score = deltaMag !== null ? Math.min(0.99, Math.max(0.1, Math.abs(deltaMag) / 2)) : null;

  return {
    object_id: fallbackId,
    score,
    score_uncertainty: score !== null ? 0.06 : null,
    delta_mag: deltaMag,
    restframe_timescale_days: restDuration,
    intrinsic_probability: deltaMag !== null && deltaMag > 0.6 ? 0.78 : 0.45,
    obscuration_probability: deltaMag !== null && deltaMag > 0.6 ? 0.22 : 0.55,
    intrinsic_obscuration_flag: deltaMag !== null && deltaMag > 0.6 ? 'intrinsic' : 'obscuration',
    intrinsic_obscuration_evidence: 'Derived from uploaded IRSA light curve.',
    qa_badge: 'yellow',
    qa_badge_reason: 'Uploaded light curve only.',
  };
};

const median = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (!sorted.length) return 0;
  if (sorted.length % 2) return sorted[mid];
  return (sorted[mid - 1] + sorted[mid]) / 2;
};

export const normalizeDataset = (input: unknown, fileName?: string): AgnDataset => {
  if (!input || typeof input !== 'object') {
    return { candidates: [] };
  }

  if (Array.isArray(input)) {
    const candidates = input.map((raw, index) =>
      normalizeCandidate(raw as Record<string, unknown>, `${fileName ?? 'IRSA'}_${index + 1}`)
    );
    return { candidates };
  }

  const data = input as Record<string, unknown>;
  const rawCandidates = extractCandidates(data);
  const lightCurves = extractLightCurves(data);
  const candidates = rawCandidates.map((raw, index) => normalizeCandidate(raw, `${fileName ?? 'IRSA'}_${index + 1}`));

  return {
    metadata: (data.metadata as Record<string, unknown>) ?? {},
    candidates,
    light_curves: lightCurves,
  };
};
