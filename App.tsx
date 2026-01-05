import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Activity, BarChart3, CloudUpload, Sparkles, Telescope } from 'lucide-react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import PulseGraph from './components/PulseGraph';
import LightCurveChart from './components/LightCurveChart';
import { FILTER_COLORS, SAMPLE_DATASET } from './constants';
import { AgnCandidate, AgnDataset, LightCurvePoint } from './types';
import {
  buildCandidateFromLightCurve,
  normalizeDataset,
  parseLightCurveCsv,
} from './services/irsaDataset';

const DEFAULT_FILTERS = new Set(['g', 'r', 'i', 'unknown']);

const formatNumber = (value: unknown, digits = 2) => {
  if (value === null || value === undefined || value === '') return 'N/A';
  const num = Number(value);
  if (!Number.isFinite(num)) return 'N/A';
  return num.toFixed(digits);
};

const formatPercent = (value: unknown) => {
  if (value === null || value === undefined || value === '') return 'N/A';
  const num = Number(value);
  if (!Number.isFinite(num)) return 'N/A';
  return `${(num * 100).toFixed(0)}%`;
};

const formatValue = (value: unknown) => {
  if (value === null || value === undefined || value === '') return 'N/A';
  if (typeof value === 'number') return formatNumber(value);
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (Array.isArray(value)) return value.slice(0, 6).map(String).join(', ');
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
};

const parseEvidence = (value?: string | null) => {
  if (!value) return [];
  return value
    .split(/[;|\n]/)
    .map((entry) => entry.trim())
    .filter(Boolean);
};

const buildInlineCurves = (dataset: AgnDataset) => {
  const curves: Record<string, LightCurvePoint[]> = { ...(dataset.light_curves ?? {}) };
  dataset.candidates.forEach((candidate) => {
    const inline =
      (candidate.light_curve as LightCurvePoint[] | undefined) ??
      (candidate.light_curve_points as LightCurvePoint[] | undefined) ??
      (candidate.lightcurve as LightCurvePoint[] | undefined);
    if (Array.isArray(inline) && inline.length) {
      curves[candidate.object_id] = inline
        .map((point) => ({
          mjd: Number((point as LightCurvePoint).mjd),
          mag: Number((point as LightCurvePoint).mag),
          magerr: Number.isFinite(Number((point as LightCurvePoint).magerr))
            ? Number((point as LightCurvePoint).magerr)
            : undefined,
          filter: (point as LightCurvePoint).filter ?? 'unknown',
        }))
        .filter((point) => Number.isFinite(point.mjd) && Number.isFinite(point.mag));
    }
  });
  return curves;
};

const App: React.FC = () => {
  const [dataset, setDataset] = useState<AgnDataset>(SAMPLE_DATASET);
  const [lightCurves, setLightCurves] = useState<Record<string, LightCurvePoint[]>>(
    SAMPLE_DATASET.light_curves ?? {}
  );
  const [selectedId, setSelectedId] = useState<string | null>(
    SAMPLE_DATASET.candidates[0]?.object_id ?? null
  );
  const [activeFilters, setActiveFilters] = useState<Set<string>>(DEFAULT_FILTERS);
  const [status, setStatus] = useState<string>('Loaded sample dataset');
  const [busy, setBusy] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    gsap.registerPlugin(ScrollTrigger);
    const ctx = gsap.context(() => {
      const tl = gsap.timeline();
      tl.fromTo(
        '.hero-title span',
        { y: 80, autoAlpha: 0 },
        {
          y: 0,
          autoAlpha: 1,
          stagger: 0.1,
          duration: 1.1,
          ease: 'power4.out',
          clearProps: 'all',
        }
      )
        .fromTo(
          '.hero-desc',
          { y: 16, autoAlpha: 0 },
          {
            y: 0,
            autoAlpha: 1,
            duration: 0.7,
            ease: 'power3.out',
            clearProps: 'all',
          },
          '-=0.6'
        )
        .fromTo(
          '.hero-btn',
          { scale: 0.96, autoAlpha: 0 },
          {
            scale: 1,
            autoAlpha: 1,
            duration: 0.5,
            ease: 'back.out(1.7)',
            clearProps: 'all',
          },
          '-=0.4'
        );

      gsap.fromTo(
        '.hero-stat-card',
        { y: 24, autoAlpha: 0 },
        {
          y: 0,
          autoAlpha: 1,
          duration: 0.7,
          ease: 'power3.out',
          clearProps: 'all',
          stagger: 0,
        }
      );

      gsap.fromTo(
        '.brand-item',
        { x: -32, autoAlpha: 0 },
        {
          x: 0,
          autoAlpha: 1,
          stagger: 0.08,
          scrollTrigger: {
            trigger: '.brand-item',
            start: 'top 85%',
          },
          clearProps: 'all',
        }
      );
    });

    return () => ctx.revert();
  }, []);

  const dotCanvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = dotCanvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext('2d');
    if (!context) return;

    let frame = 0;
    let points: { x: number; y: number }[] = [];
    let width = window.innerWidth;
    let height = window.innerHeight;
    let cursorX = width * 0.5;
    let cursorY = height * 0.4;
    let targetX = cursorX;
    let targetY = cursorY;
    let dpr = window.devicePixelRatio || 1;

    const tint = { r: 79, g: 70, b: 229 };
    const base = { r: 210, g: 217, b: 230 };
    const spacing = 78;
    const margin = 20;
    const influenceRadius = 240;
    const convergeStrength = 0.5;
    const extraRows = 2;

    const buildGrid = () => {
      width = window.innerWidth;
      height = window.innerHeight;
      dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      points = [];
      for (let y = margin; y <= height - margin + spacing * extraRows; y += spacing) {
        for (let x = margin; x <= width - margin; x += spacing) {
          points.push({ x, y });
        }
      }
    };

    const draw = () => {
      frame = window.requestAnimationFrame(draw);
      cursorX += (targetX - cursorX) * 0.12;
      cursorY += (targetY - cursorY) * 0.12;
      context.clearRect(0, 0, width, height);

      points.forEach((point) => {
        const dx = cursorX - point.x;
        const dy = cursorY - point.y;
        const dist = Math.hypot(dx, dy);
        const influence = Math.max(0, 1 - dist / influenceRadius);
        const pull = influence * convergeStrength;
        const offsetX = dx * pull;
        const offsetY = dy * pull;
        const mix = influence * 0.25;
        const r = Math.round(base.r + (tint.r - base.r) * mix);
        const g = Math.round(base.g + (tint.g - base.g) * mix);
        const b = Math.round(base.b + (tint.b - base.b) * mix);
        const alpha = 0.32 + influence * 0.15;
        context.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
        context.beginPath();
        context.arc(point.x + offsetX, point.y + offsetY, 3, 0, Math.PI * 2);
        context.fill();
      });
    };

    const onMove = (event: MouseEvent) => {
      targetX = event.clientX;
      targetY = event.clientY;
    };

    const onLeave = () => {
      targetX = width * 0.5;
      targetY = height * 0.4;
    };

    buildGrid();
    draw();
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseleave', onLeave);
    window.addEventListener('resize', buildGrid);

    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseleave', onLeave);
      window.removeEventListener('resize', buildGrid);
      if (frame) window.cancelAnimationFrame(frame);
    };
  }, []);

  const candidates = useMemo(() => dataset.candidates ?? [], [dataset]);
  const sortedCandidates = useMemo(
    () =>
      [...candidates].sort((a, b) => {
        const scoreA = Number(a.score ?? 0);
        const scoreB = Number(b.score ?? 0);
        return scoreB - scoreA;
      }),
    [candidates]
  );
  const selectedCandidate = useMemo(
    () => candidates.find((candidate) => candidate.object_id === selectedId) ?? sortedCandidates[0],
    [candidates, selectedId, sortedCandidates]
  );

  useEffect(() => {
    if (!sortedCandidates.length) {
      setSelectedId(null);
      return;
    }
    if (!selectedId || !candidates.some((candidate) => candidate.object_id === selectedId)) {
      setSelectedId(sortedCandidates[0]?.object_id ?? null);
    }
  }, [sortedCandidates, selectedId, candidates]);

  const totalCandidates = candidates.length;
  const topDelta = Math.max(
    0,
    ...candidates.map((candidate) => Math.abs(Number(candidate.delta_mag ?? 0)))
  );
  const bestScore = Math.max(...candidates.map((candidate) => Number(candidate.score ?? 0)), 0);
  const medianScore = candidates.length
    ? [...candidates]
        .map((candidate) => Number(candidate.score ?? 0))
        .sort((a, b) => a - b)[Math.floor(candidates.length / 2)]
    : 0;

  const applyDataset = (nextDataset: AgnDataset, message: string) => {
    const curves = buildInlineCurves(nextDataset);
    const topCandidate = [...nextDataset.candidates].sort((a, b) => {
      const scoreA = Number(a.score ?? 0);
      const scoreB = Number(b.score ?? 0);
      return scoreB - scoreA;
    })[0];
    setDataset(nextDataset);
    setLightCurves(curves);
    setSelectedId(topCandidate?.object_id ?? null);
    setStatus(message);
  };

  const handleDatasetUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setBusy(true);
    setStatus('Parsing dataset...');
    try {
      if (file.name.toLowerCase().endsWith('.json')) {
        const text = await file.text();
        const parsed = JSON.parse(text);
        const normalized = normalizeDataset(parsed, file.name.replace(/\.[^/.]+$/, ''));
        applyDataset(normalized, `Loaded ${file.name}`);
      } else if (file.name.toLowerCase().endsWith('.csv')) {
        const text = await file.text();
        const { objectId, points } = parseLightCurveCsv(text);
        const candidateId = objectId ?? file.name.replace(/\.[^/.]+$/, '');
        const candidate = buildCandidateFromLightCurve(points, candidateId);
        applyDataset(
          {
            metadata: {
              dataset: `IRSA CSV (${file.name})`,
              generated_at: new Date().toISOString(),
              n_candidates: 1,
            },
            candidates: [candidate],
            light_curves: { [candidateId]: points },
          },
          `Loaded ${file.name}`
        );
      } else {
        setStatus('Unsupported file. Upload a scored.json or light-curve CSV.');
      }
    } catch (error) {
      setStatus('Failed to parse dataset. Check file format.');
    } finally {
      setBusy(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };


  useEffect(() => {
    const candidate = selectedCandidate;
    if (!candidate) return;
    const existing = lightCurves[candidate.object_id];
    if (existing?.length) return;
    const path = candidate.light_curve_path || candidate.clean_light_curve_path || candidate.light_curve_csv;
    if (!path || path.startsWith('file://')) return;
    setBusy(true);
    fetch(path)
      .then((res) => res.text())
      .then((text) => {
        const { points } = parseLightCurveCsv(text);
        if (points.length) {
          setLightCurves((prev) => ({ ...prev, [candidate.object_id]: points }));
          setStatus(`Loaded light curve for ${candidate.object_id}`);
        }
      })
      .catch(() => {
        setStatus('Unable to fetch remote light curve.');
      })
      .finally(() => setBusy(false));
  }, [selectedCandidate, lightCurves]);

  const toggleFilter = (filter: string) => {
    setActiveFilters((prev) => {
      const next = new Set(prev);
      if (next.has(filter)) {
        next.delete(filter);
      } else {
        next.add(filter);
      }
      return next;
    });
  };

  const evidenceItems = parseEvidence(selectedCandidate?.intrinsic_obscuration_evidence ?? null);
  const candidateParams = selectedCandidate
    ? Object.entries(selectedCandidate)
        .filter(([key]) => !['light_curve', 'lightcurve', 'light_curve_points'].includes(key))
        .sort(([a], [b]) => a.localeCompare(b))
    : [];

  const datasetName =
    (dataset.metadata?.dataset as string | undefined) ??
    (dataset.metadata?.name as string | undefined) ??
    'IRSA dataset';
  const generatedAt = dataset.metadata?.generated_at as string | undefined;

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 canvas-bg relative overflow-hidden">
      <canvas ref={dotCanvasRef} className="dot-field" aria-hidden="true" />
      <header className="fixed top-0 left-0 w-full z-50 px-6 md:px-8 py-5 flex justify-between items-center bg-white/70 backdrop-blur-md border-b border-slate-200">
        <div className="flex items-center gap-3 group">
          <div>
            <h1 className="text-xl font-extrabold tracking-tighter text-slate-900 leading-none">
              AGN Light Analysis
            </h1>
          </div>
        </div>
        <nav className="hidden md:flex gap-10 items-center text-xs font-semibold text-slate-500 uppercase tracking-widest">
          <span>IRSA</span>
          <span>Gaia</span>
          <span>WISE</span>
          <span>Pan-STARRS</span>
        </nav>
      </header>

      <main className="relative z-10 pt-24 pb-20 px-6 md:px-12 max-w-[1600px] mx-auto">
        <section className="grid grid-cols-1 lg:grid-cols-12 gap-12 mb-28 items-end">
          <div className="lg:col-span-8">
            <h2 className="hero-title text-6xl md:text-7xl font-black tracking-tighter leading-[0.9] text-slate-900 mb-8 overflow-hidden">
              <span className="inline-block">TRACKING</span> <br />
              <span className="inline-block text-indigo-600">ACCRETION</span> <br />
              <span className="inline-block">STATE FLIPS</span>
            </h2>
            <p className="hero-desc text-lg md:text-xl text-slate-500 max-w-xl font-medium leading-relaxed">
              Feed IRSA light curves into a console that scores intrinsic transitions,
              cross-checks Gaia/WISE/Pan-STARRS, and preserves every provenance signal.
            </p>
            <div className="flex flex-wrap items-center gap-4 mt-8">
              <button
                className="hero-btn px-6 py-3 bg-slate-900 text-white font-bold rounded-2xl hover:bg-indigo-600 transition-all flex items-center gap-2"
                onClick={() => fileInputRef.current?.click()}
                disabled={busy}
              >
                <CloudUpload size={18} /> Upload dataset
              </button>
              <button
                className="hero-btn px-6 py-3 bg-white border border-slate-200 rounded-2xl font-bold text-slate-900 hover:border-indigo-400 transition-all"
                onClick={() => applyDataset(SAMPLE_DATASET, 'Loaded sample dataset')}
                disabled={busy}
              >
                Load sample
              </button>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept=".json,.csv,application/json,text/csv"
              className="hidden"
              onChange={handleDatasetUpload}
            />
          </div>
          <div className="lg:col-span-4 flex flex-col gap-4">
            <div className="hero-stat-card p-7 bg-white border border-slate-200 rounded-[2.5rem] shadow-xl shadow-slate-200/50">
              <div className="flex justify-between items-start mb-10">
                <div className="w-11 h-11 bg-indigo-50 rounded-2xl flex items-center justify-center text-indigo-600">
                  <Activity size={22} />
                </div>
                <span className="text-[10px] font-bold uppercase tracking-widest text-slate-400">
                  Best score
                </span>
              </div>
              <h4 className="text-[11px] font-black uppercase tracking-widest text-slate-400 mb-2">
                Dataset Signal
              </h4>
              <p className="text-2xl font-bold text-slate-900">{formatNumber(bestScore, 2)}</p>
              <p className="text-xs text-slate-400 mt-2">
                {datasetName}
                {generatedAt ? ` - ${generatedAt}` : ''}
              </p>
            </div>
            <div className="hero-stat-card p-7 bg-white border border-slate-200 rounded-[2.5rem] shadow-lg shadow-slate-200/40">
              <div className="flex items-center justify-between mb-6">
                <div className="w-10 h-10 bg-slate-100 rounded-2xl flex items-center justify-center text-slate-600">
                  <BarChart3 size={20} />
                </div>
                <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">Run stats</span>
              </div>
              <div className="grid grid-cols-2 gap-4 text-sm text-slate-600">
                <div>
                  <p className="text-xs uppercase tracking-widest text-slate-400">Candidates</p>
                  <p className="text-lg font-bold text-slate-900">{totalCandidates || 'N/A'}</p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-widest text-slate-400">Median score</p>
                  <p className="text-lg font-bold text-slate-900">{formatNumber(medianScore, 2)}</p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-widest text-slate-400">Max |Delta m|</p>
                  <p className="text-lg font-bold text-slate-900">{formatNumber(topDelta, 2)}</p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-widest text-slate-400">Status</p>
                  <p className="text-lg font-bold text-slate-900">{busy ? 'Working' : 'Ready'}</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="mb-28">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
            <div className="lg:col-span-3">
              <div className="flex items-center gap-2 mb-8">
                <div className="w-1 h-8 bg-indigo-600"></div>
                <h3 className="text-sm font-black uppercase tracking-[0.2em] text-slate-900">
                  Top candidates
                </h3>
              </div>
              <div className="flex flex-col gap-4">
                {sortedCandidates.slice(0, 6).map((candidate, index) => {
                  const isActive = candidate.object_id === selectedCandidate?.object_id;
                  return (
                    <button
                      key={candidate.object_id}
                      className={`brand-item group flex items-center justify-between p-4 bg-white border rounded-2xl transition-all text-left ${
                        isActive
                          ? 'border-indigo-300 shadow-lg shadow-indigo-100/60'
                          : 'border-slate-100 hover:border-indigo-200 hover:shadow-lg'
                      }`}
                      onClick={() => setSelectedId(candidate.object_id)}
                    >
                      <div className="flex items-center gap-4">
                        <span className="text-xs font-black text-slate-300 group-hover:text-indigo-600">
                          {index + 1}.
                        </span>
                        <div>
                          <p className="font-bold text-slate-900 group-hover:translate-x-1 transition-transform">
                            {candidate.object_id}
                          </p>
                          <p className="text-[11px] text-slate-400 uppercase tracking-widest">
                            {candidate.intrinsic_obscuration_flag ?? 'candidate'}
                          </p>
                        </div>
                      </div>
                      <div
                        className={`px-2 py-1 rounded-md text-[10px] font-bold ${
                          Number(candidate.delta_mag ?? 0) >= 0.6
                            ? 'bg-emerald-50 text-emerald-600'
                            : 'bg-amber-50 text-amber-600'
                        }`}
                      >
                        Delta m {formatNumber(candidate.delta_mag, 2)}
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="lg:col-span-9 relative">
              <div className="absolute top-0 right-0 p-8 text-right">
                <h2 className="text-5xl md:text-7xl font-black text-slate-900/10 tracking-tighter leading-none select-none">
                  AGN<br />VARIABILITY<br />PULSE
                </h2>
              </div>
              <PulseGraph candidates={sortedCandidates} highlightId={selectedCandidate?.object_id ?? null} />
            </div>
          </div>
        </section>

        <section className="grid grid-cols-1 lg:grid-cols-12 gap-12 mb-20">
          <div className="lg:col-span-7 space-y-6">
            <div className="bg-white border border-slate-200 rounded-3xl p-6 shadow-xl shadow-slate-200/40">
              <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
                <div>
                  <p className="text-xs uppercase tracking-widest text-slate-400">Selected candidate</p>
                  <h3 className="text-2xl font-black text-slate-900">
                    {selectedCandidate?.object_id ?? 'No candidate'}
                  </h3>
                  <p className="text-sm text-slate-500">
                    Rest delta t: {formatNumber(selectedCandidate?.restframe_timescale_days, 1)} days -
                    DRW sigma: {formatNumber(selectedCandidate?.drw_peak_sigma, 2)}
                  </p>
                </div>
                <div className="flex flex-wrap gap-2">
                  {['g', 'r', 'i'].map((filter) => (
                    <button
                      key={filter}
                      onClick={() => toggleFilter(filter)}
                      className={`px-3 py-1.5 rounded-full text-xs font-bold uppercase tracking-widest border transition-all ${
                        activeFilters.has(filter)
                          ? 'border-indigo-400 text-indigo-600 bg-indigo-50'
                          : 'border-slate-200 text-slate-400 bg-white'
                      }`}
                    >
                      <span
                        className="inline-block w-2 h-2 rounded-full mr-2"
                        style={{ backgroundColor: FILTER_COLORS[filter] ?? FILTER_COLORS.unknown }}
                      ></span>
                      {filter}
                    </button>
                  ))}
                  
                </div>
              </div>
              <LightCurveChart
                points={selectedCandidate ? lightCurves[selectedCandidate.object_id] ?? [] : []}
                activeFilters={activeFilters}
              />
              <p className="mt-4 text-xs text-slate-400 uppercase tracking-widest">
                Light curve loaded from dataset
              </p>
            </div>

            <div className="bg-white border border-slate-200 rounded-3xl p-6 shadow-lg shadow-slate-200/40">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-indigo-50 rounded-2xl flex items-center justify-center text-indigo-600">
                  <Sparkles size={20} />
                </div>
                <div>
                  <h4 className="text-lg font-black text-slate-900">Evidence synthesis</h4>
                  <p className="text-sm text-slate-500">Intrinsic vs obscuration rationale</p>
                </div>
              </div>
              <ul className="space-y-2 text-sm text-slate-600">
                {evidenceItems.length ? (
                  evidenceItems.map((item, index) => (
                    <li key={index} className="flex items-start gap-3">
                      <span className="w-2 h-2 rounded-full bg-indigo-500 mt-2"></span>
                      <span>{item}</span>
                    </li>
                  ))
                ) : (
                  <li className="text-slate-400">No evidence string provided in dataset.</li>
                )}
              </ul>
            </div>
          </div>

          <div className="lg:col-span-5 flex flex-col gap-6">
            <div className="bg-white border border-slate-200 rounded-3xl p-6 shadow-xl shadow-slate-200/40">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-slate-100 rounded-2xl flex items-center justify-center text-slate-600">
                  <CloudUpload size={20} />
                </div>
                <div>
                  <h4 className="text-lg font-black text-slate-900">Candidate metrics</h4>
                  <p className="text-sm text-slate-500">Judge-ready summary values</p>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4 text-sm text-slate-600">
                <div>
                  <p className="text-xs uppercase tracking-widest text-slate-400">Score</p>
                  <p className="text-lg font-bold text-slate-900">{formatNumber(selectedCandidate?.score, 2)}</p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-widest text-slate-400">Delta m</p>
                  <p className="text-lg font-bold text-slate-900">
                    {formatNumber(selectedCandidate?.delta_mag, 2)}
                  </p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-widest text-slate-400">Intrinsic P</p>
                  <p className="text-lg font-bold text-slate-900">
                    {formatPercent(selectedCandidate?.intrinsic_probability)}
                  </p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-widest text-slate-400">Obscuration P</p>
                  <p className="text-lg font-bold text-slate-900">
                    {formatPercent(selectedCandidate?.obscuration_probability)}
                  </p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-widest text-slate-400">IR lag (d)</p>
                  <p className="text-lg font-bold text-slate-900">
                    {formatNumber(selectedCandidate?.ir_lag_days, 1)}
                  </p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-widest text-slate-400">QA badge</p>
                  <p className="text-lg font-bold text-slate-900">
                    {selectedCandidate?.qa_badge ?? 'N/A'}
                  </p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-widest text-slate-400">Redshift</p>
                  <p className="text-lg font-bold text-slate-900">
                    {formatNumber(selectedCandidate?.redshift, 3)}
                  </p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-widest text-slate-400">RA / Dec</p>
                  <p className="text-lg font-bold text-slate-900">
                    {formatNumber(selectedCandidate?.ra, 3)} / {formatNumber(selectedCandidate?.dec, 3)}
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white border border-slate-200 rounded-3xl p-6 shadow-lg shadow-slate-200/40">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-emerald-50 rounded-2xl flex items-center justify-center text-emerald-600">
                  <Activity size={20} />
                </div>
                <div>
                  <h4 className="text-lg font-black text-slate-900">Cross-survey context</h4>
                  <p className="text-sm text-slate-500">Gaia, WISE, Pan-STARRS hooks</p>
                </div>
              </div>
              <div className="space-y-2 text-sm text-slate-600">
                <p>
                  QA note: <span className="font-semibold">{selectedCandidate?.qa_badge_reason ?? 'N/A'}</span>
                </p>
                <p>
                  Classification:{' '}
                  <span className="font-semibold">{selectedCandidate?.intrinsic_obscuration_flag ?? 'N/A'}</span>
                </p>
                <p>
                  DRW sigma: <span className="font-semibold">{formatNumber(selectedCandidate?.drw_peak_sigma, 2)}</span>
                </p>
              </div>
            </div>

            <div className="bg-white border border-slate-200 rounded-3xl p-6 shadow-lg shadow-slate-200/40">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-slate-100 rounded-2xl flex items-center justify-center text-slate-600">
                  <Sparkles size={20} />
                </div>
                <div>
                  <h4 className="text-lg font-black text-slate-900">All parameters</h4>
                  <p className="text-sm text-slate-500">Raw fields from scored.json</p>
                </div>
              </div>
              <div className="max-h-48 overflow-y-auto border border-slate-100 rounded-2xl p-3 text-xs text-slate-600">
                {candidateParams.length ? (
                  candidateParams.map(([key, value]) => (
                    <div key={key} className="flex items-start justify-between gap-4 py-1">
                      <span className="font-semibold text-slate-500">{key}</span>
                      <span className="text-right break-all">{formatValue(value)}</span>
                    </div>
                  ))
                ) : (
                  <p className="text-slate-400">No candidate selected.</p>
                )}
              </div>
            </div>
          </div>
        </section>

      </main>
    </div>
  );
};

export default App;
