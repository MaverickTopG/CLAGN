import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { AgnCandidate } from '../types';

interface PulseGraphProps {
  candidates: AgnCandidate[];
  highlightId?: string | null;
}

const PulseGraph: React.FC<PulseGraphProps> = ({ candidates, highlightId }) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current) return;

    gsap.registerPlugin(ScrollTrigger);

    const svg = d3.select(svgRef.current);
    const width = 1200;
    const height = 560;
    const baseCount = candidates.length ? candidates.length * 2 : 32;
    const curveCount = Math.max(26, Math.min(64, baseCount));
    const highlight = candidates.find((item) => item.object_id === highlightId) ?? candidates[0];
    const highlightDelta = Math.max(0.6, Math.min(2.2, Math.abs(Number(highlight?.delta_mag ?? 1))));

    svg.selectAll('*').remove();

    const generatePath = (amplitude: number, spread = 0.5) => {
      const centerX = width / 2;
      const points: [number, number][] = [];
      const resolution = 120;

      for (let i = 0; i <= resolution; i += 1) {
        const x = (i / resolution) * width;
        const normalizedX = (x - centerX) / (width / 3.6);
        const y = height - amplitude * Math.exp(-(normalizedX * normalizedX) / spread);
        points.push([x, y]);
      }

      const line = d3
        .line<[number, number]>()
        .x((d) => d[0])
        .y((d) => d[1])
        .curve(d3.curveBasis);

      return line(points);
    };

    const palette = ['#6366f1', '#7c3aed', '#a855f7', '#ec4899', '#f472b6', '#22d3ee'];

    for (let i = 0; i < curveCount; i += 1) {
      const amp = 140 + i * 6 + Math.random() * 18;
      const color = palette[i % palette.length];
      const path = svg
        .append('path')
        .attr('d', generatePath(amp) || '')
        .attr('fill', 'none')
        .attr('stroke', color)
        .attr('stroke-width', 0.6)
        .attr('opacity', 0.18);

      gsap.fromTo(
        path.node(),
        { strokeDasharray: 2200, strokeDashoffset: 2200 },
        {
          strokeDashoffset: 0,
          duration: 2.4,
          delay: i * 0.02,
          ease: 'power2.out',
          scrollTrigger: {
            trigger: svgRef.current,
            start: 'top 80%',
          },
        }
      );
    }

    const highlightAmps = [260, 320, 380].map((amp) => amp + highlightDelta * 40);

    highlightAmps.forEach((amp, idx) => {
      const path = svg
        .append('path')
        .attr('d', generatePath(amp, 0.42) || '')
        .attr('fill', 'none')
        .attr('stroke', idx === 1 ? '#ec4899' : '#8b5cf6')
        .attr('stroke-width', idx === 1 ? 2.6 : 2.2)
        .attr('opacity', idx === 1 ? 0.9 : 0.6);

      gsap.fromTo(
        path.node(),
        { strokeDasharray: 2800, strokeDashoffset: 2800 },
        {
          strokeDashoffset: 0,
          duration: 3,
          delay: 0.4 + idx * 0.2,
          ease: 'expo.out',
          scrollTrigger: {
            trigger: svgRef.current,
            start: 'top 80%',
          },
        }
      );
    });
  }, [candidates, highlightId]);

  return (
    <div className="relative w-full flex flex-col items-center py-16 min-h-[560px]">
      <div className="absolute inset-[-30%] pointer-events-none opacity-70 z-0">
        <div className="absolute left-1/2 top-1/2 h-[640px] w-[920px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-indigo-200/55 blur-[180px]"></div>
        <div className="absolute left-[38%] top-[46%] h-[520px] w-[760px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-fuchsia-200/40 blur-[200px]"></div>
        <div className="absolute left-[62%] top-[38%] h-[460px] w-[680px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-sky-200/30 blur-[210px]"></div>
      </div>
      <svg
        ref={svgRef}
        viewBox="0 0 1200 560"
        className="w-full max-w-7xl relative z-10"
        preserveAspectRatio="xMidYMid meet"
      />
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-wrap items-center justify-center gap-6 text-[10px] font-bold tracking-widest text-slate-400 uppercase">
        <div className="flex items-center gap-2">
          <div className="w-3 h-0.5 bg-indigo-400"></div>
          Baseline variability
        </div>
        <div className="flex items-center gap-2">
          <div className="w-8 h-1 bg-gradient-to-r from-indigo-500 to-fuchsia-500"></div>
          Intrinsic transition
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-0.5 bg-fuchsia-500"></div>
          Dust response
        </div>
      </div>
    </div>
  );
};

export default PulseGraph;
