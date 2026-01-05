import React, { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { FILTER_COLORS } from '../constants';
import { LightCurvePoint } from '../types';

interface LightCurveChartProps {
  points: LightCurvePoint[];
  activeFilters: Set<string>;
}

const LightCurveChart: React.FC<LightCurveChartProps> = ({ points, activeFilters }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [tooltip, setTooltip] = useState({
    visible: false,
    x: 0,
    y: 0,
    content: '',
  });
  const formatNumber = (value: number, digits = 3) => {
    if (!Number.isFinite(value)) return 'N/A';
    return value.toFixed(digits);
  };
  const filtered = useMemo(
    () => points.filter((point) => (point.filter ? activeFilters.has(point.filter) : true)),
    [points, activeFilters]
  );

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    if (!filtered.length) return;

    const width = 760;
    const height = 320;
    const margin = { top: 24, right: 24, bottom: 32, left: 48 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const mjds = filtered.map((point) => point.mjd);
    const mags = filtered.map((point) => point.mag);
    const xScale = d3
      .scaleLinear()
      .domain([Math.min(...mjds), Math.max(...mjds)])
      .range([0, innerWidth])
      .nice();
    const yMax = Math.max(...mags);
    const yMin = Math.min(...mags);
    const padding = (yMax - yMin) * 0.08 || 0.2;
    const yScale = d3
      .scaleLinear()
      .domain([yMax + padding, yMin - padding])
      .range([innerHeight, 0])
      .nice();

    const root = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const xAxis = d3.axisBottom(xScale).ticks(6).tickFormat((d) => String(d));
    const yAxis = d3.axisLeft(yScale).ticks(5);

    root
      .append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(xAxis)
      .selectAll('text')
      .attr('fill', '#64748b')
      .attr('font-size', 10);

    root
      .append('g')
      .call(yAxis)
      .selectAll('text')
      .attr('fill', '#64748b')
      .attr('font-size', 10);

    root.selectAll('.domain').attr('stroke', '#cbd5f5');
    root.selectAll('.tick line').attr('stroke', '#e2e8f0');

    const grouped = d3.group(filtered, (point) => point.filter ?? 'unknown');
    const line = d3
      .line<LightCurvePoint>()
      .x((d) => xScale(d.mjd))
      .y((d) => yScale(d.mag))
      .curve(d3.curveMonotoneX);

    for (const [filter, pointsByFilter] of grouped.entries()) {
      const sorted = [...pointsByFilter].sort((a, b) => a.mjd - b.mjd);
      root
        .append('path')
        .datum(sorted)
        .attr('d', line)
        .attr('fill', 'none')
        .attr('stroke', FILTER_COLORS[filter] ?? FILTER_COLORS.unknown)
        .attr('stroke-width', 2)
        .attr('opacity', 0.75);

      const circles = root
        .selectAll(`circle-${filter}`)
        .data(sorted)
        .enter()
        .append('circle')
        .attr('cx', (d) => xScale(d.mjd))
        .attr('cy', (d) => yScale(d.mag))
        .attr('r', 3)
        .attr('fill', FILTER_COLORS[filter] ?? FILTER_COLORS.unknown)
        .attr('opacity', 0.85)
        .attr('pointer-events', 'all');

      circles.on('mousemove', (event, d) => {
        const rect = wrapperRef.current?.getBoundingClientRect();
        if (!rect) return;
        const err = Number.isFinite(d.magerr ?? NaN) ? ` | Err: ${formatNumber(d.magerr ?? 0)}` : '';
        const filterLabel = d.filter ?? filter ?? 'unknown';
        const content = `MJD: ${formatNumber(d.mjd)} | Mag: ${formatNumber(d.mag)}${err} | Filter: ${filterLabel}`;
        const tooltipWidth = 240;
        const tooltipHeight = 48;
        const pad = 12;
        const x = Math.min(Math.max(event.clientX - rect.left + 12, pad), rect.width - tooltipWidth - pad);
        const y = Math.min(Math.max(event.clientY - rect.top + 12, pad), rect.height - tooltipHeight - pad);
        setTooltip({ visible: true, x, y, content });
      });

      circles.on('mouseleave', () => {
        setTooltip((prev) => ({ ...prev, visible: false }));
      });
    }

    root
      .append('text')
      .attr('x', 0)
      .attr('y', -8)
      .attr('fill', '#64748b')
      .attr('font-size', 11)
      .text('Magnitude (lower is brighter)');
  }, [filtered]);

  if (!filtered.length) {
    return (
      <div className="h-[320px] w-full flex items-center justify-center text-sm text-slate-400 border border-dashed border-slate-200 rounded-2xl bg-white/70">
        No light-curve points for the selected filters.
      </div>
    );
  }

  return (
    <div ref={wrapperRef} className="relative w-full" onMouseLeave={() => setTooltip((prev) => ({ ...prev, visible: false }))}>
      <svg ref={svgRef} viewBox="0 0 760 320" className="w-full" preserveAspectRatio="xMidYMid meet" />
      {tooltip.visible && (
        <div
          className="pointer-events-none absolute rounded-xl border border-slate-200 bg-white/95 px-3 py-2 text-[11px] font-semibold text-slate-600 shadow-lg shadow-slate-200/60"
          style={{ left: tooltip.x, top: tooltip.y }}
        >
          {tooltip.content}
        </div>
      )}
    </div>
  );
};

export default LightCurveChart;
