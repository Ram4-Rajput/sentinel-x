import { useEffect, useRef } from "react";
import * as d3 from "d3";
import { Panel } from "@/components/ui/Panel";
import { useSentinel } from "@/store";
import { riskColor } from "@/lib/format";

// WHEN — the risk timeline. D3 area/line of observed risk history (solid) with
// the forecast horizon (dashed/ghosted) continuing past the anchor. Scrubbing
// the timeline moves the anchor t, which re-rolls the whole world model.
export function RiskTimeline() {
  const svgRef = useRef<SVGSVGElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const history = useSentinel((s) => s.history);
  const forecast = useSentinel((s) => s.forecast);
  const tIndex = useSentinel((s) => s.tIndex);
  const minAnchor = useSentinel((s) => s.minAnchor);
  const maxAnchor = useSentinel((s) => s.maxAnchor);
  const setAnchor = useSentinel((s) => s.setAnchor);
  const threshold = forecast?.risk.risk_threshold ?? 0.5;

  useEffect(() => {
    if (!history || !wrapRef.current || !svgRef.current) return;
    const wrap = wrapRef.current;
    const width = wrap.clientWidth;
    const height = wrap.clientHeight;
    const m = { top: 10, right: 12, bottom: 18, left: 30 };
    const iw = width - m.left - m.right;
    const ih = height - m.top - m.bottom;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg.attr("width", width).attr("height", height);
    const g = svg.append("g").attr("transform", `translate(${m.left},${m.top})`);

    const obsPts = history.points;
    // Anchor index space is not 0-based in live mode (t_index runs 161..189),
    // so derive the x-domain from the real data range instead of [0, maxAnchor].
    const dataMin = obsPts.length ? obsPts[0].t_index : minAnchor;
    const dataMax = obsPts.length
      ? obsPts[obsPts.length - 1].t_index
      : maxAnchor;

    const forecastPts = (forecast?.risk.forecast_risk ?? []).map((r, i) => ({
      t_index: tIndex + i + 1,
      risk: r,
    }));
    const minX = dataMin;
    const maxX = Math.max(dataMax, tIndex + forecastPts.length);

    const x = d3.scaleLinear().domain([minX, maxX]).range([0, iw]);
    const y = d3.scaleLinear().domain([0, 1]).range([ih, 0]);

    // Look up a history point by its real t_index (not array position).
    const riskAt = (t: number) =>
      obsPts.find((p) => p.t_index === t)?.risk ?? 0;

    // axes
    g.append("g")
      .attr("transform", `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(6).tickSize(0))
      .call((s) => s.select(".domain").attr("stroke", "#1d2b4d"))
      .selectAll("text")
      .attr("fill", "#65748f")
      .attr("font-size", 9)
      .attr("font-family", "JetBrains Mono, monospace");
    g.append("g")
      .call(d3.axisLeft(y).ticks(3).tickFormat((d) => `${(+d * 100).toFixed(0)}%`).tickSize(-iw))
      .call((s) => s.select(".domain").remove())
      .call((s) => s.selectAll(".tick line").attr("stroke", "#16203a"))
      .selectAll("text")
      .attr("fill", "#65748f")
      .attr("font-size", 9)
      .attr("font-family", "JetBrains Mono, monospace");

    // threshold line
    g.append("line")
      .attr("x1", 0)
      .attr("x2", iw)
      .attr("y1", y(threshold))
      .attr("y2", y(threshold))
      .attr("stroke", "#ff4d5e")
      .attr("stroke-dasharray", "3 3")
      .attr("stroke-opacity", 0.5);

    // observed area (solid)
    const obs = obsPts;
    const area = d3
      .area<{ t_index: number; risk: number }>()
      .x((d) => x(d.t_index))
      .y0(ih)
      .y1((d) => y(d.risk))
      .curve(d3.curveMonotoneX);
    const line = d3
      .line<{ t_index: number; risk: number }>()
      .x((d) => x(d.t_index))
      .y((d) => y(d.risk))
      .curve(d3.curveMonotoneX);

    g.append("path")
      .datum(obs)
      .attr("fill", "url(#riskgrad)")
      .attr("fill-opacity", 0.25)
      .attr("d", area);
    g.append("path")
      .datum(obs)
      .attr("fill", "none")
      .attr("stroke", "#38e1ff")
      .attr("stroke-width", 1.4)
      .attr("d", line);

    // gradient def
    const defs = svg.append("defs");
    const grad = defs
      .append("linearGradient")
      .attr("id", "riskgrad")
      .attr("x1", "0")
      .attr("x2", "0")
      .attr("y1", "0")
      .attr("y2", "1");
    grad.append("stop").attr("offset", "0%").attr("stop-color", "#38e1ff");
    grad.append("stop").attr("offset", "100%").attr("stop-color", "#05070d");

    // forecast (ghosted / dashed) continuing from anchor
    if (forecastPts.length) {
      const fc = [{ t_index: tIndex, risk: riskAt(tIndex) }, ...forecastPts];
      g.append("path")
        .datum(fc)
        .attr("fill", "none")
        .attr("stroke", "#6b7bd6")
        .attr("stroke-width", 1.4)
        .attr("stroke-dasharray", "4 3")
        .attr("d", line);
      g.selectAll(".fcpt")
        .data(forecastPts)
        .join("circle")
        .attr("cx", (d) => x(d.t_index))
        .attr("cy", (d) => y(d.risk))
        .attr("r", 2.5)
        .attr("fill", (d) => riskColor(d.risk))
        .attr("stroke", "#0a0f1c");
    }

    // anchor marker
    g.append("line")
      .attr("x1", x(tIndex))
      .attr("x2", x(tIndex))
      .attr("y1", 0)
      .attr("y2", ih)
      .attr("stroke", "#ffffff")
      .attr("stroke-opacity", 0.6)
      .attr("stroke-width", 1);
    g.append("circle")
      .attr("cx", x(tIndex))
      .attr("cy", y(riskAt(tIndex)))
      .attr("r", 4)
      .attr("fill", "#ffffff");

    // scrub interaction
    svg
      .append("rect")
      .attr("x", m.left)
      .attr("y", m.top)
      .attr("width", iw)
      .attr("height", ih)
      .attr("fill", "transparent")
      .style("cursor", "col-resize")
      .on("click", (ev) => {
        const [mx] = d3.pointer(ev);
        const t = Math.round(x.invert(mx - m.left));
        void setAnchor(t);
      });
  }, [history, forecast, tIndex, minAnchor, maxAnchor, threshold, setAnchor]);

  return (
    <Panel
      title="Risk Timeline"
      className="h-full"
      right={<span className="mono text-[10px] text-inkdim">anchor t={tIndex} · click to scrub</span>}
      bodyClassName="p-1"
    >
      <div ref={wrapRef} className="h-full w-full">
        <svg ref={svgRef} />
      </div>
    </Panel>
  );
}
