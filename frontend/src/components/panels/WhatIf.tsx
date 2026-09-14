import { useEffect, useRef } from "react";
import * as d3 from "d3";
import { motion, AnimatePresence } from "framer-motion";
import { Panel } from "@/components/ui/Panel";
import { useSentinel } from "@/store";

// WHAT IF — counterfactual intervention. Compares baseline forecast risk vs the
// simulated post-intervention risk across the horizon. All values come from the
// /counterfactual endpoint (or its schema-faithful mock).
export function WhatIf() {
  const cf = useSentinel((s) => s.counterfactual);
  const whatIfNode = useSentinel((s) => s.whatIfNode);
  const clearWhatIf = useSentinel((s) => s.clearWhatIf);
  const svgRef = useRef<SVGSVGElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!cf || !svgRef.current || !wrapRef.current) return;
    const wrap = wrapRef.current;
    const width = wrap.clientWidth;
    const height = wrap.clientHeight;
    const m = { top: 8, right: 8, bottom: 16, left: 26 };
    const iw = width - m.left - m.right;
    const ih = height - m.top - m.bottom;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg.attr("width", width).attr("height", height);
    const g = svg.append("g").attr("transform", `translate(${m.left},${m.top})`);

    const K = cf.K;
    const x = d3.scaleLinear().domain([1, K]).range([0, iw]);
    // Auto-zoom the Y-axis to the two lines. The model's risk sits in a narrow
    // band, so a fixed 0..1 axis hides the (real, small) intervention effect.
    // We fit the domain to the data with a little padding so the gap is visible.
    const allVals = [...cf.baseline_risk, ...cf.intervention_risk];
    const lo = Math.min(...allVals);
    const hi = Math.max(...allVals);
    const padDom = Math.max((hi - lo) * 0.25, 0.002); // never a zero-height axis
    const y = d3
      .scaleLinear()
      .domain([Math.max(0, lo - padDom), Math.min(1, hi + padDom)])
      .range([ih, 0]);

    g.append("g")
      .call(d3.axisLeft(y).ticks(3).tickFormat((d) => `${(+d * 100).toFixed(1)}`).tickSize(-iw))
      .call((s) => s.select(".domain").remove())
      .call((s) => s.selectAll(".tick line").attr("stroke", "#16203a"))
      .selectAll("text")
      .attr("fill", "#65748f")
      .attr("font-size", 8)
      .attr("font-family", "JetBrains Mono, monospace");

    const line = d3
      .line<number>()
      .x((_, i) => x(i + 1))
      .y((d) => y(d))
      .curve(d3.curveMonotoneX);

    g.append("path")
      .datum(cf.baseline_risk)
      .attr("fill", "none")
      .attr("stroke", "#ff4d5e")
      .attr("stroke-width", 1.6)
      .attr("d", line);
    g.append("path")
      .datum(cf.intervention_risk)
      .attr("fill", "none")
      .attr("stroke", "#3ddc97")
      .attr("stroke-width", 1.6)
      .attr("stroke-dasharray", "4 3")
      .attr("d", line);
  }, [cf]);

  return (
    <Panel
      title="What-If · Counterfactual"
      className="h-full"
      right={
        cf && (
          <button
            onClick={clearWhatIf}
            className="mono text-[10px] text-inkdim hover:text-ink"
          >
            clear
          </button>
        )
      }
      bodyClassName="p-3 flex flex-col"
    >
      <AnimatePresence mode="wait">
        {cf ? (
          <motion.div
            key="result"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex flex-col h-full"
          >
            <div className="flex items-center justify-between mb-1">
              <span className="mono text-[11px] text-uncertainty">
                isolate n{whatIfNode}
              </span>
              <span className="mono text-[10px] text-inkdim">{cf.label}</span>
            </div>
            <div ref={wrapRef} className="flex-1 min-h-[70px]">
              <svg ref={svgRef} />
            </div>
            <div className="flex items-center justify-between text-[10px] mono pt-1">
              <span className="text-risk">— baseline</span>
              <span className="text-risklow">--- intervention</span>
            </div>
            {/* Real per-node effect. Δrisk is shown in basis points because the
                model's risk is near-flat, so % rounds to 0.0 and hides the
                genuine (small) difference between nodes. "state impact" is the
                mean L2 shift of the forecast latent — the more responsive
                signal that clearly differs per node. Both are straight from
                the /counterfactual response; nothing is scaled or faked. */}
            <div className="flex items-center justify-between text-[10px] mono pt-1 border-t border-[#16203a] mt-1">
              <span className="text-inkdim">
                Δ risk{" "}
                <span
                  className="font-semibold"
                  style={{
                    color: cf.mean_risk_delta < 0 ? "#3ddc97" : cf.mean_risk_delta > 0 ? "#ff4d5e" : "#65748f",
                  }}
                >
                  {cf.mean_risk_delta > 0 ? "+" : ""}
                  {(cf.mean_risk_delta * 10000).toFixed(2)} bp
                </span>
              </span>
              {cf.latent_shift && cf.latent_shift.length > 0 && (
                <span className="text-inkdim">
                  state impact{" "}
                  <span className="font-semibold text-uncertainty">
                    {(
                      cf.latent_shift.reduce((a, b) => a + b, 0) /
                      cf.latent_shift.length
                    ).toFixed(4)}
                  </span>
                </span>
              )}
            </div>
          </motion.div>
        ) : (
          <motion.p
            key="empty"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-[11px] text-inkdim leading-relaxed m-auto text-center px-2"
          >
            Double-click a node in the graph (or use “isolate” in Node
            Intelligence) to simulate quarantining it and see the forecast risk
            change.
          </motion.p>
        )}
      </AnimatePresence>
    </Panel>
  );
}
