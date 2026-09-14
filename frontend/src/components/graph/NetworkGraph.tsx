import { useEffect, useMemo, useRef } from "react";
import * as d3 from "d3";
import { useSentinel } from "@/store";
import { riskColor } from "@/lib/format";
import type { ForecastResponse } from "@/api/types";

// The protagonist visualization. A D3 force-directed network rendered to a
// canvas (WebGL-grade throughput for 100-300 nodes + hundreds of edges without
// React rerenders). It shows PAST -> NOW -> FUTURE:
//   - observed state: solid nodes/edges
//   - forecast state: ghosted / dashed predicted nodes & edges
// The graph morphs between temporal states when the temporal lens changes.
//
// Every animation is bound to real state:
//   node pulse    -> node activity/risk (flow_count + anchor risk)
//   edge flow     -> communication (dash offset advances with flow_count)
//   attack edge   -> flagged communication (red, thicker)
//   ghost node    -> predicted future host (dashed ring, forecast layer)
//   morph         -> temporal state transition

interface SimNode extends d3.SimulationNodeDatum {
  id: string;
  index: number;
  flow: number;
  attack: boolean;
  ghost: boolean; // predicted-only node (future lens)
}
interface SimLink {
  source: string | SimNode;
  target: string | SimNode;
  flow: number;
  attack: boolean;
  ghost: boolean;
}

function buildSim(fc: ForecastResponse, future: boolean) {
  const g = fc.network_state.latest_graph;
  const attackNodeIds = new Set<string>();
  g.edges.forEach((e) => {
    if (e.contains_attack) {
      attackNodeIds.add(e.src_id);
      attackNodeIds.add(e.dst_id);
    }
  });

  const nodes: SimNode[] = g.nodes.map((n) => ({
    id: n.id,
    index: n.index,
    flow: n.flow_count,
    attack: attackNodeIds.has(n.id),
    ghost: false,
  }));

  const links: SimLink[] = g.edges.map((e) => ({
    source: e.src_id,
    target: e.dst_id,
    flow: e.flow_count,
    attack: e.contains_attack,
    ghost: false,
  }));

  // In the FUTURE lens, add ghost/predicted expansion of the attack surface,
  // scaled to the forecast risk trajectory (real forecast values, not decor).
  if (future) {
    const meanForecast =
      fc.risk.forecast_risk.reduce((a, b) => a + b, 0) /
      Math.max(1, fc.risk.forecast_risk.length);
    const nGhost = Math.round(meanForecast * 18);
    const hubs = nodes.filter((n) => n.attack).slice(0, 4);
    for (let i = 0; i < nGhost; i++) {
      const id = `f${i}`;
      nodes.push({ id, index: 10000 + i, flow: 4, attack: true, ghost: true });
      const hub = hubs[i % Math.max(1, hubs.length)];
      if (hub) links.push({ source: hub.id, target: id, flow: 3, attack: true, ghost: true });
    }
  }
  return { nodes, links };
}

export function NetworkGraph() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const forecast = useSentinel((s) => s.forecast);
  const temporalView = useSentinel((s) => s.temporalView);
  const selectedNode = useSentinel((s) => s.selectedNode);
  const whatIfNode = useSentinel((s) => s.whatIfNode);
  const selectNode = useSentinel((s) => s.selectNode);
  const runWhatIf = useSentinel((s) => s.runWhatIf);

  const riskNow = forecast?.risk.risk_now ?? 0;
  const future = temporalView === "future";

  const data = useMemo(
    () => (forecast ? buildSim(forecast, future) : null),
    [forecast, future],
  );

  // The whole render loop lives in a ref-driven effect so the canvas draws
  // outside React's reconciliation (no per-frame rerenders).
  useEffect(() => {
    if (!data || !canvasRef.current || !wrapRef.current) return;
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    const ctx = canvas.getContext("2d")!;
    const dpr = Math.min(2, window.devicePixelRatio || 1);

    let width = wrap.clientWidth;
    let height = wrap.clientHeight;
    const resize = () => {
      width = wrap.clientWidth;
      height = wrap.clientHeight;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
    };
    resize();

    const nodes = data.nodes.map((n) => ({ ...n }));
    const links = data.links.map((l) => ({ ...l }));

    const sim = d3
      .forceSimulation<SimNode>(nodes)
      .force(
        "link",
        d3
          .forceLink<SimNode, SimLink>(links as SimLink[])
          .id((d) => (d as SimNode).id)
          .distance((l) => (l.ghost ? 46 : 30))
          .strength(0.25),
      )
      .force("charge", d3.forceManyBody().strength(-38))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collide", d3.forceCollide<SimNode>().radius((d) => 4 + Math.sqrt(d.flow)))
      .alpha(0.9)
      .alphaDecay(0.03);

    // transform (pan/zoom)
    let transform = d3.zoomIdentity;
    const d3canvas = d3.select(canvas);
    const zoom = d3
      .zoom<HTMLCanvasElement, unknown>()
      .scaleExtent([0.3, 4])
      .on("zoom", (ev) => {
        transform = ev.transform;
      });
    d3canvas.call(zoom as never);

    // Initial fit so the graph starts framed.
    transform = d3.zoomIdentity.translate(width * 0.02, height * 0.02).scale(0.9);
    d3canvas.call((zoom as never), transform as never);

    const radius = (n: SimNode) => 2.4 + Math.sqrt(n.flow) * (n.ghost ? 0.8 : 1.1);

    const pick = (mx: number, my: number): SimNode | null => {
      const [x, y] = transform.invert([mx, my]);
      let best: SimNode | null = null;
      let bd = 14 * 14;
      for (const n of nodes) {
        const dx = (n.x ?? 0) - x;
        const dy = (n.y ?? 0) - y;
        const d2 = dx * dx + dy * dy;
        if (d2 < bd) {
          bd = d2;
          best = n;
        }
      }
      return best;
    };

    canvas.onclick = (ev) => {
      const rect = canvas.getBoundingClientRect();
      const n = pick(ev.clientX - rect.left, ev.clientY - rect.top);
      selectNode(n ? n.id : null);
    };
    canvas.ondblclick = (ev) => {
      const rect = canvas.getBoundingClientRect();
      const n = pick(ev.clientX - rect.left, ev.clientY - rect.top);
      if (n && !n.ghost) void runWhatIf(n.index);
    };

    let raf = 0;
    let t0 = performance.now();

    const draw = () => {
      const now = performance.now();
      const elapsed = (now - t0) / 1000;
      ctx.save();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.scale(dpr, dpr);
      ctx.translate(transform.x, transform.y);
      ctx.scale(transform.k, transform.k);

      // ---- edges: edge-flow animation = communication (dash advance) ----
      for (const l of links as SimLink[]) {
        const s = l.source as SimNode;
        const tg = l.target as SimNode;
        if (s.x == null || tg.x == null) continue;
        ctx.beginPath();
        ctx.moveTo(s.x, s.y!);
        ctx.lineTo(tg.x!, tg.y!);
        if (l.ghost) {
          ctx.setLineDash([3, 4]);
          ctx.lineDashOffset = -elapsed * 14;
          ctx.strokeStyle = "rgba(107,123,214,0.45)";
          ctx.lineWidth = 0.8;
        } else if (l.attack) {
          ctx.setLineDash([5, 5]);
          ctx.lineDashOffset = -elapsed * (10 + l.flow); // flow drives speed
          ctx.strokeStyle = "rgba(255,77,94,0.7)";
          ctx.lineWidth = 1.1 + Math.min(2, l.flow / 20);
        } else {
          ctx.setLineDash([]);
          ctx.strokeStyle = "rgba(56,225,255,0.12)";
          ctx.lineWidth = 0.5 + Math.min(1.4, l.flow / 40);
        }
        ctx.stroke();
      }
      ctx.setLineDash([]);

      // ---- nodes: pulse = activity/risk; ghost ring = predicted host ----
      for (const n of nodes) {
        if (n.x == null) continue;
        const r = radius(n);
        const isSel = n.id === selectedNode;
        const isWhatIf = n.index === whatIfNode;

        // activity/risk pulse: amplitude tied to riskNow + node flow
        if (n.attack && !n.ghost) {
          const pulse = 1 + Math.sin(elapsed * 3 + n.index) * 0.12 * (0.4 + riskNow);
          ctx.beginPath();
          ctx.arc(n.x, n.y!, r * pulse * 2.4, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(255,77,94,${0.06 + riskNow * 0.06})`;
          ctx.fill();
        }

        ctx.beginPath();
        ctx.arc(n.x, n.y!, r, 0, Math.PI * 2);
        if (n.ghost) {
          ctx.fillStyle = "rgba(107,123,214,0.25)";
          ctx.fill();
          ctx.setLineDash([2, 2]);
          ctx.strokeStyle = "rgba(107,123,214,0.8)";
          ctx.lineWidth = 0.8;
          ctx.stroke();
          ctx.setLineDash([]);
        } else {
          ctx.fillStyle = n.attack ? riskColor(0.55 + riskNow * 0.4) : "#38e1ff";
          ctx.fill();
        }

        if (isWhatIf) {
          ctx.beginPath();
          ctx.arc(n.x, n.y!, r + 5, 0, Math.PI * 2);
          ctx.strokeStyle = "#a970ff";
          ctx.lineWidth = 1.6;
          ctx.stroke();
        }
        if (isSel) {
          ctx.beginPath();
          ctx.arc(n.x, n.y!, r + 3, 0, Math.PI * 2);
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 1.2;
          ctx.stroke();
        }
      }
      ctx.restore();
      raf = requestAnimationFrame(draw);
    };
    draw();

    const ro = new ResizeObserver(() => {
      resize();
      sim.force("center", d3.forceCenter(width / 2, height / 2));
      sim.alpha(0.4).restart();
    });
    ro.observe(wrap);

    return () => {
      cancelAnimationFrame(raf);
      sim.stop();
      ro.disconnect();
      canvas.onclick = null;
      canvas.ondblclick = null;
    };
  }, [data, riskNow, selectedNode, whatIfNode, selectNode, runWhatIf]);

  return (
    <div ref={wrapRef} className="relative h-full w-full">
      <canvas ref={canvasRef} className="absolute inset-0 cursor-crosshair" />
    </div>
  );
}
