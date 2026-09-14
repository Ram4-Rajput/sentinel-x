// Schema-faithful scenario generator for Sentinel-X.
//
// Every value the UI renders originates here (or from the live backend) and
// conforms to src/sentinelx/serving/schemas.py. Nothing is hardcoded into the
// components. The generator builds a coherent attack progression across a
// sequence of anchor windows so the demo flow reads as a real world-model
// trajectory:
//
//   normal -> suspicious -> reconnaissance -> initial-access forecast ->
//   lateral-movement forecast -> propagation -> novelty warning
//
// Node/edge feature columns match the Phase-2 graph cache exactly:
//   node: out_deg,in_deg,bytes_sent,bytes_recv,flow_count
//   edge: flow_count,total_bytes,contains_attack

import type {
  ForecastResponse,
  GraphEdge,
  GraphNode,
  MitreStage,
  NetworkGraph,
  NetworkHistoryResponse,
  WindowPropagation,
} from "./types";

const SEQ_LEN = 8;
const K = 6;
const LATENT_DIM = 16;
const RISK_THRESHOLD = 0.5;
const DATASET = "ctu-13 (scenario 11)";
const N_ANCHORS = 60;

// A small deterministic PRNG so the scenario is reproducible per anchor.
function mulberry32(seed: number) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Scenario "phase" for a given anchor t (0..N_ANCHORS-1). This drives the
// entire demo narrative and every derived value.
export type Phase =
  | "normal"
  | "suspicious"
  | "recon"
  | "initial-access"
  | "lateral"
  | "propagation";

export function phaseForAnchor(t: number): Phase {
  const f = t / (N_ANCHORS - 1);
  if (f < 0.35) return "normal";
  if (f < 0.5) return "suspicious";
  if (f < 0.62) return "recon";
  if (f < 0.74) return "initial-access";
  if (f < 0.86) return "lateral";
  return "propagation";
}

const PHASE_INTENSITY: Record<Phase, number> = {
  normal: 0.06,
  suspicious: 0.22,
  recon: 0.38,
  "initial-access": 0.55,
  lateral: 0.72,
  propagation: 0.9,
};

function baseRiskForAnchor(t: number): number {
  const phase = phaseForAnchor(t);
  const rng = mulberry32(t * 7919 + 13);
  const jitter = (rng() - 0.5) * 0.06;
  return Math.min(0.98, Math.max(0.01, PHASE_INTENSITY[phase] + jitter));
}

function nodeCountForAnchor(t: number): number {
  const phase = phaseForAnchor(t);
  const rng = mulberry32(t * 104729 + 1);
  const base =
    phase === "normal"
      ? 90
      : phase === "propagation"
        ? 240
        : 120 + Math.floor(PHASE_INTENSITY[phase] * 120);
  return base + Math.floor(rng() * 25);
}

// ---- graph construction ---------------------------------------------------
function buildGraph(t: number, timestep: number): NetworkGraph {
  const rng = mulberry32(t * 2654435761 + timestep * 40503);
  const phase = phaseForAnchor(t);
  const intensity = PHASE_INTENSITY[phase];
  const numNodes = nodeCountForAnchor(t);

  const nodes: GraphNode[] = [];
  for (let i = 0; i < numNodes; i++) {
    // A few "hub" hosts carry most traffic; the rest are peripheral.
    const isHub = i < Math.max(3, Math.floor(numNodes * 0.06));
    const scale = isHub ? 4 + rng() * 8 : 0.3 + rng() * 1.5;
    const flow = Math.round(scale * (4 + rng() * 10));
    nodes.push({
      id: `n${i}`,
      index: i,
      out_degree: Math.round(scale * (1 + rng() * 4)),
      in_degree: Math.round(scale * (1 + rng() * 4)),
      bytes_sent: Math.round(scale * (800 + rng() * 6000)),
      bytes_received: Math.round(scale * (800 + rng() * 6000)),
      flow_count: flow,
    });
  }

  // Attacker index 0 is the suspicion source; it fans out as intensity rises.
  const edges: GraphEdge[] = [];
  const targetEdges = Math.floor(numNodes * (1.4 + intensity * 1.8));
  const attackFanout = Math.floor(intensity * numNodes * 0.35);

  const seen = new Set<string>();
  const addEdge = (s: number, d: number, attack: boolean) => {
    if (s === d) return;
    const key = `${s}->${d}`;
    if (seen.has(key)) return;
    seen.add(key);
    const scale = 1 + rng() * 6;
    edges.push({
      src: s,
      dst: d,
      src_id: `n${s}`,
      dst_id: `n${d}`,
      flow_count: Math.round(scale * (2 + rng() * 8)),
      total_bytes: Math.round(scale * (1000 + rng() * 9000)),
      contains_attack: attack,
    });
  };

  // Benign backbone: peripheral -> hub traffic.
  for (let e = 0; e < targetEdges; e++) {
    const s = Math.floor(rng() * numNodes);
    const d = Math.floor(rng() * Math.max(3, Math.floor(numNodes * 0.1)));
    addEdge(s, d, false);
  }

  // Attack edges emanate from node 0 (and, in later phases, from compromised
  // secondaries) — this is what the propagation/MITRE layers key off.
  const compromised = [0];
  if (phase === "lateral" || phase === "propagation") {
    compromised.push(1, 2, 3);
  }
  for (const src of compromised) {
    const fan = phase === "propagation" ? attackFanout : Math.floor(attackFanout * 0.5);
    for (let e = 0; e < fan; e++) {
      const d = Math.floor(rng() * numNodes);
      addEdge(src, d, phase !== "normal");
    }
  }

  return {
    timestep,
    window_index: t - (SEQ_LEN - 1) + timestep,
    num_nodes: numNodes,
    num_edges: edges.length,
    nodes,
    edges,
  };
}

function latentForAnchor(t: number, offset: number): number[] {
  const rng = mulberry32(t * 22801 + offset * 131);
  const drift = PHASE_INTENSITY[phaseForAnchor(t)];
  return Array.from({ length: LATENT_DIM }, (_, i) =>
    Number((Math.sin(i + offset * 0.5) * 0.4 + (rng() - 0.5) * 0.3 + drift * 0.6).toFixed(4)),
  );
}

// ---- MITRE trajectory ------------------------------------------------------
function buildStages(t: number, forecastRisks: number[]): MitreStage[] {
  const phase = phaseForAnchor(t);
  const stages: MitreStage[] = [];

  const observedStage = (
    stage: string,
    confidence: number,
    evidence: string[],
  ): MitreStage => ({
    stage,
    status: "observed",
    confidence: Number(confidence.toFixed(4)),
    evidence,
    horizon: 0,
    window_index: t,
    source: "observed-state",
    timestamp: null,
  });

  if (phase === "normal") {
    stages.push(
      observedStage("Benign / No Attack Behaviour", 0.9, [
        "flow volume within baseline",
        "no flagged edges in observed windows",
      ]),
    );
  } else if (phase === "suspicious") {
    stages.push(
      observedStage("Suspicious Activity", 0.55, [
        "elevated out-degree from a single host",
        "flagged edge ratio above baseline",
      ]),
    );
  } else if (phase === "recon") {
    stages.push(
      observedStage("Reconnaissance", 0.68, [
        "one host contacting many distinct peers",
        "short-lived low-byte flows (scan-like)",
      ]),
    );
  } else if (phase === "initial-access") {
    stages.push(
      observedStage("Reconnaissance", 0.7, ["sustained scanning subsided"]),
    );
  } else {
    stages.push(
      observedStage("Lateral Movement", 0.74, [
        "multiple secondary hosts now originating flagged flows",
        "fan-out from previously-contacted peers",
      ]),
    );
  }

  // Forecast stages track the rolled-out risk. They only assert a stage when
  // the forecast risk crosses meaningful confidence.
  const forecastCatalog: Record<Phase, string> = {
    normal: "Suspicious Activity",
    suspicious: "Reconnaissance",
    recon: "Initial Access",
    "initial-access": "Lateral Movement",
    lateral: "Command & Control",
    propagation: "Exfiltration",
  };
  forecastRisks.forEach((r, k) => {
    if (r < 0.35) return;
    stages.push({
      stage: forecastCatalog[phase],
      status: "forecast",
      confidence: Number(r.toFixed(4)),
      evidence: [`predicted risk ${(r * 100).toFixed(0)}% at t+${k + 1}`],
      horizon: k + 1,
      window_index: null,
      source: "forecast-rollout",
      timestamp: null,
    });
  });

  return stages;
}

// ---- propagation -----------------------------------------------------------
function buildPropagation(t: number, observed: NetworkGraph[]) {
  const perWindow: WindowPropagation[] = [];
  const cumulative = new Set<string>();
  let out = 0;
  let inn = 0;

  observed.forEach((g, ts) => {
    const affected = new Set<string>();
    const attackEdges = g.edges.filter((e) => e.contains_attack);
    attackEdges.forEach((e) => {
      affected.add(e.src_id);
      affected.add(e.dst_id);
      if (e.src === 0) out++;
      else inn++;
    });
    const newly: string[] = [];
    affected.forEach((a) => {
      if (!cumulative.has(a)) {
        cumulative.add(a);
        newly.push(a);
      }
    });
    perWindow.push({
      timestep: ts,
      window_index: g.window_index,
      affected: [...affected],
      newly_affected: newly,
      n_affected: affected.size,
      n_newly_affected: newly.length,
      cumulative_affected: cumulative.size,
      suspicious_edges: attackEdges.length,
      out_edges: attackEdges.filter((e) => e.src === 0).length,
      in_edges: attackEdges.filter((e) => e.dst === 0).length,
      branching: Number((attackEdges.length / Math.max(1, affected.size)).toFixed(4)),
    });
  });

  const velocities = perWindow.map((w) => w.n_newly_affected);
  const velocity = velocities.reduce((a, b) => a + b, 0) / Math.max(1, velocities.length);
  const ratio = out / Math.max(1, out + inn);
  return {
    t_index: t,
    suspicion_source: "label",
    total_affected: cumulative.size,
    velocity: Number(velocity.toFixed(4)),
    peak_velocity: Math.max(0, ...velocities),
    direction: ratio > 0.6 ? "fan-out" : ratio < 0.4 ? "fan-in" : "balanced",
    direction_ratio: Number(ratio.toFixed(4)),
    mean_branching:
      Number((perWindow.reduce((a, w) => a + w.branching, 0) / Math.max(1, perWindow.length)).toFixed(4)),
    persistence: Number((cumulative.size / Math.max(1, nodeCountForAnchor(t))).toFixed(4)),
    growth: Number((velocity).toFixed(4)),
    is_spreading: velocity > 0.5,
    per_window: perWindow,
  };
}

// ---- explainability --------------------------------------------------------
const NODE_FEATURE_NAMES = ["out_degree", "in_degree", "bytes_sent", "bytes_received", "flow_count"];
function buildExplainability(t: number, latest: NetworkGraph) {
  const rng = mulberry32(t * 40961 + 3);
  const intensity = PHASE_INTENSITY[phaseForAnchor(t)];
  const topFeatures = NODE_FEATURE_NAMES.map((name, i) => {
    const attribution = Number(((rng() - 0.3) * intensity * (i === 0 ? 1.6 : 1)).toFixed(4));
    return {
      name,
      scope: "node" as const,
      attribution,
      abs_attribution: Number(Math.abs(attribution).toFixed(4)),
    };
  }).sort((a, b) => b.abs_attribution - a.abs_attribution);

  const importantNodes = latest.nodes
    .slice(0, 6)
    .map((n) => ({
      node_index: n.index,
      timestep: SEQ_LEN - 1,
      importance: Number(((n.flow_count / 100) * intensity + rng() * 0.1).toFixed(4)),
      node_id: n.id,
    }))
    .sort((a, b) => b.importance - a.importance);

  const importantEdges = latest.edges
    .filter((e) => e.contains_attack)
    .slice(0, 6)
    .map((e) => ({
      src: e.src,
      dst: e.dst,
      timestep: SEQ_LEN - 1,
      importance: Number((e.flow_count / 60 + rng() * 0.2).toFixed(4)),
      attention: Number((rng() * intensity).toFixed(4)),
    }))
    .sort((a, b) => b.importance - a.importance);

  const temporal = Array.from({ length: SEQ_LEN }, (_, ts) => ({
    timestep: ts,
    relative_horizon: ts - (SEQ_LEN - 1),
    importance: Number(((ts / SEQ_LEN) * intensity + rng() * 0.08).toFixed(4)),
  }));

  return {
    t_index: t,
    target: "state",
    target_value: Number((intensity).toFixed(4)),
    method: "gradient-x-input",
    uses_attention: true,
    attention_note:
      "Attention indicates association (where the model attended), not cause.",
    top_features: topFeatures,
    important_nodes: importantNodes,
    important_edges: importantEdges,
    temporal_evidence: temporal,
    metadata: {},
  };
}

// ---- full aggregate forecast ----------------------------------------------
export function makeForecast(t: number, K_over?: number): ForecastResponse {
  const k = K_over ?? K;
  const observed: NetworkGraph[] = Array.from({ length: SEQ_LEN }, (_, ts) =>
    buildGraph(t, ts),
  );
  const latest = observed[SEQ_LEN - 1];
  const riskNow = baseRiskForAnchor(t);

  // Forecast risk ramps from risk_now toward the next phase's intensity.
  const nextPhaseTarget = PHASE_INTENSITY[phaseForAnchor(Math.min(N_ANCHORS - 1, t + 4))];
  const forecastRisk = Array.from({ length: k }, (_, i) => {
    const w = (i + 1) / k;
    const rng = mulberry32(t * 96079 + i * 17);
    const v = riskNow * (1 - w) + nextPhaseTarget * w + (rng() - 0.5) * 0.05;
    return Number(Math.min(0.99, Math.max(0.01, v)).toFixed(4));
  });

  const horizons = Array.from({ length: k }, (_, i) => i + 1);
  const stages = buildStages(t, forecastRisk);

  // Uncertainty (predictive variance) grows with horizon distance & novelty.
  const noveltyScore = Number(
    (PHASE_INTENSITY[phaseForAnchor(t)] * 3.2 + (phaseForAnchor(t) === "propagation" ? 1.5 : 0)).toFixed(4),
  );
  const uncStd = Number(
    (0.04 + PHASE_INTENSITY[phaseForAnchor(t)] * 0.18).toFixed(4),
  );

  const propagation = buildPropagation(t, observed);

  return {
    t_index: t,
    dataset: DATASET,
    K: k,
    risk_threshold: RISK_THRESHOLD,
    network_state: {
      t_index: t,
      seq_len: SEQ_LEN,
      latest_graph: latest,
      observed_graphs: observed,
      risk_now: Number(riskNow.toFixed(4)),
      label_any_attack: phaseForAnchor(t) === "normal" ? 0 : 1,
    },
    graph_nodes: latest.nodes,
    graph_edges: latest.edges,
    forecast: {
      t_index: t,
      K: k,
      current_latent: latentForAnchor(t, 0),
      steps: forecastRisk.map((r, i) => ({
        horizon: i + 1,
        target_index: t + i + 1,
        risk: r,
        latent: latentForAnchor(t, i + 1),
      })),
      risk_threshold: RISK_THRESHOLD,
    },
    horizons,
    risk: {
      t_index: t,
      risk_now: Number(riskNow.toFixed(4)),
      risk_threshold: RISK_THRESHOLD,
      alert: riskNow >= RISK_THRESHOLD,
      forecast_risk: forecastRisk,
      horizons,
    },
    uncertainty: {
      t_index: t,
      method: "mc-dropout",
      n_passes: 30,
      risk_mean: Number(riskNow.toFixed(4)),
      uncertainty_variance: Number((uncStd * uncStd).toFixed(6)),
      uncertainty_std: uncStd,
      note: "Risk (point estimate) and uncertainty (predictive variance) are reported separately.",
    },
    novelty: {
      t_index: t,
      method: "mahalanobis",
      novelty_score: noveltyScore,
      is_novel: noveltyScore > 3.0,
      threshold: 3.0,
      fit_percentile: 95.0,
      note: "Novelty is a Mahalanobis distance in latent space (in-distribution threshold).",
    },
    attack_trajectory: {
      t_index: t,
      n_observed: stages.filter((s) => s.status === "observed").length,
      n_forecast: stages.filter((s) => s.status === "forecast").length,
      stages,
    },
    mitre: {
      t_index: t,
      n_observed: stages.filter((s) => s.status === "observed").length,
      n_forecast: stages.filter((s) => s.status === "forecast").length,
      stages,
      note: "High-level ATT&CK tactics only; observed vs forecast is explicit.",
    },
    propagation,
    explainability: buildExplainability(t, latest),
    stability: {
      t_index: t,
      K: k,
      n_trials: 16,
      epsilon: 0.05,
      mean_latent_drift: Number((0.02 + PHASE_INTENSITY[phaseForAnchor(t)] * 0.05).toFixed(4)),
      mean_risk_drift: Number((0.01 + uncStd * 0.3).toFixed(4)),
      stability_score: Number((1 - (0.02 + uncStd * 0.4)).toFixed(4)),
      method: "controlled-perturbation",
      note: "Forecast stability under bounded input perturbation (not MC-Dropout).",
    },
  };
}

export function makeHistory(upTo: number): NetworkHistoryResponse {
  const points = Array.from({ length: upTo + 1 }, (_, t) => ({
    t_index: t,
    risk: Number(baseRiskForAnchor(t).toFixed(4)),
    num_nodes: nodeCountForAnchor(t),
    num_edges: Math.floor(nodeCountForAnchor(t) * 1.6),
    label_any_attack: phaseForAnchor(t) === "normal" ? 0 : 1,
  }));
  return { dataset: DATASET, n: points.length, points };
}

// Simulate a counterfactual isolation: isolating the attacker hub node should
// suppress the forecast risk. Delta is derived from the intervention target.
export function makeCounterfactual(
  t: number,
  intervention: string,
  nodeIndex: number | null,
): ForecastResponse["propagation"] extends never ? never : import("./types").CounterfactualResponse {
  const base = makeForecast(t);
  const baseline = base.risk.forecast_risk;
  const isHub = nodeIndex === 0 || nodeIndex === null;
  const suppress = isHub ? 0.62 : 0.18;
  const intervRisk = baseline.map((r, i) =>
    Number(Math.max(0.01, r * (1 - suppress * ((i + 1) / baseline.length))).toFixed(4)),
  );
  const delta = intervRisk.map((r, i) => Number((r - baseline[i]).toFixed(4)));
  return {
    t_index: t,
    intervention,
    K: base.K,
    baseline_risk: baseline,
    intervention_risk: intervRisk,
    risk_delta: delta,
    mean_risk_delta: Number((delta.reduce((a, b) => a + b, 0) / delta.length).toFixed(4)),
    max_abs_risk_delta: Number(Math.max(...delta.map(Math.abs)).toFixed(4)),
    label: "Modelled / simulated outcome.",
  };
}

export const SCENARIO_META = { SEQ_LEN, K, N_ANCHORS, RISK_THRESHOLD, DATASET };
