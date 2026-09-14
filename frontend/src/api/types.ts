// TypeScript mirror of the Sentinel-X serving API (Pydantic v2 schemas).
// Field names & shapes match src/sentinelx/serving/schemas.py 1:1 so the app
// speaks the real contract whether data comes from the live model or the mock.

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  data_usable: boolean;
  checkpoint_available: boolean;
  combo?: string | null;
  dataset?: string | null;
  seq_len?: number | null;
  horizon?: number | null;
  serving_k?: number | null;
  num_test_anchors?: number | null;
  risk_threshold?: number | null;
  version: string;
}

export interface GraphNode {
  id: string;
  index: number;
  out_degree: number;
  in_degree: number;
  bytes_sent: number;
  bytes_received: number;
  flow_count: number;
}

export interface GraphEdge {
  src: number;
  dst: number;
  src_id: string;
  dst_id: string;
  flow_count: number;
  total_bytes: number;
  contains_attack: boolean;
}

export interface NetworkGraph {
  timestep: number;
  window_index?: number | null;
  num_nodes: number;
  num_edges: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface NetworkState {
  t_index: number;
  seq_len: number;
  latest_graph: NetworkGraph;
  observed_graphs: NetworkGraph[];
  risk_now: number;
  label_any_attack?: number | null;
}

export interface NetworkHistoryPoint {
  t_index: number;
  risk: number;
  num_nodes: number;
  num_edges: number;
  label_any_attack?: number | null;
}

export interface NetworkHistoryResponse {
  dataset?: string | null;
  n: number;
  points: NetworkHistoryPoint[];
}

export interface ForecastStep {
  horizon: number;
  target_index: number;
  risk: number;
  latent: number[];
}

export interface ForecastTrajectory {
  t_index: number;
  K: number;
  current_latent: number[];
  steps: ForecastStep[];
  risk_threshold: number;
}

export interface RiskResponse {
  t_index: number;
  risk_now: number;
  risk_threshold: number;
  alert: boolean;
  forecast_risk: number[];
  horizons: number[];
}

export interface UncertaintyResponse {
  t_index: number;
  method: string;
  n_passes: number;
  risk_mean: number;
  uncertainty_variance: number;
  uncertainty_std: number;
  note: string;
}

export interface NoveltyResponse {
  t_index: number;
  method: string;
  novelty_score: number;
  is_novel: boolean;
  threshold?: number | null;
  fit_percentile?: number | null;
  note: string;
}

export interface MitreStage {
  stage: string;
  status: "observed" | "forecast";
  confidence: number;
  evidence: string[];
  horizon: number;
  window_index?: number | null;
  source: string;
  timestamp?: string | null;
}

export interface TrajectoryResponse {
  t_index: number;
  n_observed: number;
  n_forecast: number;
  stages: MitreStage[];
}

export interface MitreResponse extends TrajectoryResponse {
  note?: string;
}

export interface WindowPropagation {
  timestep: number;
  window_index?: number | null;
  affected: string[];
  newly_affected: string[];
  n_affected: number;
  n_newly_affected: number;
  cumulative_affected: number;
  suspicious_edges: number;
  out_edges: number;
  in_edges: number;
  branching: number;
}

export interface PropagationResponse {
  t_index: number;
  suspicion_source: string;
  total_affected: number;
  velocity: number;
  peak_velocity?: number;
  direction: string;
  direction_ratio?: number;
  mean_branching?: number;
  persistence?: number;
  growth?: number;
  is_spreading: boolean;
  per_window: WindowPropagation[];
}

export interface FeatureAttribution {
  name: string;
  scope: "node" | "edge" | "temporal";
  attribution: number;
  abs_attribution: number;
}

export interface NodeImportance {
  node_index: number;
  timestep: number;
  importance: number;
  node_id?: string | null;
}

export interface EdgeImportance {
  src: number;
  dst: number;
  timestep: number;
  importance: number;
  attention?: number;
}

export interface TemporalEvidence {
  timestep: number;
  relative_horizon: number;
  importance: number;
}

export interface ExplainabilityResponse {
  t_index: number;
  target: string;
  target_value: number;
  method: string;
  uses_attention: boolean;
  attention_note?: string;
  top_features: FeatureAttribution[];
  important_nodes: NodeImportance[];
  important_edges: EdgeImportance[];
  temporal_evidence: TemporalEvidence[];
  metadata?: Record<string, unknown>;
}

export interface CounterfactualResponse {
  t_index: number;
  intervention: string;
  K: number;
  baseline_risk: number[];
  intervention_risk: number[];
  risk_delta: number[];
  mean_risk_delta: number;
  max_abs_risk_delta: number;
  latent_shift?: number[];
  label: string;
}

export interface StabilityResponse {
  t_index: number;
  K: number;
  n_trials: number;
  epsilon: number;
  mean_latent_drift: number;
  mean_risk_drift: number;
  stability_score: number;
  method: string;
  note: string;
}

export interface ForecastResponse {
  t_index: number;
  dataset?: string | null;
  K: number;
  risk_threshold: number;
  network_state: NetworkState;
  graph_nodes: GraphNode[];
  graph_edges: GraphEdge[];
  forecast: ForecastTrajectory;
  horizons: number[];
  risk: RiskResponse;
  uncertainty?: UncertaintyResponse | null;
  novelty?: NoveltyResponse | null;
  attack_trajectory: TrajectoryResponse;
  mitre: MitreResponse;
  propagation?: PropagationResponse | null;
  explainability?: ExplainabilityResponse | null;
  stability?: StabilityResponse | null;
}

// Intervention request for /counterfactual (what-if).
export interface CounterfactualRequest {
  t_index?: number | null;
  intervention: "isolate_node" | "remove_edge" | "suppress_path";
  horizons?: number | null;
  node_index?: number | null;
  src?: number | null;
  dst?: number | null;
  path?: number[] | null;
  timestep?: number | null;
}
