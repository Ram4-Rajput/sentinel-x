"""Typed Pydantic schemas for the Sentinel-X serving API.

These are the ONLY shapes that cross the API boundary. Internal tensors are
never exposed; latent vectors appear solely as ``list[float]`` where the
frontend genuinely needs them (forecast trajectory). Every field is named and
typed so the frontend has a stable contract.

Pydantic v2 models. ``model_config = ConfigDict(extra="forbid")`` is used on
request bodies (reject unknown fields) and left permissive on deeply-nested
analytical payloads that mirror the worldmodel dataclasses' ``as_dict()``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
class HealthResponse(BaseModel):
    # 'model_loaded' starts with the reserved 'model_' prefix; opt out of
    # Pydantic's protected namespace so no warning is raised for this ops field.
    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(..., description="'ok' when the service is live.")
    model_loaded: bool
    data_usable: bool
    checkpoint_available: bool
    combo: Optional[str] = None
    dataset: Optional[str] = None
    seq_len: Optional[int] = None
    horizon: Optional[int] = None
    serving_k: Optional[int] = None
    num_test_anchors: Optional[int] = None
    risk_threshold: Optional[float] = None
    version: str = "0.1.0"


# --------------------------------------------------------------------------- #
# Shared request pieces
# --------------------------------------------------------------------------- #
class AnchorSelector(BaseModel):
    """Selects which anchor window to operate on.

    ``t_index`` picks a specific test anchor; when omitted the latest (most
    recent) test anchor is used as the 'current' network state.
    """

    model_config = ConfigDict(extra="forbid")
    t_index: Optional[int] = Field(
        None, description="Anchor window index; latest test anchor if omitted.")


class ForecastRequest(BaseModel):
    """Body for POST /forecast — the primary aggregate endpoint."""

    model_config = ConfigDict(extra="forbid")
    t_index: Optional[int] = Field(
        None, description="Anchor window index; latest test anchor if omitted.")
    horizons: Optional[int] = Field(
        None, ge=1, le=10,
        description="How many future steps (K) to roll out. Bounded [1,10].")
    include_explainability: bool = True
    include_propagation: bool = True
    include_uncertainty: bool = True
    include_novelty: bool = True
    include_mitre: bool = True
    include_stability: bool = True
    mc_passes: int = Field(30, ge=1, le=200,
                           description="MC-Dropout passes for uncertainty.")


class IngestWindow(BaseModel):
    """A single network-state window in the Phase-2 cached graph format.

    This mirrors the on-disk cache dict so callers can stream in a live window
    without any new feature engineering. Node/edge feature column order matches
    ``graph_cache`` (node: out_deg,in_deg,bytes_sent,bytes_recv,flow_count;
    edge: flow_count,total_bytes,contains_attack).
    """

    model_config = ConfigDict(extra="forbid")
    num_nodes: int = Field(..., ge=0)
    node_features: List[List[float]] = Field(default_factory=list)
    edge_index: List[List[int]] = Field(default_factory=list)
    edge_features: List[List[float]] = Field(default_factory=list)
    node_ids: Optional[List[str]] = None
    window_index: Optional[int] = None
    label_any_attack: Optional[int] = None


class IngestRequest(BaseModel):
    """Body for POST /ingest — a sequence of observed windows ending at t.

    The sequence length should match the model's ``seq_len``; shorter sequences
    are left-padded with empty windows, longer ones use the last ``seq_len``.
    """

    model_config = ConfigDict(extra="forbid")
    windows: List[IngestWindow] = Field(..., min_length=1)
    horizons: Optional[int] = Field(None, ge=1, le=10)


class CounterfactualRequest(BaseModel):
    """Body for POST /counterfactual — a simulation-only intervention."""

    model_config = ConfigDict(extra="forbid")
    t_index: Optional[int] = None
    intervention: str = Field(
        ..., description="'isolate_node' | 'remove_edge' | 'suppress_path'.")
    horizons: Optional[int] = Field(None, ge=1, le=10)
    node_index: Optional[int] = None
    src: Optional[int] = None
    dst: Optional[int] = None
    path: Optional[List[int]] = None
    timestep: Optional[int] = None


# --------------------------------------------------------------------------- #
# Network state / graph
# --------------------------------------------------------------------------- #
class GraphNode(BaseModel):
    id: str
    index: int
    out_degree: float
    in_degree: float
    bytes_sent: float
    bytes_received: float
    flow_count: float


class GraphEdge(BaseModel):
    src: int
    dst: int
    src_id: str
    dst_id: str
    flow_count: float
    total_bytes: float
    contains_attack: bool


class NetworkGraph(BaseModel):
    """One window's graph as nodes + edges (typed; no tensors)."""

    timestep: int = Field(..., description="Position in the observed sequence.")
    window_index: Optional[int] = None
    num_nodes: int
    num_edges: int
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)


class NetworkState(BaseModel):
    """The current observed network state at anchor ``t``."""

    t_index: int
    seq_len: int
    latest_graph: NetworkGraph
    observed_graphs: List[NetworkGraph] = Field(default_factory=list)
    risk_now: float = Field(..., description="Deterministic risk at anchor t.")
    label_any_attack: Optional[int] = None


class NetworkHistoryPoint(BaseModel):
    t_index: int
    risk: float
    num_nodes: int
    num_edges: int
    label_any_attack: Optional[int] = None


class NetworkHistoryResponse(BaseModel):
    dataset: Optional[str]
    n: int
    points: List[NetworkHistoryPoint] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Forecast trajectory (future graph states + risk per horizon)
# --------------------------------------------------------------------------- #
class ForecastStep(BaseModel):
    horizon: int
    target_index: int
    risk: float = Field(..., description="Predicted risk at t+k.")
    latent: List[float] = Field(
        ..., description="Predicted latent network state z_hat_{t+k}.")


class ForecastTrajectory(BaseModel):
    t_index: int
    K: int
    current_latent: List[float] = Field(
        ..., description="Encoded current latent state z_t.")
    steps: List[ForecastStep] = Field(default_factory=list)
    risk_threshold: float


# --------------------------------------------------------------------------- #
# Risk / uncertainty / novelty
# --------------------------------------------------------------------------- #
class RiskResponse(BaseModel):
    t_index: int
    risk_now: float
    risk_threshold: float
    alert: bool = Field(..., description="risk_now >= risk_threshold.")
    forecast_risk: List[float] = Field(
        default_factory=list, description="Predicted risk per horizon t+1..t+K.")
    horizons: List[int] = Field(default_factory=list)


class UncertaintyResponse(BaseModel):
    t_index: int
    method: str = "mc-dropout"
    n_passes: int
    risk_mean: float
    uncertainty_variance: float
    uncertainty_std: float
    note: str = (
        "Risk (point estimate) and uncertainty (predictive variance over "
        "MC-Dropout passes) are reported separately.")


class NoveltyResponse(BaseModel):
    t_index: int
    method: str = "mahalanobis"
    novelty_score: float
    is_novel: bool
    threshold: Optional[float] = None
    fit_percentile: Optional[float] = None
    note: str = (
        "Novelty is a Mahalanobis distance in latent space; the threshold is "
        "derived from in-distribution (train/val) latents only.")


# --------------------------------------------------------------------------- #
# MITRE / trajectory / propagation / explainability / counterfactual
# (deep analytical payloads mirror the worldmodel dataclasses' as_dict())
# --------------------------------------------------------------------------- #
class MitreStage(BaseModel):
    model_config = ConfigDict(extra="allow")
    stage: str
    status: str
    confidence: float
    evidence: List[str] = Field(default_factory=list)
    horizon: int = 0
    window_index: Optional[int] = None
    source: str = ""


class TrajectoryResponse(BaseModel):
    t_index: int
    n_observed: int
    n_forecast: int
    stages: List[MitreStage] = Field(default_factory=list)


class MitreResponse(BaseModel):
    """High-level MITRE interpretation for the anchor (same as trajectory)."""

    t_index: int
    n_observed: int
    n_forecast: int
    stages: List[MitreStage] = Field(default_factory=list)
    note: str = (
        "High-level ATT&CK tactics only; observed vs forecast is explicit and "
        "precise technique IDs are never asserted.")


class PropagationResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    t_index: int
    suspicion_source: str
    total_affected: int
    velocity: float
    direction: str
    is_spreading: bool
    per_window: List[Dict[str, Any]] = Field(default_factory=list)


class ExplainabilityResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    t_index: int
    target: str
    target_value: float
    method: str
    uses_attention: bool
    top_features: List[Dict[str, Any]] = Field(default_factory=list)
    important_nodes: List[Dict[str, Any]] = Field(default_factory=list)
    important_edges: List[Dict[str, Any]] = Field(default_factory=list)
    temporal_evidence: List[Dict[str, Any]] = Field(default_factory=list)


class CounterfactualResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    t_index: int
    intervention: str
    K: int
    baseline_risk: List[float] = Field(default_factory=list)
    intervention_risk: List[float] = Field(default_factory=list)
    risk_delta: List[float] = Field(default_factory=list)
    mean_risk_delta: float
    max_abs_risk_delta: float
    label: str = "Modelled / simulated outcome."


class StabilityResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    t_index: int
    K: int
    n_trials: int
    epsilon: float
    mean_latent_drift: float
    mean_risk_drift: float
    stability_score: float
    method: str
    note: str


# --------------------------------------------------------------------------- #
# Experiments (Phase-9 registry + comparison)
# --------------------------------------------------------------------------- #
class ExperimentsResponse(BaseModel):
    available_experiments: List[str] = Field(default_factory=list)
    comparison_columns: List[str] = Field(default_factory=list)
    comparison: List[Dict[str, Any]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# The aggregate /forecast response — everything the frontend needs.
# --------------------------------------------------------------------------- #
class ForecastResponse(BaseModel):
    t_index: int
    dataset: Optional[str]
    K: int
    risk_threshold: float

    # current network state + graph
    network_state: NetworkState
    graph_nodes: List[GraphNode] = Field(default_factory=list)
    graph_edges: List[GraphEdge] = Field(default_factory=list)

    # future graph states + horizons
    forecast: ForecastTrajectory
    horizons: List[int] = Field(default_factory=list)

    # analytical layers
    risk: RiskResponse
    uncertainty: Optional[UncertaintyResponse] = None
    novelty: Optional[NoveltyResponse] = None
    attack_trajectory: TrajectoryResponse
    mitre: MitreResponse
    propagation: Optional[PropagationResponse] = None
    explainability: Optional[ExplainabilityResponse] = None
    stability: Optional[StabilityResponse] = None


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class ErrorResponse(BaseModel):
    detail: str
