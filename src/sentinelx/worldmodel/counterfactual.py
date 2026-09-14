"""Phase-8 (3/4 + 4/4): Counterfactual simulation + forecast stability.

COUNTERFACTUAL (simulation-only interventions)
----------------------------------------------
Answer "what would the model forecast if we intervened on the graph?" by
cloning the observed graph sequence, applying a *simulated* structural
intervention, re-running the trained World Model, and comparing the baseline
K-step forecast against the intervention forecast.

    Current G_t
        -> clone graph
        -> apply simulated intervention
        -> run trained world model
        -> K-step forecast
        -> compare baseline vs intervention

Supported interventions (all simulation-only, never touching real infra):

* ``isolate_node``   : remove all edges incident to a node (and zero its
  activity features) — simulates quarantining a host.
* ``remove_edge``    : drop a specific (src, dst) communication edge.
* ``suppress_path``  : remove a chain of edges forming a communication path.

Every result is labelled ``"Modelled / simulated outcome."`` and we NEVER claim
causal certainty. The comparison reports the modelled *difference* the
intervention would make under the learned dynamics — a hypothesis, not proof.

FORECAST STABILITY
------------------
Evaluate sensitivity of the forecast to *controlled, valid* perturbations of the
inputs (small feature jitter that keeps inputs in a plausible range). We compare
the original forecast against the perturbed forecast and report a documented
stability metric (mean relative latent drift + risk drift across trials).

Stability here is a **deterministic input-sensitivity** measure: the model is in
eval mode (dropout OFF) throughout, so this is explicitly NOT MC-Dropout
uncertainty (which perturbs the *model* by sampling dropout masks). The two
answer different questions and are kept separate.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .model import GraphTensors, SentinelXWorldModel
from .rollout import RolloutResult, rollout_latents

SIMULATION_LABEL = "Modelled / simulated outcome."

# Intervention kinds.
ISOLATE_NODE = "isolate_node"
REMOVE_EDGE = "remove_edge"
SUPPRESS_PATH = "suppress_path"
VALID_INTERVENTIONS = (ISOLATE_NODE, REMOVE_EDGE, SUPPRESS_PATH)


# --------------------------------------------------------------------------- #
# Graph cloning + simulated interventions (never mutate the caller's tensors).
# --------------------------------------------------------------------------- #
def clone_sequence(input_seq: Sequence[GraphTensors]) -> List[GraphTensors]:
    """Deep-clone an observed graph sequence so interventions are isolated."""
    out: List[GraphTensors] = []
    for g in input_seq:
        out.append(GraphTensors(
            x=g.x.detach().clone(),
            edge_index=g.edge_index.detach().clone(),
            edge_attr=g.edge_attr.detach().clone(),
            num_nodes=int(g.num_nodes),
        ))
    return out


def _drop_edges(g: GraphTensors, keep_mask: torch.Tensor) -> GraphTensors:
    """Return a copy of ``g`` keeping only edges where ``keep_mask`` is True."""
    if g.edge_index.numel() == 0:
        return g
    ei = g.edge_index[:, keep_mask]
    ea = g.edge_attr[keep_mask] if g.edge_attr.numel() else g.edge_attr
    return GraphTensors(x=g.x, edge_index=ei, edge_attr=ea, num_nodes=g.num_nodes)


def isolate_node(seq: List[GraphTensors], node_index: int) -> List[GraphTensors]:
    """Simulate isolating ``node_index``: drop incident edges + zero its features.

    Applied to EVERY window in the sequence (the node stays quarantined across
    the observed history). Node rows are kept (so indexing/graph size is stable)
    but their activity features are zeroed — a simulated "no traffic" host.
    """
    for i, g in enumerate(seq):
        if g.num_nodes == 0:
            continue
        if g.edge_index.numel():
            src, dst = g.edge_index[0], g.edge_index[1]
            keep = (src != node_index) & (dst != node_index)
            g = _drop_edges(g, keep)
        if 0 <= node_index < g.num_nodes:
            x = g.x.clone()
            x[node_index] = 0.0
            g = GraphTensors(x=x, edge_index=g.edge_index,
                             edge_attr=g.edge_attr, num_nodes=g.num_nodes)
        seq[i] = g
    return seq


def remove_edge(seq: List[GraphTensors], src: int, dst: int,
                timestep: Optional[int] = None) -> List[GraphTensors]:
    """Simulate removing the directed edge (src -> dst).

    If ``timestep`` is given, only that window is affected; otherwise the edge
    is removed from every window it appears in.
    """
    for i, g in enumerate(seq):
        if timestep is not None and i != timestep:
            continue
        if g.edge_index.numel() == 0:
            continue
        s, d = g.edge_index[0], g.edge_index[1]
        keep = ~((s == src) & (d == dst))
        seq[i] = _drop_edges(g, keep)
    return seq


def suppress_path(seq: List[GraphTensors], path: Sequence[int],
                  timestep: Optional[int] = None) -> List[GraphTensors]:
    """Simulate suppressing a communication path (chain of nodes).

    ``path`` = [a, b, c] removes edges (a->b) and (b->c) in the affected
    window(s). Reverse-direction edges along the path are left intact (only the
    forward communication path is suppressed).
    """
    edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    for (s, d) in edges:
        remove_edge(seq, s, d, timestep=timestep)
    return seq


def apply_intervention(seq: List[GraphTensors], kind: str, **params) -> List[GraphTensors]:
    """Dispatch a simulated intervention onto a (already cloned) sequence."""
    if kind == ISOLATE_NODE:
        return isolate_node(seq, int(params["node_index"]))
    if kind == REMOVE_EDGE:
        return remove_edge(seq, int(params["src"]), int(params["dst"]),
                           timestep=params.get("timestep"))
    if kind == SUPPRESS_PATH:
        return suppress_path(seq, [int(x) for x in params["path"]],
                             timestep=params.get("timestep"))
    raise ValueError(f"unknown intervention kind {kind!r}; "
                     f"valid: {VALID_INTERVENTIONS}")


# --------------------------------------------------------------------------- #
# Counterfactual comparison
# --------------------------------------------------------------------------- #
@dataclass
class CounterfactualResult:
    """Baseline vs simulated-intervention forecast comparison."""

    intervention: str
    params: Dict
    K: int
    baseline_risk: List[float]           # per-horizon baseline risk
    intervention_risk: List[float]       # per-horizon intervention risk
    risk_delta: List[float]              # intervention - baseline (per horizon)
    latent_shift: List[float]            # L2 distance of latent per horizon
    mean_risk_delta: float
    max_abs_risk_delta: float
    label: str = SIMULATION_LABEL
    metadata: Dict = field(default_factory=dict)

    def as_dict(self) -> Dict:
        return {
            "intervention": self.intervention,
            "params": dict(self.params),
            "K": self.K,
            "baseline_risk": [round(float(x), 6) for x in self.baseline_risk],
            "intervention_risk": [round(float(x), 6) for x in self.intervention_risk],
            "risk_delta": [round(float(x), 6) for x in self.risk_delta],
            "latent_shift": [round(float(x), 6) for x in self.latent_shift],
            "mean_risk_delta": round(float(self.mean_risk_delta), 6),
            "max_abs_risk_delta": round(float(self.max_abs_risk_delta), 6),
            "label": self.label,
            "metadata": dict(self.metadata),
        }


@torch.no_grad()
def simulate_intervention(
    model: SentinelXWorldModel,
    input_seq: Sequence[GraphTensors],
    intervention: str,
    *,
    K: int = 5,
    **params,
) -> CounterfactualResult:
    """Run the baseline vs simulated-intervention forecast comparison.

    Workflow (per the phase brief): clone G_t -> apply simulated intervention ->
    run the trained world model -> K-step forecast -> compare baseline vs
    intervention. The real graph is never modified; the intervention is applied
    only to a clone. The result is a MODELLED outcome, not a causal guarantee.
    """
    if intervention not in VALID_INTERVENTIONS:
        raise ValueError(f"unknown intervention {intervention!r}; "
                         f"valid: {VALID_INTERVENTIONS}")

    # Baseline rollout on the untouched (cloned) sequence.
    base_seq = clone_sequence(input_seq)
    base_roll: RolloutResult = rollout_latents(model, [base_seq], K)
    base_risk = base_roll.risk_probs[:, 0].cpu().tolist()          # (K,)
    base_latents = base_roll.latents[:, 0, :]                      # (K, latent)

    # Intervention rollout on a fresh clone with the simulated change applied.
    cf_seq = clone_sequence(input_seq)
    cf_seq = apply_intervention(cf_seq, intervention, **params)
    cf_roll: RolloutResult = rollout_latents(model, [cf_seq], K)
    cf_risk = cf_roll.risk_probs[:, 0].cpu().tolist()
    cf_latents = cf_roll.latents[:, 0, :]

    risk_delta = [c - b for c, b in zip(cf_risk, base_risk)]
    latent_shift = torch.linalg.vector_norm(
        cf_latents - base_latents, dim=1).cpu().tolist()           # (K,)

    mean_delta = sum(risk_delta) / len(risk_delta) if risk_delta else 0.0
    max_abs = max((abs(x) for x in risk_delta), default=0.0)

    return CounterfactualResult(
        intervention=intervention,
        params=params,
        K=K,
        baseline_risk=base_risk,
        intervention_risk=cf_risk,
        risk_delta=risk_delta,
        latent_shift=latent_shift,
        mean_risk_delta=mean_delta,
        max_abs_risk_delta=max_abs,
        metadata={
            "gnn_type": model.cfg.gnn_type,
            "temporal_type": model.cfg.temporal_type,
            "latent_dim": model.cfg.latent_dim,
            "deterministic": True,
            "note": ("Difference is the modelled effect under the learned "
                     "dynamics; not a causal guarantee."),
        },
    )


# --------------------------------------------------------------------------- #
# Forecast stability (controlled valid perturbations)  — NOT MC-Dropout.
# --------------------------------------------------------------------------- #
@dataclass
class StabilityResult:
    """Sensitivity of the forecast to controlled valid input perturbations."""

    n_trials: int
    epsilon: float
    K: int
    mean_latent_drift: float             # mean relative L2 latent drift over trials
    mean_risk_drift: float               # mean |risk_perturbed - risk_baseline|
    max_latent_drift: float
    max_risk_drift: float
    stability_score: float               # in (0,1]; 1 = perfectly stable
    per_horizon_risk_drift: List[float] = field(default_factory=list)
    method: str = "controlled-input-perturbation"
    note: str = (
        "Deterministic input-sensitivity (dropout OFF). This is NOT MC-Dropout "
        "model uncertainty; it measures how the forecast responds to small, "
        "valid perturbations of the observed inputs."
    )
    metadata: Dict = field(default_factory=dict)

    def as_dict(self) -> Dict:
        return {
            "n_trials": self.n_trials,
            "epsilon": self.epsilon,
            "K": self.K,
            "mean_latent_drift": round(float(self.mean_latent_drift), 6),
            "mean_risk_drift": round(float(self.mean_risk_drift), 6),
            "max_latent_drift": round(float(self.max_latent_drift), 6),
            "max_risk_drift": round(float(self.max_risk_drift), 6),
            "stability_score": round(float(self.stability_score), 6),
            "per_horizon_risk_drift": [round(float(x), 6)
                                       for x in self.per_horizon_risk_drift],
            "method": self.method,
            "note": self.note,
            "metadata": dict(self.metadata),
        }


def _perturb_sequence(
    input_seq: Sequence[GraphTensors],
    epsilon: float,
    generator: torch.Generator,
) -> List[GraphTensors]:
    """Apply small multiplicative jitter to non-negative activity features.

    Perturbation is *controlled and valid*: each feature f -> f * (1 + eps*u),
    u ~ Uniform(-1, 1), clamped at 0 so counts/bytes stay non-negative. Graph
    structure (edges) is left intact — we perturb magnitudes, not topology, so
    the input stays a plausible graph of the same shape.
    """
    out: List[GraphTensors] = []
    for g in input_seq:
        if g.num_nodes == 0:
            out.append(GraphTensors(x=g.x.clone(), edge_index=g.edge_index.clone(),
                                    edge_attr=g.edge_attr.clone(), num_nodes=0))
            continue
        ux = (torch.rand(g.x.shape, generator=generator) * 2 - 1) * epsilon
        x = (g.x * (1.0 + ux)).clamp_min(0.0)
        if g.edge_attr.numel():
            ue = (torch.rand(g.edge_attr.shape, generator=generator) * 2 - 1) * epsilon
            ea = (g.edge_attr * (1.0 + ue)).clamp_min(0.0)
        else:
            ea = g.edge_attr.clone()
        out.append(GraphTensors(x=x, edge_index=g.edge_index.clone(),
                                edge_attr=ea, num_nodes=g.num_nodes))
    return out


@torch.no_grad()
def evaluate_stability(
    model: SentinelXWorldModel,
    input_seq: Sequence[GraphTensors],
    *,
    K: int = 5,
    n_trials: int = 16,
    epsilon: float = 0.05,
    seed: int = 42,
) -> StabilityResult:
    """Measure forecast stability under controlled valid input perturbations.

    For each trial we jitter the observed inputs by <= ``epsilon`` (relative),
    re-run the deterministic rollout, and measure how far the forecast moved
    from the unperturbed baseline. Returns a documented stability metric.

    ``stability_score = 1 / (1 + mean_latent_drift)`` maps drift into (0, 1]:
    a score near 1 means the forecast barely moved (stable); a lower score means
    the forecast is sensitive to small input changes.
    """
    if epsilon < 0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon}")
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")

    was_training = model.training
    model.eval()  # dropout OFF: this is input-sensitivity, not MC-Dropout.
    try:
        base = rollout_latents(model, [clone_sequence(input_seq)], K)
        base_latents = base.latents[:, 0, :]          # (K, latent)
        base_risk = base.risk_probs[:, 0]             # (K,)
        base_norm = torch.linalg.vector_norm(base_latents, dim=1).clamp_min(1e-8)

        gen = torch.Generator().manual_seed(seed)
        latent_drifts: List[float] = []
        risk_drifts: List[float] = []
        horizon_risk_drift = torch.zeros(K)
        max_latent = 0.0
        max_risk = 0.0

        for _ in range(n_trials):
            pert = _perturb_sequence(input_seq, epsilon, gen)
            roll = rollout_latents(model, [pert], K)
            lat = roll.latents[:, 0, :]
            rsk = roll.risk_probs[:, 0]
            # relative latent drift per horizon, averaged over horizons
            rel = (torch.linalg.vector_norm(lat - base_latents, dim=1)
                   / base_norm)                       # (K,)
            rd = (rsk - base_risk).abs()               # (K,)
            latent_drifts.append(float(rel.mean()))
            risk_drifts.append(float(rd.mean()))
            horizon_risk_drift += rd
            max_latent = max(max_latent, float(rel.max()))
            max_risk = max(max_risk, float(rd.max()))

        mean_latent = sum(latent_drifts) / len(latent_drifts)
        mean_risk = sum(risk_drifts) / len(risk_drifts)
        per_h = (horizon_risk_drift / n_trials).cpu().tolist()
        stability_score = 1.0 / (1.0 + mean_latent)
    finally:
        if was_training:
            model.train()

    return StabilityResult(
        n_trials=n_trials, epsilon=epsilon, K=K,
        mean_latent_drift=mean_latent, mean_risk_drift=mean_risk,
        max_latent_drift=max_latent, max_risk_drift=max_risk,
        stability_score=stability_score, per_horizon_risk_drift=per_h,
        metadata={
            "gnn_type": model.cfg.gnn_type,
            "latent_dim": model.cfg.latent_dim,
            "seed": seed,
            "deterministic_rollout": True,
        },
    )
