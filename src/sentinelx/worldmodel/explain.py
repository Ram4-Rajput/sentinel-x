"""Phase-8 (1/4): Explainability for the trained world model.

Answers **"why did the model produce this forecast?"** with model-attributed
evidence — not hand-written rules. Every quantity here is derived from the
trained Phase-4 checkpoint applied to real cached windows; nothing is
fabricated.

What we attribute
-----------------
* **node features / edge features** — feature-level attribution via a gradient
  x input sensitivity of the forecast objective w.r.t. the inputs. This is the
  standard "how much did moving this input move the output" attribution; it is
  a *sensitivity*, not a causal claim.
* **graph structure** — attribution rolled up per node / per edge so we can rank
  ``important_nodes`` and ``important_edges``.
* **temporal evidence** — sensitivity of the forecast to each observed timestep
  in the sequence, showing which past windows mattered most.
* **attention** — when the encoder is a GAT we additionally expose the learned
  attention weights as *supporting* evidence. Attention is reported as
  "association / where the model looked", explicitly NOT as a causal signal
  (see ``ATTENTION_DISCLAIMER``). GraphSAGE encoders have no attention, so that
  channel is simply absent and we fall back to gradient attribution only.

Target of the explanation
--------------------------
By default we explain the model's PRIMARY output — the predicted next latent
state ``z_hat_{t+1}`` — via a scalar summary (its L2 norm / movement), because
the world model's job is future-state forecasting, not attack yes/no. The
auxiliary risk logit can be explained instead by passing
``target="risk"`` — useful when the operator cares about "why did risk go up".

Design constraints (from the phase brief)
------------------------------------------
* Do NOT modify the core forecasting architecture. We only *read* gradients and
  (optionally) attention off the existing modules.
* Do NOT call attention causal.
* Return ``top_features``, ``important_nodes``, ``important_edges``,
  ``temporal_evidence``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..pipeline.graph_cache import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES
from .model import GraphTensors, SentinelXWorldModel


ATTENTION_DISCLAIMER = (
    "Attention weights indicate where the model attended (association), not "
    "cause. They are reported as supporting evidence only and must not be "
    "interpreted as a causal explanation of the forecast."
)

# Which scalar of the forecast we differentiate. "state" = movement of the
# predicted next latent (primary objective); "risk" = auxiliary risk logit.
VALID_TARGETS = ("state", "risk")


@dataclass
class FeatureAttribution:
    """Attribution for one named input feature (node- or edge-level)."""

    name: str
    scope: str                 # "node" | "edge" | "temporal"
    attribution: float         # signed sensitivity aggregated over the input
    abs_attribution: float     # magnitude used for ranking

    def as_dict(self) -> Dict:
        return {
            "name": self.name,
            "scope": self.scope,
            "attribution": round(float(self.attribution), 6),
            "abs_attribution": round(float(self.abs_attribution), 6),
        }


@dataclass
class NodeImportance:
    node_index: int
    timestep: int              # which observed window in the sequence
    importance: float          # magnitude of gradient-based attribution
    node_id: Optional[str] = None

    def as_dict(self) -> Dict:
        return {
            "node_index": self.node_index,
            "timestep": self.timestep,
            "importance": round(float(self.importance), 6),
            "node_id": self.node_id,
        }


@dataclass
class EdgeImportance:
    src: int
    dst: int
    timestep: int
    importance: float          # gradient magnitude + optional attention support
    attention: Optional[float] = None   # supporting association signal (GAT only)

    def as_dict(self) -> Dict:
        d = {
            "src": self.src,
            "dst": self.dst,
            "timestep": self.timestep,
            "importance": round(float(self.importance), 6),
        }
        if self.attention is not None:
            d["attention"] = round(float(self.attention), 6)
        return d


@dataclass
class TemporalEvidence:
    timestep: int              # position in the observed sequence (0 = oldest)
    relative_horizon: int      # <=0; 0 is the anchor window t
    importance: float          # how much the forecast depends on this window

    def as_dict(self) -> Dict:
        return {
            "timestep": self.timestep,
            "relative_horizon": self.relative_horizon,
            "importance": round(float(self.importance), 6),
        }


@dataclass
class Explanation:
    """The Phase-8 explainability contract for one forecast."""

    target: str                                  # "state" | "risk"
    target_value: float                          # scalar being explained
    top_features: List[FeatureAttribution] = field(default_factory=list)
    important_nodes: List[NodeImportance] = field(default_factory=list)
    important_edges: List[EdgeImportance] = field(default_factory=list)
    temporal_evidence: List[TemporalEvidence] = field(default_factory=list)
    method: str = "gradient-x-input"
    uses_attention: bool = False
    attention_note: str = ""
    metadata: Dict = field(default_factory=dict)

    def as_dict(self) -> Dict:
        return {
            "target": self.target,
            "target_value": round(float(self.target_value), 6),
            "method": self.method,
            "uses_attention": self.uses_attention,
            "attention_note": self.attention_note,
            "top_features": [f.as_dict() for f in self.top_features],
            "important_nodes": [n.as_dict() for n in self.important_nodes],
            "important_edges": [e.as_dict() for e in self.important_edges],
            "temporal_evidence": [t.as_dict() for t in self.temporal_evidence],
            "metadata": dict(self.metadata),
        }


def _validate_target(target: str) -> str:
    t = str(target).lower()
    if t not in VALID_TARGETS:
        raise ValueError(f"target must be one of {VALID_TARGETS}, got {target!r}")
    return t


def _clone_input_with_grad(
    input_seq: Sequence[GraphTensors],
) -> List[GraphTensors]:
    """Deep-copy an observed sequence with x / edge_attr set to require grad.

    We never mutate the caller's tensors; a fresh leaf tensor is created per
    timestep so gradients attach cleanly.
    """
    cloned: List[GraphTensors] = []
    for g in input_seq:
        if g.num_nodes == 0:
            cloned.append(GraphTensors(
                x=g.x.clone(), edge_index=g.edge_index.clone(),
                edge_attr=g.edge_attr.clone(), num_nodes=0))
            continue
        x = g.x.detach().clone().requires_grad_(True)
        ea = (g.edge_attr.detach().clone().requires_grad_(True)
              if g.edge_attr.numel() else g.edge_attr.detach().clone())
        cloned.append(GraphTensors(
            x=x, edge_index=g.edge_index.detach().clone(),
            edge_attr=ea, num_nodes=g.num_nodes))
    return cloned


def _target_scalar(model: SentinelXWorldModel,
                   batch_seq: List[List[GraphTensors]],
                   target: str) -> Tuple[torch.Tensor, float]:
    """Compute the differentiable scalar being explained for a single sample.

    * "state" -> L2 norm of the predicted next latent z_hat_{t+1}. This is a
      faithful scalar summary of the PRIMARY forecast: how far the network
      state is predicted to move. Larger norm = a more pronounced predicted
      transition, and its gradient tells us which inputs drove that movement.
    * "risk" -> the auxiliary risk logit (pre-sigmoid), the natural scalar for
      "why did predicted risk change".
    """
    out = model(batch_seq)
    if target == "risk":
        scalar = out.risk_logit.squeeze()
    else:
        scalar = torch.linalg.vector_norm(out.z_next_pred.squeeze(0))
    return scalar, float(scalar.detach())


def _collect_attention(
    model: SentinelXWorldModel,
    input_seq: Sequence[GraphTensors],
) -> Optional[List[Optional[torch.Tensor]]]:
    """Best-effort capture of GAT attention per timestep (supporting evidence).

    Returns a list aligned with ``input_seq``; each element is a 1-D tensor of
    per-edge attention (averaged over heads) or ``None`` for empty/edgeless
    graphs. Returns ``None`` entirely when the encoder is not attention-based.
    The core architecture is NOT modified — we register temporary forward hooks
    that request ``return_attention_weights`` from the first GAT layer, then
    remove them.
    """
    if model.cfg.gnn_type != "gat":
        return None

    encoder = model.graph_encoder
    if not encoder.convs:
        return None
    first_conv = encoder.convs[0]

    captured: Dict[str, torch.Tensor] = {}

    def hook(module, inputs, output):
        # We re-run the conv with return_attention_weights to read alpha without
        # altering the module's normal forward output.
        try:
            x = inputs[0]
            edge_index = inputs[1]
            edge_attr = inputs[2] if len(inputs) > 2 else None
            with torch.no_grad():
                if edge_attr is not None:
                    _, (ei, alpha) = module(
                        x, edge_index, edge_attr=edge_attr,
                        return_attention_weights=True)
                else:
                    _, (ei, alpha) = module(
                        x, edge_index, return_attention_weights=True)
            captured["alpha"] = alpha.detach()
            captured["edge_index"] = ei.detach()
        except Exception:
            pass  # attention is optional supporting evidence

    per_ts: List[Optional[torch.Tensor]] = []
    handle = first_conv.register_forward_hook(hook)
    try:
        for g in input_seq:
            captured.clear()
            if g.num_nodes == 0 or g.edge_index.numel() == 0:
                per_ts.append(None)
                continue
            bvec = torch.zeros((g.num_nodes,), dtype=torch.long)
            with torch.no_grad():
                model.graph_encoder(g.x, g.edge_index, bvec, g.edge_attr)
            alpha = captured.get("alpha")
            if alpha is None:
                per_ts.append(None)
            else:
                # (E, heads) -> mean over heads, keep only the original E edges
                a = alpha.mean(dim=1) if alpha.dim() > 1 else alpha
                per_ts.append(a[: g.edge_index.shape[1]].cpu())
    finally:
        handle.remove()
    return per_ts


def explain_forecast(
    model: SentinelXWorldModel,
    input_seq: Sequence[GraphTensors],
    *,
    target: str = "state",
    top_k_features: int = 10,
    top_k_nodes: int = 10,
    top_k_edges: int = 10,
    node_ids_by_ts: Optional[Sequence[Optional[Sequence[str]]]] = None,
) -> Explanation:
    """Explain one forecast with model-attributed evidence.

    Parameters
    ----------
    model : trained Sentinel-X world model.
    input_seq : the observed graph sequence [t-L+1 .. t] for ONE sample.
    target : "state" (primary forecast movement) or "risk" (auxiliary).
    node_ids_by_ts : optional per-timestep node id lists (from the cache) so
        ``important_nodes`` can carry human-readable entity ids.

    Returns an :class:`Explanation` with ``top_features``, ``important_nodes``,
    ``important_edges`` and ``temporal_evidence``.
    """
    target = _validate_target(target)
    was_training = model.training
    model.eval()  # deterministic attribution (dropout off)
    try:
        cloned = _clone_input_with_grad(input_seq)
        model.zero_grad(set_to_none=True)
        scalar, scalar_val = _target_scalar(model, [cloned], target)
        scalar.backward()

        L = len(cloned)
        node_dim = model.cfg.node_feature_dim
        edge_dim = model.cfg.edge_feature_dim

        # ---- feature-level attribution (grad x input), aggregated ----
        node_feat_attr = torch.zeros(node_dim)
        edge_feat_attr = torch.zeros(edge_dim)
        temporal = []          # per-timestep importance
        node_imps: List[NodeImportance] = []
        edge_imps: List[EdgeImportance] = []

        attentions = _collect_attention(model, input_seq)

        for ts, g in enumerate(cloned):
            ts_importance = 0.0
            if g.num_nodes > 0 and g.x.grad is not None:
                gi = g.x.grad * g.x                     # (N, node_dim)
                node_feat_attr += gi.sum(dim=0).detach().cpu()
                per_node = gi.abs().sum(dim=1).detach().cpu()   # (N,)
                ts_importance += float(per_node.sum())
                ids = None
                if node_ids_by_ts is not None and ts < len(node_ids_by_ts):
                    ids = node_ids_by_ts[ts]
                for ni in range(g.num_nodes):
                    node_imps.append(NodeImportance(
                        node_index=ni, timestep=ts,
                        importance=float(per_node[ni]),
                        node_id=(ids[ni] if ids and ni < len(ids) else None),
                    ))
            if g.edge_attr.numel() and g.edge_attr.grad is not None:
                ge = g.edge_attr.grad * g.edge_attr     # (E, edge_dim)
                edge_feat_attr += ge.sum(dim=0).detach().cpu()
                per_edge = ge.abs().sum(dim=1).detach().cpu()   # (E,)
                ts_importance += float(per_edge.sum())
                att_ts = attentions[ts] if attentions is not None else None
                ei = g.edge_index.cpu()
                for e in range(ei.shape[1]):
                    att = (float(att_ts[e]) if att_ts is not None
                           and e < att_ts.shape[0] else None)
                    edge_imps.append(EdgeImportance(
                        src=int(ei[0, e]), dst=int(ei[1, e]), timestep=ts,
                        importance=float(per_edge[e]), attention=att))
            temporal.append(ts_importance)

        # ---- assemble ranked outputs ----
        feat_attrs: List[FeatureAttribution] = []
        for i, nm in enumerate(NODE_FEATURE_NAMES[:node_dim]):
            v = float(node_feat_attr[i])
            feat_attrs.append(FeatureAttribution(
                name=f"node.{nm}", scope="node", attribution=v, abs_attribution=abs(v)))
        for i, nm in enumerate(EDGE_FEATURE_NAMES[:edge_dim]):
            v = float(edge_feat_attr[i])
            feat_attrs.append(FeatureAttribution(
                name=f"edge.{nm}", scope="edge", attribution=v, abs_attribution=abs(v)))
        feat_attrs.sort(key=lambda f: f.abs_attribution, reverse=True)

        node_imps.sort(key=lambda n: n.importance, reverse=True)
        edge_imps.sort(key=lambda e: e.importance, reverse=True)

        temporal_evidence = [
            TemporalEvidence(timestep=ts, relative_horizon=ts - (L - 1),
                             importance=imp)
            for ts, imp in enumerate(temporal)
        ]

        expl = Explanation(
            target=target,
            target_value=scalar_val,
            top_features=feat_attrs[:top_k_features],
            important_nodes=node_imps[:top_k_nodes],
            important_edges=edge_imps[:top_k_edges],
            temporal_evidence=temporal_evidence,
            method="gradient-x-input",
            uses_attention=bool(attentions is not None),
            attention_note=(ATTENTION_DISCLAIMER if attentions is not None else ""),
            metadata={
                "gnn_type": model.cfg.gnn_type,
                "temporal_type": model.cfg.temporal_type,
                "latent_dim": model.cfg.latent_dim,
                "seq_len": L,
                "deterministic": True,
            },
        )
    finally:
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()
    return expl
