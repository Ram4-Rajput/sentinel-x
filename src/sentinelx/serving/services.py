"""Service layer for the Sentinel-X serving API.

This is the ONLY module that runs ML operations for the API. Route handlers
call these functions and serialise the result; they never touch the model,
tensors, rollouts, or the worldmodel package directly.

Everything here is a thin, well-typed adapter over the already-built worldmodel
+ research packages (Phases 4-9). No training, no cache rebuild: we reuse the
loaded checkpoint and the K-step samples held by :class:`SentinelRuntime`.

Latent vectors are the only numeric arrays that leave this layer, and only as
plain Python ``list[float]`` for the forecast trajectory the frontend renders.
Raw ``torch.Tensor`` objects never escape.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from ..pipeline.graph_cache import (
    EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES,
)
from ..worldmodel.calibration import evaluate_calibration
from ..worldmodel.counterfactual import (
    VALID_INTERVENTIONS, simulate_intervention, evaluate_stability,
)
from ..worldmodel.explain import explain_forecast
from ..worldmodel.kstep_data import KStepSample, collate_inputs
from ..worldmodel.model import GraphTensors, SentinelXWorldModel
from ..worldmodel.ood import MahalanobisOOD, encode_latents
from ..worldmodel.propagation import SUSPICION_LABEL, analyze_propagation
from ..worldmodel.rollout import rollout_latents
from ..worldmodel.trajectory import build_trajectory
from ..worldmodel.uncertainty import mc_dropout_risk
from .runtime import SentinelRuntime


class ServiceError(Exception):
    """Raised for expected, user-facing service failures (mapped to 4xx)."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


# --------------------------------------------------------------------------- #
# Small internal helpers (tensor -> plain python; no tensors escape)
# --------------------------------------------------------------------------- #
def _deterministic_risk(rt: SentinelRuntime, input_seq: List[GraphTensors]) -> float:
    """Single dropout-OFF risk pass for one observed sequence."""
    model = rt.model
    model.eval()
    with torch.no_grad():
        out = model([input_seq])
        return float(torch.sigmoid(out.risk_logit)[0].item())


def _graph_dict_from_sample_tensor(g: GraphTensors, timestep: int) -> Dict:
    """Reconstruct a cache-style graph dict from a sample's tensors.

    Mirrors the reconstruction the Phase-7/8 drivers use so propagation and the
    node/edge serialisers share one representation. Feature column order follows
    graph_cache: node = NODE_FEATURE_NAMES, edge = EDGE_FEATURE_NAMES.
    """
    n_nodes = int(g.num_nodes)
    node_features = g.x.cpu().tolist() if n_nodes > 0 else []
    edge_features = g.edge_attr.cpu().tolist() if g.edge_attr.numel() else []
    edge_index = g.edge_index.t().cpu().tolist() if g.edge_index.numel() else []
    node_ids = [f"n{i}" for i in range(n_nodes)]
    attack_edges = sum(1 for e in edge_features if len(e) >= 3 and e[2] >= 0.5)
    return {
        "timestep": timestep,
        "window_index": None,
        "num_nodes": n_nodes,
        "num_edges": len(edge_index),
        "node_ids": node_ids,
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "attack_ratio": (attack_edges / len(edge_features)) if edge_features else 0.0,
        "label_any_attack": 1 if attack_edges > 0 else 0,
    }


def _serialize_nodes(graph_dict: Dict) -> List[Dict]:
    nodes: List[Dict] = []
    feats = graph_dict.get("node_features") or []
    ids = graph_dict.get("node_ids") or []
    for i, row in enumerate(feats):
        vals = list(row) + [0.0] * (len(NODE_FEATURE_NAMES) - len(row))
        nodes.append({
            "id": ids[i] if i < len(ids) else f"n{i}",
            "index": i,
            "out_degree": float(vals[0]),
            "in_degree": float(vals[1]),
            "bytes_sent": float(vals[2]),
            "bytes_received": float(vals[3]),
            "flow_count": float(vals[4]),
        })
    return nodes


def _serialize_edges(graph_dict: Dict) -> List[Dict]:
    edges: List[Dict] = []
    ei = graph_dict.get("edge_index") or []
    ef = graph_dict.get("edge_features") or []
    ids = graph_dict.get("node_ids") or []

    def _nid(idx: int) -> str:
        return ids[idx] if 0 <= idx < len(ids) else f"n{idx}"

    for k, (src, dst) in enumerate(ei):
        feat = ef[k] if k < len(ef) else [0.0, 0.0, 0.0]
        feat = list(feat) + [0.0] * (len(EDGE_FEATURE_NAMES) - len(feat))
        edges.append({
            "src": int(src),
            "dst": int(dst),
            "src_id": _nid(int(src)),
            "dst_id": _nid(int(dst)),
            "flow_count": float(feat[0]),
            "total_bytes": float(feat[1]),
            "contains_attack": bool(feat[2] >= 0.5),
        })
    return edges


def _serialize_graph(graph_dict: Dict) -> Dict:
    return {
        "timestep": int(graph_dict.get("timestep", 0)),
        "window_index": graph_dict.get("window_index"),
        "num_nodes": int(graph_dict.get("num_nodes", 0)),
        "num_edges": int(graph_dict.get("num_edges", 0)),
        "nodes": _serialize_nodes(graph_dict),
        "edges": _serialize_edges(graph_dict),
    }


def _require_sample(rt: SentinelRuntime, t_index: Optional[int]) -> KStepSample:
    if not rt.data_usable:
        raise ServiceError(
            "Serving data is not usable for this dataset "
            f"({rt.samples.info.get('reason', 'no usable windows')}).",
            status_code=503)
    sample = rt.resolve_sample(t_index)
    if sample is None:
        raise ServiceError(
            f"No test anchor found for t_index={t_index}.", status_code=404)
    return sample


def _resolve_k(rt: SentinelRuntime, horizons: Optional[int]) -> int:
    if horizons is None:
        return rt.K
    k = int(horizons)
    if not (1 <= k <= rt.K):
        raise ServiceError(
            f"horizons must be in [1, {rt.K}] (serving K); got {k}.")
    return k


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
def health(rt: SentinelRuntime) -> Dict:
    """Liveness + readiness. Never raises; degrades gracefully."""
    payload: Dict = {
        "status": "ok",
        "checkpoint_available": rt.checkpoint_available,
        "model_loaded": False,
        "data_usable": False,
    }
    if not rt.checkpoint_available:
        payload["status"] = "degraded"
        return payload
    try:
        model = rt.model  # triggers load
        cfg = rt.config
        payload.update({
            "model_loaded": True,
            "combo": rt.combo,
            "dataset": cfg.data.dataset,
            "seq_len": cfg.data.seq_len,
            "horizon": cfg.data.horizon,
            "serving_k": rt.K,
            "risk_threshold": rt.threshold,
        })
        payload["data_usable"] = rt.data_usable
        payload["num_test_anchors"] = len(rt.samples.test)
    except Exception as exc:  # pragma: no cover - defensive
        payload["status"] = "degraded"
        payload["error"] = str(exc)
    return payload


# --------------------------------------------------------------------------- #
# Network state / graph / history
# --------------------------------------------------------------------------- #
def network_state(rt: SentinelRuntime, t_index: Optional[int]) -> Dict:
    sample = _require_sample(rt, t_index)
    graphs = [_graph_dict_from_sample_tensor(g, ts)
              for ts, g in enumerate(sample.input_seq)]
    latest = graphs[-1] if graphs else {"timestep": 0, "num_nodes": 0,
                                        "num_edges": 0}
    return {
        "t_index": int(sample.t_index),
        "seq_len": len(sample.input_seq),
        "latest_graph": _serialize_graph(latest),
        "observed_graphs": [_serialize_graph(g) for g in graphs],
        "risk_now": _deterministic_risk(rt, sample.input_seq),
        "label_any_attack": latest.get("label_any_attack"),
    }


def network_history(rt: SentinelRuntime, limit: Optional[int] = None) -> Dict:
    if not rt.data_usable:
        return {"dataset": rt.config.data.dataset, "n": 0, "points": []}
    model = rt.model
    model.eval()
    test = rt.all_test_samples()
    ordered = sorted(test, key=lambda s: int(s.t_index))
    points: List[Dict] = []
    # Batched deterministic risk over the latest observed window per anchor.
    with torch.no_grad():
        for start in range(0, len(ordered), 32):
            batch = ordered[start:start + 32]
            out = model(collate_inputs(batch))
            risks = torch.sigmoid(out.risk_logit).cpu().tolist()
            for s, r in zip(batch, risks):
                last = s.input_seq[-1]
                gd = _graph_dict_from_sample_tensor(last, len(s.input_seq) - 1)
                points.append({
                    "t_index": int(s.t_index),
                    "risk": float(r),
                    "num_nodes": gd["num_nodes"],
                    "num_edges": gd["num_edges"],
                    "label_any_attack": gd["label_any_attack"],
                })
    if limit is not None and limit > 0:
        points = points[-limit:]
    return {"dataset": rt.config.data.dataset, "n": len(points), "points": points}


# --------------------------------------------------------------------------- #
# Forecast trajectory (future latent states + risk per horizon)
# --------------------------------------------------------------------------- #
def forecast_trajectory(rt: SentinelRuntime, t_index: Optional[int],
                        horizons: Optional[int] = None) -> Dict:
    sample = _require_sample(rt, t_index)
    K = _resolve_k(rt, horizons)
    roll = rollout_latents(rt.model, [sample.input_seq], K)
    latents = roll.latents  # (K, 1, latent_dim)
    risks = roll.risk_probs  # (K, 1)
    steps: List[Dict] = []
    for k in range(1, K + 1):
        steps.append({
            "horizon": k,
            "target_index": int(sample.t_index) + k,
            "risk": float(risks[k - 1, 0].item()),
            "latent": [float(v) for v in latents[k - 1, 0].cpu().tolist()],
        })
    return {
        "t_index": int(sample.t_index),
        "K": K,
        "current_latent": [float(v) for v in roll.z0[0].cpu().tolist()],
        "steps": steps,
        "risk_threshold": rt.threshold,
    }


# --------------------------------------------------------------------------- #
# Risk
# --------------------------------------------------------------------------- #
def risk(rt: SentinelRuntime, t_index: Optional[int],
         horizons: Optional[int] = None) -> Dict:
    sample = _require_sample(rt, t_index)
    K = _resolve_k(rt, horizons)
    risk_now = _deterministic_risk(rt, sample.input_seq)
    roll = rollout_latents(rt.model, [sample.input_seq], K)
    forecast_risk = [float(roll.risk_probs[k, 0].item()) for k in range(K)]
    return {
        "t_index": int(sample.t_index),
        "risk_now": risk_now,
        "risk_threshold": rt.threshold,
        "alert": bool(risk_now >= rt.threshold),
        "forecast_risk": forecast_risk,
        "horizons": list(range(1, K + 1)),
    }


# --------------------------------------------------------------------------- #
# Uncertainty (MC-Dropout) — kept distinct from the point risk estimate
# --------------------------------------------------------------------------- #
def uncertainty(rt: SentinelRuntime, t_index: Optional[int],
                n_passes: int = 30) -> Dict:
    sample = _require_sample(rt, t_index)
    res = mc_dropout_risk(rt.model, [sample.input_seq],
                          n_passes=int(n_passes), seed=42)
    return {
        "t_index": int(sample.t_index),
        "method": "mc-dropout",
        "n_passes": int(n_passes),
        "risk_mean": float(res["risk_mean"][0]),
        "uncertainty_variance": float(res["variance"][0]),
        "uncertainty_std": float(res["std"][0]),
    }


# --------------------------------------------------------------------------- #
# Novelty / OOD (Mahalanobis; detector fit on in-distribution train latents)
# --------------------------------------------------------------------------- #
def _get_ood_detector(rt: SentinelRuntime) -> Optional[MahalanobisOOD]:
    """Fit (and cache on the runtime) a Mahalanobis OOD detector.

    Fit uses TRAIN latents; the threshold uses train+val (in-distribution) only,
    never test — identical discipline to the Phase-6 experiment.
    """
    cached = getattr(rt, "_ood_detector", None)
    if cached is not None:
        return cached
    train = rt.samples.train
    if len(train) < 2:
        return None
    z_train = encode_latents(rt.model, train, collate_inputs=collate_inputs)
    detector = MahalanobisOOD().fit(z_train)
    indist = list(train) + list(rt.samples.val)
    z_indist = encode_latents(rt.model, indist, collate_inputs=collate_inputs)
    detector.set_threshold_from_indist(z_indist, percentile=95.0)
    setattr(rt, "_ood_detector", detector)
    return detector


def novelty(rt: SentinelRuntime, t_index: Optional[int]) -> Dict:
    sample = _require_sample(rt, t_index)
    detector = _get_ood_detector(rt)
    if detector is None:
        raise ServiceError(
            "Novelty detector unavailable (need >=2 training latents).",
            status_code=503)
    z = encode_latents(rt.model, [sample], collate_inputs=collate_inputs)
    score = detector.score(z)[0]
    return {
        "t_index": int(sample.t_index),
        "method": "mahalanobis",
        "novelty_score": float(score.novelty_score),
        "is_novel": bool(score.is_ood),
        "threshold": (None if detector.threshold is None
                      else float(detector.threshold)),
        "fit_percentile": float(detector.fit_percentile),
    }


# --------------------------------------------------------------------------- #
# Attack trajectory + high-level MITRE interpretation
# --------------------------------------------------------------------------- #
def attack_trajectory(rt: SentinelRuntime, t_index: Optional[int],
                      horizons: Optional[int] = None) -> Dict:
    sample = _require_sample(rt, t_index)
    K = _resolve_k(rt, horizons)
    roll = rollout_latents(rt.model, [sample.input_seq], K)
    forecast_risks = [float(roll.risk_probs[k, 0].item()) for k in range(K)]
    observed = [_graph_dict_from_sample_tensor(g, ts)
                for ts, g in enumerate(sample.input_seq)]
    traj = build_trajectory(
        observed_windows=observed,
        forecast_risks=forecast_risks,
        t_index=int(sample.t_index),
    )
    return traj.as_dict()


def mitre(rt: SentinelRuntime, t_index: Optional[int],
          horizons: Optional[int] = None) -> Dict:
    """MITRE interpretation == the observed→forecast trajectory of stages."""
    return attack_trajectory(rt, t_index, horizons)


# --------------------------------------------------------------------------- #
# Propagation
# --------------------------------------------------------------------------- #
def propagation(rt: SentinelRuntime, t_index: Optional[int]) -> Dict:
    sample = _require_sample(rt, t_index)
    observed = [_graph_dict_from_sample_tensor(g, ts)
                for ts, g in enumerate(sample.input_seq)]
    analysis = analyze_propagation(observed, suspicion_source=SUSPICION_LABEL)
    d = analysis.as_dict()
    d["t_index"] = int(sample.t_index)
    return d


# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #
def explainability(rt: SentinelRuntime, t_index: Optional[int],
                   target: str = "state") -> Dict:
    sample = _require_sample(rt, t_index)
    node_ids_by_ts = [[f"n{i}" for i in range(int(g.num_nodes))]
                      for g in sample.input_seq]
    expl = explain_forecast(rt.model, sample.input_seq, target=target,
                            node_ids_by_ts=node_ids_by_ts)
    d = expl.as_dict()
    d["t_index"] = int(sample.t_index)
    return d


# --------------------------------------------------------------------------- #
# Counterfactual (simulation-only)
# --------------------------------------------------------------------------- #
def counterfactual(rt: SentinelRuntime, req_intervention: str,
                   t_index: Optional[int], horizons: Optional[int],
                   *, node_index: Optional[int] = None,
                   src: Optional[int] = None, dst: Optional[int] = None,
                   path: Optional[List[int]] = None,
                   timestep: Optional[int] = None) -> Dict:
    sample = _require_sample(rt, t_index)
    K = _resolve_k(rt, horizons)
    kind = str(req_intervention).lower()
    if kind not in VALID_INTERVENTIONS:
        raise ServiceError(
            f"intervention must be one of {sorted(VALID_INTERVENTIONS)}, "
            f"got {req_intervention!r}.")

    params: Dict = {}
    if kind == "isolate_node":
        if node_index is None:
            raise ServiceError("isolate_node requires 'node_index'.")
        params["node_index"] = int(node_index)
    elif kind == "remove_edge":
        if src is None or dst is None:
            raise ServiceError("remove_edge requires 'src' and 'dst'.")
        params["src"], params["dst"] = int(src), int(dst)
        if timestep is not None:
            params["timestep"] = int(timestep)
    elif kind == "suppress_path":
        if not path:
            raise ServiceError("suppress_path requires a non-empty 'path'.")
        params["path"] = [int(p) for p in path]
        if timestep is not None:
            params["timestep"] = int(timestep)

    try:
        result = simulate_intervention(rt.model, sample.input_seq, kind,
                                       K=K, **params)
    except (ValueError, IndexError) as exc:
        raise ServiceError(f"Invalid intervention parameters: {exc}")
    d = result.as_dict()
    d["t_index"] = int(sample.t_index)
    return d


# --------------------------------------------------------------------------- #
# Forecast stability (controlled perturbations; NOT MC-Dropout)
# --------------------------------------------------------------------------- #
def stability(rt: SentinelRuntime, t_index: Optional[int],
              horizons: Optional[int] = None, *, n_trials: int = 16,
              epsilon: float = 0.05) -> Dict:
    sample = _require_sample(rt, t_index)
    K = _resolve_k(rt, horizons)
    result = evaluate_stability(rt.model, sample.input_seq, K=K,
                                n_trials=int(n_trials), epsilon=float(epsilon),
                                seed=42)
    d = result.as_dict()
    d["t_index"] = int(sample.t_index)
    return d


# --------------------------------------------------------------------------- #
# Ingest: run the forecast pipeline over a caller-supplied window sequence
# --------------------------------------------------------------------------- #
def _windows_to_input_seq(rt: SentinelRuntime, windows: List[Dict]
                          ) -> List[GraphTensors]:
    """Convert caller-supplied cache-format windows to a model input sequence.

    Pads/truncates to the model's ``seq_len`` (left-pad with empty graphs), and
    builds GraphTensors with the model's node/edge feature dims.
    """
    node_dim = rt.model.cfg.node_feature_dim
    edge_dim = rt.model.cfg.edge_feature_dim
    seq_len = rt.config.data.seq_len

    def to_tensors(w: Dict) -> GraphTensors:
        n = int(w.get("num_nodes", 0))
        if n == 0:
            return GraphTensors(
                x=torch.empty((0, node_dim), dtype=torch.float32),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, edge_dim), dtype=torch.float32),
                num_nodes=0)
        x = torch.tensor(w.get("node_features") or [], dtype=torch.float32)
        if x.numel() == 0:
            x = torch.zeros((n, node_dim), dtype=torch.float32)
        ei = w.get("edge_index") or []
        if ei:
            edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(w.get("edge_features") or [],
                                     dtype=torch.float32)
            if edge_attr.numel() == 0:
                edge_attr = torch.zeros((edge_index.shape[1], edge_dim),
                                        dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, edge_dim), dtype=torch.float32)
        return GraphTensors(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            num_nodes=n)

    seq = [to_tensors(w) for w in windows]
    if len(seq) > seq_len:
        seq = seq[-seq_len:]
    elif len(seq) < seq_len:
        pad = GraphTensors(
            x=torch.empty((0, node_dim), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, edge_dim), dtype=torch.float32),
            num_nodes=0)
        seq = [pad] * (seq_len - len(seq)) + seq
    return seq


def ingest(rt: SentinelRuntime, windows: List[Dict],
           horizons: Optional[int] = None) -> Dict:
    """Ingest a live observed window sequence and return a forecast summary.

    Runs the same forecast + risk pipeline as /forecast but on caller-supplied
    windows (no anchor lookup), so a live sensor can push a sequence directly.
    """
    if not windows:
        raise ServiceError("ingest requires at least one window.")
    K = _resolve_k(rt, horizons)
    input_seq = _windows_to_input_seq(rt, windows)
    risk_now = _deterministic_risk(rt, input_seq)
    roll = rollout_latents(rt.model, [input_seq], K)
    forecast_risk = [float(roll.risk_probs[k, 0].item()) for k in range(K)]
    steps = [{
        "horizon": k + 1,
        "target_index": k + 1,
        "risk": forecast_risk[k],
        "latent": [float(v) for v in roll.latents[k, 0].cpu().tolist()],
    } for k in range(K)]
    graphs = [_graph_dict_from_sample_tensor(g, ts)
              for ts, g in enumerate(input_seq)]
    latest = graphs[-1]
    return {
        "accepted": True,
        "seq_len": len(input_seq),
        "K": K,
        "risk_now": risk_now,
        "risk_threshold": rt.threshold,
        "alert": bool(risk_now >= rt.threshold),
        "forecast_risk": forecast_risk,
        "horizons": list(range(1, K + 1)),
        "current_latent": [float(v) for v in roll.z0[0].cpu().tolist()],
        "steps": steps,
        "latest_graph": _serialize_graph(latest),
    }


# --------------------------------------------------------------------------- #
# Experiments registry + comparison table (Phase-9)
# --------------------------------------------------------------------------- #
def warmup(rt: SentinelRuntime) -> Dict:
    """Pay every one-time lazy cost up front (called at server startup).

    The first request would otherwise trigger: checkpoint load, K-step sample
    build from the cache, and the Mahalanobis OOD fit (which encodes all
    train+val latents). Cold, that adds up to tens of seconds on CPU — long
    enough that a client with a request timeout aborts the first /forecast,
    resets the proxy connection, and retries into the still-initialising server
    (an apparent 'app won't open' hang). Priming here makes the first real
    request sub-second and identical in cost to every subsequent one.

    Never raises: warmup is best-effort. If the checkpoint or data is missing,
    the endpoints still degrade gracefully via their own guards.
    """
    result: Dict = {"model_loaded": False, "samples_built": False,
                    "ood_ready": False, "primed_forecast": False}
    try:
        _ = rt.model                      # load checkpoint
        result["model_loaded"] = True
        if rt.data_usable:                # build K-step samples from cache
            result["samples_built"] = True
            if _get_ood_detector(rt) is not None:   # fit Mahalanobis OOD once
                result["ood_ready"] = True
            sample = rt.latest_sample()   # exercise the full path once
            if sample is not None:
                full_forecast(rt, int(sample.t_index), mc_passes=15)
                result["primed_forecast"] = True
    except Exception as exc:  # pragma: no cover - defensive; warmup never fails
        result["error"] = str(exc)
    return result


def experiments(rt: SentinelRuntime) -> Dict:
    from ..research.experiments import ALL_EXPERIMENTS
    from ..research.comparison import COLUMNS
    from ..pipeline.config import REPO_ROOT
    import csv

    comparison: List[Dict] = []
    comp_path = REPO_ROOT / "experiments" / "model_comparison.csv"
    if comp_path.exists():
        with open(comp_path, "r", encoding="utf-8", newline="") as fh:
            comparison = list(csv.DictReader(fh))
    return {
        "available_experiments": list(ALL_EXPERIMENTS),
        "comparison_columns": list(COLUMNS),
        "comparison": comparison,
    }


# --------------------------------------------------------------------------- #
# The aggregate forecast — everything the frontend needs in one payload
# --------------------------------------------------------------------------- #
def full_forecast(
    rt: SentinelRuntime,
    t_index: Optional[int],
    *,
    horizons: Optional[int] = None,
    include_explainability: bool = True,
    include_propagation: bool = True,
    include_uncertainty: bool = True,
    include_novelty: bool = True,
    include_mitre: bool = True,
    include_stability: bool = True,
    mc_passes: int = 30,
) -> Dict:
    """Assemble the complete /forecast payload from the service functions.

    Each optional analytical layer is guarded so a single failing/unavailable
    layer never sinks the whole response (it is returned as ``None`` instead).
    """
    sample = _require_sample(rt, t_index)
    ti = int(sample.t_index)
    K = _resolve_k(rt, horizons)

    state = network_state(rt, ti)
    fc = forecast_trajectory(rt, ti, K)
    rk = risk(rt, ti, K)
    traj = attack_trajectory(rt, ti, K)

    def _safe(fn):
        try:
            return fn()
        except ServiceError:
            return None
        except Exception:  # pragma: no cover - defensive isolation
            return None

    unc = _safe(lambda: uncertainty(rt, ti, mc_passes)) if include_uncertainty else None
    nov = _safe(lambda: novelty(rt, ti)) if include_novelty else None
    prop = _safe(lambda: propagation(rt, ti)) if include_propagation else None
    expl = _safe(lambda: explainability(rt, ti)) if include_explainability else None
    stab = _safe(lambda: stability(rt, ti, K)) if include_stability else None

    mitre_block = dict(traj)  # trajectory IS the MITRE interpretation
    return {
        "t_index": ti,
        "dataset": rt.config.data.dataset,
        "K": K,
        "risk_threshold": rt.threshold,
        "network_state": state,
        "graph_nodes": state["latest_graph"]["nodes"],
        "graph_edges": state["latest_graph"]["edges"],
        "forecast": fc,
        "horizons": list(range(1, K + 1)),
        "risk": rk,
        "uncertainty": unc,
        "novelty": nov,
        "attack_trajectory": traj,
        "mitre": mitre_block,
        "propagation": prop,
        "explainability": expl,
        "stability": stab,
    }
