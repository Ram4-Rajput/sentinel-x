"""Phase-8 experiment driver: explainability + propagation + counterfactual +
forecast stability, all layered around the trained World Model.

Reuses everything already built — no retraining, no new model outputs:

  * loads the trained Phase-4 world-model checkpoint,
  * rebuilds the SAME K-step evaluation samples as Phase-5/7 (leakage-safe),
  * for each selected TEST anchor:
      - EXPLAINABILITY : gradient x input attribution (+ GAT attention as
        supporting evidence) -> top_features / important_nodes / important_edges
        / temporal_evidence,
      - PROPAGATION    : how flagged suspicious behaviour spreads across the
        observed window graphs,
      - COUNTERFACTUAL : clone -> simulated intervention (isolate the most
        important node) -> rerun -> compare baseline vs intervention K-step
        forecast (labelled "Modelled / simulated outcome."),
      - STABILITY      : sensitivity of the forecast to controlled valid input
        perturbations (documented metric; NOT MC-Dropout).
  * writes:
        experiments/phase8_results.csv       (one row per anchor, headline metrics)
        experiments/phase8_report.md         (narrative + examples)
        experiments/phase8_metrics.json      (full nested results)

Discipline: attention is never called causal; counterfactuals are modelled/
simulated (never touch real infra, never claim causal certainty); stability is
kept conceptually separate from MC-Dropout uncertainty.
"""

from __future__ import annotations

import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..experiments.common import seed_everything
from .config import WorldModelConfig, load_config
from .counterfactual import (
    ISOLATE_NODE, SIMULATION_LABEL, evaluate_stability, simulate_intervention,
)
from .explain import explain_forecast
from .kstep_data import KStepDataset, KStepSample
from .model import GraphTensors, SentinelXWorldModel
from .propagation import SUSPICION_LABEL, analyze_propagation

DEFAULT_K = 5


def _observed_dicts(sample: KStepSample) -> List[Dict]:
    """Recover observed window dicts from a KStepSample's tensors.

    Mirrors the Phase-7 driver's reconstruction so propagation can run on the
    same behaviour signals the world model consumed. Node/edge feature column
    order follows graph_cache: node = [out_deg, in_deg, bytes_sent,
    bytes_recv, flow_count]; edge = [flow_count, total_bytes, contains_attack].
    """
    dicts: List[Dict] = []
    for g in sample.input_seq:
        n_nodes = int(g.num_nodes)
        node_features = g.x.cpu().tolist() if n_nodes > 0 else []
        edge_features = g.edge_attr.cpu().tolist() if g.edge_attr.numel() else []
        edge_index = g.edge_index.t().cpu().tolist() if g.edge_index.numel() else []
        dicts.append({
            "window_index": None,
            "num_nodes": n_nodes,
            "num_edges": len(edge_index),
            "node_ids": [f"n{i}" for i in range(n_nodes)],
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_features": edge_features,
        })
    return dicts


def analyze_anchor(
    model: SentinelXWorldModel,
    sample: KStepSample,
    *,
    K: int,
    stability_trials: int,
    stability_epsilon: float,
    seed: int,
) -> Dict:
    """Run all four Phase-8 layers for one anchor sample."""
    input_seq: List[GraphTensors] = sample.input_seq

    # 1. EXPLAINABILITY (explain the primary forecast: latent-state movement).
    explanation = explain_forecast(model, input_seq, target="state")

    # 2. PROPAGATION over the observed window graphs (label-flagged suspicion).
    obs = _observed_dicts(sample)
    propagation = analyze_propagation(obs, suspicion_source=SUSPICION_LABEL)

    # 3. COUNTERFACTUAL: isolate the model's most important node (if any) and
    # compare baseline vs simulated intervention. Falls back to remove_edge on
    # the top important edge if no node attribution is available.
    counterfactual = None
    if explanation.important_nodes:
        top = explanation.important_nodes[0]
        # Only isolate a node index valid in at least one window.
        max_nodes = max((int(g.num_nodes) for g in input_seq), default=0)
        if 0 <= top.node_index < max_nodes:
            counterfactual = simulate_intervention(
                model, input_seq, ISOLATE_NODE, K=K, node_index=top.node_index)
    if counterfactual is None and explanation.important_edges:
        te = explanation.important_edges[0]
        counterfactual = simulate_intervention(
            model, input_seq, "remove_edge", K=K, src=te.src, dst=te.dst)

    # 4. STABILITY under controlled valid perturbations (NOT MC-Dropout).
    stability = evaluate_stability(
        model, input_seq, K=K, n_trials=stability_trials,
        epsilon=stability_epsilon, seed=seed)

    return {
        "t_index": sample.t_index,
        "explanation": explanation.as_dict(),
        "propagation": propagation.as_dict(),
        "counterfactual": counterfactual.as_dict() if counterfactual else None,
        "stability": stability.as_dict(),
    }


def run_phase8_experiment(
    checkpoint_path: Path,
    processed_root: Path,
    *,
    K: int = DEFAULT_K,
    max_anchors: Optional[int] = 25,
    stability_trials: int = 16,
    stability_epsilon: float = 0.05,
    seed: int = 42,
    data_cfg=None,
    experiments_dir: Optional[Path] = None,
    log=print,
) -> Dict:
    """Run the full Phase-8 analytical suite around the trained World Model."""
    seed_everything(seed)
    K = int(K)
    if K < 1:
        raise ValueError(f"K must be a positive integer, got {K}")

    model, extra = SentinelXWorldModel.load_checkpoint(checkpoint_path)
    model.eval()
    risk_threshold = float(extra.get("chosen_threshold", 0.5))
    node_dim = model.cfg.node_feature_dim
    edge_dim = model.cfg.edge_feature_dim

    if data_cfg is None:
        cfg_path = Path(checkpoint_path).parent / "config.yaml"
        data_cfg = (load_config(cfg_path).data if cfg_path.exists()
                    else WorldModelConfig().data)

    ds = KStepDataset(data_cfg, processed_root, node_dim, edge_dim, K=K)
    train, val, test, info = ds.build()

    results: Dict = {
        "phase": "8-explainability-propagation-counterfactual-stability",
        "checkpoint": str(checkpoint_path),
        "combo": extra.get("combo"),
        "dataset": data_cfg.dataset,
        "seq_len": data_cfg.seq_len,
        "K": K,
        "risk_threshold": risk_threshold,
        "gnn_type": model.cfg.gnn_type,
        "attention_available": model.cfg.gnn_type == "gat",
        "stability_trials": stability_trials,
        "stability_epsilon": stability_epsilon,
        "split_info": info,
        "simulation_label": SIMULATION_LABEL,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "anchors": [],
    }

    if not info.get("usable"):
        results["skipped"] = True
        results["reason"] = info.get("reason", "unusable")
        log(f"[phase8] {data_cfg.dataset}: SKIPPED — {results['reason']}")
        if experiments_dir is not None:
            _write_outputs(results, experiments_dir, log)
        return results

    samples: List[KStepSample] = list(test)
    if max_anchors is not None:
        samples = samples[:max_anchors]

    log(f"[phase8] {data_cfg.dataset}: analysing {len(samples)} anchors "
        f"(K={K}, L={data_cfg.seq_len}, gnn={model.cfg.gnn_type})")

    anchors: List[Dict] = []
    for s in samples:
        anchors.append(analyze_anchor(
            model, s, K=K, stability_trials=stability_trials,
            stability_epsilon=stability_epsilon, seed=seed))
    results["anchors"] = anchors
    results["summary"] = _summarize(anchors)

    log(f"[phase8]  anchors={results['summary']['n_anchors']} "
        f"mean_stability={results['summary']['mean_stability_score']:.4f} "
        f"spreading={results['summary']['n_spreading']} "
        f"counterfactuals={results['summary']['n_counterfactuals']}")

    if experiments_dir is not None:
        _write_outputs(results, experiments_dir, log)
    return results


def _summarize(anchors: Sequence[Dict]) -> Dict:
    n = len(anchors)
    if n == 0:
        return {"n_anchors": 0, "mean_stability_score": 0.0, "n_spreading": 0,
                "n_counterfactuals": 0}
    stab = [a["stability"]["stability_score"] for a in anchors]
    spreading = sum(1 for a in anchors if a["propagation"]["is_spreading"])
    cfs = [a for a in anchors if a["counterfactual"] is not None]
    mean_cf_delta = (sum(a["counterfactual"]["mean_risk_delta"] for a in cfs)
                     / len(cfs)) if cfs else 0.0
    # Which feature is most often the top attributed feature?
    top_feat_counts: Dict[str, int] = {}
    for a in anchors:
        tf = a["explanation"]["top_features"]
        if tf:
            name = tf[0]["name"]
            top_feat_counts[name] = top_feat_counts.get(name, 0) + 1
    return {
        "n_anchors": n,
        "mean_stability_score": sum(stab) / n,
        "min_stability_score": min(stab),
        "max_stability_score": max(stab),
        "n_spreading": spreading,
        "n_counterfactuals": len(cfs),
        "mean_counterfactual_risk_delta": mean_cf_delta,
        "top_feature_counts": top_feat_counts,
    }


# --------------------------------------------------------------------------- #
# Artifact writers
# --------------------------------------------------------------------------- #
def _write_outputs(results: Dict, experiments_dir: Path, log) -> None:
    experiments_dir = Path(experiments_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    csv_path = experiments_dir / "phase8_results.csv"
    fields = [
        "t_index", "target", "target_value", "top_feature", "top_feature_attr",
        "top_node", "top_node_importance", "uses_attention",
        "prop_total_affected", "prop_velocity", "prop_direction",
        "prop_is_spreading", "cf_intervention", "cf_mean_risk_delta",
        "cf_max_abs_risk_delta", "stability_score", "stability_mean_latent_drift",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for a in results.get("anchors", []):
            expl = a["explanation"]
            prop = a["propagation"]
            cf = a["counterfactual"]
            stab = a["stability"]
            tf = expl["top_features"][0] if expl["top_features"] else {}
            tn = expl["important_nodes"][0] if expl["important_nodes"] else {}
            w.writerow({
                "t_index": a["t_index"],
                "target": expl["target"],
                "target_value": expl["target_value"],
                "top_feature": tf.get("name"),
                "top_feature_attr": tf.get("attribution"),
                "top_node": tn.get("node_index"),
                "top_node_importance": tn.get("importance"),
                "uses_attention": expl["uses_attention"],
                "prop_total_affected": prop["total_affected"],
                "prop_velocity": prop["velocity"],
                "prop_direction": prop["direction"],
                "prop_is_spreading": prop["is_spreading"],
                "cf_intervention": cf["intervention"] if cf else None,
                "cf_mean_risk_delta": cf["mean_risk_delta"] if cf else None,
                "cf_max_abs_risk_delta": cf["max_abs_risk_delta"] if cf else None,
                "stability_score": stab["stability_score"],
                "stability_mean_latent_drift": stab["mean_latent_drift"],
            })

    (experiments_dir / "phase8_metrics.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")
    (experiments_dir / "phase8_report.md").write_text(
        _render_report(results), encoding="utf-8")
    log(f"[phase8]  wrote {csv_path.name}, phase8_report.md, phase8_metrics.json")


def _render_report(results: Dict) -> str:
    L: List[str] = []
    L.append("# Sentinel-X — Phase 8: Explainability, Propagation, "
             "Counterfactual & Stability\n")
    L.append("**Goal.** Layer four analytical capabilities around the trained "
             "world model *without* changing the core forecasting architecture: "
             "(1) explain why the model produced a forecast, (2) analyse how "
             "suspicious behaviour propagates on the graph, (3) run "
             "simulation-only counterfactual interventions, and (4) measure "
             "forecast stability under controlled perturbations.\n")

    if results.get("skipped"):
        L.append(f"\n**SKIPPED** — {results.get('reason')}\n")
        return "\n".join(L) + "\n"

    summ = results.get("summary", {})
    L.append("## Setup\n")
    L.append(f"- checkpoint: `{results['checkpoint']}` (combo "
             f"**{results.get('combo')}**, gnn=**{results['gnn_type']}**)")
    L.append(f"- dataset: **{results['dataset']}** | observed L="
             f"{results['seq_len']} | forecast depth K={results['K']}")
    L.append(f"- attention available (GAT only): "
             f"**{results['attention_available']}**")
    L.append(f"- stability: {results['stability_trials']} trials, "
             f"epsilon={results['stability_epsilon']}")
    L.append(f"- anchors analysed: {summ.get('n_anchors')}\n")

    L.append("## 1. Explainability\n")
    L.append("Attribution uses **gradient x input** sensitivity of the primary "
             "forecast (predicted next-latent movement) w.r.t. node/edge/temporal "
             "inputs. For GAT encoders, attention weights are additionally "
             "exposed as *supporting* evidence.\n")
    L.append("> **Attention is not causal.** It indicates where the model "
             "attended (association), reported as supporting evidence only.\n")
    if summ.get("top_feature_counts"):
        L.append("Most-frequent top-attributed feature across anchors:\n")
        L.append("| Feature | Times ranked #1 |")
        L.append("|---|---|")
        for name, c in sorted(summ["top_feature_counts"].items(),
                              key=lambda kv: -kv[1]):
            L.append(f"| {name} | {c} |")

    L.append("\n## 2. Propagation\n")
    L.append("Supporting analytical layer describing how label-flagged "
             "suspicious behaviour spreads across the observed window graphs "
             "(affected/newly-affected nodes, velocity, direction, branching, "
             "persistence, growth). This does **not** replace the world model "
             "with reachability rules.\n")
    L.append(f"- anchors whose observed suspicious footprint is spreading "
             f"(growth>0 & new nodes appear): **{summ.get('n_spreading')}** / "
             f"{summ.get('n_anchors')}\n")

    L.append("## 3. Counterfactual (simulation-only)\n")
    L.append(f"For each anchor we clone the graph, apply a **simulated** "
             f"intervention (isolate the model's most-important node, or remove "
             f"its top edge), rerun the trained world model, and compare the "
             f"baseline vs intervention K-step forecast. Every such result is "
             f"labelled: *\"{results['simulation_label']}\"*\n")
    L.append("> No real network infrastructure is ever modified, and no causal "
             "certainty is claimed — the reported difference is the modelled "
             "effect under the learned dynamics.\n")
    L.append(f"- counterfactuals produced: **{summ.get('n_counterfactuals')}** "
             f"(mean modelled risk delta = "
             f"{round(summ.get('mean_counterfactual_risk_delta', 0.0), 4)})\n")

    L.append("## 4. Forecast stability\n")
    L.append("Sensitivity of the forecast to controlled, valid input "
             "perturbations (small multiplicative feature jitter, structure "
             "preserved). Stability score = 1/(1+mean relative latent drift) in "
             "(0,1]; higher = more stable.\n")
    L.append("> **Not MC-Dropout.** Dropout is OFF throughout; this measures "
             "input-sensitivity, a different question from model uncertainty.\n")
    L.append(f"- mean stability score: "
             f"**{round(summ.get('mean_stability_score', 0.0), 4)}** "
             f"(min {round(summ.get('min_stability_score', 0.0), 4)}, "
             f"max {round(summ.get('max_stability_score', 0.0), 4)})\n")

    # A couple of worked examples.
    L.append("## Example anchors\n")
    for a in results.get("anchors", [])[:3]:
        expl = a["explanation"]
        prop = a["propagation"]
        cf = a["counterfactual"]
        stab = a["stability"]
        L.append(f"\n**Anchor t={a['t_index']}**\n")
        L.append("```")
        L.append(f"explain target : {expl['target']} "
                 f"(value {expl['target_value']:.4f})")
        if expl["top_features"]:
            tops = ", ".join(f"{f['name']}={f['attribution']:.3f}"
                             for f in expl["top_features"][:3])
            L.append(f"top features   : {tops}")
        if expl["important_nodes"]:
            n0 = expl["important_nodes"][0]
            L.append(f"top node       : idx {n0['node_index']} "
                     f"@ts{n0['timestep']} (importance {n0['importance']:.3f})")
        L.append(f"attention used : {expl['uses_attention']}")
        L.append(f"propagation    : affected={prop['total_affected']} "
                 f"velocity={prop['velocity']:.2f} dir={prop['direction']} "
                 f"spreading={prop['is_spreading']}")
        if cf:
            L.append(f"counterfactual : {cf['intervention']} -> mean risk "
                     f"delta {cf['mean_risk_delta']:+.4f} "
                     f"(max|Δ| {cf['max_abs_risk_delta']:.4f})  [{cf['label']}]")
        L.append(f"stability      : score {stab['stability_score']:.4f} "
                 f"(mean latent drift {stab['mean_latent_drift']:.4f})")
        L.append("```")

    L.append("\n## Interpretation & discipline\n")
    L.append("- **Explainability** is model-attributed (gradients), not "
             "hand-written rules; attention is supporting association only, "
             "never causal.")
    L.append("- **Propagation** is a descriptive supporting layer over the real "
             "graphs; it does not forecast and does not replace the world model.")
    L.append("- **Counterfactuals** are simulation-only, applied to graph "
             "clones, and always labelled a modelled/simulated outcome with no "
             "causal-certainty claim.")
    L.append("- **Stability** is deterministic input-sensitivity, explicitly "
             "distinct from MC-Dropout model uncertainty (Phase 6).")
    L.append("\n*All quantities are derived from the trained checkpoint applied "
             "to real cached windows. Nothing is fabricated.*\n")
    return "\n".join(L) + "\n"
