"""Phase-4 experiment driver: train the Sentinel-X world model across the four
architecture combinations and write reproducible outputs.

Combinations (same evaluation protocol as Phase-3 baselines):
    GAT + GRU | GAT + LSTM | GraphSAGE + GRU | GraphSAGE + LSTM

Outputs:
    models/sentinel_x/model.pt         (best combo checkpoint)
    models/sentinel_x/config.yaml      (its config)
    models/sentinel_x/metadata.json    (metrics, params, provenance)
    experiments/world_model_results.csv
    experiments/world_model_report.md
"""

from __future__ import annotations

import csv
import json
import platform
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ..experiments.common import seed_everything
from .config import DataConfig, ModelConfig, TrainingConfig, WorldModelConfig, dump_config_yaml
from .data import WorldModelDataset
from .model import SentinelXWorldModel
from .train import TrainResult, train_world_model

COMBOS = [
    ("gat", "gru"),
    ("gat", "lstm"),
    ("graphsage", "gru"),
    ("graphsage", "lstm"),
]


def _metrics_dict(m) -> Dict:
    if m is None:
        return {}
    d = m.as_dict()
    # NaN -> None for clean JSON/CSV
    if d.get("pr_auc") != d.get("pr_auc"):
        d["pr_auc"] = None
    return d


def run_world_model_experiments(
    base_cfg: WorldModelConfig,
    processed_root: Path,
    *,
    combos: Optional[List] = None,
    models_dir: Optional[Path] = None,
    experiments_dir: Optional[Path] = None,
    log=print,
) -> Dict:
    combos = combos or COMBOS
    seed_everything(base_cfg.training.seed)

    node_dim = base_cfg.model.node_feature_dim
    edge_dim = base_cfg.model.edge_feature_dim
    ds = WorldModelDataset(base_cfg.data, processed_root, node_dim, edge_dim)
    train, val, test, info = ds.build()

    results: Dict = {
        "dataset": base_cfg.data.dataset,
        "split_info": info,
        "data_config": asdict(base_cfg.data),
        "training_config": asdict(base_cfg.training),
        "runs": [],
    }

    if not info.get("usable"):
        results["skipped"] = True
        results["reason"] = info.get("reason", "unusable")
        log(f"[worldmodel] {base_cfg.data.dataset}: SKIPPED — {results['reason']}")
        return results

    log(f"[worldmodel] {base_cfg.data.dataset}: train={len(train)} val={len(val)} "
        f"test={len(test)} | positives(test)={info['positives']['test']} "
        f"| graph_capable={info.get('graph_capable')}")

    best = None  # (val_loss, run_record, model, cfg)
    for gnn_type, temporal_type in combos:
        seed_everything(base_cfg.training.seed)
        model_cfg = replace(base_cfg.model, gnn_type=gnn_type,
                             temporal_type=temporal_type).normalized()
        model = SentinelXWorldModel(model_cfg)
        name = f"{gnn_type}_{temporal_type}"
        log(f"[worldmodel]  training {name} ({model.num_parameters()} params)")
        res: TrainResult = train_world_model(
            model, train, val, test, base_cfg.training,
            checkpoint_path=None, log=log,
        )
        record = {
            "combo": name,
            "gnn_type": gnn_type,
            "temporal_type": temporal_type,
            "num_parameters": res.num_parameters,
            "best_epoch": res.best_epoch,
            "best_val_loss": res.best_val_loss,
            "best_val_state_loss": res.best_val_state_loss,
            "best_val_total_loss": res.best_val_total_loss,
            "stopped_early": res.stopped_early,
            "train_seconds": res.train_seconds,
            "chosen_threshold": res.chosen_threshold,
            "val": _metrics_dict(res.val_metrics),
            "test": _metrics_dict(res.test_metrics),
            "final_train_loss": res.history[-1].train_loss if res.history else None,
            "final_val_state_loss": res.history[-1].val_state if res.history else None,
            "final_val_risk_loss": res.history[-1].val_risk if res.history else None,
        }
        results["runs"].append(record)

        # Selection by the PRIMARY objective: validation future-state loss.
        # Lower is better. (The auxiliary risk metric is reported but does not
        # drive selection — future-state modelling is the primary objective.)
        score = res.best_val_state_loss
        if best is None or score < best[0]:
            best = (score, record, model, model_cfg)

    if best is not None:
        _, best_record, best_model, best_model_cfg = best
        results["best_combo"] = best_record["combo"]
        if models_dir is not None:
            _save_best(best_model, best_model_cfg, base_cfg, best_record, info,
                       models_dir, log)

    if experiments_dir is not None:
        _write_experiment_outputs(results, experiments_dir, log)

    return results


def _save_best(model: SentinelXWorldModel, model_cfg: ModelConfig,
               base_cfg: WorldModelConfig, best_record: Dict, info: Dict,
               models_dir: Path, log) -> None:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    # model.pt
    model.save_checkpoint(models_dir / "model.pt", extra={
        "combo": best_record["combo"],
        "best_epoch": best_record["best_epoch"],
        "chosen_threshold": best_record["chosen_threshold"],
    })
    # config.yaml (the winning configuration)
    winning_cfg = WorldModelConfig(model=model_cfg, training=base_cfg.training,
                                   data=base_cfg.data)
    dump_config_yaml(winning_cfg, models_dir / "config.yaml")
    # metadata.json
    try:
        import torch
        torch_v = torch.__version__
        import torch_geometric
        pyg_v = torch_geometric.__version__
    except Exception:
        torch_v = pyg_v = "unknown"
    metadata = {
        "model": "sentinel_x_world_model",
        "phase": "4-world-model",
        "objective": "future-state modelling P(S_{t+1}|S_t) with auxiliary risk",
        "best_combo": best_record["combo"],
        "num_parameters": best_record["num_parameters"],
        "dataset": base_cfg.data.dataset,
        "split_info": info,
        "test_metrics": best_record["test"],
        "val_metrics": best_record["val"],
        "chosen_threshold": best_record["chosen_threshold"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch_v,
        "torch_geometric": pyg_v,
        "config": winning_cfg.to_dict(),
    }
    (models_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    log(f"[worldmodel]  saved best combo '{best_record['combo']}' -> {models_dir}")


def _fmt(v):
    return "n/a" if v is None or (isinstance(v, float) and v != v) else v


def _write_experiment_outputs(results: Dict, experiments_dir: Path, log) -> None:
    experiments_dir = Path(experiments_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = experiments_dir / "world_model_results.csv"
    fields = ["dataset", "combo", "gnn_type", "temporal_type", "num_parameters",
              "best_epoch", "best_val_state_loss", "best_val_total_loss",
              "train_seconds", "chosen_threshold",
              "test_precision", "test_recall", "test_f1", "test_pr_auc",
              "test_roc_auc", "test_fpr"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in results.get("runs", []):
            t = r.get("test", {})
            w.writerow({
                "dataset": results["dataset"], "combo": r["combo"],
                "gnn_type": r["gnn_type"], "temporal_type": r["temporal_type"],
                "num_parameters": r["num_parameters"], "best_epoch": r["best_epoch"],
                "best_val_state_loss": round(r["best_val_state_loss"], 6) if r.get("best_val_state_loss") is not None else None,
                "best_val_total_loss": round(r["best_val_total_loss"], 6) if r.get("best_val_total_loss") is not None else None,
                "train_seconds": r["train_seconds"], "chosen_threshold": round(r["chosen_threshold"], 4),
                "test_precision": _fmt(round(t["precision"], 4)) if t else "n/a",
                "test_recall": _fmt(round(t["recall"], 4)) if t else "n/a",
                "test_f1": _fmt(round(t["f1"], 4)) if t else "n/a",
                "test_pr_auc": _fmt(round(t["pr_auc"], 4)) if (t and t.get("pr_auc") is not None) else "n/a",
                "test_roc_auc": _fmt(round(t["roc_auc"], 4)) if (t and t.get("roc_auc") is not None) else "n/a",
                "test_fpr": _fmt(round(t["false_positive_rate"], 4)) if t else "n/a",
            })

    # JSON (full)
    (experiments_dir / "world_model_metrics.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")

    # Markdown report
    (experiments_dir / "world_model_report.md").write_text(
        _render_report(results), encoding="utf-8")
    log(f"[worldmodel]  wrote {csv_path.name}, world_model_report.md")


def _render_report(results: Dict) -> str:
    L: List[str] = []
    L.append("# Sentinel-X — Phase 4 World Model Report\n")
    L.append("**Objective.** Learn `P(S_{t+1} | S_t)` over a *dynamic network "
             "state* — how network behaviour evolves — rather than classifying "
             "the current window. Architecture:\n")
    L.append("```\n"
             "dynamic graph G_t -> GAT/GraphSAGE -> node embeddings -> graph pooling\n"
             " -> graph state h_t -> GRU/LSTM -> temporal latent z_t\n"
             " -> forecasting head -> predicted next state z_{t+1}\n"
             " (+ auxiliary risk head -> attack@t+H)\n```\n")
    L.append("**Loss (documented, multi-objective).** "
             "`L = λ_state · (1 − cos(ẑ_{t+1}, sg(z_{t+1}))) + λ_risk · wBCE(risk, attack@t+H)`. "
             "The primary term is self-supervised future-state prediction "
             "(stop-grad target prevents collapse); the auxiliary term is a "
             "class-weighted risk BCE so the model stays comparable to the "
             "Phase-3 baselines on the SAME target. The model is never reduced "
             "to `z_t → attack yes/no`.\n")
    L.append("**Protocol.** Identical windows / seq_len / horizon / chronological "
             "split / seed / leakage policy as Phase-3 "
             "(`sentinelx.experiments.common`), so results are apples-to-apples "
             "with the baselines.\n")

    ds = results["dataset"]
    L.append(f"\n## Dataset: {ds}\n")
    if results.get("skipped"):
        L.append(f"**SKIPPED** — {results.get('reason')}\n")
        return "\n".join(L) + "\n"

    info = results["split_info"]
    dc = results["data_config"]
    L.append(f"- windows: {info.get('num_windows')} | forecasting samples: "
             f"{info.get('num_samples')} | seq_len={dc['seq_len']} horizon={dc['horizon']}")
    L.append(f"- split counts: {info.get('counts')} | positives per split: "
             f"{info.get('positives')} | graph_capable={info.get('graph_capable')}")
    L.append(f"- best combo (by PRIMARY future-state validation loss): "
             f"**{results.get('best_combo')}**\n")

    L.append("\n### Architecture comparison\n")
    L.append("*Selection column is the PRIMARY future-state validation loss "
             "(lower = better forecasting of the next network state). The "
             "remaining columns are the AUXILIARY risk head on TEST.*\n")
    L.append("| Combo | Params | Best epoch | Val state loss | Val total loss | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results.get("runs", []):
        t = r.get("test", {})
        L.append(
            f"| {r['combo']} | {r['num_parameters']} | {r['best_epoch']} | "
            f"{_fmt(round(r.get('best_val_state_loss'),4) if r.get('best_val_state_loss') is not None else None)} | "
            f"{_fmt(round(r.get('best_val_total_loss'),4) if r.get('best_val_total_loss') is not None else None)} | "
            f"{_fmt(round(t['precision'],4)) if t else 'n/a'} | "
            f"{_fmt(round(t['recall'],4)) if t else 'n/a'} | "
            f"{_fmt(round(t['f1'],4)) if t else 'n/a'} | "
            f"{_fmt(round(t['pr_auc'],4)) if (t and t.get('pr_auc') is not None) else 'n/a'} | "
            f"{_fmt(round(t['roc_auc'],4)) if (t and t.get('roc_auc') is not None) else 'n/a'} | "
            f"{_fmt(round(t['false_positive_rate'],4)) if t else 'n/a'} |")

    L.append("\n### Interpreting these numbers\n")
    L.append("- **Val state loss** reflects the PRIMARY future-state objective "
             "(1 − cosine similarity between the predicted and actual next latent "
             "state) and is what model selection uses. The risk-head columns are "
             "the AUXILIARY metrics, shown so the world model can be compared to "
             "the Phase-3 baseline bar (CTU-13 scenario 11: LogReg PR-AUC 0.917, "
             "GRU 0.955, GraphSAGE 0.736).")
    L.append("- A world model that forecasts future state well AND matches/beats "
             "the baselines on the shared risk target is the desired outcome; "
             "either result is reported honestly rather than tuned for the "
             "classifier alone.")
    L.append("\n### Reproducibility\n")
    L.append(f"- Fixed seed {results['training_config']['seed']}, deterministic "
             "RNGs, gradient clipping, early stopping on validation objective, "
             "best-epoch checkpointing. Config is saved to "
             "`models/sentinel_x/config.yaml`.")
    return "\n".join(L) + "\n"
