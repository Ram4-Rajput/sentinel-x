"""Phase-3 baseline runner: trains all three baselines on a dataset with an
identical protocol, tracks to MLflow, and writes experiments/ outputs.

Outputs:
    experiments/baseline_results.csv    (one row per model x dataset)
    experiments/baseline_metrics.json   (full nested metrics + config)
    experiments/baseline_report.md      (comparison LogReg vs LSTM/GRU vs GNN)
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from .common import ExperimentConfig, build_window_samples, seed_everything
from .baseline_logreg import run_logreg
from .baseline_sequence import run_sequence
from .baseline_gnn import run_gnn
from .tracking import MlflowTracker


def _metrics_row(dataset: str, model: str, split_info: Dict, m, extra: Dict) -> Dict:
    d = {
        "dataset": dataset,
        "model": model,
        "n_test": m.n,
        "test_positives": m.positives,
        "precision": round(m.precision, 4),
        "recall": round(m.recall, 4),
        "f1": round(m.f1, 4),
        "pr_auc": None if m.pr_auc != m.pr_auc else round(m.pr_auc, 4),  # NaN guard
        "roc_auc": None if m.roc_auc is None else round(m.roc_auc, 4),
        "false_positive_rate": round(m.false_positive_rate, 4),
        "threshold": round(m.threshold, 4),
        "tp": m.tp, "fp": m.fp, "tn": m.tn, "fn": m.fn,
    }
    d.update(extra)
    return d


def run_all_baselines(
    cfg: ExperimentConfig,
    *,
    temporal_type: str = "gru",
    gnn_type: str = "sage",
    epochs: int = 30,
    mlflow_uri: Optional[str] = None,
    mlflow_artifacts: Optional[str] = None,
    experiment_name: str = "sentinelx-phase3-baselines",
    log=print,
) -> Dict:
    """Run all three baselines for one dataset. Returns a results dict."""
    seed_everything(cfg.seed)
    tracker = MlflowTracker(mlflow_uri, experiment_name, artifact_location=mlflow_artifacts)

    train, val, test, info = build_window_samples(cfg)
    results: Dict = {"dataset": cfg.dataset, "split_info": info,
                     "config": {"seq_len": cfg.seq_len, "horizon": cfg.horizon,
                                "train_frac": cfg.train_frac, "val_frac": cfg.val_frac,
                                "seed": cfg.seed}, "models": {}, "rows": []}

    if not info.get("usable"):
        results["skipped"] = True
        results["reason"] = info.get("reason", "unusable")
        log(f"[runner] {cfg.dataset}: SKIPPED — {results['reason']}")
        return results

    log(f"[runner] {cfg.dataset}: samples train={len(train)} val={len(val)} test={len(test)} "
        f"| positives(test)={info['positives']['test']}")

    common_tags = {"dataset": cfg.dataset, "phase": "3-baselines",
                   "target": f"attack@t+{cfg.horizon}"}
    split_cfg = {"seq_len": cfg.seq_len, "horizon": cfg.horizon,
                 "train_frac": cfg.train_frac, "val_frac": cfg.val_frac,
                 "n_train": len(train), "n_val": len(val), "n_test": len(test),
                 "seed": cfg.seed}

    # -------- Baseline 1: Logistic Regression --------
    log("[runner]   baseline 1: logistic regression")
    lr_res = run_logreg(train, val, test, seed=cfg.seed)
    results["models"]["logistic_regression"] = {
        "params": lr_res.params, "threshold": lr_res.chosen_threshold,
        "val": lr_res.val_metrics.as_dict(), "test": lr_res.test_metrics.as_dict(),
        "confusion_matrix": lr_res.test_metrics.confusion_matrix,
    }
    results["rows"].append(_metrics_row(cfg.dataset, "logistic_regression", info,
                                        lr_res.test_metrics, {"train_seconds": 0.0, "params": lr_res.params.get("n_features", 0)}))
    with tracker.run(f"{cfg.dataset}-logreg") as ml:
        tracker.log(ml, params={**lr_res.params, **split_cfg},
                    metrics={**lr_res.test_metrics.as_dict()}, tags=common_tags)

    # -------- Baseline 2: LSTM/GRU --------
    log(f"[runner]   baseline 2: sequence ({temporal_type})")
    seq_res = run_sequence(train, val, test, temporal_type=temporal_type,
                           epochs=epochs, seed=cfg.seed)
    results["models"][f"sequence_{temporal_type}"] = {
        "params": seq_res.params, "threshold": seq_res.chosen_threshold,
        "val": seq_res.val_metrics.as_dict(), "test": seq_res.test_metrics.as_dict(),
        "confusion_matrix": seq_res.test_metrics.confusion_matrix,
        "train_seconds": seq_res.train_seconds, "num_parameters": seq_res.num_parameters,
    }
    results["rows"].append(_metrics_row(cfg.dataset, f"sequence_{temporal_type}", info,
                                        seq_res.test_metrics,
                                        {"train_seconds": seq_res.train_seconds,
                                         "params": seq_res.num_parameters}))
    with tracker.run(f"{cfg.dataset}-seq-{temporal_type}") as ml:
        tracker.log(ml, params={**seq_res.params, **split_cfg},
                    metrics={**seq_res.test_metrics.as_dict(),
                             "train_seconds": seq_res.train_seconds,
                             "num_parameters": seq_res.num_parameters}, tags=common_tags)

    # -------- Baseline 3: Temporal GNN --------
    log(f"[runner]   baseline 3: temporal GNN ({gnn_type})")
    gnn_res = run_gnn(train, val, test, gnn_type=gnn_type, epochs=epochs, seed=cfg.seed)
    results["models"][f"temporal_gnn_{gnn_type}"] = {
        "params": gnn_res.params, "threshold": gnn_res.chosen_threshold,
        "val": gnn_res.val_metrics.as_dict(), "test": gnn_res.test_metrics.as_dict(),
        "confusion_matrix": gnn_res.test_metrics.confusion_matrix,
        "train_seconds": gnn_res.train_seconds, "num_parameters": gnn_res.num_parameters,
        "graph_capable": gnn_res.graph_capable, "note": gnn_res.note,
    }
    row = _metrics_row(cfg.dataset, f"temporal_gnn_{gnn_type}", info, gnn_res.test_metrics,
                       {"train_seconds": gnn_res.train_seconds, "params": gnn_res.num_parameters})
    row["graph_capable"] = gnn_res.graph_capable
    results["rows"].append(row)
    with tracker.run(f"{cfg.dataset}-gnn-{gnn_type}") as ml:
        tracker.log(ml, params={**gnn_res.params, **split_cfg, "graph_capable": gnn_res.graph_capable},
                    metrics={**gnn_res.test_metrics.as_dict(),
                             "train_seconds": gnn_res.train_seconds,
                             "num_parameters": gnn_res.num_parameters}, tags=common_tags)

    return results


def write_outputs(all_results: List[Dict], out_dir: Path) -> Dict[str, Path]:
    """Write baseline_results.csv, baseline_metrics.json, baseline_report.md."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. CSV (one row per model x dataset)
    rows: List[Dict] = []
    for res in all_results:
        rows.extend(res.get("rows", []))
    csv_path = out_dir / "baseline_results.csv"
    if rows:
        fieldnames = sorted({k for r in rows for k in r})
        # keep a stable, readable column order
        preferred = ["dataset", "model", "n_test", "test_positives", "precision",
                     "recall", "f1", "pr_auc", "roc_auc", "false_positive_rate",
                     "threshold", "tp", "fp", "tn", "fn", "train_seconds", "params",
                     "graph_capable"]
        ordered = [c for c in preferred if c in fieldnames] + \
                  [c for c in fieldnames if c not in preferred]
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=ordered)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    else:
        csv_path.write_text("dataset,model\n", encoding="utf-8")

    # 2. JSON (full nested)
    json_path = out_dir / "baseline_metrics.json"
    json_path.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")

    # 3. Markdown report
    md_path = out_dir / "baseline_report.md"
    md_path.write_text(_render_report(all_results), encoding="utf-8")

    return {"csv": csv_path, "json": json_path, "md": md_path}


def _fmt(v):
    return "n/a" if v is None or (isinstance(v, float) and v != v) else v


def _render_report(all_results: List[Dict]) -> str:
    lines: List[str] = []
    lines.append("# Sentinel-X — Phase 3 Baseline Report\n")
    lines.append("**Question:** does modelling temporal structure and network topology "
                 "provide measurable benefit over simpler approaches?\n")
    lines.append("**Target (all baselines, identical):** given information up to and "
                 "including window *t*, predict whether window *t+H* contains attack "
                 "activity (`label_any_attack`). Inputs never include *t+H* → no future "
                 "leakage. Same chronological split, seed, imbalance handling, and "
                 "train-only preprocessing across all three baselines.\n")
    lines.append("**Baselines:** (1) Logistic Regression on aggregated window features, "
                 "(2) LSTM/GRU on the feature sequence, (3) Temporal GNN (GraphSAGE/GAT) "
                 "on the graph sequence. The Sentinel-X world model is intentionally NOT "
                 "included in this phase.\n")

    for res in all_results:
        ds = res["dataset"]
        lines.append(f"\n## Dataset: {ds}\n")
        if res.get("skipped"):
            lines.append(f"**SKIPPED** — {res.get('reason')}\n")
            continue
        info = res["split_info"]
        cfg = res["config"]
        lines.append(f"- windows: {info.get('num_windows')} | forecasting samples: "
                     f"{info.get('num_samples')} | seq_len={cfg['seq_len']} horizon={cfg['horizon']}")
        lines.append(f"- split counts: {info.get('counts')} | positives per split: "
                     f"{info.get('positives')}")
        lines.append("\n| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR | TP | FP | TN | FN |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for model_name, md in res["models"].items():
            t = md["test"]
            lines.append(
                f"| {model_name} | {_fmt(round(t['precision'],4))} | "
                f"{_fmt(round(t['recall'],4))} | {_fmt(round(t['f1'],4))} | "
                f"{_fmt(round(t['pr_auc'],4) if t['pr_auc']==t['pr_auc'] else None)} | "
                f"{_fmt(round(t['roc_auc'],4) if t['roc_auc'] is not None else None)} | "
                f"{_fmt(round(t['false_positive_rate'],4))} | "
                f"{t['tp']} | {t['fp']} | {t['tn']} | {t['fn']} |")
        # GNN capability note
        for model_name, md in res["models"].items():
            if md.get("note"):
                lines.append(f"\n> **{model_name}**: {md['note']}")

    lines.append("\n## Fairness & reproducibility\n")
    lines.append("- Identical temporal split policy, leakage policy, target definition, "
                 "seed, and train-only preprocessing for every baseline.")
    lines.append("- Class imbalance handled uniformly (LogReg `class_weight=balanced`; "
                 "sequence/GNN weighted BCE with train `pos_weight`).")
    lines.append("- Decision threshold tuned on VALIDATION PR curve, applied to TEST "
                 "(never tuned on test).")
    lines.append("- No baseline intentionally weakened; no cherry-picking (all runs + "
                 "seeds recorded to MLflow and this report).")
    lines.append("\n## Interpretation guidance\n")
    lines.append("- Compare each model's PR-AUC / recall at matched FPR. If temporal "
                 "(LSTM/GRU) and topological (GNN) baselines do NOT beat Logistic "
                 "Regression, that is a legitimate finding to report — it sets the bar "
                 "the Sentinel-X world model must clear in a later phase.")
    lines.append("- Datasets flagged `graph_capable=false` (no entities, e.g. CIC-IDS2018) "
                 "cannot fairly support the GNN baseline; this is documented, not hidden.")

    lines.append("\n## Dataset applicability to the three baselines\n")
    lines.append("Not every dataset can fairly support every baseline (from the "
                 "Phase-1 audit + Phase-2 build):\n")
    lines.append("| Dataset | LogReg | LSTM/GRU | Temporal GNN | Reason |")
    lines.append("|---|---|---|---|---|")
    lines.append("| CTU-13 | ✅ | ✅ | ✅ | Has timestamps AND src/dst entities → windows + graphs. Fair for all three. |")
    lines.append("| UNSW-NB15 (raw) | ✅ | ✅ | ✅ | Raw files have timestamps + IPs → graph-capable (partition CSVs lack IPs). |")
    lines.append("| CIC-IDS2018 | ✅ | ✅ | ⚠️ | Has timestamps but NO src/dst IPs → windows form but graphs have 0 nodes; GNN not fair. |")
    lines.append("| CICIoT2023 | ⚠️ | ❌ | ❌ | NO timestamps → cannot build temporal windows or graph sequences at all; only IID classification is possible. |")
    lines.append("\nCTU-13 is therefore the primary dataset for the head-to-head "
                 "comparison of all three baselines; the others are reported with their "
                 "documented limitations rather than forced into an unfair comparison.")
    return "\n".join(lines) + "\n"
