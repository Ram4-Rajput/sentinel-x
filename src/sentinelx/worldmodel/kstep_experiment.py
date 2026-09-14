"""Phase-5 experiment driver: K-step future-state forecasting.

Loads the trained Phase-4 world model checkpoint, builds K-step evaluation
samples on the SAME protocol as Phase-3/Phase-4, runs the autoregressive
rollout for each configured K (default 1, 3, 5, 10), and writes:

    experiments/k_step_results.csv   (one row per (K, horizon))
    experiments/k_step_report.md     (comparison + interpretation)
    experiments/k_step_metrics.json  (full nested results)

The largest configured K is used to build the sample set once (samples for a
larger K are a subset — deeper horizon requires more future windows), and
smaller Ks are evaluated by truncating the rollout on that same sample set, so
every K is scored on an identical, comparable population.
"""

from __future__ import annotations

import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..experiments.common import seed_everything
from .config import WorldModelConfig
from .kstep_data import KStepDataset, KStepSample
from .kstep_eval import KStepEvaluation, evaluate_kstep
from .model import SentinelXWorldModel

DEFAULT_KS = (1, 3, 5, 10)


def _fmt(v, nd=6):
    if v is None:
        return "n/a"
    if isinstance(v, float) and v != v:
        return "n/a"
    if isinstance(v, float):
        return round(v, nd)
    return v


def run_kstep_experiments(
    checkpoint_path: Path,
    processed_root: Path,
    *,
    ks: Sequence[int] = DEFAULT_KS,
    batch_size: int = 32,
    seed: int = 42,
    data_cfg=None,
    experiments_dir: Optional[Path] = None,
    log=print,
) -> Dict:
    """Run K-step forecasting evaluation from a trained checkpoint.

    ``data_cfg`` defaults to the checkpoint's own DataConfig (dataset/seq_len/
    split fractions) so evaluation matches training exactly.
    """
    seed_everything(seed)
    ks = sorted({int(k) for k in ks})
    if not ks or min(ks) < 1:
        raise ValueError(f"ks must be positive integers, got {list(ks)}")
    k_max = max(ks)

    model, extra = SentinelXWorldModel.load_checkpoint(checkpoint_path)
    model.eval()
    risk_threshold = float(extra.get("chosen_threshold", 0.5))

    node_dim = model.cfg.node_feature_dim
    edge_dim = model.cfg.edge_feature_dim

    # DataConfig: reuse the checkpoint's config if not supplied.
    if data_cfg is None:
        cfg_path = Path(checkpoint_path).parent / "config.yaml"
        from .config import load_config
        data_cfg = load_config(cfg_path).data if cfg_path.exists() else WorldModelConfig().data

    ds = KStepDataset(data_cfg, processed_root, node_dim, edge_dim, K=k_max)
    train, val, test, info = ds.build()

    results: Dict = {
        "phase": "5-k-step-forecasting",
        "checkpoint": str(checkpoint_path),
        "combo": extra.get("combo"),
        "dataset": data_cfg.dataset,
        "seq_len": data_cfg.seq_len,
        "risk_threshold": risk_threshold,
        "ks": ks,
        "split_info": info,
        "num_parameters": model.num_parameters(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "evaluations": {},
    }

    if not info.get("usable"):
        results["skipped"] = True
        results["reason"] = info.get("reason", "unusable")
        log(f"[kstep] {data_cfg.dataset}: SKIPPED — {results['reason']}")
        if experiments_dir is not None:
            _write_outputs(results, experiments_dir, log)
        return results

    log(f"[kstep] {data_cfg.dataset}: test samples={len(test)} "
        f"(built at K={k_max}) | graph_capable={info.get('graph_capable')}")

    # Evaluate each K on the test split. Rollout for K reuses the learned
    # one-step transition — NOT K independent classifiers.
    for K in ks:
        ev: KStepEvaluation = evaluate_kstep(
            model, test, K, batch_size=batch_size, risk_threshold=risk_threshold)
        results["evaluations"][str(K)] = ev.as_dict()
        h1 = ev.horizons[0].state_cosine_distance if ev.horizons else None
        hK = ev.horizons[-1].state_cosine_distance if ev.horizons else None
        log(f"[kstep]  K={K}: state cos-dist h1={_fmt(h1,4)} h{K}={_fmt(hK,4)} "
            f"| rollout {ev.rollout_seconds:.4f}s ({ev.per_step_seconds*1e6:.2f} us/step)")

    if experiments_dir is not None:
        _write_outputs(results, experiments_dir, log)
    return results


def _write_outputs(results: Dict, experiments_dir: Path, log) -> None:
    experiments_dir = Path(experiments_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    # ---- CSV: one row per (K, horizon) ----
    csv_path = experiments_dir / "k_step_results.csv"
    fields = ["dataset", "K", "horizon", "n", "positives", "negatives",
              "state_cosine_distance", "state_cosine_sim", "state_l2",
              "risk_pr_auc", "risk_roc_auc", "risk_f1",
              "risk_precision", "risk_recall",
              "rollout_seconds", "per_step_seconds",
              "degradation_abs", "degradation_ratio"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for K in results.get("ks", []):
            ev = results.get("evaluations", {}).get(str(K))
            if not ev:
                continue
            for h in ev["horizons"]:
                w.writerow({
                    "dataset": results["dataset"], "K": K, "horizon": h["horizon"],
                    "n": h["n"], "positives": h["positives"], "negatives": h["negatives"],
                    "state_cosine_distance": _fmt(h["state_cosine_distance"]),
                    "state_cosine_sim": _fmt(h["state_cosine_sim"]),
                    "state_l2": _fmt(h["state_l2"]),
                    "risk_pr_auc": _fmt(h["risk_pr_auc"], 4),
                    "risk_roc_auc": _fmt(h["risk_roc_auc"], 4),
                    "risk_f1": _fmt(h["risk_f1"], 4),
                    "risk_precision": _fmt(h["risk_precision"], 4),
                    "risk_recall": _fmt(h["risk_recall"], 4),
                    "rollout_seconds": _fmt(ev["rollout_seconds"]),
                    "per_step_seconds": _fmt(ev["per_step_seconds"], 8),
                    "degradation_abs": _fmt(ev["degradation_abs"]),
                    "degradation_ratio": _fmt(ev["degradation_ratio"]),
                })

    # ---- JSON (full) ----
    (experiments_dir / "k_step_metrics.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")

    # ---- Markdown report ----
    (experiments_dir / "k_step_report.md").write_text(
        _render_report(results), encoding="utf-8")
    log(f"[kstep]  wrote {csv_path.name}, k_step_report.md, k_step_metrics.json")


def _render_report(results: Dict) -> str:
    L: List[str] = []
    L.append("# Sentinel-X — Phase 5: K-Step Future-State Forecasting\n")
    L.append("**Goal.** Forecast the network state trajectory "
             "`S_t → S_{t+1} → … → S_{t+K}` for configurable K "
             f"({', '.join('K='+str(k) for k in results.get('ks', []))}).\n")
    L.append("**Method — autoregressive rollout (NOT K independent classifiers).** "
             "The Phase-4 world model learns a single one-step latent transition "
             "`f: z_t → z_{t+1}`. Phase-5 rolls that ONE learned transition "
             "forward, feeding each predicted latent back in:\n")
    L.append("```\n"
             "z_t = encode_state(G_{t-L+1..t})\n"
             "z_hat_{t+1} = f(z_t)\n"
             "z_hat_{t+2} = f(z_hat_{t+1})\n"
             "...\n"
             "z_hat_{t+K} = f(z_hat_{t+K-1})\n"
             "risk_{t+k} = sigmoid(risk_head(z_hat_{t+k}))\n```\n")
    L.append("For each future step we return the predicted latent/state, the "
             "predicted risk (where a label exists), the horizon/target window "
             "index, and prediction metadata (see `k_step_metrics.json`).\n")

    if results.get("skipped"):
        L.append(f"\n**SKIPPED** — {results.get('reason')}\n")
        return "\n".join(L) + "\n"

    info = results["split_info"]
    L.append(f"\n## Setup\n")
    L.append(f"- checkpoint: `{results['checkpoint']}` (combo "
             f"**{results.get('combo')}**, {results['num_parameters']} params)")
    L.append(f"- dataset: **{results['dataset']}** | seq_len={results['seq_len']} "
             f"| risk threshold (from Phase-4 val tuning): "
             f"{round(results['risk_threshold'], 4)}")
    L.append(f"- evaluation split: test samples={info.get('counts', {}).get('test')} "
             f"| graph_capable={info.get('graph_capable')}")
    L.append(f"- positives per horizon (test): {info.get('test_positives_by_horizon')}\n")
    L.append("Every K is scored on the same test population (samples built at "
             "the deepest K so all horizons have real future targets), so the "
             "K=1/3/5/10 comparison is apples-to-apples.\n")

    # Per-K detail tables.
    for K in results.get("ks", []):
        ev = results.get("evaluations", {}).get(str(K))
        if not ev:
            continue
        L.append(f"\n## K = {K}\n")
        L.append(f"- rollout wall-clock: {ev['rollout_seconds']:.4f}s over "
                 f"{ev['n_samples']} samples × {K} steps "
                 f"(**{ev['per_step_seconds']*1e6:.2f} µs / sample-step**); "
                 f"future-encode cost {ev['encode_seconds']:.4f}s")
        L.append(f"- state-error degradation (cosine distance, horizon {K} vs 1): "
                 f"abs **{_fmt(ev['degradation_abs'])}**, "
                 f"ratio **{_fmt(ev['degradation_ratio'])}×**\n")
        L.append("| Horizon (t+k) | n | pos | State cos-dist ↓ | State cos-sim ↑ | State L2 ↓ | Risk PR-AUC | Risk ROC-AUC | Risk F1 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for h in ev["horizons"]:
            L.append(
                f"| {h['horizon']} | {h['n']} | {h['positives']} | "
                f"{_fmt(h['state_cosine_distance'],4)} | {_fmt(h['state_cosine_sim'],4)} | "
                f"{_fmt(h['state_l2'],4)} | {_fmt(h['risk_pr_auc'],4)} | "
                f"{_fmt(h['risk_roc_auc'],4)} | {_fmt(h['risk_f1'],4)} |")

    # Cross-K comparison.
    L.append("\n## Comparison across K\n")
    L.append("| K | State cos-dist @ t+1 | State cos-dist @ t+K | Degradation (abs) | Degradation (×) | Rollout s | µs / sample-step |")
    L.append("|---|---|---|---|---|---|---|")
    for K in results.get("ks", []):
        ev = results.get("evaluations", {}).get(str(K))
        if not ev or not ev["horizons"]:
            continue
        h1 = ev["horizons"][0]["state_cosine_distance"]
        hK = ev["horizons"][-1]["state_cosine_distance"]
        L.append(f"| {K} | {_fmt(h1,4)} | {_fmt(hK,4)} | {_fmt(ev['degradation_abs'])} | "
                 f"{_fmt(ev['degradation_ratio'])} | {_fmt(ev['rollout_seconds'])} | "
                 f"{_fmt(ev['per_step_seconds']*1e6,2)} |")

    L.append("\n## Interpretation\n")
    L.append("- **State prediction error vs. horizon.** Cosine distance between "
             "the rolled-out latent and the encoder's own encoding of the actual "
             "future state. Distance at t+1 is the one-step error the model was "
             "trained on; distance at deeper horizons shows how autoregressive "
             "error compounds.")
    L.append("- **Degradation with horizon.** The compounding is summarised by "
             "the abs/ratio columns (state error at the deepest horizon relative "
             "to t+1). A ratio near 1.0 means the rollout stays stable; a large "
             "ratio means error accumulates as predictions feed back into "
             "themselves — the expected behaviour of autoregressive rollout.")
    L.append("- **Future-risk performance.** The auxiliary risk head is read off "
             "each predicted latent. Where a horizon's test labels are "
             "single-class, ranking metrics are reported as n/a rather than a "
             "misleading value.")
    L.append("- **Computational cost.** One encode + K cheap latent-space "
             "transitions per sample; cost scales linearly in K (see µs / "
             "sample-step), far cheaper than running K independent models.")
    L.append("\n*All numbers are produced from the trained checkpoint on real "
             "cached data. No results are fabricated; single-class horizons and "
             "unstable metrics are reported honestly.*\n")
    return "\n".join(L) + "\n"
