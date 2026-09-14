"""Phase-9 named research experiments (registry + implementations).

Each experiment reuses what earlier phases already built:

  * the trained Phase-4 world-model checkpoint (models/sentinel_x/model.pt),
  * the Phase-2 window cache (data/processed/<dataset>/), and
  * the existing phase drivers (baselines, k-step, phase6, phase8, trajectory).

Nothing is retrained or rebuilt here. New experiments that earlier phases did
not cover (unseen attack types, early-warning lead time, missing telemetry,
cross-dataset generalization) are implemented directly against the frozen
checkpoint + cached windows, on the SAME chronological, leakage-safe protocol.

Discipline (project rules):
  * chronological splits only (inherited from experiments.common);
  * thresholds are NEVER tuned on the final test set — the checkpoint's own
    validation-tuned threshold is reused;
  * results are never fabricated — where a measurement is not possible (e.g.
    single-class split, non-graph dataset) the experiment reports it as skipped
    / n/a rather than inventing a number.

The registry maps experiment names to callables. ``run_experiment(name, ...)``
dispatches; ``ALL_EXPERIMENTS`` lists everything ``run_experiment.py`` can run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..experiments.common import ExperimentConfig, seed_everything
from ..experiments.metrics import compute_binary_metrics
from ..experiments.runner import run_all_baselines, write_outputs as write_baseline_outputs
from ..worldmodel.config import WorldModelConfig, load_config
from ..worldmodel.data import WorldModelDataset, collate_inputs as wm_collate
from ..worldmodel.kstep_data import KStepDataset
from ..worldmodel.kstep_experiment import run_kstep_experiments
from ..worldmodel.model import GraphTensors, SentinelXWorldModel
from ..worldmodel.phase6_experiment import run_phase6_experiments
from ..worldmodel.phase8_experiment import run_phase8_experiment
from ..worldmodel.rollout import rollout_latents

DEFAULT_CHECKPOINT_REL = Path("models") / "sentinel_x" / "model.pt"


# --------------------------------------------------------------------------- #
# Helpers shared by the directly-implemented experiments
# --------------------------------------------------------------------------- #
def _load_checkpoint(checkpoint_path: Path) -> Tuple[SentinelXWorldModel, Dict, float]:
    model, extra = SentinelXWorldModel.load_checkpoint(checkpoint_path)
    model.eval()
    threshold = float(extra.get("chosen_threshold", 0.5))
    return model, extra, threshold


def _data_cfg_from_checkpoint(checkpoint_path: Path):
    cfg_path = Path(checkpoint_path).parent / "config.yaml"
    if cfg_path.exists():
        return load_config(cfg_path).data
    return WorldModelConfig().data


def _risk_probs(model: SentinelXWorldModel, samples, batch_size: int = 32) -> np.ndarray:
    """Deterministic risk probabilities for a list of WorldModelSample."""
    import torch
    probs: List[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = list(samples[start:start + batch_size])
            out = model(wm_collate(batch))
            probs.extend(torch.sigmoid(out.risk_logit).cpu().numpy().tolist())
    return np.asarray(probs, dtype=float)


def _metrics_block(y: np.ndarray, scores: np.ndarray, threshold: float) -> Dict:
    """Full metric panel, guarding single-class splits (report n/a, not fake)."""
    y = np.asarray(y, dtype=int)
    if y.size == 0:
        return {"usable": False, "reason": "no samples"}
    m = compute_binary_metrics(y, scores, threshold=threshold)
    d = m.as_dict()
    if d.get("pr_auc") != d.get("pr_auc"):  # NaN guard
        d["pr_auc"] = None
    d["single_class"] = bool(y.sum() == 0 or y.sum() == len(y))
    return {"usable": True, **d}


# --------------------------------------------------------------------------- #
# 1. Known attacks  (reuse baselines + world model on the standard split)
# --------------------------------------------------------------------------- #
def run_known_attacks(
    *, dataset: str, processed_root: Path, checkpoint_path: Path,
    experiments_dir: Path, seed: int = 42, log=print,
) -> Dict:
    """Known-attack detection on the standard chronological test split.

    This is the world model's auxiliary risk head evaluated at the checkpoint's
    validation-tuned threshold (never re-tuned on test), on the same split the
    baselines use.
    """
    seed_everything(seed)
    model, extra, threshold = _load_checkpoint(checkpoint_path)
    data_cfg = _data_cfg_from_checkpoint(checkpoint_path)
    data_cfg.dataset = dataset

    ds = WorldModelDataset(data_cfg, processed_root,
                           model.cfg.node_feature_dim, model.cfg.edge_feature_dim)
    train, val, test, info = ds.build()
    result: Dict = {"experiment": "known_attacks", "dataset": dataset,
                    "combo": extra.get("combo"), "threshold": threshold,
                    "split_info": info}
    if not info.get("usable"):
        result["skipped"] = True
        result["reason"] = info.get("reason", "unusable")
        log(f"[known_attacks] {dataset}: SKIPPED — {result['reason']}")
        return result

    scores = _risk_probs(model, test)
    y = np.array([s.y_risk for s in test], dtype=int)
    result["test"] = _metrics_block(y, scores, threshold)
    log(f"[known_attacks] {dataset}: n={len(test)} pos={int(y.sum())} "
        f"pr_auc={result['test'].get('pr_auc')} recall={result['test'].get('recall')}")
    return result


# --------------------------------------------------------------------------- #
# 2. Unseen attack types  (generalization to attack behaviour held out of fit)
# --------------------------------------------------------------------------- #
def run_unseen_attacks(
    *, dataset: str, processed_root: Path, checkpoint_path: Path,
    experiments_dir: Path, seed: int = 42, log=print,
) -> Dict:
    """Evaluate detection on *unseen* attack windows via the OOD/novelty signal.

    The world model's known distribution was fit on BENIGN training behaviour;
    attack windows in the test split are, by construction, unseen behaviour.
    We reuse the Phase-6 Mahalanobis novelty detector (fit on benign-train
    latents only, threshold from in-distribution percentile — never on test) and
    report how well genuinely-unseen attack behaviour is flagged as novel. This
    measures generalization to attack types the model never saw as positives.
    """
    seed_everything(seed)
    # Reuse the Phase-6 driver but keep artifacts in a scoped subdir so we don't
    # overwrite the canonical Phase-6 outputs.
    sub = Path(experiments_dir) / "phase9" / "unseen_attacks"
    res = run_phase6_experiments(
        Path(checkpoint_path), Path(processed_root),
        experiments_dir=sub, seed=seed, log=log,
    )
    out: Dict = {"experiment": "unseen_attacks", "dataset": dataset,
                 "combo": res.get("combo")}
    if res.get("skipped"):
        out["skipped"] = True
        out["reason"] = res.get("reason")
        return out
    ood = res.get("ood", {})
    if not ood.get("usable"):
        out["skipped"] = True
        out["reason"] = ood.get("reason", "ood not usable")
        return out
    out["ood"] = {
        "known_distribution": ood.get("known_distribution"),
        "unseen_behaviour": ood.get("unseen_behaviour"),
        "n_attack_test": ood.get("n_attack_test"),
        "metrics": ood.get("metrics"),
    }
    m = ood.get("metrics", {})
    log(f"[unseen_attacks] {dataset}: OOD AUROC={m.get('auroc')} "
        f"detection_rate={m.get('detection_rate')}")
    return out


# --------------------------------------------------------------------------- #
# 3. K = 1 / 3 / 5 / 10   (autoregressive rollout — reuse Phase-5 driver)
# --------------------------------------------------------------------------- #
def run_kstep(
    *, dataset: str, processed_root: Path, checkpoint_path: Path,
    experiments_dir: Path, ks: Sequence[int] = (1, 3, 5, 10),
    seed: int = 42, log=print,
) -> Dict:
    res = run_kstep_experiments(
        Path(checkpoint_path), Path(processed_root),
        ks=tuple(ks), seed=seed, experiments_dir=Path(experiments_dir), log=log,
    )
    return {"experiment": "kstep", "dataset": dataset, "result": res,
            "skipped": bool(res.get("skipped")), "reason": res.get("reason")}


# --------------------------------------------------------------------------- #
# 4. Early-warning lead time  (how early is an attack flagged before it lands?)
# --------------------------------------------------------------------------- #
def run_early_warning(
    *, dataset: str, processed_root: Path, checkpoint_path: Path,
    experiments_dir: Path, K: int = 10, seed: int = 42, log=print,
) -> Dict:
    """Measure early-warning lead time from the autoregressive rollout.

    For each test anchor t whose future actually contains an attack, we roll the
    world model forward K steps and find the EARLIEST horizon k at which the
    forecast risk crosses the (validation-tuned, not re-tuned) threshold while
    the ground-truth attack occurs at some horizon >= k. The lead time is
    (horizon_of_first_true_attack - k): how many windows ahead the model raised
    the alarm before the attack window. We report the mean/median lead time in
    windows over anchors that have a real future attack, plus detection coverage.

    Leakage-safe: inputs still end at t; future labels are used only to score.
    """
    seed_everything(seed)
    model, extra, threshold = _load_checkpoint(checkpoint_path)
    data_cfg = _data_cfg_from_checkpoint(checkpoint_path)
    data_cfg.dataset = dataset

    ds = KStepDataset(data_cfg, processed_root,
                      model.cfg.node_feature_dim, model.cfg.edge_feature_dim, K=K)
    train, val, test, info = ds.build()
    out: Dict = {"experiment": "early_warning", "dataset": dataset,
                 "combo": extra.get("combo"), "threshold": threshold, "K": K,
                 "split_info": info}
    if not info.get("usable"):
        out["skipped"] = True
        out["reason"] = info.get("reason", "unusable")
        log(f"[early_warning] {dataset}: SKIPPED — {out['reason']}")
        return out

    import torch
    lead_times: List[int] = []            # windows of warning before first attack
    anchors_with_attack = 0
    anchors_warned_in_time = 0
    per_anchor: List[Dict] = []

    with torch.no_grad():
        batch_size = 32
        samples = list(test)
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            roll = rollout_latents(model, [s.input_seq for s in batch], K)
            risks = roll.risk_probs.cpu().numpy()   # (K, B)
            for j, s in enumerate(batch):
                labels = [int(s.future_labels[k]) for k in range(1, K + 1)]
                if 1 not in labels:
                    continue  # no future attack for this anchor — nothing to warn about
                anchors_with_attack += 1
                first_attack_k = labels.index(1) + 1  # horizon (1-based) of first true attack
                # earliest horizon the forecast crosses threshold
                warn_k = None
                for k in range(1, K + 1):
                    if risks[k - 1, j] >= threshold:
                        warn_k = k
                        break
                lead = None
                if warn_k is not None and warn_k <= first_attack_k:
                    lead = first_attack_k - warn_k  # >=0 windows of early warning
                    lead_times.append(lead)
                    anchors_warned_in_time += 1
                per_anchor.append({
                    "t_index": s.t_index,
                    "first_attack_horizon": first_attack_k,
                    "warn_horizon": warn_k,
                    "lead_time_windows": lead,
                })

    out["anchors_with_future_attack"] = anchors_with_attack
    out["anchors_warned_in_time"] = anchors_warned_in_time
    if lead_times:
        lt = np.array(lead_times, dtype=float)
        out["lead_time"] = {
            "n": int(lt.size),
            "mean_windows": float(lt.mean()),
            "median_windows": float(np.median(lt)),
            "max_windows": int(lt.max()),
            "min_windows": int(lt.min()),
            "coverage": float(anchors_warned_in_time / anchors_with_attack)
                        if anchors_with_attack else None,
        }
    else:
        out["lead_time"] = {"n": 0, "reason":
                            "no anchors both had a future attack and were warned "
                            "at or before the first attack horizon"}
    out["per_anchor"] = per_anchor
    log(f"[early_warning] {dataset}: attack-anchors={anchors_with_attack} "
        f"warned_in_time={anchors_warned_in_time} "
        f"mean_lead={out['lead_time'].get('mean_windows')}")
    return out


# --------------------------------------------------------------------------- #
# 5. Missing telemetry  (robustness when input features/graphs are dropped)
# --------------------------------------------------------------------------- #
def run_missing_telemetry(
    *, dataset: str, processed_root: Path, checkpoint_path: Path,
    experiments_dir: Path, drop_fractions: Sequence[float] = (0.0, 0.25, 0.5, 0.75),
    seed: int = 42, log=print,
) -> Dict:
    """Degrade telemetry by zeroing node/edge features and re-scoring.

    Simulates missing telemetry by masking a fraction of node features to zero
    (structure preserved) across the observed window graphs, then re-evaluating
    the risk head at the fixed validation-tuned threshold. Reports how PR-AUC /
    recall degrade as telemetry is progressively removed. No re-tuning, no
    retraining, deterministic masking from the seed.
    """
    seed_everything(seed)
    model, extra, threshold = _load_checkpoint(checkpoint_path)
    data_cfg = _data_cfg_from_checkpoint(checkpoint_path)
    data_cfg.dataset = dataset

    ds = WorldModelDataset(data_cfg, processed_root,
                           model.cfg.node_feature_dim, model.cfg.edge_feature_dim)
    train, val, test, info = ds.build()
    out: Dict = {"experiment": "missing_telemetry", "dataset": dataset,
                 "combo": extra.get("combo"), "threshold": threshold,
                 "split_info": info, "levels": []}
    if not info.get("usable"):
        out["skipped"] = True
        out["reason"] = info.get("reason", "unusable")
        log(f"[missing_telemetry] {dataset}: SKIPPED — {out['reason']}")
        return out

    import torch
    y = np.array([s.y_risk for s in test], dtype=int)
    rng = np.random.default_rng(seed)

    def mask_sample(sample, frac: float):
        """Return a copy of the sample's input_seq with a fraction of node
        feature rows zeroed (deterministic per level)."""
        new_seq: List[GraphTensors] = []
        for g in sample.input_seq:
            if g.num_nodes == 0 or frac <= 0.0:
                new_seq.append(g)
                continue
            x = g.x.clone()
            n = x.shape[0]
            k = int(round(n * frac))
            if k > 0:
                idx = rng.choice(n, size=k, replace=False)
                x[idx] = 0.0
            new_seq.append(GraphTensors(x=x, edge_index=g.edge_index,
                                        edge_attr=g.edge_attr, num_nodes=g.num_nodes))
        return new_seq

    for frac in drop_fractions:
        probs: List[float] = []
        with torch.no_grad():
            for s in test:
                seq = mask_sample(s, frac)
                o = model([seq])
                probs.append(float(torch.sigmoid(o.risk_logit)[0]))
        block = _metrics_block(y, np.asarray(probs), threshold)
        block["drop_fraction"] = float(frac)
        out["levels"].append(block)
        log(f"[missing_telemetry] {dataset}: drop={frac:.2f} "
            f"pr_auc={block.get('pr_auc')} recall={block.get('recall')}")
    return out


# --------------------------------------------------------------------------- #
# 6. Uncertainty vs error   (reuse Phase-6 MC-Dropout)
# --------------------------------------------------------------------------- #
def run_uncertainty_error(
    *, dataset: str, processed_root: Path, checkpoint_path: Path,
    experiments_dir: Path, seed: int = 42, log=print,
) -> Dict:
    sub = Path(experiments_dir) / "phase9" / "uncertainty_error"
    res = run_phase6_experiments(
        Path(checkpoint_path), Path(processed_root),
        experiments_dir=sub, seed=seed, log=log,
    )
    out: Dict = {"experiment": "uncertainty_error", "dataset": dataset,
                 "combo": res.get("combo")}
    if res.get("skipped"):
        out["skipped"] = True
        out["reason"] = res.get("reason")
        return out
    unc = res.get("uncertainty", {})
    cal = res.get("calibration", {})
    out["uncertainty"] = {
        "mean_uncertainty_std": unc.get("mean_uncertainty_std"),
        "uncertainty_error_correlation": unc.get("uncertainty_error_correlation"),
        "mc_passes": unc.get("mc_passes"),
    }
    out["calibration"] = {
        "ece": (cal.get("before") or {}).get("ece"),
        "brier": (cal.get("before") or {}).get("brier"),
    }
    log(f"[uncertainty_error] {dataset}: unc/err corr="
        f"{out['uncertainty']['uncertainty_error_correlation']} "
        f"ece={out['calibration']['ece']}")
    return out


# --------------------------------------------------------------------------- #
# 7. OOD detection   (reuse Phase-6 Mahalanobis novelty)
# --------------------------------------------------------------------------- #
def run_ood_detection(
    *, dataset: str, processed_root: Path, checkpoint_path: Path,
    experiments_dir: Path, seed: int = 42, log=print,
) -> Dict:
    sub = Path(experiments_dir) / "phase9" / "ood_detection"
    res = run_phase6_experiments(
        Path(checkpoint_path), Path(processed_root),
        experiments_dir=sub, seed=seed, log=log,
    )
    out: Dict = {"experiment": "ood_detection", "dataset": dataset,
                 "combo": res.get("combo")}
    if res.get("skipped"):
        out["skipped"] = True
        out["reason"] = res.get("reason")
        return out
    ood = res.get("ood", {})
    if not ood.get("usable"):
        out["skipped"] = True
        out["reason"] = ood.get("reason", "ood not usable")
        return out
    out["ood"] = {"detector": ood.get("detector"), "metrics": ood.get("metrics")}
    log(f"[ood_detection] {dataset}: AUROC={ood.get('metrics', {}).get('auroc')}")
    return out


# --------------------------------------------------------------------------- #
# 8. Forecast stability   (reuse Phase-8 stability under perturbation)
# --------------------------------------------------------------------------- #
def run_forecast_stability(
    *, dataset: str, processed_root: Path, checkpoint_path: Path,
    experiments_dir: Path, K: int = 5, max_anchors: int = 25,
    seed: int = 42, log=print,
) -> Dict:
    sub = Path(experiments_dir) / "phase9" / "forecast_stability"
    res = run_phase8_experiment(
        Path(checkpoint_path), Path(processed_root),
        K=K, max_anchors=max_anchors, seed=seed,
        experiments_dir=sub, log=log,
    )
    out: Dict = {"experiment": "forecast_stability", "dataset": dataset,
                 "combo": res.get("combo")}
    if res.get("skipped"):
        out["skipped"] = True
        out["reason"] = res.get("reason")
        return out
    out["summary"] = res.get("summary", {})
    log(f"[forecast_stability] {dataset}: mean_stability="
        f"{out['summary'].get('mean_stability_score')}")
    return out


# --------------------------------------------------------------------------- #
# 9. Cross-dataset generalization  (train on A, evaluate the frozen model on B)
# --------------------------------------------------------------------------- #
def run_cross_dataset(
    *, dataset: str, processed_root: Path, checkpoint_path: Path,
    experiments_dir: Path, target_datasets: Optional[Sequence[str]] = None,
    seed: int = 42, log=print,
) -> Dict:
    """Apply the frozen checkpoint (trained on its source dataset) to OTHER
    graph-capable datasets and report the risk head's transfer performance.

    The model is not retrained. A dataset that is not graph-capable or lacks a
    usable window cache is reported as skipped, never forced.
    """
    seed_everything(seed)
    model, extra, threshold = _load_checkpoint(checkpoint_path)
    src = extra.get("dataset") or dataset
    data_cfg = _data_cfg_from_checkpoint(checkpoint_path)

    candidates = list(target_datasets) if target_datasets else \
        ["ctu-13", "unsw-nb15", "cic-ids2018"]
    out: Dict = {"experiment": "cross_dataset", "source_dataset": src,
                 "combo": extra.get("combo"), "threshold": threshold,
                 "transfers": []}

    for tgt in candidates:
        data_cfg.dataset = tgt
        ds = WorldModelDataset(data_cfg, processed_root,
                               model.cfg.node_feature_dim, model.cfg.edge_feature_dim)
        try:
            train, val, test, info = ds.build()
        except FileNotFoundError as e:
            out["transfers"].append({"dataset": tgt, "skipped": True,
                                     "reason": f"no cache: {e}"})
            log(f"[cross_dataset] {tgt}: SKIPPED — no cache")
            continue
        block: Dict = {"dataset": tgt, "is_source": tgt == src,
                       "split_info": {"usable": info.get("usable"),
                                      "graph_capable": info.get("graph_capable"),
                                      "counts": info.get("counts")}}
        if not info.get("usable"):
            block["skipped"] = True
            block["reason"] = info.get("reason", "unusable")
            out["transfers"].append(block)
            log(f"[cross_dataset] {tgt}: SKIPPED — {block['reason']}")
            continue
        scores = _risk_probs(model, test)
        y = np.array([s.y_risk for s in test], dtype=int)
        block["test"] = _metrics_block(y, scores, threshold)
        out["transfers"].append(block)
        log(f"[cross_dataset] {tgt}: pr_auc={block['test'].get('pr_auc')} "
            f"recall={block['test'].get('recall')} (source={tgt == src})")
    return out


# --------------------------------------------------------------------------- #
# 10. Baseline comparison   (reuse Phase-3 baseline runner)
# --------------------------------------------------------------------------- #
def run_baseline_comparison(
    *, dataset: str, processed_root: Path, checkpoint_path: Path,
    experiments_dir: Path, epochs: int = 30, seed: int = 42, log=print,
) -> Dict:
    """Re-run the Phase-3 baselines so the world model can be compared against
    LogReg / GRU / GraphSAGE on the identical protocol."""
    cfg = ExperimentConfig(dataset=dataset, processed_root=Path(processed_root),
                           seed=seed)
    sub = Path(experiments_dir) / "phase9" / "baseline_comparison"
    sub.mkdir(parents=True, exist_ok=True)
    mlflow_uri = "sqlite:///" + str((Path(experiments_dir) / "mlflow.db").resolve()).replace("\\", "/")
    res = run_all_baselines(cfg, epochs=epochs, mlflow_uri=mlflow_uri,
                            experiment_name="sentinelx-phase9-baselines", log=log)
    write_baseline_outputs([res], sub)
    return {"experiment": "baseline_comparison", "dataset": dataset,
            "result": res, "skipped": bool(res.get("skipped")),
            "reason": res.get("reason")}


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
EXPERIMENT_REGISTRY: Dict[str, Callable[..., Dict]] = {
    "known_attacks": run_known_attacks,
    "unseen_attacks": run_unseen_attacks,
    "kstep": run_kstep,
    "early_warning": run_early_warning,
    "missing_telemetry": run_missing_telemetry,
    "uncertainty_error": run_uncertainty_error,
    "ood_detection": run_ood_detection,
    "forecast_stability": run_forecast_stability,
    "cross_dataset": run_cross_dataset,
    "baseline_comparison": run_baseline_comparison,
}

ALL_EXPERIMENTS: List[str] = list(EXPERIMENT_REGISTRY.keys())


def run_experiment(name: str, **kwargs) -> Dict:
    """Dispatch a named experiment. Raises KeyError with the valid names."""
    if name not in EXPERIMENT_REGISTRY:
        raise KeyError(f"unknown experiment '{name}'. "
                       f"Valid: {', '.join(ALL_EXPERIMENTS)}")
    return EXPERIMENT_REGISTRY[name](**kwargs)
