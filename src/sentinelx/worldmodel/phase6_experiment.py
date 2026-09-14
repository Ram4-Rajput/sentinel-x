"""Phase-6 experiment driver: Risk + Uncertainty + Calibration + Novelty/OOD.

Loads the trained Phase-4/5 world-model checkpoint and produces FOUR explicitly
separate signals on the SAME protocol/splits as the earlier phases:

  1. RISK          — the trained risk head's probability (from the model output;
                     no invented dashboard formula).
  2. UNCERTAINTY   — genuine MC-Dropout predictive mean + variance (30 stochastic
                     passes by default, configurable). Exposed separately from
                     risk; never collapsed into one "confidence" number.
  3. CALIBRATION   — ECE, Brier, reliability bins on the risk head, plus optional
                     temperature scaling (a separate post-hoc stage; before/after
                     reported so poor calibration is never hidden).
  4. NOVELTY / OOD — Mahalanobis distance in the learned latent space. The
                     known-distribution manifold + threshold are fit on
                     IN-DISTRIBUTION (benign train) latents ONLY; the held-out
                     unseen behaviour (attack windows) never leaks into the fit.

Outputs (experiments/):
    uncertainty_results.csv
    calibration_results.csv
    ood_results.csv
    uncertainty_ood_report.md

The four signals are kept distinct in every artifact: raw values plus a
*documented* derived status, with the explicit note that the interpretation is
not a universal truth.
"""

from __future__ import annotations

import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..experiments.common import seed_everything
from .calibration import (
    CalibrationReport, TemperatureScaler, evaluate_calibration, logit,
)
from .config import WorldModelConfig, load_config
from .data import (
    WorldModelDataset, WorldModelSample, collate_inputs as wm_collate_inputs,
)
from .model import SentinelXWorldModel
from .ood import (
    MahalanobisOOD, encode_latents, ood_detection_metrics,
)
from .uncertainty import (
    DEFAULT_MC_PASSES, count_dropout_layers, deterministic_risk,
    estimate_uncertainty, has_active_dropout,
)


def _fmt(v, nd=6):
    if v is None:
        return "n/a"
    if isinstance(v, float) and v != v:
        return "n/a"
    if isinstance(v, float):
        return round(v, nd)
    return v


def _batch_risk(model: SentinelXWorldModel, samples: Sequence[WorldModelSample],
                batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic risk probabilities AND logits for a set of samples."""
    import torch
    model.eval()
    probs: List[float] = []
    logits: List[float] = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = list(samples[start:start + batch_size])
            out = model(wm_collate_inputs(batch))
            logits.extend(out.risk_logit.detach().cpu().numpy().tolist())
            probs.extend(torch.sigmoid(out.risk_logit).detach().cpu().numpy().tolist())
    return np.asarray(probs), np.asarray(logits)


def run_phase6_experiments(
    checkpoint_path: Path,
    processed_root: Path,
    *,
    mc_passes: int = DEFAULT_MC_PASSES,
    n_calibration_bins: int = 10,
    ood_percentile: float = 95.0,
    use_temperature_scaling: bool = True,
    batch_size: int = 32,
    seed: int = 42,
    data_cfg=None,
    experiments_dir: Optional[Path] = None,
    log=print,
) -> Dict:
    """Compute the four Phase-6 signals from a trained checkpoint."""
    seed_everything(seed)

    model, extra = SentinelXWorldModel.load_checkpoint(checkpoint_path)
    model.eval()
    risk_threshold = float(extra.get("chosen_threshold", 0.5))
    node_dim = model.cfg.node_feature_dim
    edge_dim = model.cfg.edge_feature_dim

    if data_cfg is None:
        cfg_path = Path(checkpoint_path).parent / "config.yaml"
        data_cfg = (load_config(cfg_path).data if cfg_path.exists()
                    else WorldModelConfig().data)

    ds = WorldModelDataset(data_cfg, processed_root, node_dim, edge_dim)
    train, val, test, info = ds.build()

    results: Dict = {
        "phase": "6-risk-uncertainty-novelty-ood",
        "checkpoint": str(checkpoint_path),
        "combo": extra.get("combo"),
        "dataset": data_cfg.dataset,
        "seq_len": data_cfg.seq_len,
        "risk_threshold": risk_threshold,
        "mc_passes": mc_passes,
        "ood_percentile": ood_percentile,
        "n_calibration_bins": n_calibration_bins,
        "dropout_layers": count_dropout_layers(model),
        "dropout_active": has_active_dropout(model),
        "num_parameters": model.num_parameters(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "split_info": info,
    }

    if not info.get("usable"):
        results["skipped"] = True
        results["reason"] = info.get("reason", "unusable")
        log(f"[phase6] {data_cfg.dataset}: SKIPPED — {results['reason']}")
        if experiments_dir is not None:
            _write_outputs(results, experiments_dir, log)
        return results

    if not results["dropout_active"]:
        # MC Dropout requires active dropout — surface this rather than faking it.
        log("[phase6] WARNING: model has no active dropout (p>0); MC-Dropout "
            "variance will be ~0. Uncertainty is only meaningful with dropout.")

    log(f"[phase6] {data_cfg.dataset}: train={len(train)} val={len(val)} "
        f"test={len(test)} | dropout_layers={results['dropout_layers']} "
        f"active={results['dropout_active']}")

    # ---------------------------------------------------------------- #
    # 1 + 2. RISK and UNCERTAINTY (test split)
    # ---------------------------------------------------------------- #
    det_risk, test_logits = _batch_risk(model, test, batch_size)
    unc = estimate_uncertainty(
        model, test, collate_inputs=wm_collate_inputs,
        n_passes=mc_passes, batch_size=batch_size, seed=seed)
    y_test = np.array([s.y_risk for s in test], dtype=int)

    # per-sample uncertainty rows (risk and uncertainty kept DISTINCT)
    uncertainty_rows: List[Dict] = []
    mc_mean = np.array([u.risk_mean for u in unc])
    mc_var = np.array([u.variance for u in unc])
    mc_std = np.array([u.std for u in unc])
    for i, s in enumerate(test):
        # deterministic risk error at this sample (|mc_mean - label|)
        uncertainty_rows.append({
            "t_index": s.t_index,
            "label": int(s.y_risk),
            "risk_deterministic": float(det_risk[i]),
            "risk_mc_mean": float(mc_mean[i]),
            "uncertainty_variance": float(mc_var[i]),
            "uncertainty_std": float(mc_std[i]),
            "abs_error_mc": float(abs(mc_mean[i] - s.y_risk)),
        })

    # uncertainty-vs-error relationship (do high-uncertainty samples err more?)
    abs_err = np.abs(mc_mean - y_test)
    unc_err_corr = None
    if mc_std.std() > 1e-12 and abs_err.std() > 1e-12:
        unc_err_corr = float(np.corrcoef(mc_std, abs_err)[0, 1])
    results["uncertainty"] = {
        "n": len(test),
        "mc_passes": mc_passes,
        "mean_uncertainty_std": float(mc_std.mean()),
        "mean_uncertainty_variance": float(mc_var.mean()),
        "max_uncertainty_std": float(mc_std.max()) if len(mc_std) else None,
        "uncertainty_error_correlation": unc_err_corr,
        "rows": uncertainty_rows,
    }
    log(f"[phase6] uncertainty: mean std={_fmt(mc_std.mean(),4)} "
        f"unc/err corr={_fmt(unc_err_corr,4)}")

    # ---------------------------------------------------------------- #
    # 3. CALIBRATION (test split) — before, and optionally after T-scaling
    # ---------------------------------------------------------------- #
    cal_before: CalibrationReport = evaluate_calibration(
        y_test, det_risk, n_bins=n_calibration_bins)
    calibration_block: Dict = {
        "before": cal_before.as_dict(),
        "temperature_scaling": None,
        "after": None,
    }
    if use_temperature_scaling and len(val) > 0:
        # Fit T on VALIDATION logits (never on test).
        _, val_logits = _batch_risk(model, val, batch_size)
        y_val = np.array([s.y_risk for s in val], dtype=int)
        scaler = TemperatureScaler().fit(val_logits, y_val, seed=seed)
        calibration_block["temperature_scaling"] = scaler.as_dict()
        if scaler.fitted:
            cal_probs = scaler.transform(test_logits)
            cal_after = evaluate_calibration(
                y_test, cal_probs, n_bins=n_calibration_bins)
            calibration_block["after"] = cal_after.as_dict()
    results["calibration"] = calibration_block
    log(f"[phase6] calibration: ECE={_fmt(cal_before.ece,4)} "
        f"Brier={_fmt(cal_before.brier,4)} "
        f"(T={_fmt((calibration_block['temperature_scaling'] or {}).get('temperature'),4)})")

    # ---------------------------------------------------------------- #
    # 4. NOVELTY / OOD — Mahalanobis in latent space.
    #    Known distribution = benign TRAIN latents ONLY. The held-out unseen
    #    behaviour (attack windows) never leaks into the manifold or threshold.
    # ---------------------------------------------------------------- #
    ood_block = _run_ood(model, train, val, test, ood_percentile, batch_size, log)
    results["ood"] = ood_block

    if experiments_dir is not None:
        _write_outputs(results, experiments_dir, log)
    return results


def _run_ood(model, train, val, test, ood_percentile, batch_size, log) -> Dict:
    """Fit Mahalanobis OOD on BENIGN-train latents; score benign vs attack."""
    # Encode latents deterministically.
    train_lat = encode_latents(model, train, collate_inputs=wm_collate_inputs,
                               batch_size=batch_size)
    val_lat = encode_latents(model, val, collate_inputs=wm_collate_inputs,
                             batch_size=batch_size)
    test_lat = encode_latents(model, test, collate_inputs=wm_collate_inputs,
                              batch_size=batch_size)
    y_train = np.array([s.y_risk for s in train], dtype=int)
    y_val = np.array([s.y_risk for s in val], dtype=int)
    y_test = np.array([s.y_risk for s in test], dtype=int)

    # KNOWN (in-distribution) manifold = benign TRAIN windows only.
    benign_train = train_lat[y_train == 0]
    if benign_train.shape[0] < 2:
        return {"usable": False,
                "reason": "Fewer than 2 benign training windows; cannot fit "
                          "an in-distribution manifold."}

    det = MahalanobisOOD().fit(benign_train)
    # Threshold from IN-DISTRIBUTION (benign train+val) latents only — no test.
    benign_val = val_lat[y_val == 0]
    indist_for_thr = (np.concatenate([benign_train, benign_val], axis=0)
                      if benign_val.shape[0] > 0 else benign_train)
    det.set_threshold_from_indist(indist_for_thr, percentile=ood_percentile)

    # Evaluate on TEST: benign (known, label 0) vs attack (unseen, label 1).
    benign_test = test_lat[y_test == 0]
    attack_test = test_lat[y_test == 1]
    indist_scores = det.distance(benign_test) if benign_test.shape[0] else np.array([])
    ood_scores = det.distance(attack_test) if attack_test.shape[0] else np.array([])

    metrics = ood_detection_metrics(indist_scores, ood_scores, det.threshold)

    # Per-sample rows (all test samples) — novelty_score + is_ood exposed raw.
    all_scores = det.distance(test_lat)
    rows: List[Dict] = []
    for i, s in enumerate(test):
        rows.append({
            "t_index": s.t_index,
            "label": int(s.y_risk),
            "behaviour": "attack" if s.y_risk == 1 else "benign",
            "novelty_score": float(all_scores[i]),
            "is_ood": bool(all_scores[i] > det.threshold),
        })

    log(f"[phase6] ood: threshold={_fmt(det.threshold,4)} "
        f"AUROC={_fmt(metrics.get('auroc'),4)} "
        f"AUPRC={_fmt(metrics.get('auprc'),4)} "
        f"detect={_fmt(metrics.get('detection_rate'),4)} "
        f"FAR={_fmt(metrics.get('false_acceptance_rate'),4)}")

    return {
        "usable": True,
        "detector": det.as_dict(),
        "known_distribution": "benign_train_windows",
        "unseen_behaviour": "attack_windows_held_out_of_fit",
        "n_benign_train": int(benign_train.shape[0]),
        "n_benign_test": int(benign_test.shape[0]),
        "n_attack_test": int(attack_test.shape[0]),
        "metrics": metrics,
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #
def _write_outputs(results: Dict, experiments_dir: Path, log) -> None:
    experiments_dir = Path(experiments_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    _write_uncertainty_csv(results, experiments_dir)
    _write_calibration_csv(results, experiments_dir)
    _write_ood_csv(results, experiments_dir)
    (experiments_dir / "uncertainty_ood_metrics.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")
    (experiments_dir / "uncertainty_ood_report.md").write_text(
        _render_report(results), encoding="utf-8")
    log(f"[phase6]  wrote uncertainty_results.csv, calibration_results.csv, "
        f"ood_results.csv, uncertainty_ood_report.md")


def _write_uncertainty_csv(results: Dict, d: Path) -> None:
    path = d / "uncertainty_results.csv"
    fields = ["t_index", "label", "risk_deterministic", "risk_mc_mean",
              "uncertainty_variance", "uncertainty_std", "abs_error_mc"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in results.get("uncertainty", {}).get("rows", []):
            w.writerow({k: _fmt(r.get(k)) for k in fields})


def _write_calibration_csv(results: Dict, d: Path) -> None:
    path = d / "calibration_results.csv"
    fields = ["stage", "lower", "upper", "count", "mean_confidence",
              "observed_frequency", "gap", "ece", "mce", "brier", "temperature"]
    cal = results.get("calibration", {})
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for stage in ("before", "after"):
            rep = cal.get(stage)
            if not rep:
                continue
            temp = None
            if stage == "after" and cal.get("temperature_scaling"):
                temp = cal["temperature_scaling"].get("temperature")
            for b in rep.get("bins", []):
                w.writerow({
                    "stage": stage, "lower": _fmt(b["lower"], 3),
                    "upper": _fmt(b["upper"], 3), "count": b["count"],
                    "mean_confidence": _fmt(b["mean_confidence"], 4),
                    "observed_frequency": _fmt(b["observed_frequency"], 4),
                    "gap": _fmt(b["gap"], 4),
                    "ece": _fmt(rep["ece"], 4), "mce": _fmt(rep["mce"], 4),
                    "brier": _fmt(rep["brier"], 4), "temperature": _fmt(temp, 4),
                })


def _write_ood_csv(results: Dict, d: Path) -> None:
    path = d / "ood_results.csv"
    fields = ["t_index", "label", "behaviour", "novelty_score", "is_ood"]
    ood = results.get("ood", {})
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in ood.get("rows", []):
            w.writerow({
                "t_index": r["t_index"], "label": r["label"],
                "behaviour": r["behaviour"],
                "novelty_score": _fmt(r["novelty_score"], 4),
                "is_ood": r["is_ood"],
            })


def _render_report(results: Dict) -> str:
    L: List[str] = []
    L.append("# Sentinel-X — Phase 6: Risk + Uncertainty + Calibration + Novelty/OOD\n")
    L.append("Three (four) **explicitly separate** signals. They are NEVER "
             "collapsed into a single \"confidence\" number:\n")
    L.append("- **Risk** — the trained risk head's estimated likelihood of "
             "malicious progression at the prediction target (model output, not "
             "a dashboard formula).")
    L.append("- **Uncertainty** — genuine MC-Dropout predictive variance "
             "(dropout kept active at inference; multiple stochastic passes).")
    L.append("- **Calibration** — ECE / Brier / reliability, plus optional "
             "temperature scaling as a separate stage.")
    L.append("- **Novelty / OOD** — Mahalanobis distance in the learned latent "
             "space, threshold derived from the training distribution.\n")

    if results.get("skipped"):
        L.append(f"\n**SKIPPED** — {results.get('reason')}\n")
        return "\n".join(L) + "\n"

    info = results["split_info"]
    L.append("## Setup\n")
    L.append(f"- checkpoint: `{results['checkpoint']}` (combo "
             f"**{results.get('combo')}**, {results['num_parameters']} params)")
    L.append(f"- dataset: **{results['dataset']}** | seq_len={results['seq_len']}")
    L.append(f"- test samples: {info.get('counts', {}).get('test')} | "
             f"dropout layers: {results['dropout_layers']} "
             f"(active p>0: {results['dropout_active']})")
    L.append(f"- MC passes: **{results['mc_passes']}** | OOD percentile: "
             f"{results['ood_percentile']} | calibration bins: "
             f"{results['n_calibration_bins']}\n")

    # Example of the three signals staying distinct (illustrative, per spec).
    _render_signal_example(L, results)

    # --- Uncertainty ---
    u = results.get("uncertainty", {})
    L.append("\n## 1-2. Risk & Uncertainty (MC Dropout)\n")
    L.append("Risk and uncertainty are reported as **distinct** quantities. The "
             "MC-Dropout mean is an uncertainty-aware risk estimate; the "
             "variance/std is the uncertainty itself. Inputs are identical "
             "across passes — only dropout masks change (no input perturbation).\n")
    L.append(f"- mean predictive std (uncertainty): "
             f"**{_fmt(u.get('mean_uncertainty_std'),4)}**")
    L.append(f"- mean predictive variance: {_fmt(u.get('mean_uncertainty_variance'),6)}")
    L.append(f"- uncertainty↔error correlation: "
             f"**{_fmt(u.get('uncertainty_error_correlation'),4)}** "
             f"(positive ⇒ the model is more uncertain where it errs more).\n")
    L.append("Per-sample values in `uncertainty_results.csv`.\n")

    # --- Calibration ---
    cal = results.get("calibration", {})
    before = cal.get("before", {})
    after = cal.get("after")
    ts = cal.get("temperature_scaling")
    L.append("## 3. Calibration\n")
    L.append("| Stage | ECE ↓ | MCE ↓ | Brier ↓ | Temperature |")
    L.append("|---|---|---|---|---|")
    L.append(f"| before | {_fmt(before.get('ece'),4)} | {_fmt(before.get('mce'),4)} "
             f"| {_fmt(before.get('brier'),4)} | 1.0 |")
    if after:
        L.append(f"| after (T-scaled) | {_fmt(after.get('ece'),4)} | "
                 f"{_fmt(after.get('mce'),4)} | {_fmt(after.get('brier'),4)} | "
                 f"{_fmt((ts or {}).get('temperature'),4)} |")
    else:
        reason = ("temperature scaling not applied (single-class validation or "
                  "disabled)")
        L.append(f"\n*After-stage: {reason}.*")
    L.append("\nReliability bins (confidence vs observed frequency) are in "
             "`calibration_results.csv`. Poor calibration is reported, not "
             "hidden — on this small, imbalanced split ECE/Brier should be read "
             "with the sample count in mind.\n")

    # --- OOD ---
    ood = results.get("ood", {})
    L.append("## 4. Novelty / OOD (Mahalanobis in latent space)\n")
    if not ood.get("usable"):
        L.append(f"*OOD not evaluated — {ood.get('reason')}.*\n")
    else:
        det = ood.get("detector", {})
        m = ood.get("metrics", {})
        L.append("Known-training distribution = **benign train windows only**; "
                 "the held-out unseen behaviour = **attack windows**, which never "
                 "enter the manifold fit or the threshold. The threshold is the "
                 f"{det.get('fit_percentile')}th percentile of in-distribution "
                 "Mahalanobis distances (train/val), NOT tuned on test.\n")
        L.append(f"- detector: Mahalanobis, latent dim {det.get('dim')}, "
                 f"fit on {det.get('n_fit')} benign latents, "
                 f"threshold **{_fmt(det.get('threshold'),4)}**")
        L.append(f"- test: {ood.get('n_benign_test')} benign vs "
                 f"{ood.get('n_attack_test')} attack windows\n")
        L.append("| Metric | Value |")
        L.append("|---|---|")
        L.append(f"| OOD AUROC | {_fmt(m.get('auroc'),4)} |")
        L.append(f"| OOD AUPRC | {_fmt(m.get('auprc'),4)} |")
        L.append(f"| Detection rate (recall on unseen) | {_fmt(m.get('detection_rate'),4)} |")
        L.append(f"| False acceptance rate | {_fmt(m.get('false_acceptance_rate'),4)} |")
        L.append("\nPer-sample `novelty_score` and `is_ood` in `ood_results.csv`.\n")

    L.append("## Keeping the signals distinct\n")
    L.append("Risk, uncertainty, and novelty answer different questions and can "
             "disagree: a sample can be high-risk with low uncertainty (a "
             "confident attack call), or low-risk with high novelty (benign but "
             "unfamiliar behaviour). We expose the RAW values plus a documented "
             "derived status and deliberately avoid hardcoding any single "
             "interpretation as universal truth.\n")
    L.append("*All numbers come from the trained checkpoint on real cached data. "
             "Single-class splits and unstable metrics are reported as n/a, not "
             "fabricated.*\n")
    return "\n".join(L) + "\n"


def _render_signal_example(L: List[str], results: Dict) -> None:
    """Show one concrete sample's three raw signals side by side (illustrative)."""
    u_rows = results.get("uncertainty", {}).get("rows", [])
    ood_rows = {r["t_index"]: r for r in results.get("ood", {}).get("rows", [])}
    if not u_rows:
        return
    # pick the highest-risk test sample as the example
    ex = max(u_rows, key=lambda r: r.get("risk_mc_mean", 0.0))
    ood = ood_rows.get(ex["t_index"], {})
    L.append("### Signals stay separate (example)\n")
    L.append(f"For test window t={ex['t_index']} (label={ex['label']}):\n")
    L.append(f"- Risk = **{_fmt(ex.get('risk_mc_mean'),4)}**  "
             f"(deterministic {_fmt(ex.get('risk_deterministic'),4)})")
    L.append(f"- Uncertainty (std) = **{_fmt(ex.get('uncertainty_std'),4)}**")
    L.append(f"- Novelty = **{_fmt(ood.get('novelty_score'),4)}** "
             f"(is_ood={ood.get('is_ood')})\n")
    L.append("These are three independent numbers; the interpretation below is "
             "documented, not a universal rule.\n")
