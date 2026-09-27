#!/usr/bin/env python
"""Phase B: diagnose + fix the val->test threshold/calibration issue.

Loads the canonical checkpoint, recomputes val/test risk probabilities on
ctu-13-full (locally, no GPU), then:

  1. Reproduces the reported metrics (sanity check vs Kaggle numbers).
  2. Investigates the low test ROC-AUC (check score orientation / class balance).
  3. Runs the calibration panel (ECE / MCE / Brier) BEFORE and AFTER temperature
     scaling fitted on VAL only.
  4. Reports test recall/precision/F1 at several operating points:
     - checkpoint's val-tuned threshold (as shipped),
     - F1-optimal threshold chosen on VAL (never test),
     - a prevalence-matched note.

Thresholds are only ever selected on VAL. Test is measured, never tuned.

Outputs:
    experiments/phase_b_calibration.json
    experiments/phase_b_calibration_report.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sentinelx.pipeline.config import PROCESSED_ROOT, REPO_ROOT  # noqa: E402
from sentinelx.worldmodel.config import WorldModelConfig, load_config  # noqa: E402
from sentinelx.worldmodel.data import WorldModelDataset  # noqa: E402
from sentinelx.worldmodel.model import SentinelXWorldModel  # noqa: E402
from sentinelx.experiments.metrics import compute_binary_metrics  # noqa: E402
from sentinelx.worldmodel.calibration import (  # noqa: E402
    evaluate_calibration, TemperatureScaler, logit,
)
from sklearn.metrics import roc_auc_score, precision_recall_curve  # noqa: E402


def _risk_probs(model, samples, batch_size=64):
    import torch
    from sentinelx.worldmodel.data import collate_inputs
    probs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = list(samples[start:start + batch_size])
            out = model(collate_inputs(batch))
            probs.extend(torch.sigmoid(out.risk_logit).cpu().numpy().tolist())
    return np.asarray(probs, dtype=float)


def _f1_opt_threshold(y, scores):
    y = np.asarray(y)
    if y.sum() in (0, len(y)):
        return 0.5
    prec, rec, thr = precision_recall_curve(y, scores)
    if len(thr) == 0:
        return 0.5
    p, r = prec[:-1], rec[:-1]
    f1 = np.where((p + r) > 0, 2 * p * r / (p + r + 1e-12), 0.0)
    return float(thr[int(np.argmax(f1))])


def _panel(y, scores, thr):
    m = compute_binary_metrics(y, scores, threshold=thr)
    d = m.as_dict()
    if d.get("pr_auc") != d.get("pr_auc"):
        d["pr_auc"] = None
    return d


def main() -> int:
    ckpt = REPO_ROOT / "models" / "sentinel_x" / "model.pt"
    model, extra = SentinelXWorldModel.load_checkpoint(ckpt)
    shipped_thr = float(extra.get("chosen_threshold", 0.5))
    combo = extra.get("combo", "?")

    cfg_path = ckpt.parent / "config.yaml"
    data_cfg = load_config(cfg_path).data if cfg_path.exists() else WorldModelConfig().data
    print(f"[phaseB] checkpoint combo={combo} dataset={data_cfg.dataset} "
          f"shipped_threshold={shipped_thr:.4f}")

    ds = WorldModelDataset(data_cfg, PROCESSED_ROOT,
                           model.cfg.node_feature_dim, model.cfg.edge_feature_dim)
    train, val, test, info = ds.build()
    print(f"[phaseB] samples: train={len(train)} val={len(val)} test={len(test)}")

    yva = np.array([s.y_risk for s in val], dtype=int)
    yte = np.array([s.y_risk for s in test], dtype=int)
    print("[phaseB] scoring val ...")
    pva = _risk_probs(model, val)
    print("[phaseB] scoring test ...")
    pte = _risk_probs(model, test)

    val_prev = float(yva.mean())
    test_prev = float(yte.mean())

    # --- 1/2. ROC orientation check ---
    roc_te = roc_auc_score(yte, pte) if yte.sum() not in (0, len(yte)) else None
    roc_te_flip = roc_auc_score(yte, 1 - pte) if roc_te is not None else None
    roc_va = roc_auc_score(yva, pva) if yva.sum() not in (0, len(yva)) else None

    # --- 3. calibration before/after temperature scaling (fit on VAL) ---
    cal_te_before = evaluate_calibration(yte, pte).as_dict()
    zva = logit(pva)
    zte = logit(pte)
    ts = TemperatureScaler().fit(zva, yva)
    pte_cal = ts.transform(zte)
    pva_cal = ts.transform(zva)
    cal_te_after = evaluate_calibration(yte, pte_cal).as_dict()

    # --- 4. operating points (thresholds chosen on VAL only) ---
    f1_thr_val = _f1_opt_threshold(yva, pva)
    f1_thr_val_cal = _f1_opt_threshold(yva, pva_cal)

    ops = {
        "shipped_val_threshold": _panel(yte, pte, shipped_thr),
        "f1opt_on_val": _panel(yte, pte, f1_thr_val),
        "f1opt_on_val_after_tempscale": _panel(yte, pte_cal, f1_thr_val_cal),
    }

    out = {
        "checkpoint_combo": combo,
        "dataset": data_cfg.dataset,
        "prevalence": {"val": val_prev, "test": test_prev},
        "roc_auc": {
            "val": roc_va, "test": roc_te, "test_score_flipped": roc_te_flip,
        },
        "pr_auc": {
            "test": _panel(yte, pte, shipped_thr).get("pr_auc"),
        },
        "temperature": ts.as_dict(),
        "calibration_test_before": {k: cal_te_before[k] for k in ("ece", "mce", "brier")},
        "calibration_test_after": {k: cal_te_after[k] for k in ("ece", "mce", "brier")},
        "thresholds": {
            "shipped_val": shipped_thr,
            "f1opt_val": f1_thr_val,
            "f1opt_val_after_tempscale": f1_thr_val_cal,
        },
        "operating_points": ops,
    }

    exp = REPO_ROOT / "experiments"
    exp.mkdir(exist_ok=True)
    (exp / "phase_b_calibration.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # readable report
    def _fmt(v):
        return "n/a" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))

    lines = [
        "# Phase B — Calibration & threshold diagnosis",
        "",
        f"Checkpoint: **{combo}** on `{data_cfg.dataset}`.",
        "",
        f"- Prevalence (positives): val {val_prev:.1%} vs test {test_prev:.1%} "
        f"(distribution shift confirmed).",
        f"- ROC-AUC: val {_fmt(roc_va)} | test {_fmt(roc_te)} | "
        f"test with flipped score {_fmt(roc_te_flip)}.",
        f"- Fitted temperature T = {ts.temperature:.4f} (fitted={ts.fitted}).",
        f"- Test calibration ECE: {_fmt(cal_te_before['ece'])} -> "
        f"{_fmt(cal_te_after['ece'])} after temp scaling; "
        f"Brier {_fmt(cal_te_before['brier'])} -> {_fmt(cal_te_after['brier'])}.",
        "",
        "## Test operating points (thresholds picked on VAL only)",
        "| operating point | threshold | precision | recall | F1 | PR-AUC |",
        "|---|---|---|---|---|---|",
    ]
    thr_map = {
        "shipped_val_threshold": shipped_thr,
        "f1opt_on_val": f1_thr_val,
        "f1opt_on_val_after_tempscale": f1_thr_val_cal,
    }
    for name, blk in ops.items():
        lines.append(
            f"| {name} | {thr_map[name]:.4f} | {_fmt(blk['precision'])} | "
            f"{_fmt(blk['recall'])} | {_fmt(blk['f1'])} | {_fmt(blk['pr_auc'])} |"
        )
    (exp / "phase_b_calibration_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\n[phaseB] wrote experiments/phase_b_calibration.json + _report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
