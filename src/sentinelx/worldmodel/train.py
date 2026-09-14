"""Training loop for the Sentinel-X world model.

Implements (per Phase-4 spec):
  * PyTorch training loop + validation loop
  * early stopping on validation objective
  * checkpointing (best model by validation)
  * reproducible seeds
  * gradient clipping (justified: recurrent + message-passing stacks can produce
    large gradients; clipping the global norm stabilises training)
  * configurable learning rate / batch size / epochs

Model selection metric
-----------------------
Because the PRIMARY objective is future-state prediction, early stopping tracks
the validation *total* loss (state + risk). We additionally record validation
risk PR-AUC each epoch (auxiliary head) for reporting and threshold tuning, but
selection is not driven solely by the classifier — this keeps future-state
modelling the primary objective.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..experiments.common import seed_everything
from ..experiments.metrics import BinaryMetrics, compute_binary_metrics
from .config import TrainingConfig
from .data import (
    WorldModelSample,
    collate_inputs,
    collate_risk,
    collate_targets,
    iter_batches,
)
from .losses import world_model_loss
from .model import SentinelXWorldModel


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    train_state: float
    train_risk: float
    val_loss: float
    val_state: float
    val_risk: float
    val_pr_auc: float


@dataclass
class TrainResult:
    model: SentinelXWorldModel
    history: List[EpochLog] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")       # value of the selection metric
    best_val_state_loss: float = float("inf")
    best_val_total_loss: float = float("inf")
    chosen_threshold: float = 0.5
    train_seconds: float = 0.0
    num_parameters: int = 0
    val_metrics: Optional[BinaryMetrics] = None
    test_metrics: Optional[BinaryMetrics] = None
    stopped_early: bool = False


def _select_device() -> torch.device:
    """GPU 0 when CUDA is available, otherwise CPU (single-GPU only for now)."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _pos_weight(train: Sequence[WorldModelSample]) -> torch.Tensor:
    y = np.array([s.y_risk for s in train], dtype=float)
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    return torch.tensor([n_neg / n_pos]) if n_pos > 0 else torch.tensor([1.0])


def _tune_threshold(y_val, scores_val, target_precision: float = 0.5) -> float:
    """Pick a decision threshold on the VALIDATION set (never on test).

    Prefer the lowest threshold reaching ``target_precision``; if none does
    (common on tiny/imbalanced splits), fall back to the F1-optimal threshold
    rather than a blind 0.5, so the reported operating point is meaningful and
    consistent with the model's actual score distribution.
    """
    y_val = np.asarray(y_val)
    if y_val.sum() == 0 or y_val.sum() == len(y_val):
        return 0.5
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_val, scores_val)
    if len(thr) == 0:
        return 0.5
    ok = np.where(prec[:-1] >= target_precision)[0]
    if len(ok):
        return float(thr[ok[0]])
    # F1-optimal fallback (prec/rec have one more entry than thr)
    p, r = prec[:-1], rec[:-1]
    f1 = np.where((p + r) > 0, 2 * p * r / (p + r + 1e-12), 0.0)
    return float(thr[int(np.argmax(f1))])


@torch.no_grad()
def _risk_scores(model: SentinelXWorldModel, samples: Sequence[WorldModelSample],
                 batch_size: int) -> np.ndarray:
    model.eval()
    scores: List[float] = []
    for batch in iter_batches(samples, batch_size):
        out = model(collate_inputs(batch))
        scores.extend(torch.sigmoid(out.risk_logit).cpu().numpy().tolist())
    return np.asarray(scores)


@torch.no_grad()
def _eval_loss(model: SentinelXWorldModel, samples: Sequence[WorldModelSample],
               batch_size: int, lambda_state: float, lambda_risk: float,
               pos_weight: torch.Tensor) -> Dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    tot = st = rk = 0.0
    n = 0
    for batch in iter_batches(samples, batch_size):
        out = model(collate_inputs(batch))
        z_target = model.encode_target_state(collate_targets(batch))
        y = collate_risk(batch).to(device)
        parts = world_model_loss(out.z_next_pred, z_target, out.risk_logit, y,
                                 lambda_state=lambda_state, lambda_risk=lambda_risk,
                                 pos_weight=pos_weight)
        bs = len(batch)
        tot += float(parts.total) * bs
        st += float(parts.state) * bs
        rk += float(parts.risk) * bs
        n += bs
    n = max(n, 1)
    return {"total": tot / n, "state": st / n, "risk": rk / n}


def train_world_model(
    model: SentinelXWorldModel,
    train: Sequence[WorldModelSample],
    val: Sequence[WorldModelSample],
    test: Sequence[WorldModelSample],
    cfg: TrainingConfig,
    *,
    checkpoint_path: Optional[Path] = None,
    log=lambda *a, **k: None,
) -> TrainResult:
    seed_everything(cfg.seed)

    result = TrainResult(model=model, num_parameters=model.num_parameters())
    if not train:
        return result

    # ---- device selection (GPU 0 if available, else CPU) ----
    # We move only the MODEL and per-batch tensors to the device; the dataset
    # stays on CPU (graphs are moved per batch inside the model).
    device = _select_device()
    model.to(device)
    if device.type == "cuda":
        log(f"  device: cuda:0 ({torch.cuda.get_device_name(0)})")
    else:
        log("  device: cpu (CUDA not available)")

    pos_weight = _pos_weight(train).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                           weight_decay=cfg.weight_decay)

    best_state = None
    best_val = float("inf")
    epochs_no_improve = 0
    t0 = time.perf_counter()

    for epoch in range(cfg.epochs):
        model.train()
        tot = st = rk = 0.0
        n = 0
        for batch in iter_batches(train, cfg.batch_size, shuffle=True,
                                  seed=cfg.seed + epoch):
            opt.zero_grad()
            out = model(collate_inputs(batch))
            z_target = model.encode_target_state(collate_targets(batch))
            y = collate_risk(batch).to(device)
            parts = world_model_loss(
                out.z_next_pred, z_target, out.risk_logit, y,
                lambda_state=cfg.lambda_state, lambda_risk=cfg.lambda_risk,
                pos_weight=pos_weight,
            )
            parts.total.backward()
            if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            opt.step()
            bs = len(batch)
            tot += float(parts.total.detach()) * bs
            st += float(parts.state.detach()) * bs
            rk += float(parts.risk.detach()) * bs
            n += bs
        n = max(n, 1)

        # ---- validation ----
        if val:
            vl = _eval_loss(model, val, cfg.batch_size, cfg.lambda_state,
                            cfg.lambda_risk, pos_weight)
            yva = np.array([s.y_risk for s in val])
            if yva.sum() not in (0, len(yva)):
                from sklearn.metrics import average_precision_score
                val_pr_auc = float(average_precision_score(yva, _risk_scores(model, val, cfg.batch_size)))
            else:
                val_pr_auc = float("nan")
        else:
            vl = {"total": tot / n, "state": st / n, "risk": rk / n}
            val_pr_auc = float("nan")

        result.history.append(EpochLog(
            epoch=epoch, train_loss=tot / n, train_state=st / n, train_risk=rk / n,
            val_loss=vl["total"], val_state=vl["state"], val_risk=vl["risk"],
            val_pr_auc=val_pr_auc,
        ))
        log(f"  epoch {epoch:03d} | train {tot/n:.4f} (state {st/n:.4f} risk {rk/n:.4f}) "
            f"| val {vl['total']:.4f} (state {vl['state']:.4f} risk {vl['risk']:.4f}) "
            f"| val_pr_auc {val_pr_auc:.4f}")

        # ---- early stopping on the selection metric ----
        # Default "state" honours the PRIMARY future-state objective; the
        # auxiliary risk term on a tiny/imbalanced val split is noisy and should
        # not drive selection away from the world-model objective.
        sel_key = "state" if cfg.selection_metric == "state" else "total"
        sel_value = vl[sel_key]
        if sel_value < best_val - cfg.early_stopping_min_delta:
            best_val = sel_value
            result.best_epoch = epoch
            result.best_val_state_loss = vl["state"]
            result.best_val_total_loss = vl["total"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stopping_patience:
                result.stopped_early = True
                log(f"  early stopping at epoch {epoch} (no val improvement in "
                    f"{cfg.early_stopping_patience} epochs)")
                break

    result.train_seconds = round(time.perf_counter() - t0, 3)
    if best_state is not None:
        model.load_state_dict(best_state)
    result.best_val_loss = best_val

    # ---- threshold tuning on VAL, metrics on VAL + TEST (aux risk head) ----
    if val:
        yva = np.array([s.y_risk for s in val])
        sva = _risk_scores(model, val, cfg.batch_size)
        thr = _tune_threshold(yva, sva)
        result.chosen_threshold = thr
        result.val_metrics = compute_binary_metrics(yva, sva, threshold=thr)
    else:
        thr = 0.5
    if test:
        yte = np.array([s.y_risk for s in test])
        ste = _risk_scores(model, test, cfg.batch_size)
        result.test_metrics = compute_binary_metrics(yte, ste, threshold=result.chosen_threshold)

    if checkpoint_path is not None:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        model.save_checkpoint(checkpoint_path, extra={
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "chosen_threshold": result.chosen_threshold,
        })
    return result
