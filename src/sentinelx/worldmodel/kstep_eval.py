"""Phase-5 evaluation: score the autoregressive K-step rollout by horizon.

For each requested K in the configured set (default 1, 3, 5, 10) we:

  1. Encode the observed history -> z_t (Phase-4 world model, frozen).
  2. Roll the learned one-step transition forward K times (rollout.py).
  3. At every horizon k in 1..K, measure
       * state prediction error   : cosine + L2 distance between the predicted
         latent z_hat_{t+k} and the encoder's own encoding of the ACTUAL future
         window sequence ending at t+k (self-consistent target).
       * future-risk performance   : PR-AUC / ROC-AUC / F1 of the risk head read
         off the predicted latent vs. attack@(t+k) (where labels vary).
  4. Report degradation of state error with horizon and the computational cost
     (wall-clock rollout time, per-step cost).

Nothing here is fabricated: metrics come from real model outputs on real cached
data; where a horizon's labels are single-class (no positives or no negatives)
the ranking metrics are reported as None rather than a misleading number.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..experiments.metrics import compute_binary_metrics
from .kstep_data import (
    KStepSample,
    collate_future_labels,
    collate_future_seqs,
    collate_inputs,
    iter_kstep_batches,
)
from .model import SentinelXWorldModel
from .rollout import encode_future_states, rollout_latents, validate_k


@dataclass
class HorizonMetrics:
    """State + risk metrics at a single horizon k."""

    horizon: int
    n: int
    # state prediction error (predicted latent vs. actual future encoding)
    state_cosine_distance: float        # mean (1 - cos) — matches training loss
    state_cosine_sim: float             # mean cosine similarity
    state_l2: float                     # mean L2 distance
    # future-risk performance (auxiliary head off the predicted latent)
    positives: int
    negatives: int
    risk_pr_auc: Optional[float]
    risk_roc_auc: Optional[float]
    risk_f1: Optional[float]
    risk_precision: Optional[float]
    risk_recall: Optional[float]

    def as_dict(self) -> Dict:
        return {
            "horizon": self.horizon, "n": self.n,
            "state_cosine_distance": self.state_cosine_distance,
            "state_cosine_sim": self.state_cosine_sim,
            "state_l2": self.state_l2,
            "positives": self.positives, "negatives": self.negatives,
            "risk_pr_auc": self.risk_pr_auc, "risk_roc_auc": self.risk_roc_auc,
            "risk_f1": self.risk_f1, "risk_precision": self.risk_precision,
            "risk_recall": self.risk_recall,
        }


@dataclass
class KStepEvaluation:
    """Full evaluation for one configured K."""

    K: int
    n_samples: int
    horizons: List[HorizonMetrics] = field(default_factory=list)
    # computational cost
    rollout_seconds: float = 0.0
    encode_seconds: float = 0.0
    per_step_seconds: float = 0.0
    # degradation summary (state cosine distance: horizon K vs horizon 1)
    degradation_abs: Optional[float] = None
    degradation_ratio: Optional[float] = None

    def as_dict(self) -> Dict:
        return {
            "K": self.K, "n_samples": self.n_samples,
            "rollout_seconds": self.rollout_seconds,
            "encode_seconds": self.encode_seconds,
            "per_step_seconds": self.per_step_seconds,
            "degradation_abs": self.degradation_abs,
            "degradation_ratio": self.degradation_ratio,
            "horizons": [h.as_dict() for h in self.horizons],
        }


def _safe_ranking_metrics(y: np.ndarray, scores: np.ndarray,
                          threshold: float) -> Dict[str, Optional[float]]:
    """Compute risk metrics, guarding single-class horizons (report None)."""
    y = np.asarray(y)
    if y.size == 0 or y.sum() == 0 or y.sum() == len(y):
        return {"pr_auc": None, "roc_auc": None, "f1": None,
                "precision": None, "recall": None}
    m = compute_binary_metrics(y, scores, threshold=threshold)
    d = m.as_dict()
    pr = d.get("pr_auc")
    if pr is not None and pr != pr:  # NaN guard
        pr = None
    return {"pr_auc": pr, "roc_auc": d.get("roc_auc"), "f1": d.get("f1"),
            "precision": d.get("precision"), "recall": d.get("recall")}


def evaluate_kstep(
    model: SentinelXWorldModel,
    samples: Sequence[KStepSample],
    K: int,
    *,
    batch_size: int = 32,
    risk_threshold: float = 0.5,
) -> KStepEvaluation:
    """Evaluate the autoregressive K-step rollout over ``samples``.

    Collects, per horizon, the predicted latents/risk and the ground-truth
    future encodings/labels across all batches, then computes state error and
    risk metrics. Returns a :class:`KStepEvaluation`.
    """
    validate_k(K)
    model.eval()

    # Per-horizon accumulators (concatenated across batches).
    pred_latents: Dict[int, List[torch.Tensor]] = {k: [] for k in range(1, K + 1)}
    true_latents: Dict[int, List[torch.Tensor]] = {k: [] for k in range(1, K + 1)}
    pred_risk: Dict[int, List[float]] = {k: [] for k in range(1, K + 1)}
    true_label: Dict[int, List[int]] = {k: [] for k in range(1, K + 1)}

    rollout_seconds = 0.0
    encode_seconds = 0.0
    n_samples = 0

    for batch in iter_kstep_batches(samples, batch_size):
        n_samples += len(batch)
        t0 = time.perf_counter()
        roll = rollout_latents(model, collate_inputs(batch), K)
        rollout_seconds += time.perf_counter() - t0

        # Ground-truth future encodings per horizon.
        fut_by_h = {k: collate_future_seqs(batch, k) for k in range(1, K + 1)}
        t1 = time.perf_counter()
        true = encode_future_states(model, fut_by_h)
        encode_seconds += time.perf_counter() - t1

        for k in range(1, K + 1):
            pred_latents[k].append(roll.latents[k - 1])       # (B, latent)
            true_latents[k].append(true[k])                   # (B, latent)
            pred_risk[k].extend(roll.risk_probs[k - 1].cpu().numpy().tolist())
            true_label[k].extend(collate_future_labels(batch, k))

    horizons: List[HorizonMetrics] = []
    for k in range(1, K + 1):
        if not pred_latents[k]:
            continue
        pl = torch.cat(pred_latents[k], dim=0)
        tl = torch.cat(true_latents[k], dim=0)
        cos = F.cosine_similarity(pl, tl, dim=-1)             # (N,)
        l2 = torch.linalg.norm(pl - tl, dim=-1)               # (N,)
        y = np.asarray(true_label[k], dtype=int)
        scores = np.asarray(pred_risk[k], dtype=float)
        rk = _safe_ranking_metrics(y, scores, risk_threshold)
        horizons.append(HorizonMetrics(
            horizon=k, n=int(pl.shape[0]),
            state_cosine_distance=float((1.0 - cos).mean()),
            state_cosine_sim=float(cos.mean()),
            state_l2=float(l2.mean()),
            positives=int(y.sum()), negatives=int(len(y) - y.sum()),
            risk_pr_auc=rk["pr_auc"], risk_roc_auc=rk["roc_auc"],
            risk_f1=rk["f1"], risk_precision=rk["precision"],
            risk_recall=rk["recall"],
        ))

    ev = KStepEvaluation(K=K, n_samples=n_samples, horizons=horizons,
                         rollout_seconds=round(rollout_seconds, 6),
                         encode_seconds=round(encode_seconds, 6))
    if n_samples > 0 and K > 0:
        ev.per_step_seconds = round(rollout_seconds / (n_samples * K), 8)
    # Degradation of the primary state error from horizon 1 -> K.
    if horizons:
        h1 = horizons[0].state_cosine_distance
        hK = horizons[-1].state_cosine_distance
        ev.degradation_abs = round(hK - h1, 6)
        ev.degradation_ratio = round((hK / h1), 6) if h1 > 1e-9 else None
    return ev
