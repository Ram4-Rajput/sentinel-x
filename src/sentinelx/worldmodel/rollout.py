"""Phase-5: K-step future-state forecasting via autoregressive rollout.

The Phase-4 world model learns a one-step transition of the *latent network
state*:

    z_t            = encode_state(G_{t-L+1..t})     (observed history -> latent)
    z_hat_{t+1}    = forecast_head(z_t)             (one-step latent transition)
    risk_{t+1}     = risk_head(z_hat_{t+1})         (auxiliary risk at that state)

Phase-5 forecasts K steps into the future WITHOUT training K separate
classifiers. Instead it *rolls the learned one-step transition forward*,
feeding each predicted latent back in as the next state:

    z_hat_{t+1} = f(z_t)
    z_hat_{t+2} = f(z_hat_{t+1})
    ...
    z_hat_{t+K} = f(z_hat_{t+K-1})

At every horizon k we also read the auxiliary risk head off the predicted
latent, so a single learned world model produces the whole trajectory
``S_t -> S_{t+1} -> ... -> S_{t+K}`` plus its risk profile. This is the
autoregressive-rollout requirement: one model, applied recursively, not K
independent predictors.

Design notes
------------
* ``forecast_head`` maps z -> z (same latent space), so it composes with
  itself. We apply it in ``eval`` mode under ``no_grad`` for deterministic,
  cheap inference.
* Determinism: given a fixed model in eval mode and identical inputs, the
  rollout is fully deterministic (no sampling, dropout off). Tested.
* Ground-truth targets for evaluating rollout error are the encoder's OWN
  encoding of the actual future window sequence ending at ``t+k`` (the same
  quantity Phase-4 used as its stop-grad training target). This keeps the
  error metric self-consistent with what the model was trained to predict and
  never leaks future inputs into the *prediction* path — the observed history
  only ever ends at ``t``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

from .model import GraphTensors, SentinelXWorldModel


# Supported / recommended horizons per the Phase-5 spec.
SUPPORTED_K = (1, 3, 5, 10)


@dataclass
class StepPrediction:
    """One step of the autoregressive rollout for a single sample."""

    horizon: int                 # k in {1..K}: how many steps ahead (t+k)
    t_index: int                 # anchor window index t (last observed window)
    target_index: int            # absolute window index being predicted (t+k)
    latent: List[float]          # predicted latent state z_hat_{t+k}
    risk_prob: float             # predicted risk (sigmoid of risk head) @ t+k
    risk_available: bool         # whether a ground-truth risk label exists here
    metadata: Dict               # provenance (combo, latent_dim, deterministic, ...)


@dataclass
class RolloutResult:
    """Full K-step rollout for a batch of samples.

    ``latents`` has shape (K, batch, latent_dim); ``risk_probs`` (K, batch).
    ``steps`` is a per-sample, per-horizon list of :class:`StepPrediction` for
    convenient CSV/JSON export.
    """

    K: int
    latents: torch.Tensor            # (K, B, latent_dim)  predicted future states
    risk_probs: torch.Tensor         # (K, B)              predicted future risk
    z0: torch.Tensor                 # (B, latent_dim)     encoded current state z_t
    steps: List[StepPrediction] = field(default_factory=list)


def validate_k(K: int) -> int:
    """Validate a requested horizon.

    K must be a positive integer. We do not silently coerce floats or
    non-positive values — an invalid K is a programming/config error and is
    surfaced immediately (tested).
    """
    if isinstance(K, bool) or not isinstance(K, int):
        raise ValueError(f"K must be a positive integer, got {K!r}")
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    return K


@torch.no_grad()
def rollout_latents(
    model: SentinelXWorldModel,
    batch_seq: List[List[GraphTensors]],
    K: int,
) -> RolloutResult:
    """Autoregressively roll the learned one-step transition K steps forward.

    Parameters
    ----------
    model : the trained Sentinel-X world model (used in eval mode).
    batch_seq : a batch of observed graph sequences ending at window t.
    K : number of future steps to predict (>= 1).

    Returns a :class:`RolloutResult` with predicted latents and risk per step.
    Deterministic given a fixed model and inputs.
    """
    validate_k(K)
    was_training = model.training
    model.eval()
    try:
        z = model.encode_state(batch_seq)          # (B, latent_dim) = z_t
        z0 = z
        latents: List[torch.Tensor] = []
        risks: List[torch.Tensor] = []
        for _ in range(K):
            z = model.forecast_head(z)             # z_hat_{t+k} = f(z_hat_{t+k-1})
            risk = torch.sigmoid(model.risk_head(z).squeeze(-1))  # (B,)
            latents.append(z)
            risks.append(risk)
        latents_t = torch.stack(latents, dim=0)    # (K, B, latent_dim)
        risks_t = torch.stack(risks, dim=0)        # (K, B)
    finally:
        if was_training:
            model.train()
    return RolloutResult(K=K, latents=latents_t, risk_probs=risks_t, z0=z0)


@torch.no_grad()
def rollout_steps(
    model: SentinelXWorldModel,
    input_seq: List[GraphTensors],
    K: int,
    *,
    t_index: int = -1,
    label_by_horizon: Optional[Dict[int, int]] = None,
) -> List[StepPrediction]:
    """Roll a SINGLE sample forward K steps and return per-step predictions.

    Each :class:`StepPrediction` carries the predicted latent, predicted risk,
    the horizon/target-window index, whether a ground-truth risk label exists,
    and prediction metadata — the per-future-step contract from the Phase-5
    spec. ``t_index`` anchors the absolute window; ``label_by_horizon`` (if
    provided) marks which horizons have real labels.
    """
    roll = rollout_latents(model, [input_seq], K)   # batch of 1
    combo_meta = {
        "gnn_type": model.cfg.gnn_type,
        "temporal_type": model.cfg.temporal_type,
        "latent_dim": model.cfg.latent_dim,
        "deterministic": True,
        "autoregressive": True,
    }
    steps: List[StepPrediction] = []
    for k in range(1, K + 1):
        latent = roll.latents[k - 1, 0].cpu().tolist()
        risk = float(roll.risk_probs[k - 1, 0])
        has_label = bool(label_by_horizon and k in label_by_horizon)
        steps.append(StepPrediction(
            horizon=k,
            t_index=t_index,
            target_index=(t_index + k) if t_index >= 0 else k,
            latent=latent,
            risk_prob=risk,
            risk_available=has_label,
            metadata=dict(combo_meta),
        ))
    return steps


@torch.no_grad()
def encode_future_states(
    model: SentinelXWorldModel,
    future_seqs_by_horizon: Dict[int, List[List[GraphTensors]]],
) -> Dict[int, torch.Tensor]:
    """Encode the ACTUAL future window sequences into ground-truth latents.

    ``future_seqs_by_horizon[k]`` is the batch of graph sequences ending at
    window ``t+k`` (one per sample). Returns ``{k: (B, latent_dim)}``. This is
    the self-consistent target for measuring rollout error: the model predicts
    a latent, and we compare against the encoder's own encoding of the real
    future — exactly the stop-grad target Phase-4 trained on.
    """
    was_training = model.training
    model.eval()
    out: Dict[int, torch.Tensor] = {}
    try:
        for k, seqs in future_seqs_by_horizon.items():
            out[k] = model.encode_state(seqs)
    finally:
        if was_training:
            model.train()
    return out
