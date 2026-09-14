"""Multi-objective loss for the Sentinel-X world model.

The world model's PRIMARY objective is future-state modelling, so the loss is
dominated by a self-supervised future-state term, with an auxiliary supervised
risk term where labels exist. Each component is justified below — this is NOT an
arbitrary weighted sum.

Components
----------
1. future-state representation prediction  (PRIMARY, self-supervised)
   L_state = 1 - cos_sim(z_hat_{t+1}, sg(z_{t+1}))

   Rationale: the research goal is to learn P(S_{t+1} | S_t) — how the network
   state evolves. We predict the next latent state z_hat_{t+1} from z_t and
   compare it to the encoder's own encoding of the actual future window sequence
   (a target computed with stop-gradient, ``sg``, so the encoder is not trained
   to collapse targets toward predictions — a standard predictive-coding /
   BYOL-style safeguard against representation collapse). Cosine distance is
   scale-invariant, which suits a Tanh/LayerNorm latent whose direction encodes
   state more meaningfully than magnitude.

2. future malicious-risk prediction        (AUXILIARY, supervised)
   L_risk = weighted BCE(risk_logit, attack@{t+H})

   Rationale: where attack labels exist for window t+H, a light supervised
   signal shapes the latent so it is also useful for risk forecasting and makes
   the world model directly comparable to the Phase-3 baselines (same target).
   It is auxiliary: the world model is never reduced to z_t -> attack yes/no.
   Class imbalance is handled with a positive-class weight (pos_weight) computed
   on TRAIN, matching the baselines' imbalance policy.

Total
-----
   L = lambda_state * L_state + lambda_risk * L_risk

lambda_state / lambda_risk are configurable (default 1.0 / 1.0). Setting
lambda_risk = 0 recovers a pure self-supervised world model; setting
lambda_state = 0 is explicitly discouraged because it degrades the model into a
classifier, defeating the phase objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class LossParts:
    total: torch.Tensor
    state: torch.Tensor
    risk: torch.Tensor


def future_state_loss(z_next_pred: torch.Tensor, z_next_target: torch.Tensor) -> torch.Tensor:
    """1 - mean cosine similarity between predicted and (stop-grad) target state.

    z_next_target must already be detached by the caller (see model.encode_target_state).
    """
    pred = F.normalize(z_next_pred, dim=-1, eps=1e-8)
    tgt = F.normalize(z_next_target, dim=-1, eps=1e-8)
    cos = (pred * tgt).sum(dim=-1)          # (batch,)
    return (1.0 - cos).mean()


def risk_loss(risk_logit: torch.Tensor, y_risk: torch.Tensor,
              pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Weighted BCE for the auxiliary malicious-risk head."""
    return F.binary_cross_entropy_with_logits(risk_logit, y_risk, pos_weight=pos_weight)


def world_model_loss(
    z_next_pred: torch.Tensor,
    z_next_target: torch.Tensor,
    risk_logit: torch.Tensor,
    y_risk: torch.Tensor,
    *,
    lambda_state: float = 1.0,
    lambda_risk: float = 1.0,
    pos_weight: Optional[torch.Tensor] = None,
) -> LossParts:
    l_state = future_state_loss(z_next_pred, z_next_target)
    l_risk = risk_loss(risk_logit, y_risk, pos_weight=pos_weight)
    total = lambda_state * l_state + lambda_risk * l_risk
    return LossParts(total=total, state=l_state, risk=l_risk)
