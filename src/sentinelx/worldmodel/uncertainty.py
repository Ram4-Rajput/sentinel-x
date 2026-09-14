"""Phase-6: predictive uncertainty via genuine MC Dropout.

This module estimates *epistemic* uncertainty of the auxiliary risk head using
Monte-Carlo Dropout (Gal & Ghahramani, 2016). The idea is simple and honest:

  * the model already contains ``nn.Dropout`` layers (graph encoder, forecast
    head, risk head);
  * at inference we KEEP those dropout layers active (train mode) while all
    other stochastic-free layers behave normally;
  * we run ``n_passes`` stochastic forward passes, each sampling a different
    dropout mask, and read the risk probability every time;
  * the predictive **mean** across passes is the uncertainty-aware risk estimate;
  * the predictive **variance** (and std) across passes is the *uncertainty*.

Crucial design decisions (per the Phase-6 requirements):

  * Uncertainty is derived ONLY from dropout stochasticity in the neural model.
    We do NOT perturb the inputs to fake variance — the graph inputs are fixed
    across all passes; only the dropout masks change.
  * Uncertainty is exposed SEPARATELY from risk. ``UncertaintyEstimate`` carries
    ``risk_mean`` and ``uncertainty`` (variance/std) as distinct fields; they are
    never collapsed into a single "confidence" number.
  * The number of passes defaults to 30 and is fully configurable.

Only dropout modules are re-enabled; BatchNorm/LayerNorm running stats are left
in eval mode (the model uses LayerNorm, which is deterministic anyway, but we
guard explicitly so future BatchNorm additions stay correct).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .model import GraphTensors, SentinelXWorldModel

# Default number of stochastic forward passes (configurable everywhere).
DEFAULT_MC_PASSES = 30


@contextmanager
def dropout_active(model: nn.Module):
    """Context manager: enable ONLY dropout layers, restore state afterwards.

    Puts the model in eval mode (so LayerNorm / any BatchNorm use fixed stats),
    then flips every ``nn.Dropout*`` module back to train mode so its mask keeps
    sampling. On exit the model's original training flag is restored.

    This is the mechanism that makes MC Dropout *genuine*: the exact same fixed
    input produces different outputs solely because dropout masks differ.
    """
    was_training = model.training
    # Base: everything deterministic (eval), then selectively re-enable dropout.
    model.eval()
    dropout_layers = [m for m in model.modules()
                      if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d,
                                        nn.Dropout3d, nn.AlphaDropout))]
    for m in dropout_layers:
        m.train()
    try:
        yield dropout_layers
    finally:
        model.train(was_training)


def count_dropout_layers(model: nn.Module) -> int:
    """How many dropout modules the model exposes (used to assert MC is real)."""
    return sum(1 for m in model.modules()
               if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d,
                                 nn.Dropout3d, nn.AlphaDropout)))


def has_active_dropout(model: nn.Module) -> bool:
    """True iff the model contains at least one dropout module with p > 0.

    A model with no dropout (or all p == 0) cannot produce MC-Dropout variance;
    we surface that explicitly rather than returning silent zeros.
    """
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d,
                          nn.Dropout3d, nn.AlphaDropout)):
            if float(getattr(m, "p", 0.0)) > 0.0:
                return True
    return False


@dataclass
class UncertaintyEstimate:
    """Per-sample MC-Dropout result — risk and uncertainty kept DISTINCT.

    * risk_mean : predictive mean risk probability across passes (uncertainty-
                  aware point estimate of risk).
    * variance  : predictive variance of the risk probability across passes
                  (the epistemic uncertainty signal).
    * std       : sqrt(variance), same units as probability.
    * n_passes  : number of stochastic passes used.
    * samples   : the raw per-pass probabilities (kept so callers can compute
                  richer statistics; never discarded / hidden).
    """

    risk_mean: float
    variance: float
    std: float
    n_passes: int
    samples: List[float]

    def as_dict(self) -> Dict:
        return {
            "risk_mean": self.risk_mean,
            "uncertainty_variance": self.variance,
            "uncertainty_std": self.std,
            "n_passes": self.n_passes,
        }


@torch.no_grad()
def mc_dropout_risk(
    model: SentinelXWorldModel,
    batch_seq: List[List[GraphTensors]],
    *,
    n_passes: int = DEFAULT_MC_PASSES,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Run ``n_passes`` MC-Dropout forward passes over a batch.

    Returns a dict of numpy arrays (all shape (B,) unless noted):

        * ``risk_mean``  : predictive mean risk probability
        * ``variance``   : predictive variance of risk probability
        * ``std``        : predictive std of risk probability
        * ``samples``    : (n_passes, B) raw per-pass probabilities

    The input ``batch_seq`` is IDENTICAL on every pass — only dropout masks
    differ. This keeps the uncertainty epistemic (model-driven), never an
    artefact of input perturbation.
    """
    if not isinstance(n_passes, int) or isinstance(n_passes, bool) or n_passes < 1:
        raise ValueError(f"n_passes must be a positive integer, got {n_passes!r}")

    if seed is not None:
        torch.manual_seed(seed)

    per_pass: List[np.ndarray] = []
    with dropout_active(model):
        for _ in range(n_passes):
            out = model(batch_seq)                        # dropout masks resampled
            probs = torch.sigmoid(out.risk_logit)         # (B,)
            per_pass.append(probs.detach().cpu().numpy())

    samples = np.stack(per_pass, axis=0)                  # (n_passes, B)
    # Population variance across passes (ddof=0): the MC estimate of Var[p].
    mean = samples.mean(axis=0)                           # (B,)
    var = samples.var(axis=0, ddof=0)                     # (B,)
    std = np.sqrt(var)
    return {"risk_mean": mean, "variance": var, "std": std, "samples": samples}


def estimate_uncertainty(
    model: SentinelXWorldModel,
    samples: Sequence,
    *,
    collate_inputs,
    n_passes: int = DEFAULT_MC_PASSES,
    batch_size: int = 32,
    seed: Optional[int] = 42,
) -> List[UncertaintyEstimate]:
    """MC-Dropout uncertainty for a sequence of dataset samples.

    ``collate_inputs`` extracts the observed input sequence batch from the
    dataset samples (so this works for both the Phase-4 and K-step data layers).
    Returns one :class:`UncertaintyEstimate` per input sample, in order.
    """
    out: List[UncertaintyEstimate] = []
    idx = 0
    for start in range(0, len(samples), batch_size):
        batch = list(samples[start:start + batch_size])
        batch_seq = collate_inputs(batch)
        # Deterministic-but-distinct seed per batch so runs are reproducible.
        bseed = None if seed is None else seed + idx
        res = mc_dropout_risk(model, batch_seq, n_passes=n_passes, seed=bseed)
        for j in range(len(batch)):
            out.append(UncertaintyEstimate(
                risk_mean=float(res["risk_mean"][j]),
                variance=float(res["variance"][j]),
                std=float(res["std"][j]),
                n_passes=n_passes,
                samples=[float(v) for v in res["samples"][:, j]],
            ))
        idx += 1
    return out


@torch.no_grad()
def deterministic_risk(
    model: SentinelXWorldModel,
    batch_seq: List[List[GraphTensors]],
) -> np.ndarray:
    """Single deterministic (dropout-OFF) risk pass — the plain Phase-4 risk.

    Provided so callers can compare the deterministic risk against the MC mean
    and keep the two clearly labelled.
    """
    model.eval()
    out = model(batch_seq)
    return torch.sigmoid(out.risk_logit).detach().cpu().numpy()
