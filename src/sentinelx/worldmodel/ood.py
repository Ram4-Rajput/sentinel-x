"""Phase-6: novelty / out-of-distribution detection in latent space.

Sentinel-X already learns a latent network state

    G_t  --encoder-->  z_t

The world model's job is to model the *training distribution* of network
behaviour. Anything whose latent lies far from that distribution is novel /
out-of-distribution (OOD) — a candidate previously-unseen behaviour.

Method (defensible, per the Phase-6 preference): **Mahalanobis distance in the
learned embedding space.** We fit a Gaussian to the training-set latents
(mean ``mu`` and covariance ``Sigma``) and score a new latent by

    d_M(z) = sqrt( (z - mu)^T Sigma^{-1} (z - mu) )

Large Mahalanobis distance = far from the known training distribution = novel.

Why Mahalanobis (not raw Euclidean): it accounts for the covariance structure of
the embedding — directions the training data varies a lot in are down-weighted,
rare directions up-weighted. This is a standard, well-motivated OOD detector on
learned features (Lee et al., 2018).

Threshold discipline (critical): the OOD threshold is derived from
training/validation latents ONLY — a high percentile of the *in-distribution*
Mahalanobis distances (default 95th). It is NEVER tuned on the test/unseen set,
which would leak. ``novelty_score`` (the raw distance) and ``is_ood`` (distance
> threshold) are both exposed.

Numerical care: covariance is regularised with a small ridge (``shrinkage``) so
the inverse is stable even when the number of training points is close to the
embedding dimension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .model import GraphTensors, SentinelXWorldModel


@dataclass
class OODScore:
    """Per-sample OOD result — score and decision kept explicit."""

    novelty_score: float   # Mahalanobis distance in latent space
    is_ood: bool           # novelty_score > threshold (threshold from train/val)

    def as_dict(self) -> Dict:
        return {"novelty_score": self.novelty_score, "is_ood": self.is_ood}


@dataclass
class MahalanobisOOD:
    """Mahalanobis OOD detector fitted on training latents.

    Fields after :meth:`fit`:
      * ``mu``            : (D,) mean of training latents
      * ``precision``     : (D, D) inverse (regularised) covariance
      * ``threshold``     : distance above which a sample is flagged OOD
      * ``train_percentile`` / ``fit_percentile`` : provenance of the threshold
    """

    shrinkage: float = 1e-3
    mu: Optional[np.ndarray] = None
    precision: Optional[np.ndarray] = None
    threshold: Optional[float] = None
    fit_percentile: float = 95.0
    dim: int = 0
    n_fit: int = 0

    # ---------------------------------------------------------------- fit --- #
    def fit(self, train_latents: np.ndarray) -> "MahalanobisOOD":
        """Estimate mean + regularised precision from IN-DISTRIBUTION latents."""
        Z = np.asarray(train_latents, dtype=np.float64)
        if Z.ndim != 2 or Z.shape[0] < 2:
            raise ValueError(
                f"Need >=2 training latents shaped (N, D); got {Z.shape}")
        self.dim = int(Z.shape[1])
        self.n_fit = int(Z.shape[0])
        self.mu = Z.mean(axis=0)
        cov = np.cov(Z, rowvar=False)
        if cov.ndim == 0:  # D == 1 edge case
            cov = cov.reshape(1, 1)
        # Ridge regularisation for a stable, invertible covariance.
        ridge = self.shrinkage * np.trace(cov) / max(self.dim, 1)
        cov_reg = cov + np.eye(self.dim) * (ridge if ridge > 0 else self.shrinkage)
        self.precision = np.linalg.inv(cov_reg)
        return self

    # --------------------------------------------------------------- score -- #
    def distance(self, latents: np.ndarray) -> np.ndarray:
        """Mahalanobis distance of each latent to the training distribution."""
        if self.mu is None or self.precision is None:
            raise RuntimeError("MahalanobisOOD.fit must be called before scoring.")
        Z = np.asarray(latents, dtype=np.float64)
        if Z.ndim == 1:
            Z = Z[None, :]
        diff = Z - self.mu[None, :]                       # (N, D)
        # sqrt( diff @ precision @ diff^T ) per row, vectorised.
        m = np.einsum("nd,de,ne->n", diff, self.precision, diff)
        m = np.clip(m, 0.0, None)                         # guard tiny negatives
        return np.sqrt(m)

    # ----------------------------------------------------------- threshold -- #
    def set_threshold_from_indist(
        self,
        indist_latents: np.ndarray,
        percentile: float = 95.0,
    ) -> float:
        """Derive the OOD threshold from IN-DISTRIBUTION (train/val) latents.

        The threshold is a high percentile of in-distribution Mahalanobis
        distances — so by construction only ~(100 - percentile)% of known
        behaviour is (falsely) flagged. Never computed on test/unseen data.
        """
        d = self.distance(indist_latents)
        self.threshold = float(np.percentile(d, percentile))
        self.fit_percentile = float(percentile)
        return self.threshold

    def score(self, latents: np.ndarray) -> List[OODScore]:
        """Return per-sample :class:`OODScore` (novelty_score + is_ood)."""
        if self.threshold is None:
            raise RuntimeError(
                "Threshold not set; call set_threshold_from_indist first.")
        d = self.distance(latents)
        return [OODScore(novelty_score=float(x), is_ood=bool(x > self.threshold))
                for x in d]

    def as_dict(self) -> Dict:
        return {
            "method": "mahalanobis",
            "dim": self.dim, "n_fit": self.n_fit,
            "shrinkage": self.shrinkage,
            "threshold": self.threshold,
            "fit_percentile": self.fit_percentile,
        }


# --------------------------------------------------------------------------- #
# Latent extraction helper
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_latents(
    model: SentinelXWorldModel,
    samples: Sequence,
    *,
    collate_inputs,
    batch_size: int = 32,
) -> np.ndarray:
    """Encode dataset samples -> (N, latent_dim) latent matrix (z_t per sample).

    Deterministic (dropout OFF) — the OOD manifold is defined on the model's
    stable embedding, not on stochastic passes.
    """
    model.eval()
    out: List[np.ndarray] = []
    for start in range(0, len(samples), batch_size):
        batch = list(samples[start:start + batch_size])
        z = model.encode_state(collate_inputs(batch))     # (B, latent_dim)
        out.append(z.detach().cpu().numpy())
    if not out:
        return np.empty((0, model.cfg.latent_dim), dtype=np.float64)
    return np.concatenate(out, axis=0)


# --------------------------------------------------------------------------- #
# OOD ranking metrics (known vs unseen) — AUROC / AUPRC / detection & FAR
# --------------------------------------------------------------------------- #
def ood_detection_metrics(
    indist_scores: Sequence[float],
    ood_scores: Sequence[float],
    threshold: float,
) -> Dict[str, Optional[float]]:
    """Score how well ``novelty_score`` separates known from unseen behaviour.

    Labels: OOD = 1 (positive), in-distribution = 0. AUROC/AUPRC are
    threshold-free ranking metrics; detection rate (recall on OOD) and false
    acceptance rate (FAR: fraction of in-distribution wrongly flagged OOD) use
    the train/val-derived threshold.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    ind = np.asarray(indist_scores, dtype=float)
    ood = np.asarray(ood_scores, dtype=float)
    n_ind, n_ood = ind.size, ood.size
    result: Dict[str, Optional[float]] = {
        "n_indist": int(n_ind), "n_ood": int(n_ood), "threshold": float(threshold),
    }
    if n_ind == 0 or n_ood == 0:
        result.update({"auroc": None, "auprc": None, "detection_rate": None,
                       "false_acceptance_rate": None})
        return result

    y = np.concatenate([np.zeros(n_ind), np.ones(n_ood)])
    s = np.concatenate([ind, ood])
    result["auroc"] = float(roc_auc_score(y, s))
    result["auprc"] = float(average_precision_score(y, s))
    # Threshold-based operating point (threshold from in-distribution only).
    detected = float((ood > threshold).mean())            # recall on OOD
    far = float((ind > threshold).mean())                 # in-dist flagged OOD
    result["detection_rate"] = detected
    result["false_acceptance_rate"] = far
    return result
