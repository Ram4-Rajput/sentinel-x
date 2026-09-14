"""Phase-6: probability calibration analysis.

Calibration asks a different question than accuracy or ranking (PR-AUC): *when
the model says 0.8, does the event actually happen ~80% of the time?* A model
can rank well yet be badly calibrated. Per the Phase-6 requirement we measure
this honestly and never hide poor calibration.

Implemented:

  * **Expected Calibration Error (ECE)** — bin predictions by confidence, take
    the weighted average gap between mean predicted probability and observed
    frequency in each bin. Maximum Calibration Error (MCE) is reported too.
  * **Brier score** — mean squared error between predicted probability and the
    0/1 outcome (a proper scoring rule). Reported where both classes exist.
  * **Reliability analysis** — the per-bin table (confidence vs. accuracy vs.
    count) that a reliability diagram would plot, exposed as data so it can be
    written to CSV / rendered without a plotting dependency.
  * **Temperature scaling** (optional, SEPARATE stage) — fit a single scalar
    temperature T on a held-out (validation) set by minimising NLL of the
    logits, then apply ``sigmoid(logit / T)``. It is a post-hoc recalibration
    step kept distinct from the base model; we report calibration BEFORE and
    AFTER so any improvement (or lack of it) is visible.

All statistics degrade gracefully on single-class splits (return ``None`` rather
than a misleading number).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Reliability / ECE
# --------------------------------------------------------------------------- #
@dataclass
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float     # mean predicted probability in the bin
    observed_frequency: float  # fraction of positives actually observed
    gap: float                 # |mean_confidence - observed_frequency|

    def as_dict(self) -> Dict:
        return {
            "lower": self.lower, "upper": self.upper, "count": self.count,
            "mean_confidence": self.mean_confidence,
            "observed_frequency": self.observed_frequency, "gap": self.gap,
        }


@dataclass
class CalibrationReport:
    n: int
    positives: int
    negatives: int
    n_bins: int
    ece: Optional[float]
    mce: Optional[float]
    brier: Optional[float]
    bins: List[ReliabilityBin] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return {
            "n": self.n, "positives": self.positives, "negatives": self.negatives,
            "n_bins": self.n_bins, "ece": self.ece, "mce": self.mce,
            "brier": self.brier,
            "bins": [b.as_dict() for b in self.bins],
        }


def reliability_bins(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    n_bins: int = 10,
) -> List[ReliabilityBin]:
    """Bin predictions into ``n_bins`` equal-width confidence bins.

    Bin edges span [0, 1]; a probability of exactly 1.0 falls in the last bin.
    Empty bins are omitted from the returned list (count == 0 contributes 0 to
    ECE anyway).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: List[ReliabilityBin] = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        conf = float(y_prob[mask].mean())
        freq = float(y_true[mask].mean())
        out.append(ReliabilityBin(
            lower=float(lo), upper=float(hi), count=count,
            mean_confidence=conf, observed_frequency=freq,
            gap=abs(conf - freq),
        ))
    return out


def expected_calibration_error(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    n_bins: int = 10,
) -> Tuple[float, float, List[ReliabilityBin]]:
    """Return (ECE, MCE, bins).

    ECE = sum_b (n_b / N) * |conf_b - freq_b|
    MCE = max_b |conf_b - freq_b|
    """
    bins = reliability_bins(y_true, y_prob, n_bins=n_bins)
    n = len(y_true)
    if n == 0 or not bins:
        return 0.0, 0.0, bins
    ece = sum((b.count / n) * b.gap for b in bins)
    mce = max(b.gap for b in bins)
    return float(ece), float(mce), bins


def brier_score(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    """Mean squared error between predicted probability and 0/1 outcome."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean((y_prob - y_true) ** 2))


def evaluate_calibration(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    *,
    n_bins: int = 10,
) -> CalibrationReport:
    """Full calibration panel: ECE, MCE, Brier, reliability bins.

    Brier is always computable; ECE/MCE need at least one populated bin. On a
    single-class set the reliability table and Brier are still meaningful, so we
    report them; ECE/MCE remain valid (they compare predicted vs observed
    frequency regardless of class balance).
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    n = int(y_true_arr.shape[0])
    positives = int(y_true_arr.sum())
    negatives = int(n - positives)
    ece, mce, bins = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
    brier = brier_score(y_true, y_prob)
    return CalibrationReport(
        n=n, positives=positives, negatives=negatives, n_bins=n_bins,
        ece=(None if n == 0 else float(ece)),
        mce=(None if n == 0 else float(mce)),
        brier=(None if n == 0 or brier != brier else float(brier)),
        bins=bins,
    )


# --------------------------------------------------------------------------- #
# Temperature scaling (optional, separate post-hoc stage)
# --------------------------------------------------------------------------- #
@dataclass
class TemperatureScaler:
    """A single-parameter post-hoc calibrator: ``sigmoid(logit / T)``.

    Kept SEPARATE from the model — it never changes the network weights, only
    rescales its logits. ``temperature`` is fit on validation logits/labels by
    minimising binary cross-entropy (equivalently NLL).
    """

    temperature: float = 1.0
    fitted: bool = False

    def fit(
        self,
        logits: Sequence[float],
        y_true: Sequence[int],
        *,
        lr: float = 0.01,
        max_iter: int = 300,
        seed: int = 42,
    ) -> "TemperatureScaler":
        """Fit T by gradient descent on validation NLL (torch LBFGS-free).

        Uses a simple, dependency-light gradient loop on ``log T`` (so T stays
        positive). Requires both classes present; otherwise leaves T = 1.0 and
        marks itself unfitted (documented, not silently 'calibrated').
        """
        import torch

        y = np.asarray(y_true, dtype=float)
        if y.size == 0 or y.sum() == 0 or y.sum() == len(y):
            self.temperature = 1.0
            self.fitted = False
            return self

        torch.manual_seed(seed)
        z = torch.tensor(np.asarray(logits, dtype=float), dtype=torch.float64)
        t = torch.tensor(np.asarray(y_true, dtype=float), dtype=torch.float64)
        log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)  # T = exp(0)=1
        opt = torch.optim.Adam([log_t], lr=lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        for _ in range(max_iter):
            opt.zero_grad()
            temp = torch.exp(log_t)
            loss = loss_fn(z / temp, t)
            loss.backward()
            opt.step()
        self.temperature = float(torch.exp(log_t).item())
        self.fitted = True
        return self

    def transform(self, logits: Sequence[float]) -> np.ndarray:
        """Apply temperature scaling: return calibrated probabilities."""
        z = np.asarray(logits, dtype=float)
        t = self.temperature if self.temperature > 1e-6 else 1.0
        return 1.0 / (1.0 + np.exp(-z / t))

    def as_dict(self) -> Dict:
        return {"temperature": self.temperature, "fitted": self.fitted}


def logit(p: Sequence[float], eps: float = 1e-6) -> np.ndarray:
    """Inverse-sigmoid, used when only probabilities (not logits) are available."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))
