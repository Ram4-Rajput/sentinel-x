"""Evaluation metrics for the binary forecasting target.

Per model-evaluation + imbalanced-data skills: for rare-positive security data
PR-AUC (average precision) and recall matter more than accuracy; we report the
full panel including confusion matrix and false-positive rate. Metrics are
computed identically for every baseline so comparisons are fair.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class BinaryMetrics:
    n: int
    positives: int
    negatives: int
    threshold: float
    precision: float
    recall: float
    f1: float
    pr_auc: float                 # average precision
    roc_auc: Optional[float]      # None if only one class present
    false_positive_rate: float
    tp: int
    fp: int
    tn: int
    fn: int

    def as_dict(self) -> Dict:
        return asdict(self)

    @property
    def confusion_matrix(self) -> List[List[int]]:
        # rows = actual [neg, pos], cols = predicted [neg, pos]
        return [[self.tn, self.fp], [self.fn, self.tp]]


def compute_binary_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    threshold: float = 0.5,
) -> BinaryMetrics:
    """Compute the full metric panel from labels + predicted probabilities.

    Uses scikit-learn for AUCs (with graceful handling of single-class splits).
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n = int(y_true.shape[0])
    positives = int(y_true.sum())
    negatives = int(n - positives)

    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    # AUCs need both classes present in y_true.
    if positives == 0 or negatives == 0:
        pr_auc = float("nan")
        roc_auc = None
    else:
        pr_auc = float(average_precision_score(y_true, y_score))
        try:
            roc_auc = float(roc_auc_score(y_true, y_score))
        except ValueError:
            roc_auc = None

    return BinaryMetrics(
        n=n, positives=positives, negatives=negatives, threshold=threshold,
        precision=float(precision), recall=float(recall), f1=float(f1),
        pr_auc=float(pr_auc), roc_auc=roc_auc, false_positive_rate=float(fpr),
        tp=tp, fp=fp, tn=tn, fn=fn,
    )
