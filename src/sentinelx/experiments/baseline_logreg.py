"""Baseline 1 — Logistic Regression (leakage-safe).

Input : aggregated features of window t (WINDOW_FEATURE_NAMES).
Output: P(attack activity in window t+horizon).
No future information is used (input strictly <= t; target at t+horizon).

Leakage-safety: the StandardScaler is fit on TRAIN ONLY and reused on val/test
(sklearn-pipelines skill). Class imbalance handled via class_weight="balanced"
(imbalanced-data skill). Threshold tuned on VALIDATION PR curve, then applied to
test (never tuned on test).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from .common import WindowSample, stack_features, stack_labels
from .metrics import BinaryMetrics, compute_binary_metrics


@dataclass
class LogRegResult:
    val_metrics: BinaryMetrics
    test_metrics: BinaryMetrics
    chosen_threshold: float
    params: Dict


def _tune_threshold(y_val: np.ndarray, scores_val: np.ndarray,
                    target_precision: float = 0.5) -> float:
    """Pick the smallest threshold on the VAL PR curve achieving >= target
    precision; fall back to 0.5 if unattainable/degenerate."""
    if y_val.sum() == 0 or y_val.sum() == len(y_val):
        return 0.5
    from sklearn.metrics import precision_recall_curve
    prec, _rec, thr = precision_recall_curve(y_val, scores_val)
    ok = np.where(prec[:-1] >= target_precision)[0]
    if len(ok) and len(thr):
        return float(thr[ok[0]])
    return 0.5


def run_logreg(
    train: Sequence[WindowSample],
    val: Sequence[WindowSample],
    test: Sequence[WindowSample],
    *,
    seed: int = 42,
    max_iter: int = 1000,
    C: float = 1.0,
) -> LogRegResult:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Xtr, ytr = stack_features(train), stack_labels(train)
    Xva, yva = stack_features(val), stack_labels(val)
    Xte, yte = stack_features(test), stack_labels(test)

    # train-only scaling
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

    clf = LogisticRegression(
        class_weight="balanced",   # imbalance handling
        max_iter=max_iter, C=C, random_state=seed,
    )
    # Guard: LogisticRegression needs >=2 classes in train.
    if len(np.unique(ytr)) < 2:
        # degenerate: predict the prior; still produce metrics honestly
        prior = float(ytr.mean())
        scores_va = np.full(len(yva), prior)
        scores_te = np.full(len(yte), prior)
        thr = 0.5
    else:
        clf.fit(Xtr_s, ytr)
        scores_va = clf.predict_proba(Xva_s)[:, 1]
        scores_te = clf.predict_proba(Xte_s)[:, 1]
        thr = _tune_threshold(yva, scores_va)

    val_m = compute_binary_metrics(yva, scores_va, threshold=thr)
    test_m = compute_binary_metrics(yte, scores_te, threshold=thr)

    params = {"model": "logistic_regression", "C": C, "max_iter": max_iter,
              "class_weight": "balanced", "seed": seed,
              "n_features": int(Xtr.shape[1]) if Xtr.size else 0}
    return LogRegResult(val_metrics=val_m, test_metrics=test_m,
                        chosen_threshold=thr, params=params)
