"""Phase-3 metrics tests: correctness of the binary panel + edge cases."""

import numpy as np

from sentinelx.experiments.metrics import compute_binary_metrics


def test_perfect_predictions():
    y = [0, 0, 1, 1]
    s = [0.1, 0.2, 0.9, 0.8]
    m = compute_binary_metrics(y, s, threshold=0.5)
    assert m.precision == 1.0 and m.recall == 1.0 and m.f1 == 1.0
    assert m.tp == 2 and m.fp == 0 and m.tn == 2 and m.fn == 0
    assert m.false_positive_rate == 0.0
    assert m.roc_auc == 1.0
    assert m.pr_auc == 1.0


def test_confusion_matrix_layout():
    y = [1, 0, 1, 0]
    s = [0.9, 0.8, 0.2, 0.1]   # one TP, one FP, one FN, one TN
    m = compute_binary_metrics(y, s, threshold=0.5)
    assert m.tp == 1 and m.fp == 1 and m.fn == 1 and m.tn == 1
    assert m.confusion_matrix == [[1, 1], [1, 1]]  # [[tn,fp],[fn,tp]]


def test_single_class_gives_nan_aucs():
    y = [0, 0, 0, 0]
    s = [0.1, 0.4, 0.6, 0.9]
    m = compute_binary_metrics(y, s, threshold=0.5)
    assert m.roc_auc is None
    assert m.pr_auc != m.pr_auc  # NaN


def test_false_positive_rate():
    y = [0, 0, 0, 1]
    s = [0.9, 0.9, 0.1, 0.9]    # 2 FP out of 3 negatives
    m = compute_binary_metrics(y, s, threshold=0.5)
    assert m.false_positive_rate == 2 / 3
