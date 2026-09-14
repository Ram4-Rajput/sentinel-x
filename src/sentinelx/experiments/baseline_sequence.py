"""Baseline 2 — Temporal sequence model (LSTM or GRU).

Input : feature sequence X_{t-n} ... X_t  (aggregated window features).
Output: P(attack in window t+horizon).
temporal_type: "gru" | "lstm" (configurable).

Same chronological samples/splits and leakage policy as the LogReg baseline.
Train-only feature scaling. Class imbalance handled with a positive-class
weighted BCE loss (pos_weight = n_neg/n_pos on TRAIN). Best epoch selected on
VAL PR-AUC; threshold tuned on VAL, applied to TEST.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from .common import WindowSample, stack_labels, stack_sequences
from .metrics import BinaryMetrics, compute_binary_metrics


@dataclass
class SequenceResult:
    val_metrics: BinaryMetrics
    test_metrics: BinaryMetrics
    chosen_threshold: float
    params: Dict
    train_seconds: float
    num_parameters: int


def _tune_threshold(y_val, scores_val, target_precision=0.5) -> float:
    y_val = np.asarray(y_val)
    if y_val.sum() == 0 or y_val.sum() == len(y_val):
        return 0.5
    from sklearn.metrics import precision_recall_curve
    prec, _rec, thr = precision_recall_curve(y_val, scores_val)
    ok = np.where(prec[:-1] >= target_precision)[0]
    return float(thr[ok[0]]) if (len(ok) and len(thr)) else 0.5


def run_sequence(
    train: Sequence[WindowSample],
    val: Sequence[WindowSample],
    test: Sequence[WindowSample],
    *,
    temporal_type: str = "gru",
    hidden_size: int = 32,
    num_layers: int = 1,
    epochs: int = 30,
    lr: float = 1e-3,
    seed: int = 42,
) -> SequenceResult:
    import time
    import torch
    import torch.nn as nn

    temporal_type = temporal_type.lower()
    if temporal_type not in ("gru", "lstm"):
        raise ValueError("temporal_type must be 'gru' or 'lstm'")

    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr, ytr = stack_sequences(train), stack_labels(train)   # (N, L, F)
    Xva, yva = stack_sequences(val), stack_labels(val)
    Xte, yte = stack_sequences(test), stack_labels(test)

    # train-only standardization (per feature, over N*L)
    F = Xtr.shape[2]
    flat = Xtr.reshape(-1, F)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std == 0] = 1.0

    def norm(X):
        return (X - mean) / std

    Xtr_t = torch.tensor(norm(Xtr), dtype=torch.float32)
    Xva_t = torch.tensor(norm(Xva), dtype=torch.float32)
    Xte_t = torch.tensor(norm(Xte), dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)

    class SeqNet(nn.Module):
        def __init__(self):
            super().__init__()
            rnn_cls = nn.GRU if temporal_type == "gru" else nn.LSTM
            self.rnn = rnn_cls(F, hidden_size, num_layers=num_layers, batch_first=True)
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.rnn(x)
            last = out[:, -1, :]           # last timestep hidden state
            return self.head(last).squeeze(-1)

    model = SeqNet()
    n_params = sum(p.numel() for p in model.parameters())

    # class imbalance: pos_weight from TRAIN
    n_pos = float(ytr.sum())
    n_neg = float(len(ytr) - n_pos)
    pos_weight = torch.tensor([n_neg / n_pos]) if n_pos > 0 else torch.tensor([1.0])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def val_scores():
        model.eval()
        with torch.no_grad():
            return torch.sigmoid(model(Xva_t)).numpy() if len(val) else np.array([])

    best_ap = -1.0
    best_state = None
    t0 = time.perf_counter()
    can_train = len(train) > 0 and len(np.unique(ytr)) >= 1
    for _epoch in range(epochs if can_train else 0):
        model.train()
        opt.zero_grad()
        logits = model(Xtr_t)
        loss = loss_fn(logits, ytr_t)
        loss.backward()
        opt.step()
        # select best epoch on VAL PR-AUC (fallback to loss if val single-class)
        if len(val) and yva.sum() not in (0, len(yva)):
            from sklearn.metrics import average_precision_score
            ap = average_precision_score(yva, val_scores())
            if ap > best_ap:
                best_ap = ap
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    train_seconds = time.perf_counter() - t0

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        sva = torch.sigmoid(model(Xva_t)).numpy() if len(val) else np.zeros(0)
        ste = torch.sigmoid(model(Xte_t)).numpy() if len(test) else np.zeros(0)

    thr = _tune_threshold(yva, sva) if len(val) else 0.5
    val_m = compute_binary_metrics(yva, sva, threshold=thr)
    test_m = compute_binary_metrics(yte, ste, threshold=thr)

    params = {"model": f"sequence_{temporal_type}", "temporal_type": temporal_type,
              "hidden_size": hidden_size, "num_layers": num_layers,
              "epochs": epochs, "lr": lr, "seed": seed,
              "pos_weight": float(pos_weight.item()), "seq_len": int(Xtr.shape[1]) if Xtr.size else 0}
    return SequenceResult(val_metrics=val_m, test_metrics=test_m, chosen_threshold=thr,
                          params=params, train_seconds=round(train_seconds, 3),
                          num_parameters=int(n_params))
