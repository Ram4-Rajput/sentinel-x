"""Baseline 3 — Temporal GNN (GraphSAGE or GAT).

Input : graph sequence G_{t-n} ... G_t (cached window graphs).
Output: P(attack in window t+horizon).
gnn_type: "sage" (GraphSAGE) | "gat" (GAT).

This is a deliberately SIMPLE graph baseline — NOT the full Sentinel-X world
model. Per window graph: a 1-layer GNN encodes node features, mean-pools to a
graph embedding. The sequence of graph embeddings is mean-pooled over time and
passed to a linear head. (No forecasting head, no rollout, no uncertainty/OOD —
those belong to the Sentinel-X model in a later phase.)

Same samples/splits/leakage policy/imbalance handling as the other baselines.
Graphs with 0 nodes (e.g. CIC-IDS2018, which has no entities) yield a zero
embedding — documented; such datasets are flagged as not fairly GNN-capable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .common import NODE_FEATURE_DIM, WindowSample, stack_labels
from .metrics import BinaryMetrics, compute_binary_metrics


@dataclass
class GNNResult:
    val_metrics: BinaryMetrics
    test_metrics: BinaryMetrics
    chosen_threshold: float
    params: Dict
    train_seconds: float
    num_parameters: int
    graph_capable: bool
    note: str = ""


def _tune_threshold(y_val, scores_val, target_precision=0.5) -> float:
    y_val = np.asarray(y_val)
    if y_val.sum() == 0 or y_val.sum() == len(y_val):
        return 0.5
    from sklearn.metrics import precision_recall_curve
    prec, _rec, thr = precision_recall_curve(y_val, scores_val)
    ok = np.where(prec[:-1] >= target_precision)[0]
    return float(thr[ok[0]]) if (len(ok) and len(thr)) else 0.5


def _graph_to_tensors(g: dict):
    import torch
    n = int(g.get("num_nodes", 0))
    if n == 0:
        x = torch.zeros((1, NODE_FEATURE_DIM), dtype=torch.float32)  # dummy node
        ei = torch.empty((2, 0), dtype=torch.long)
        return x, ei, 0
    x = torch.tensor(g["node_features"], dtype=torch.float32)
    if g["edge_index"]:
        ei = torch.tensor(g["edge_index"], dtype=torch.long).t().contiguous()
    else:
        ei = torch.empty((2, 0), dtype=torch.long)
    return x, ei, n


def _graph_capable(train: Sequence[WindowSample]) -> bool:
    """A dataset is fairly GNN-capable only if graphs actually have nodes."""
    total_nodes = sum(
        int(g.get("num_nodes", 0)) for s in train for g in s.graph_seq
    )
    return total_nodes > 0


def run_gnn(
    train: Sequence[WindowSample],
    val: Sequence[WindowSample],
    test: Sequence[WindowSample],
    *,
    gnn_type: str = "sage",
    hidden_size: int = 32,
    epochs: int = 30,
    lr: float = 1e-3,
    seed: int = 42,
) -> GNNResult:
    import time
    import torch
    import torch.nn as nn
    from torch_geometric.nn import GATConv, SAGEConv, global_mean_pool
    from torch_geometric.data import Batch, Data

    gnn_type = gnn_type.lower()
    if gnn_type not in ("sage", "gat"):
        raise ValueError("gnn_type must be 'sage' or 'gat'")

    torch.manual_seed(seed)
    np.random.seed(seed)

    ytr = stack_labels(train)
    yva = stack_labels(val)
    yte = stack_labels(test)

    capable = _graph_capable(train)
    note = ("" if capable else
            "Dataset graphs have no nodes (no src/dst entities); GNN runs on "
            "empty graphs and cannot fairly exploit topology.")

    class TemporalGNN(nn.Module):
        def __init__(self):
            super().__init__()
            if gnn_type == "sage":
                self.conv = SAGEConv(NODE_FEATURE_DIM, hidden_size)
            else:
                self.conv = GATConv(NODE_FEATURE_DIM, hidden_size, heads=1)
            self.head = nn.Linear(hidden_size, 1)

        def encode_graph(self, x, edge_index, batch):
            h = torch.relu(self.conv(x, edge_index))
            return global_mean_pool(h, batch)      # (num_graphs, hidden)

        def forward(self, sample_graph_batches: List):
            # sample_graph_batches: list over time-steps of (Batch) -> we pass
            # per-sample; here we process one sample's sequence and temporal-mean.
            embs = []
            for (x, ei) in sample_graph_batches:
                batch = torch.zeros(x.size(0), dtype=torch.long)
                embs.append(self.encode_graph(x, ei, batch))  # (1, hidden)
            seq = torch.cat(embs, dim=0)          # (L, hidden)
            temporal = seq.mean(dim=0, keepdim=True)  # (1, hidden)
            return self.head(temporal).squeeze(-1)    # (1,)

    model = TemporalGNN()
    n_params = sum(p.numel() for p in model.parameters())

    # Precompute per-sample tensor sequences (memory-safe: small graphs).
    def sample_tensors(samples):
        out = []
        for s in samples:
            seq = [_graph_to_tensors(g)[:2] for g in s.graph_seq]
            out.append(seq)
        return out

    tr_t = sample_tensors(train)
    va_t = sample_tensors(val)
    te_t = sample_tensors(test)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)

    n_pos = float(ytr.sum())
    n_neg = float(len(ytr) - n_pos)
    pos_weight = torch.tensor([n_neg / n_pos]) if n_pos > 0 else torch.tensor([1.0])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def scores_for(tensors):
        model.eval()
        outs = []
        with torch.no_grad():
            for seq in tensors:
                outs.append(torch.sigmoid(model(seq)).item())
        return np.asarray(outs)

    best_ap = -1.0
    best_state = None
    t0 = time.perf_counter()
    can_train = len(train) > 0
    for _epoch in range(epochs if can_train else 0):
        model.train()
        opt.zero_grad()
        logits = []
        for seq in tr_t:
            logits.append(model(seq))
        logits = torch.cat(logits, dim=0)
        loss = loss_fn(logits, ytr_t)
        loss.backward()
        opt.step()
        if len(val) and yva.sum() not in (0, len(yva)):
            from sklearn.metrics import average_precision_score
            ap = average_precision_score(yva, scores_for(va_t))
            if ap > best_ap:
                best_ap = ap
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    train_seconds = time.perf_counter() - t0

    if best_state is not None:
        model.load_state_dict(best_state)

    sva = scores_for(va_t) if len(val) else np.zeros(0)
    ste = scores_for(te_t) if len(test) else np.zeros(0)
    thr = _tune_threshold(yva, sva) if len(val) else 0.5
    val_m = compute_binary_metrics(yva, sva, threshold=thr)
    test_m = compute_binary_metrics(yte, ste, threshold=thr)

    params = {"model": f"temporal_gnn_{gnn_type}", "gnn_type": gnn_type,
              "hidden_size": hidden_size, "epochs": epochs, "lr": lr, "seed": seed,
              "pos_weight": float(pos_weight.item())}
    return GNNResult(val_metrics=val_m, test_metrics=test_m, chosen_threshold=thr,
                     params=params, train_seconds=round(train_seconds, 3),
                     num_parameters=int(n_params), graph_capable=capable, note=note)
