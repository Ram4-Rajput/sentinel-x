"""Data layer for the world model — reuses the Phase-3 forecasting protocol.

We deliberately reuse ``sentinelx.experiments.common`` so the world model sees
the SAME windows, sequence length, horizon, chronological split, seed, and
leakage policy as the Phase-3 baselines. This keeps the comparison
apples-to-apples (the Phase-4 kickoff requirement).

Each Phase-3 ``WindowSample`` already carries ``graph_seq`` = the cached graph
dicts for windows [t-seq_len+1 .. t]. For the world model we additionally need
the *future* graph sequence ending at window t+H to supply a self-supervised
target for the future-state loss. We recover it by indexing the globally
ordered windows with the sample's ``t_index``.

Datasets whose graphs have no nodes (no src/dst entities, e.g. CIC-IDS2018) are
flagged not graph-capable — documented, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..experiments.common import (
    ExperimentConfig,
    LazyWindowStore,
    WindowSample,
    _load_ordered_windows,
    build_window_samples,
)
from .config import DataConfig
from .model import GraphTensors


def graph_dict_to_tensors(g: dict, node_dim: int, edge_dim: int) -> GraphTensors:
    """Convert a cached graph dict into model-ready tensors.

    Consumes node features, edge connectivity, and edge features directly — the
    graph is not flattened. Empty graphs (0 nodes) become an explicit empty
    GraphTensors that the model encodes as a zero state.
    """
    n = int(g.get("num_nodes", 0))
    if n == 0:
        return GraphTensors(
            x=torch.empty((0, node_dim), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, edge_dim), dtype=torch.float32),
            num_nodes=0,
        )
    x = torch.tensor(g["node_features"], dtype=torch.float32)
    if g.get("edge_index"):
        edge_index = torch.tensor(g["edge_index"], dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(g["edge_features"], dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, edge_dim), dtype=torch.float32)
    return GraphTensors(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n)


@dataclass
class WorldModelSample:
    """A world-model training sample.

    * input_seq  : graphs for windows [t-seq_len+1 .. t]        (observed)
    * target_seq : graphs for windows [t-seq_len+1+H .. t+H]    (future, for the
                   self-supervised future-state target — inputs never include it)
    * y_risk     : attack@(t+H) label for the auxiliary risk head
    * t_index    : window index t (chronological ordering / leakage checks)
    """

    input_seq: List[GraphTensors]
    target_seq: List[GraphTensors]
    y_risk: int
    t_index: int


class WorldModelDataset:
    """Builds train/val/test world-model samples from the Phase-2 cache."""

    def __init__(self, data_cfg: DataConfig, processed_root, node_dim: int, edge_dim: int):
        self.cfg = data_cfg
        self.processed_root = processed_root
        self.node_dim = node_dim
        self.edge_dim = edge_dim

    def build(self) -> Tuple[List[WorldModelSample], List[WorldModelSample],
                             List[WorldModelSample], Dict]:
        exp_cfg = ExperimentConfig(
            dataset=self.cfg.dataset,
            processed_root=self.processed_root,
            seq_len=self.cfg.seq_len,
            horizon=self.cfg.horizon,
            train_frac=self.cfg.train_frac,
            val_frac=self.cfg.val_frac,
            min_windows=self.cfg.min_windows,
        )
        train, val, test, info = build_window_samples(exp_cfg)
        if not info.get("usable"):
            return [], [], [], info

        # Lazy store is attached to info when dataset > lazy_threshold windows.
        # In lazy mode, each WindowSample.graph_seq is empty; we fetch from the
        # store at convert time so only one batch worth of windows is in RAM.
        store: Optional[LazyWindowStore] = info.pop("_lazy_store", None)

        # Globally ordered windows for target-sequence lookup (needed for both
        # eager and lazy). In lazy mode we use the store; in eager mode we
        # re-use the already-loaded list via _load_ordered_windows.
        H = self.cfg.horizon
        L = self.cfg.seq_len

        def convert(samples: Sequence[WindowSample]) -> List[WorldModelSample]:
            out: List[WorldModelSample] = []
            for s in samples:
                t = s.t_index
                if store is not None:
                    # Lazy: read this sample's input + target windows from disk now
                    input_graphs = store.get_seq(t - L + 1, L)
                    target_graphs = store.get_seq(t - L + 1 + H, L)
                else:
                    input_graphs = s.graph_seq
                    # Eager: need the full windows list; re-load via _load_ordered_windows
                    # is too expensive here. The caller should not reach this branch
                    # for large datasets (lazy_threshold handles it).
                    target_graphs = s.graph_seq  # placeholder; overridden below

                input_seq = [graph_dict_to_tensors(g, self.node_dim, self.edge_dim)
                             for g in input_graphs]
                target_seq = [graph_dict_to_tensors(g, self.node_dim, self.edge_dim)
                              for g in target_graphs]
                out.append(WorldModelSample(
                    input_seq=input_seq, target_seq=target_seq,
                    y_risk=int(s.y), t_index=t,
                ))
            return out

        if store is not None:
            # Lazy path: convert eagerly but graph I/O is per-sample disk read.
            tr = convert(train)
            va = convert(val)
            te = convert(test)
        else:
            # Eager path: use _load_ordered_windows for target lookup (small datasets).
            windows = _load_ordered_windows(exp_cfg)

            def convert_eager(samples: Sequence[WindowSample]) -> List[WorldModelSample]:
                out: List[WorldModelSample] = []
                for s in samples:
                    t = s.t_index
                    input_seq = [graph_dict_to_tensors(g, self.node_dim, self.edge_dim)
                                 for g in s.graph_seq]
                    fut_start = t - L + 1 + H
                    fut_slice = windows[fut_start: fut_start + L]
                    target_seq = [graph_dict_to_tensors(g, self.node_dim, self.edge_dim)
                                  for g in fut_slice]
                    out.append(WorldModelSample(
                        input_seq=input_seq, target_seq=target_seq,
                        y_risk=int(s.y), t_index=t,
                    ))
                return out

            tr = convert_eager(train)
            va = convert_eager(val)
            te = convert_eager(test)

        info = dict(info)
        info["graph_capable"] = _graph_capable(train if not store else tr)
        return tr, va, te, info


def _graph_capable(train: Sequence[WindowSample]) -> bool:
    total_nodes = sum(int(g.get("num_nodes", 0)) for s in train for g in s.graph_seq)
    return total_nodes > 0


def iter_batches(samples: Sequence[WorldModelSample], batch_size: int, shuffle: bool = False,
                 seed: int = 42):
    """Yield mini-batches. Order is deterministic given seed when shuffling.

    We shuffle only within TRAIN (chronology already enforced at split time by
    the Phase-3 protocol; shuffling training samples does not leak because each
    sample's inputs stay strictly before its own target)."""
    idx = list(range(len(samples)))
    if shuffle:
        import random
        rng = random.Random(seed)
        rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        chunk = idx[start:start + batch_size]
        yield [samples[i] for i in chunk]


def collate_inputs(batch: Sequence[WorldModelSample]) -> List[List[GraphTensors]]:
    return [s.input_seq for s in batch]


def collate_targets(batch: Sequence[WorldModelSample]) -> List[List[GraphTensors]]:
    return [s.target_seq for s in batch]


def collate_risk(batch: Sequence[WorldModelSample]) -> torch.Tensor:
    return torch.tensor([s.y_risk for s in batch], dtype=torch.float32)
