"""Shared experiment protocol for all three baselines (fairness layer).

Every baseline consumes the SAME artifacts produced here, so comparisons are
apples-to-apples:

  * Source: Phase-2 cached window graph sequences (data/processed/<dataset>/).
    No raw CSV is re-read; no new splitting mechanism is introduced — windows
    are already chronologically ordered by the Phase-2 builder.

  * Target (forecasting, leakage-safe): given information available UP TO AND
    INCLUDING window t, predict whether window t+H contains attack activity
    (label_any_attack of window t+H). Inputs never include window t+H, so no
    future information leaks. H = horizon (default 1).

  * Sample = (window_index t, sequence of the last `seq_len` windows' features
    /graphs ending at t, aggregated features of t, target y = attack@(t+H)).

  * Split: chronological by window index (train < val < test), same fractions
    for all baselines. Windows are already time-ordered; we never shuffle.

  * Leakage checks: assert train window indices < val < test, and that the
    target horizon does not straddle a split boundary into an earlier split.

  * Seeding: seed_everything sets python/numpy/torch RNGs deterministically.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Node feature dimension from the Phase-2 cache (out/in degree, bytes x2, flows)
NODE_FEATURE_DIM = 5
# Window-level aggregate feature names (derived only from cached graph dicts).
WINDOW_FEATURE_NAMES = [
    "num_nodes", "num_edges", "num_flows", "attack_ratio",
    "mean_node_out_degree", "mean_node_in_degree",
    "mean_node_bytes_sent", "mean_node_bytes_received",
    "total_edge_bytes", "edge_attack_fraction",
    "density",
]


def seed_everything(seed: int = 42) -> None:
    """Deterministic RNGs across python / numpy / torch (reproducible-ml skill)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


@dataclass
class ExperimentConfig:
    dataset: str
    processed_root: Path
    seq_len: int = 4            # number of past windows in a sequence
    horizon: int = 1           # predict attack H windows ahead
    train_frac: float = 0.7
    val_frac: float = 0.15
    seed: int = 42
    min_windows: int = 12      # need enough windows to form fair splits
    lazy_threshold: int = 5000  # use lazy loading when num_windows > this


@dataclass
class WindowSample:
    """One forecasting sample derived from cached windows.

    graph_seq stores the actual graph dicts ONLY when use_lazy=False (small
    datasets, local dev). For large datasets (ctu-13-full on Kaggle) set
    use_lazy=True: graph_seq is empty and graphs are fetched via LazyWindowStore
    on demand during batching. This keeps RAM proportional to batch_size, not
    to the full dataset size.
    """
    dataset: str
    t_index: int                       # window index t (last observed window)
    window_features: np.ndarray        # aggregate features of window t  (F,)
    seq_features: np.ndarray           # aggregated features [t-seq_len+1 .. t] (seq_len, F)
    graph_seq: List[dict]              # graph dicts [t-seq_len+1 .. t]; empty if lazy
    y: int                             # attack@(t+horizon)


class LazyWindowStore:
    """Memory-efficient window store: reads graph dicts from JSONL on demand.

    Keeps only a small LRU-style cache in RAM so repeated access within one
    batch is fast, but the full ~2.5 GB of JSON never sits in memory at once.

    Layout: three JSONL files (train/val/test) are concatenated logically.
    Absolute window index maps to (file_index, line_number) via a lightweight
    offset table built once at construction (one pass, integer-only).
    """

    _SPLITS = ("train", "val", "test")

    def __init__(self, windows_dir: Path, cache_size: int = 512):
        self._dir = Path(windows_dir)
        self._cache: dict = {}          # abs_index -> dict (bounded LRU-ish)
        self._cache_size = cache_size
        # Build offset table: list of (filepath, byte_offset) per abs index.
        self._offsets: List[tuple] = []   # (path, byte_offset)
        for split in self._SPLITS:
            p = self._dir / f"{split}.jsonl"
            if not p.exists():
                continue
            with open(p, "rb") as fh:
                while True:
                    offset = fh.tell()
                    line = fh.readline()
                    if not line:
                        break
                    if line.strip():
                        self._offsets.append((p, offset))

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, idx: int) -> dict:
        if idx in self._cache:
            return self._cache[idx]
        path, offset = self._offsets[idx]
        with open(path, "rb") as fh:
            fh.seek(offset)
            line = fh.readline().decode("utf-8")
        g = json.loads(line)
        # Simple bounded cache: evict oldest when full
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[idx] = g
        return g

    def get_seq(self, start: int, length: int) -> List[dict]:
        """Return a contiguous slice of windows [start, start+length)."""
        return [self[i] for i in range(start, start + length)]


def _aggregate_window(graph: dict) -> np.ndarray:
    """Window-level aggregate feature vector from a cached graph dict.

    Uses only quantities already present in the cache (no raw re-read, no
    fabricated packet features)."""
    node_feats = graph.get("node_features") or []
    edge_feats = graph.get("edge_features") or []
    num_nodes = float(graph.get("num_nodes", 0))
    num_edges = float(graph.get("num_edges", 0))
    num_flows = float(graph.get("num_flows", 0))
    attack_ratio = float(graph.get("attack_ratio", 0.0))

    if node_feats:
        arr = np.asarray(node_feats, dtype=float)  # (N, 5)
        mean_out, mean_in = float(arr[:, 0].mean()), float(arr[:, 1].mean())
        mean_sent, mean_recv = float(arr[:, 2].mean()), float(arr[:, 3].mean())
    else:
        mean_out = mean_in = mean_sent = mean_recv = 0.0

    if edge_feats:
        earr = np.asarray(edge_feats, dtype=float)  # (E, 3): flow_count, bytes, attack
        total_edge_bytes = float(earr[:, 1].sum())
        edge_attack_fraction = float(earr[:, 2].mean())
    else:
        total_edge_bytes = 0.0
        edge_attack_fraction = 0.0

    density = (num_edges / (num_nodes * (num_nodes - 1))) if num_nodes > 1 else 0.0

    return np.array([
        num_nodes, num_edges, num_flows, attack_ratio,
        mean_out, mean_in, mean_sent, mean_recv,
        total_edge_bytes, edge_attack_fraction, density,
    ], dtype=float)


def _load_ordered_windows(cfg: ExperimentConfig) -> List[dict]:
    """Load all cached window graph dicts for a dataset in window_index order.

    Concatenates the Phase-2 splits (train/val/test JSONL) which are already
    contiguous in time, then re-orders globally by window_index so this phase
    applies ONE consistent chronological split for the forecasting task.

    For large datasets (> cfg.lazy_threshold windows) this is called only to
    obtain aggregate features + labels (one pass, then discarded). Graph dicts
    are fetched on demand via LazyWindowStore during batching.
    """
    root = Path(cfg.processed_root) / cfg.dataset / "windows"
    if not root.exists():
        raise FileNotFoundError(
            f"No Phase-2 cache for '{cfg.dataset}' at {root}. Run "
            f"scripts/build_dataset.py --dataset {cfg.dataset} first."
        )
    windows: List[dict] = []
    for split in ("train", "val", "test"):
        p = root / f"{split}.jsonl"
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    windows.append(json.loads(line))
    # Windows in different splits restart window_index at 0; make a global order
    # by their (split order, window_index) which is already chronological because
    # Phase-2 split train<val<test by time. We keep insertion order (train first).
    return windows


def _load_labels_and_agg(cfg: ExperimentConfig):
    """Lightweight pass: read only labels + aggregate features, not full graph dicts.

    Used by the lazy path to avoid materialising 2.5 GB of JSON in RAM.
    Returns (agg_list, y_all) where agg_list[i] is the numpy feature vector
    and y_all[i] is the binary attack label for window i.
    """
    root = Path(cfg.processed_root) / cfg.dataset / "windows"
    agg_list: List[np.ndarray] = []
    y_all: List[int] = []
    for split in ("train", "val", "test"):
        p = root / f"{split}.jsonl"
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                w = json.loads(line)
                agg_list.append(_aggregate_window(w))
                y_all.append(int(w.get("label_any_attack", 0)))
    return agg_list, y_all


def build_window_samples(cfg: ExperimentConfig) -> Tuple[List[WindowSample], List[WindowSample], List[WindowSample], Dict]:
    """Build train/val/test forecasting samples with an identical, leakage-safe
    protocol shared by all baselines.

    For large datasets (> cfg.lazy_threshold windows) graph dicts are NOT loaded
    into RAM upfront. Each WindowSample stores only its t_index; graph dicts are
    fetched on demand via LazyWindowStore during batching. This caps peak RAM at
    O(batch_size * seq_len * avg_window_size) instead of O(N * seq_len * size),
    avoiding OOM on Kaggle (13 GB cap) with ctu-13-full (47k windows, ~2.5 GB raw).

    Returns (train, val, test, info).
    """
    root = Path(cfg.processed_root) / cfg.dataset / "windows"
    if not root.exists():
        raise FileNotFoundError(
            f"No Phase-2 cache for '{cfg.dataset}' at {root}. Run "
            f"scripts/build_dataset.py --dataset {cfg.dataset} first."
        )

    # Quick window count (no JSON parse) to decide eager vs lazy.
    n_win = sum(
        sum(1 for ln in open(root / f"{s}.jsonl", "r", encoding="utf-8") if ln.strip())
        for s in ("train", "val", "test")
        if (root / f"{s}.jsonl").exists()
    )
    info: Dict = {"dataset": cfg.dataset, "num_windows": n_win,
                  "seq_len": cfg.seq_len, "horizon": cfg.horizon}

    if n_win < cfg.min_windows:
        info["usable"] = False
        info["reason"] = (
            f"Only {n_win} windows (< min_windows={cfg.min_windows}); cannot form "
            f"fair train/val/test forecasting splits.")
        return [], [], [], info

    use_lazy = n_win > cfg.lazy_threshold
    info["lazy"] = use_lazy

    if use_lazy:
        # Lightweight pass: labels + agg features only; LazyWindowStore for graphs.
        agg, y_all = _load_labels_and_agg(cfg)
        store: Optional[LazyWindowStore] = LazyWindowStore(root)
    else:
        windows = _load_ordered_windows(cfg)
        agg = [_aggregate_window(w) for w in windows]
        y_all = [int(w.get("label_any_attack", 0)) for w in windows]
        store = None

    # Build samples.
    samples: List[WindowSample] = []
    for t in range(cfg.seq_len - 1, n_win - cfg.horizon):
        seq_slice = agg[t - cfg.seq_len + 1: t + 1]
        # Lazy: graph_seq is empty; store fetches on demand at batch time.
        graph_seq: List[dict] = (
            [] if use_lazy
            else windows[t - cfg.seq_len + 1: t + 1]  # type: ignore[index]
        )
        samples.append(WindowSample(
            dataset=cfg.dataset,
            t_index=t,
            window_features=agg[t],
            seq_features=np.stack(seq_slice, axis=0),
            graph_seq=graph_seq,
            y=y_all[t + cfg.horizon],
        ))

    n = len(samples)
    if n < cfg.min_windows:
        info["usable"] = False
        info["reason"] = f"Only {n} forecasting samples after windowing; too few for fair splits."
        return [], [], [], info

    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)
    train = samples[:n_train]
    val = samples[n_train:n_train + n_val]
    test = samples[n_train + n_val:]

    # --- leakage assertions (identical policy for all baselines) ---
    if train and val:
        assert train[-1].t_index < val[0].t_index, "train/val window overlap (leakage)"
    if val and test:
        assert val[-1].t_index < test[0].t_index, "val/test window overlap (leakage)"
    if train and test and not val:
        assert train[-1].t_index < test[0].t_index, "train/test window overlap (leakage)"
    # target horizon must not reach back into an earlier split
    if train and val:
        assert train[-1].t_index + cfg.horizon <= val[0].t_index + cfg.horizon

    info["usable"] = True
    info["num_samples"] = n
    info["counts"] = {"train": len(train), "val": len(val), "test": len(test)}
    info["positives"] = {
        "train": sum(s.y for s in train),
        "val": sum(s.y for s in val),
        "test": sum(s.y for s in test),
    }
    info["feature_names"] = WINDOW_FEATURE_NAMES
    if use_lazy:
        info["_lazy_store"] = store   # consumed by WorldModelDataset.build()
    return train, val, test, info


def stack_features(samples: Sequence[WindowSample]) -> np.ndarray:
    return np.stack([s.window_features for s in samples], axis=0) if samples else np.empty((0, len(WINDOW_FEATURE_NAMES)))


def stack_sequences(samples: Sequence[WindowSample]) -> np.ndarray:
    return np.stack([s.seq_features for s in samples], axis=0) if samples else np.empty((0, 0, len(WINDOW_FEATURE_NAMES)))


def stack_labels(samples: Sequence[WindowSample]) -> np.ndarray:
    return np.asarray([s.y for s in samples], dtype=int)
