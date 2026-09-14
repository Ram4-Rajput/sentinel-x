"""Phase-5 data layer: K-step forecasting samples.

Reuses the SAME Phase-3/Phase-4 protocol (``sentinelx.experiments.common``):
identical windows, ``seq_len``, chronological split, seed, and leakage policy.
The one addition is that, for each anchor window ``t``, we materialise the
actual future window sequences ending at ``t+1, t+2, ..., t+K`` together with
the attack label at each of those horizons, so we can score the autoregressive
rollout as a function of horizon.

Leakage safety
--------------
The *observed input* for a sample still ends at window ``t`` and never includes
any future window. The future sequences are ground-truth ONLY — used to
evaluate the rollout, never fed into the prediction path. A sample at anchor
``t`` is only created when the deepest horizon ``t+K`` exists in the globally
ordered windows, so every horizon has a real target (no padding, no fabrication).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..experiments.common import ExperimentConfig, _load_ordered_windows
from .config import DataConfig
from .data import graph_dict_to_tensors
from .model import GraphTensors


@dataclass
class KStepSample:
    """A K-step forecasting sample anchored at window ``t``.

    * input_seq          : graphs for windows [t-L+1 .. t]                (observed)
    * future_seqs[k]     : graphs for windows [t-L+1+k .. t+k]  (ground-truth future)
    * future_labels[k]   : attack@(t+k) label (auxiliary risk target at horizon k)
    * t_index            : anchor window index t
    """

    input_seq: List[GraphTensors]
    future_seqs: Dict[int, List[GraphTensors]]
    future_labels: Dict[int, int]
    t_index: int


class KStepDataset:
    """Builds K-step train/val/test samples from the Phase-2 cache.

    The chronological split boundaries match the Phase-4 ``WorldModelDataset``
    exactly (same fractions applied to the same base sample set), so K-step
    evaluation is apples-to-apples with Phase-4/Phase-3.
    """

    def __init__(self, data_cfg: DataConfig, processed_root, node_dim: int,
                 edge_dim: int, K: int):
        self.cfg = data_cfg
        self.processed_root = processed_root
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.K = int(K)

    def _exp_cfg(self) -> ExperimentConfig:
        return ExperimentConfig(
            dataset=self.cfg.dataset,
            processed_root=self.processed_root,
            seq_len=self.cfg.seq_len,
            horizon=self.cfg.horizon,
            train_frac=self.cfg.train_frac,
            val_frac=self.cfg.val_frac,
            min_windows=self.cfg.min_windows,
        )

    def build(self) -> Tuple[List[KStepSample], List[KStepSample],
                             List[KStepSample], Dict]:
        exp_cfg = self._exp_cfg()
        windows = _load_ordered_windows(exp_cfg)
        n_win = len(windows)
        L = self.cfg.seq_len
        K = self.K

        info: Dict = {
            "dataset": self.cfg.dataset, "num_windows": n_win,
            "seq_len": L, "K": K,
        }
        if n_win < self.cfg.min_windows:
            info["usable"] = False
            info["reason"] = (f"Only {n_win} windows (< min_windows="
                              f"{self.cfg.min_windows}); cannot form K-step splits.")
            return [], [], [], info

        y_all = [int(w.get("label_any_attack", 0)) for w in windows]

        def to_tensors(g):
            return graph_dict_to_tensors(g, self.node_dim, self.edge_dim)

        # Anchor windows: need a full observed history [t-L+1 .. t] AND the
        # deepest future window t+K to exist.
        samples: List[KStepSample] = []
        for t in range(L - 1, n_win - K):
            input_seq = [to_tensors(windows[i]) for i in range(t - L + 1, t + 1)]
            future_seqs: Dict[int, List[GraphTensors]] = {}
            future_labels: Dict[int, int] = {}
            for k in range(1, K + 1):
                fut_start = t - L + 1 + k
                future_seqs[k] = [to_tensors(windows[i])
                                  for i in range(fut_start, fut_start + L)]
                future_labels[k] = y_all[t + k]
            samples.append(KStepSample(
                input_seq=input_seq, future_seqs=future_seqs,
                future_labels=future_labels, t_index=t,
            ))

        n = len(samples)
        if n < self.cfg.min_windows:
            info["usable"] = False
            info["reason"] = (f"Only {n} K-step samples after windowing "
                              f"(K={K}); too few for fair splits.")
            return [], [], [], info

        n_train = int(n * self.cfg.train_frac)
        n_val = int(n * self.cfg.val_frac)
        train = samples[:n_train]
        val = samples[n_train:n_train + n_val]
        test = samples[n_train + n_val:]

        # Leakage assertions (chronological, identical policy).
        if train and val:
            assert train[-1].t_index < val[0].t_index, "train/val overlap (leakage)"
        if val and test:
            assert val[-1].t_index < test[0].t_index, "val/test overlap (leakage)"
        if train and test and not val:
            assert train[-1].t_index < test[0].t_index, "train/test overlap (leakage)"

        info["usable"] = True
        info["num_samples"] = n
        info["counts"] = {"train": len(train), "val": len(val), "test": len(test)}
        info["graph_capable"] = _graph_capable(train)
        # positives per horizon (test split) — useful context for risk metrics
        info["test_positives_by_horizon"] = {
            k: sum(s.future_labels[k] for s in test) for k in range(1, K + 1)
        }
        return train, val, test, info


def _graph_capable(samples: Sequence[KStepSample]) -> bool:
    total_nodes = sum(g.num_nodes for s in samples for g in s.input_seq)
    return total_nodes > 0


def iter_kstep_batches(samples: Sequence[KStepSample], batch_size: int):
    """Yield deterministic mini-batches (no shuffle — evaluation only)."""
    for start in range(0, len(samples), batch_size):
        yield list(samples[start:start + batch_size])


def collate_inputs(batch: Sequence[KStepSample]) -> List[List[GraphTensors]]:
    return [s.input_seq for s in batch]


def collate_future_seqs(batch: Sequence[KStepSample], k: int) -> List[List[GraphTensors]]:
    return [s.future_seqs[k] for s in batch]


def collate_future_labels(batch: Sequence[KStepSample], k: int) -> List[int]:
    return [s.future_labels[k] for s in batch]
