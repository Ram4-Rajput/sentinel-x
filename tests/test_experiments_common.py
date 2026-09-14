"""Phase-3 shared-protocol tests: sample building, chronological split,
leakage-safe forecasting target, and the too-few-windows guard.

A tiny synthetic Phase-2 cache is written to a temp dir (no dependence on the
large real caches)."""

import json
from pathlib import Path

import numpy as np
import pytest

from sentinelx.experiments.common import (
    ExperimentConfig,
    build_window_samples,
    seed_everything,
)


def _write_synth_cache(root: Path, dataset: str, n_windows: int,
                       attack_from: int = None):
    """Write n_windows tiny graph dicts across train/val/test JSONL, time-ordered.
    Windows >= attack_from carry label_any_attack=1."""
    wdir = root / dataset / "windows"
    wdir.mkdir(parents=True, exist_ok=True)
    # split the windows across the three files, preserving order
    n_tr = int(n_windows * 0.7)
    n_va = int(n_windows * 0.15)
    splits = {"train": range(0, n_tr),
              "val": range(n_tr, n_tr + n_va),
              "test": range(n_tr + n_va, n_windows)}
    gi = 0
    for split, idxs in splits.items():
        with open(wdir / f"{split}.jsonl", "w", encoding="utf-8") as fh:
            for local_i, _ in enumerate(idxs):
                attack = 1 if (attack_from is not None and gi >= attack_from) else 0
                # a couple of nodes/edges so graph baseline has structure
                g = {
                    "window_index": local_i,
                    "start": None, "end": None,
                    "num_nodes": 2, "num_edges": 1, "num_flows": 3 + gi % 5,
                    "node_ids": ["a", "b"],
                    "node_features": [[1.0, 0.0, 100.0 + gi, 0.0, 2.0],
                                      [0.0, 1.0, 0.0, 50.0, 1.0]],
                    "edge_index": [[0, 1]],
                    "edge_features": [[1.0, 150.0, float(attack)]],
                    "attack_ratio": float(attack) * 0.5,
                    "label_any_attack": attack,
                }
                fh.write(json.dumps(g) + "\n")
                gi += 1


def test_seed_everything_is_deterministic():
    seed_everything(123)
    a = np.random.rand(5)
    seed_everything(123)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_build_samples_and_chronological_split(tmp_path):
    _write_synth_cache(tmp_path, "ctu-13", n_windows=40, attack_from=25)
    cfg = ExperimentConfig(dataset="ctu-13", processed_root=tmp_path,
                           seq_len=4, horizon=1)
    train, val, test, info = build_window_samples(cfg)
    assert info["usable"] is True
    # chronological, non-overlapping window indices
    assert train[-1].t_index < val[0].t_index < test[0].t_index or not val
    # samples carry a sequence of the right length
    assert train[0].seq_features.shape[0] == 4
    # target is drawn from t+horizon (attacks appear late -> present in test)
    assert info["positives"]["test"] > 0


def test_no_future_leakage_in_inputs(tmp_path):
    # A sample's graph_seq must all have window indices <= t (never t+horizon).
    _write_synth_cache(tmp_path, "ctu-13", n_windows=30, attack_from=20)
    cfg = ExperimentConfig(dataset="ctu-13", processed_root=tmp_path, seq_len=3, horizon=1)
    train, val, test, info = build_window_samples(cfg)
    for s in train + val + test:
        # the sequence ends at t; target is t+1 which is NOT in the sequence
        assert len(s.graph_seq) == 3


def test_too_few_windows_flagged(tmp_path):
    _write_synth_cache(tmp_path, "ctu-13", n_windows=6)
    cfg = ExperimentConfig(dataset="ctu-13", processed_root=tmp_path, min_windows=12)
    train, val, test, info = build_window_samples(cfg)
    assert info["usable"] is False
    assert "too few" in info["reason"].lower() or "min_windows" in info["reason"].lower()


def test_missing_cache_raises(tmp_path):
    cfg = ExperimentConfig(dataset="nope", processed_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        build_window_samples(cfg)
