"""Phase-3 baseline model tests: each of the three baselines trains, produces
metrics, is reproducible with a fixed seed, and handles imbalance/edge cases.
Uses a small synthetic Phase-2 cache (deterministic)."""

import json
from pathlib import Path

import numpy as np
import pytest

from sentinelx.experiments.common import ExperimentConfig, build_window_samples, seed_everything
from sentinelx.experiments.baseline_logreg import run_logreg
from sentinelx.experiments.baseline_sequence import run_sequence
from sentinelx.experiments.baseline_gnn import run_gnn


def _write_synth_cache(root: Path, dataset: str, n_windows: int, attack_from: int,
                       with_nodes: bool = True):
    wdir = root / dataset / "windows"
    wdir.mkdir(parents=True, exist_ok=True)
    n_tr = int(n_windows * 0.7)
    n_va = int(n_windows * 0.15)
    splits = {"train": range(0, n_tr), "val": range(n_tr, n_tr + n_va),
              "test": range(n_tr + n_va, n_windows)}
    gi = 0
    rng = np.random.RandomState(0)
    for split, idxs in splits.items():
        with open(wdir / f"{split}.jsonl", "w", encoding="utf-8") as fh:
            for local_i, _ in enumerate(idxs):
                attack = 1 if gi >= attack_from else 0
                # signal: attack windows have higher edge bytes / attack fraction
                nb = 2 if with_nodes else 0
                node_features = ([[1.0, 0.0, 100.0 + 50 * attack + rng.rand(), 0.0, 2.0],
                                  [0.0, 1.0, 0.0, 50.0, 1.0]] if with_nodes else [])
                edge_features = [[1.0, 150.0 + 200.0 * attack, float(attack)]] if with_nodes else []
                edge_index = [[0, 1]] if with_nodes else []
                g = {
                    "window_index": local_i, "start": None, "end": None,
                    "num_nodes": nb, "num_edges": len(edge_index),
                    "num_flows": 3 + 5 * attack,
                    "node_ids": ["a", "b"][:nb],
                    "node_features": node_features,
                    "edge_index": edge_index, "edge_features": edge_features,
                    "attack_ratio": 0.5 * attack, "label_any_attack": attack,
                }
                fh.write(json.dumps(g) + "\n")
                gi += 1


@pytest.fixture
def samples(tmp_path):
    _write_synth_cache(tmp_path, "ctu-13", n_windows=60, attack_from=30)
    cfg = ExperimentConfig(dataset="ctu-13", processed_root=tmp_path, seq_len=4, horizon=1)
    seed_everything(42)
    return build_window_samples(cfg)


def test_logreg_trains_and_scores(samples):
    train, val, test, info = samples
    res = run_logreg(train, val, test, seed=42)
    assert 0.0 <= res.test_metrics.precision <= 1.0
    assert res.test_metrics.n == len(test)
    assert res.params["class_weight"] == "balanced"  # imbalance handled


def test_logreg_reproducible(samples):
    train, val, test, _ = samples
    r1 = run_logreg(train, val, test, seed=42)
    r2 = run_logreg(train, val, test, seed=42)
    assert r1.test_metrics.f1 == r2.test_metrics.f1
    assert r1.test_metrics.pr_auc == r2.test_metrics.pr_auc or (
        r1.test_metrics.pr_auc != r1.test_metrics.pr_auc)  # NaN==NaN guard


@pytest.mark.parametrize("temporal_type", ["gru", "lstm"])
def test_sequence_trains(samples, temporal_type):
    train, val, test, _ = samples
    res = run_sequence(train, val, test, temporal_type=temporal_type, epochs=5, seed=42)
    assert res.num_parameters > 0
    assert res.train_seconds >= 0.0
    assert res.test_metrics.n == len(test)
    assert res.params["temporal_type"] == temporal_type


def test_sequence_reproducible(samples):
    train, val, test, _ = samples
    r1 = run_sequence(train, val, test, temporal_type="gru", epochs=5, seed=42)
    r2 = run_sequence(train, val, test, temporal_type="gru", epochs=5, seed=42)
    assert r1.test_metrics.f1 == r2.test_metrics.f1


def test_sequence_invalid_type(samples):
    train, val, test, _ = samples
    with pytest.raises(ValueError):
        run_sequence(train, val, test, temporal_type="rnn")


@pytest.mark.parametrize("gnn_type", ["sage", "gat"])
def test_gnn_trains_and_flags_capability(samples, gnn_type):
    train, val, test, _ = samples
    res = run_gnn(train, val, test, gnn_type=gnn_type, epochs=5, seed=42)
    assert res.num_parameters > 0
    assert res.graph_capable is True     # synthetic cache has nodes
    assert res.test_metrics.n == len(test)


def test_gnn_marks_no_node_datasets(tmp_path):
    # graphs with 0 nodes (like CIC-IDS2018) -> graph_capable False + note
    _write_synth_cache(tmp_path, "cic-ids2018", n_windows=60, attack_from=30, with_nodes=False)
    cfg = ExperimentConfig(dataset="cic-ids2018", processed_root=tmp_path, seq_len=4, horizon=1)
    train, val, test, info = build_window_samples(cfg)
    res = run_gnn(train, val, test, gnn_type="sage", epochs=3, seed=42)
    assert res.graph_capable is False
    assert "no nodes" in res.note.lower()


def test_gnn_invalid_type(samples):
    train, val, test, _ = samples
    with pytest.raises(ValueError):
        run_gnn(train, val, test, gnn_type="gcn")
