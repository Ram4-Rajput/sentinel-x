"""Phase-4 Sentinel-X world model tests.

Covers (per the Phase-4 spec):
  * graph encoder (GAT + GraphSAGE), consumes topology + edge features
  * temporal encoder (GRU + LSTM)
  * tensor dimensions of the latent state / forecasting head
  * forward pass + backward pass
  * checkpoint save/load
  * deterministic inference (eval mode, fixed seed)
  * variable graph sizes
  * config loading / normalization
  * documented multi-objective loss behaviour

A tiny synthetic Phase-2 cache (deterministic) is used — no real data files.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sentinelx.experiments.common import ExperimentConfig, seed_everything
from sentinelx.worldmodel.config import (
    ModelConfig, TrainingConfig, DataConfig, WorldModelConfig, load_config,
    dump_config_yaml, from_dict,
)
from sentinelx.worldmodel.data import (
    WorldModelDataset, graph_dict_to_tensors, collate_inputs, collate_risk, iter_batches,
)
from sentinelx.worldmodel.graph_encoder import GraphEncoder
from sentinelx.worldmodel.temporal_encoder import TemporalEncoder
from sentinelx.worldmodel.model import SentinelXWorldModel, GraphTensors
from sentinelx.worldmodel.losses import world_model_loss, future_state_loss
from sentinelx.worldmodel.train import train_world_model
from sentinelx.worldmodel.experiment import run_world_model_experiments


NODE_DIM = 5
EDGE_DIM = 3


def _write_synth_cache(root: Path, dataset: str, n_windows: int, attack_from: int,
                       with_nodes: bool = True, seed: int = 0):
    """Synthetic cache mirroring graph_cache.py format; attack windows differ."""
    wdir = root / dataset / "windows"
    wdir.mkdir(parents=True, exist_ok=True)
    n_tr = int(n_windows * 0.7)
    n_va = int(n_windows * 0.15)
    splits = {"train": range(0, n_tr), "val": range(n_tr, n_tr + n_va),
              "test": range(n_tr + n_va, n_windows)}
    gi = 0
    rng = np.random.RandomState(seed)
    for split, idxs in splits.items():
        with open(wdir / f"{split}.jsonl", "w", encoding="utf-8") as fh:
            for local_i, _ in enumerate(idxs):
                attack = 1 if gi >= attack_from else 0
                # variable graph size: node count varies with index
                nb = (2 + (gi % 3)) if with_nodes else 0
                node_features = []
                for k in range(nb):
                    node_features.append([
                        float(1 + k), float(k), 100.0 + 50 * attack + rng.rand(),
                        float(10 * k), float(2 + attack),
                    ])
                edge_index, edge_features = [], []
                for k in range(max(0, nb - 1)):
                    edge_index.append([k, k + 1])
                    edge_features.append([1.0, 150.0 + 200.0 * attack, float(attack)])
                g = {
                    "window_index": local_i, "start": None, "end": None,
                    "num_nodes": nb, "num_edges": len(edge_index),
                    "num_flows": 3 + 5 * attack,
                    "node_ids": [f"n{k}" for k in range(nb)],
                    "node_features": node_features,
                    "edge_index": edge_index, "edge_features": edge_features,
                    "attack_ratio": 0.5 * attack, "label_any_attack": attack,
                }
                fh.write(json.dumps(g) + "\n")
                gi += 1


@pytest.fixture
def synth_root(tmp_path):
    _write_synth_cache(tmp_path, "ctu-13", n_windows=60, attack_from=30)
    return tmp_path


@pytest.fixture
def dataset_samples(synth_root):
    dc = DataConfig(dataset="ctu-13", seq_len=4, horizon=1)
    ds = WorldModelDataset(dc, synth_root, NODE_DIM, EDGE_DIM)
    seed_everything(42)
    return ds.build()


# --------------------------- config ---------------------------------------- #
def test_config_normalization_aliases():
    cfg = ModelConfig(gnn_type="sage", temporal_type="GRU").normalized()
    assert cfg.gnn_type == "graphsage"
    assert cfg.temporal_type == "gru"


def test_config_rejects_bad_types():
    with pytest.raises(ValueError):
        ModelConfig(gnn_type="gcn").normalized()
    with pytest.raises(ValueError):
        ModelConfig(temporal_type="rnn").normalized()


def test_config_yaml_roundtrip(tmp_path):
    cfg = WorldModelConfig(model=ModelConfig(gnn_type="gat", hidden_dim=64, gat_heads=4).normalized(),
                           training=TrainingConfig(epochs=3), data=DataConfig(dataset="ctu-13"))
    p = dump_config_yaml(cfg, tmp_path / "c.yaml")
    loaded = load_config(p)
    assert loaded.model.gnn_type == "gat"
    assert loaded.model.hidden_dim == 64
    assert loaded.training.epochs == 3
    assert loaded.data.dataset == "ctu-13"


# --------------------------- graph encoder --------------------------------- #
@pytest.mark.parametrize("gnn_type", ["gat", "graphsage"])
def test_graph_encoder_output_dim(gnn_type):
    cfg = ModelConfig(gnn_type=gnn_type, hidden_dim=32, gat_heads=4,
                      graph_layers=2).normalized()
    enc = GraphEncoder(cfg)
    x = torch.randn(5, NODE_DIM)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    edge_attr = torch.randn(4, EDGE_DIM)
    batch = torch.zeros(5, dtype=torch.long)
    h = enc(x, edge_index, batch, edge_attr)
    assert h.shape == (1, 32)


def test_graph_encoder_consumes_edge_features_gat():
    """GAT output changes when edge features change -> edges are actually used."""
    cfg = ModelConfig(gnn_type="gat", hidden_dim=16, gat_heads=2, graph_layers=1,
                      dropout=0.0).normalized()
    enc = GraphEncoder(cfg).eval()
    x = torch.randn(4, NODE_DIM)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    batch = torch.zeros(4, dtype=torch.long)
    ea1 = torch.zeros(3, EDGE_DIM)
    ea2 = torch.ones(3, EDGE_DIM) * 5.0
    with torch.no_grad():
        h1 = enc(x, edge_index, batch, ea1)
        h2 = enc(x, edge_index, batch, ea2)
    assert not torch.allclose(h1, h2)


def test_graph_encoder_uses_topology_graphsage():
    """GraphSAGE output changes when connectivity changes -> topology is used."""
    cfg = ModelConfig(gnn_type="graphsage", hidden_dim=16, graph_layers=1,
                      dropout=0.0).normalized()
    enc = GraphEncoder(cfg).eval()
    x = torch.randn(4, NODE_DIM)
    ei1 = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    ei2 = torch.tensor([[0, 3], [3, 0]], dtype=torch.long)
    batch = torch.zeros(4, dtype=torch.long)
    ea = torch.zeros(2, EDGE_DIM)
    with torch.no_grad():
        h1 = enc(x, ei1, batch, ea)
        h2 = enc(x, ei2, batch, ea)
    assert not torch.allclose(h1, h2)


def test_graph_encoder_empty_graph():
    cfg = ModelConfig(hidden_dim=24).normalized()
    enc = GraphEncoder(cfg)
    x = torch.empty((0, NODE_DIM))
    ei = torch.empty((2, 0), dtype=torch.long)
    batch = torch.empty((0,), dtype=torch.long)
    h = enc(x, ei, batch, torch.empty((0, EDGE_DIM)))
    assert h.shape == (1, 24)
    assert torch.count_nonzero(h) == 0


# --------------------------- temporal encoder ------------------------------ #
@pytest.mark.parametrize("temporal_type", ["gru", "lstm"])
def test_temporal_encoder_output_dim(temporal_type):
    cfg = ModelConfig(temporal_type=temporal_type, hidden_dim=32, latent_dim=16).normalized()
    enc = TemporalEncoder(cfg)
    seq = torch.randn(8, 4, 32)   # (batch, seq_len, hidden)
    z = enc(seq)
    assert z.shape == (8, 16)


# --------------------------- data layer ------------------------------------ #
def test_data_builds_input_and_future_target(dataset_samples):
    train, val, test, info = dataset_samples
    assert info["usable"] is True
    assert info["graph_capable"] is True
    assert len(train) > 0 and len(val) > 0 and len(test) > 0
    s = train[0]
    assert len(s.input_seq) == 4
    assert len(s.target_seq) == 4
    # target sequence is shifted by horizon (=1): its last window is one ahead
    assert isinstance(s.y_risk, int)


def test_graph_dict_to_tensors_variable_sizes(synth_root):
    from sentinelx.experiments.common import _load_ordered_windows
    cfg = ExperimentConfig(dataset="ctu-13", processed_root=synth_root, seq_len=4, horizon=1)
    windows = _load_ordered_windows(cfg)
    sizes = set()
    for g in windows[:6]:
        gt = graph_dict_to_tensors(g, NODE_DIM, EDGE_DIM)
        assert gt.x.shape[1] == NODE_DIM
        sizes.add(gt.num_nodes)
    assert len(sizes) > 1   # variable graph sizes present


# --------------------------- full model ------------------------------------ #
def test_model_forward_shapes(dataset_samples):
    train, _, _, _ = dataset_samples
    cfg = ModelConfig(hidden_dim=32, latent_dim=16, gat_heads=4).normalized()
    model = SentinelXWorldModel(cfg)
    batch = train[:5]
    out = model(collate_inputs(batch))
    assert out.z_t.shape == (5, 16)
    assert out.z_next_pred.shape == (5, 16)
    assert out.risk_logit.shape == (5,)


def test_model_encode_state_api(dataset_samples):
    train, _, _, _ = dataset_samples
    cfg = ModelConfig(hidden_dim=16, latent_dim=8).normalized()
    model = SentinelXWorldModel(cfg).eval()
    z = model.encode_state(collate_inputs(train[:3]))
    assert z.shape == (3, 8)


def test_model_backward_pass(dataset_samples):
    train, _, _, _ = dataset_samples
    cfg = ModelConfig(hidden_dim=16, latent_dim=8, gat_heads=2).normalized()
    model = SentinelXWorldModel(cfg)
    batch = train[:6]
    out = model(collate_inputs(batch))
    z_target = model.encode_target_state([s.target_seq for s in batch])
    y = collate_risk(batch)
    parts = world_model_loss(out.z_next_pred, z_target, out.risk_logit, y)
    parts.total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(torch.count_nonzero(g) > 0 for g in grads)


def test_variable_graph_sizes_in_batch(dataset_samples):
    """A batch mixes graphs with different node counts without error."""
    train, _, _, _ = dataset_samples
    cfg = ModelConfig(hidden_dim=16, latent_dim=8).normalized()
    model = SentinelXWorldModel(cfg).eval()
    node_counts = {g.num_nodes for s in train[:10] for g in s.input_seq}
    assert len(node_counts) > 1
    out = model(collate_inputs(train[:10]))
    assert out.z_t.shape[0] == 10


# --------------------------- loss ------------------------------------------ #
def test_future_state_loss_zero_when_identical():
    z = torch.randn(4, 8)
    loss = future_state_loss(z, z.clone())
    assert float(loss) < 1e-5


def test_future_state_loss_positive_when_different():
    z = torch.randn(4, 8)
    loss = future_state_loss(z, -z)   # opposite direction -> cos = -1 -> loss ~ 2
    assert float(loss) > 1.0


# --------------------------- checkpoint ------------------------------------ #
def test_checkpoint_save_load(dataset_samples, tmp_path):
    train, _, _, _ = dataset_samples
    cfg = ModelConfig(hidden_dim=16, latent_dim=8, gat_heads=2).normalized()
    model = SentinelXWorldModel(cfg).eval()
    ckpt = tmp_path / "m.pt"
    model.save_checkpoint(ckpt, extra={"combo": "gat_gru"})
    loaded, extra = SentinelXWorldModel.load_checkpoint(ckpt)
    assert extra["combo"] == "gat_gru"
    with torch.no_grad():
        z1 = model.encode_state(collate_inputs(train[:3]))
        z2 = loaded.encode_state(collate_inputs(train[:3]))
    assert torch.allclose(z1, z2, atol=1e-6)


# --------------------------- deterministic inference ----------------------- #
def test_deterministic_inference(dataset_samples):
    train, _, _, _ = dataset_samples
    seed_everything(42)
    cfg = ModelConfig(hidden_dim=16, latent_dim=8, dropout=0.3).normalized()
    m1 = SentinelXWorldModel(cfg).eval()
    with torch.no_grad():
        a = m1.encode_state(collate_inputs(train[:4]))
        b = m1.encode_state(collate_inputs(train[:4]))
    assert torch.allclose(a, b)   # eval mode -> dropout off -> deterministic


# --------------------------- training loop --------------------------------- #
@pytest.mark.parametrize("gnn_type,temporal_type", [("gat", "gru"), ("graphsage", "lstm")])
def test_training_loop_runs(dataset_samples, gnn_type, temporal_type):
    train, val, test, _ = dataset_samples
    cfg = ModelConfig(gnn_type=gnn_type, temporal_type=temporal_type,
                      hidden_dim=16, latent_dim=8, gat_heads=2).normalized()
    model = SentinelXWorldModel(cfg)
    tcfg = TrainingConfig(epochs=3, batch_size=8, seed=42, early_stopping_patience=5)
    res = train_world_model(model, train, val, test, tcfg)
    assert res.num_parameters > 0
    assert len(res.history) >= 1
    assert res.test_metrics is not None
    assert 0.0 <= res.test_metrics.precision <= 1.0


def test_training_reproducible(dataset_samples):
    train, val, test, _ = dataset_samples
    tcfg = TrainingConfig(epochs=3, batch_size=8, seed=123, early_stopping_patience=5)

    def run_once():
        seed_everything(123)
        cfg = ModelConfig(hidden_dim=16, latent_dim=8, gat_heads=2).normalized()
        return train_world_model(SentinelXWorldModel(cfg), train, val, test, tcfg)

    r1 = run_once()
    r2 = run_once()
    assert r1.history[-1].train_loss == pytest.approx(r2.history[-1].train_loss, abs=1e-5)


def test_checkpoint_written_by_trainer(dataset_samples, tmp_path):
    train, val, test, _ = dataset_samples
    cfg = ModelConfig(hidden_dim=16, latent_dim=8, gat_heads=2).normalized()
    tcfg = TrainingConfig(epochs=2, batch_size=8, seed=42)
    ckpt = tmp_path / "ckpt" / "model.pt"
    train_world_model(SentinelXWorldModel(cfg), train, val, test, tcfg, checkpoint_path=ckpt)
    assert ckpt.exists()
    loaded, extra = SentinelXWorldModel.load_checkpoint(ckpt)
    assert "chosen_threshold" in extra


# --------------------------- experiment driver ----------------------------- #
def test_experiment_two_combos_and_outputs(synth_root, tmp_path):
    base = WorldModelConfig(
        model=ModelConfig(hidden_dim=16, latent_dim=8, gat_heads=2).normalized(),
        training=TrainingConfig(epochs=2, batch_size=8, seed=42, early_stopping_patience=5),
        data=DataConfig(dataset="ctu-13", seq_len=4, horizon=1),
    )
    models_dir = tmp_path / "models"
    exp_dir = tmp_path / "experiments"
    res = run_world_model_experiments(
        base, synth_root, combos=[("gat", "gru"), ("graphsage", "lstm")],
        models_dir=models_dir, experiments_dir=exp_dir, log=lambda *a, **k: None,
    )
    assert not res.get("skipped")
    assert len(res["runs"]) == 2
    assert res["best_combo"] in ("gat_gru", "graphsage_lstm")
    assert (models_dir / "model.pt").exists()
    assert (models_dir / "config.yaml").exists()
    assert (models_dir / "metadata.json").exists()
    assert (exp_dir / "world_model_results.csv").exists()
    assert (exp_dir / "world_model_report.md").exists()


def test_experiment_skips_no_node_dataset(tmp_path):
    """CIC-IDS2018-like cache (no nodes) still runs but is flagged not graph-capable."""
    _write_synth_cache(tmp_path, "cic-ids2018", n_windows=60, attack_from=30, with_nodes=False)
    base = WorldModelConfig(
        model=ModelConfig(hidden_dim=16, latent_dim=8, gat_heads=2).normalized(),
        training=TrainingConfig(epochs=2, batch_size=8, seed=42),
        data=DataConfig(dataset="cic-ids2018", seq_len=4, horizon=1),
    )
    res = run_world_model_experiments(
        base, tmp_path, combos=[("gat", "gru")],
        models_dir=None, experiments_dir=None, log=lambda *a, **k: None,
    )
    assert res["split_info"]["graph_capable"] is False
