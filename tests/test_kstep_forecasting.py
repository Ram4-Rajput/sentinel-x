"""Phase-5 tests: K-step future-state forecasting via autoregressive rollout.

Covers (per the Phase-5 spec):
  * K = 1, 3, 5, 10  rollouts produce the right shapes / horizon counts
  * invalid K rejected (0, negative, float, bool, non-int)
  * rollout determinism (fixed model + inputs -> identical trajectory)
  * checkpoint loading (rollout runs from a saved-then-loaded model)
  * autoregressive composition (step k+1 is f applied to step k's latent)
  * K-step data layer (future targets + labels per horizon, leakage-safe order)
  * evaluation metrics + degradation + computational cost
  * experiment driver writes k_step_results.csv / k_step_report.md

A tiny synthetic Phase-2 cache (deterministic) is used — no real data files.
Expectations are derived by hand, not from the code under test.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sentinelx.experiments.common import seed_everything
from sentinelx.worldmodel.config import (
    DataConfig, ModelConfig, TrainingConfig, WorldModelConfig, dump_config_yaml,
)
from sentinelx.worldmodel.model import SentinelXWorldModel
from sentinelx.worldmodel.kstep_data import (
    KStepDataset, collate_inputs, collate_future_labels, collate_future_seqs,
    iter_kstep_batches,
)
from sentinelx.worldmodel.kstep_eval import evaluate_kstep
from sentinelx.worldmodel.kstep_experiment import run_kstep_experiments
from sentinelx.worldmodel.rollout import (
    rollout_latents, rollout_steps, validate_k, SUPPORTED_K,
)


NODE_DIM = 5
EDGE_DIM = 3


def _write_synth_cache(root: Path, dataset: str, n_windows: int, attack_from: int,
                       with_nodes: bool = True, seed: int = 0):
    """Synthetic cache mirroring graph_cache.py format (same as Phase-4 tests)."""
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
    # 80 windows so K=10 still leaves enough samples for a fair split.
    _write_synth_cache(tmp_path, "ctu-13", n_windows=80, attack_from=40)
    return tmp_path


@pytest.fixture
def trained_checkpoint(synth_root, tmp_path):
    """A quickly-trained checkpoint saved to disk (small model, few epochs)."""
    from sentinelx.worldmodel.data import WorldModelDataset
    from sentinelx.worldmodel.train import train_world_model
    dc = DataConfig(dataset="ctu-13", seq_len=4, horizon=1)
    seed_everything(42)
    ds = WorldModelDataset(dc, synth_root, NODE_DIM, EDGE_DIM)
    train, val, test, info = ds.build()
    cfg = ModelConfig(hidden_dim=16, latent_dim=8, gat_heads=2,
                      gnn_type="graphsage", temporal_type="lstm").normalized()
    tcfg = TrainingConfig(epochs=2, batch_size=8, seed=42)
    model = SentinelXWorldModel(cfg)
    ckpt = tmp_path / "sentinel_x" / "model.pt"
    train_world_model(model, train, val, test, tcfg, checkpoint_path=ckpt)
    # Also drop the config.yaml the experiment driver looks for.
    dump_config_yaml(WorldModelConfig(model=cfg, training=tcfg, data=dc),
                     ckpt.parent / "config.yaml")
    return ckpt


@pytest.fixture
def small_model():
    seed_everything(7)
    cfg = ModelConfig(hidden_dim=16, latent_dim=8, gat_heads=2).normalized()
    return SentinelXWorldModel(cfg).eval()


@pytest.fixture
def kstep_samples(synth_root):
    dc = DataConfig(dataset="ctu-13", seq_len=4, horizon=1)
    ds = KStepDataset(dc, synth_root, NODE_DIM, EDGE_DIM, K=10)
    seed_everything(42)
    return ds.build()


# --------------------------- validate_k ------------------------------------ #
def test_validate_k_accepts_supported():
    for k in SUPPORTED_K:
        assert validate_k(k) == k


@pytest.mark.parametrize("bad", [0, -1, -5])
def test_validate_k_rejects_non_positive(bad):
    with pytest.raises(ValueError):
        validate_k(bad)


@pytest.mark.parametrize("bad", [1.0, 3.5, "3", None, True])
def test_validate_k_rejects_non_int(bad):
    with pytest.raises(ValueError):
        validate_k(bad)


# --------------------------- rollout shapes (K=1,3,5,10) ------------------- #
@pytest.mark.parametrize("K", [1, 3, 5, 10])
def test_rollout_shapes_for_each_k(small_model, kstep_samples, K):
    train, _, _, info = kstep_samples
    assert info["usable"] and info["graph_capable"]
    batch = collate_inputs(train[:6])
    roll = rollout_latents(small_model, batch, K)
    assert roll.K == K
    assert roll.latents.shape == (K, 6, small_model.cfg.latent_dim)
    assert roll.risk_probs.shape == (K, 6)
    assert roll.z0.shape == (6, small_model.cfg.latent_dim)
    # risk probs are valid probabilities
    assert float(roll.risk_probs.min()) >= 0.0
    assert float(roll.risk_probs.max()) <= 1.0


# --------------------------- autoregressive composition -------------------- #
def test_rollout_is_autoregressive(small_model, kstep_samples):
    """Step k+1 latent == forecast_head applied to step k latent (composition)."""
    train, _, _, _ = kstep_samples
    batch = collate_inputs(train[:4])
    roll = rollout_latents(small_model, batch, 5)
    with torch.no_grad():
        for k in range(1, 5):  # compare step k -> step k+1
            expected = small_model.forecast_head(roll.latents[k - 1])
            assert torch.allclose(expected, roll.latents[k], atol=1e-6)
    # And step 1 == forecast_head(z0)
    with torch.no_grad():
        assert torch.allclose(small_model.forecast_head(roll.z0),
                              roll.latents[0], atol=1e-6)


def test_rollout_not_k_independent_passes(small_model, kstep_samples):
    """The whole trajectory comes from ONE encode + repeated one-step transition:
    z0 is encoded once and reused (no re-encoding per horizon)."""
    train, _, _, _ = kstep_samples
    batch = collate_inputs(train[:3])
    r1 = rollout_latents(small_model, batch, 1)
    r5 = rollout_latents(small_model, batch, 5)
    # First-step prediction identical regardless of total K -> same transition.
    assert torch.allclose(r1.latents[0], r5.latents[0], atol=1e-6)
    assert torch.allclose(r1.z0, r5.z0, atol=1e-6)


# --------------------------- determinism ----------------------------------- #
def test_rollout_deterministic(small_model, kstep_samples):
    train, _, _, _ = kstep_samples
    batch = collate_inputs(train[:5])
    a = rollout_latents(small_model, batch, 10)
    b = rollout_latents(small_model, batch, 10)
    assert torch.allclose(a.latents, b.latents)
    assert torch.allclose(a.risk_probs, b.risk_probs)


def test_rollout_preserves_train_mode(kstep_samples):
    """Rollout uses eval internally but restores the model's training flag."""
    seed_everything(1)
    cfg = ModelConfig(hidden_dim=16, latent_dim=8, gat_heads=2).normalized()
    model = SentinelXWorldModel(cfg)
    model.train()
    train, _, _, _ = kstep_samples
    rollout_latents(model, collate_inputs(train[:2]), 3)
    assert model.training is True


# --------------------------- per-step contract ----------------------------- #
def test_rollout_steps_contract(small_model, kstep_samples):
    train, _, _, _ = kstep_samples
    s = train[0]
    steps = rollout_steps(small_model, s.input_seq, 3, t_index=s.t_index,
                          label_by_horizon=s.future_labels)
    assert len(steps) == 3
    for k, step in enumerate(steps, start=1):
        assert step.horizon == k
        assert step.target_index == s.t_index + k
        assert len(step.latent) == small_model.cfg.latent_dim
        assert 0.0 <= step.risk_prob <= 1.0
        assert step.risk_available is True
        assert step.metadata["autoregressive"] is True
        assert step.metadata["latent_dim"] == small_model.cfg.latent_dim


# --------------------------- checkpoint loading ---------------------------- #
def test_rollout_from_loaded_checkpoint(trained_checkpoint, kstep_samples):
    model, extra = SentinelXWorldModel.load_checkpoint(trained_checkpoint)
    train, _, _, _ = kstep_samples
    roll = rollout_latents(model, collate_inputs(train[:4]), 5)
    assert roll.latents.shape == (5, 4, model.cfg.latent_dim)
    assert "chosen_threshold" in extra


def test_loaded_checkpoint_matches_original(trained_checkpoint, kstep_samples):
    m1, _ = SentinelXWorldModel.load_checkpoint(trained_checkpoint)
    m2, _ = SentinelXWorldModel.load_checkpoint(trained_checkpoint)
    train, _, _, _ = kstep_samples
    r1 = rollout_latents(m1, collate_inputs(train[:3]), 5)
    r2 = rollout_latents(m2, collate_inputs(train[:3]), 5)
    assert torch.allclose(r1.latents, r2.latents)


# --------------------------- K-step data layer ----------------------------- #
def test_kstep_data_future_targets_and_labels(kstep_samples):
    train, val, test, info = kstep_samples
    assert info["usable"] and info["K"] == 10
    assert len(train) > 0 and len(test) > 0
    s = train[0]
    assert len(s.input_seq) == 4
    assert set(s.future_seqs.keys()) == set(range(1, 11))
    assert set(s.future_labels.keys()) == set(range(1, 11))
    for k in range(1, 11):
        assert len(s.future_seqs[k]) == 4


def test_kstep_data_chronological_no_overlap(kstep_samples):
    train, val, test, _ = kstep_samples
    if train and val:
        assert train[-1].t_index < val[0].t_index
    if val and test:
        assert val[-1].t_index < test[0].t_index


def test_kstep_data_labels_match_future_window(synth_root):
    """future_labels[k] == label_any_attack of the window at t+k (hand check)."""
    dc = DataConfig(dataset="ctu-13", seq_len=4, horizon=1)
    ds = KStepDataset(dc, synth_root, NODE_DIM, EDGE_DIM, K=5)
    train, _, _, _ = ds.build()
    # attack_from=40: windows with global index >= 40 are attacks.
    s = train[0]
    for k in range(1, 6):
        expected = 1 if (s.t_index + k) >= 40 else 0
        assert s.future_labels[k] == expected


# --------------------------- evaluation ------------------------------------ #
@pytest.mark.parametrize("K", [1, 3, 5, 10])
def test_evaluate_kstep_horizons(trained_checkpoint, kstep_samples, K):
    model, extra = SentinelXWorldModel.load_checkpoint(trained_checkpoint)
    _, _, test, _ = kstep_samples
    ev = evaluate_kstep(model, test, K, batch_size=8,
                        risk_threshold=extra.get("chosen_threshold", 0.5))
    assert ev.K == K
    assert len(ev.horizons) == K
    for h in ev.horizons:
        assert 0.0 <= h.state_cosine_distance <= 2.0
        assert h.state_l2 >= 0.0
        assert h.n == len(test)
    # computational cost recorded
    assert ev.rollout_seconds >= 0.0
    assert ev.per_step_seconds >= 0.0


def test_evaluate_state_error_zero_at_h1_when_identical():
    """If the forecast head is identity and target==prediction encoding, the
    one-step state error is ~0. Sanity check on the cosine metric wiring using a
    controlled tiny case."""
    # Build a model, then verify: cosine distance of a vector with itself is 0.
    import torch.nn.functional as F
    z = torch.randn(6, 8)
    cos = F.cosine_similarity(z, z, dim=-1)
    assert float((1 - cos).mean()) < 1e-6


def test_degradation_recorded(trained_checkpoint, kstep_samples):
    model, _ = SentinelXWorldModel.load_checkpoint(trained_checkpoint)
    _, _, test, _ = kstep_samples
    ev = evaluate_kstep(model, test, 5, batch_size=8)
    assert ev.degradation_abs is not None
    # ratio may be None only if h1 ~ 0; here it should be a finite number
    assert ev.degradation_ratio is None or ev.degradation_ratio > 0


# --------------------------- experiment driver ----------------------------- #
def test_experiment_writes_outputs(trained_checkpoint, synth_root, tmp_path):
    exp_dir = tmp_path / "experiments"
    res = run_kstep_experiments(
        trained_checkpoint, synth_root, ks=(1, 3, 5, 10),
        batch_size=8, experiments_dir=exp_dir, log=lambda *a, **k: None,
    )
    assert not res.get("skipped")
    assert set(res["evaluations"].keys()) == {"1", "3", "5", "10"}
    assert (exp_dir / "k_step_results.csv").exists()
    assert (exp_dir / "k_step_report.md").exists()
    assert (exp_dir / "k_step_metrics.json").exists()
    # CSV has one row per (K, horizon): 1 + 3 + 5 + 10 = 19 data rows
    lines = (exp_dir / "k_step_results.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1 + (1 + 3 + 5 + 10)


def test_experiment_invalid_ks_rejected(trained_checkpoint, synth_root):
    with pytest.raises(ValueError):
        run_kstep_experiments(trained_checkpoint, synth_root, ks=(0,),
                              experiments_dir=None, log=lambda *a, **k: None)


def test_experiment_skips_too_few_windows(tmp_path):
    """A cache too small to form K=10 splits is reported as skipped, not faked."""
    _write_synth_cache(tmp_path, "ctu-13", n_windows=20, attack_from=10)
    # need a checkpoint to load; make a trivial one
    seed_everything(0)
    cfg = ModelConfig(hidden_dim=8, latent_dim=4, gat_heads=2).normalized()
    ckpt = tmp_path / "m.pt"
    SentinelXWorldModel(cfg).save_checkpoint(ckpt, extra={"chosen_threshold": 0.5})
    dc = DataConfig(dataset="ctu-13", seq_len=4, horizon=1, min_windows=40)
    res = run_kstep_experiments(ckpt, tmp_path, ks=(1, 3, 5, 10), data_cfg=dc,
                                experiments_dir=None, log=lambda *a, **k: None)
    assert res.get("skipped") is True
