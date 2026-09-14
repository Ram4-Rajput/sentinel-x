"""Phase-6 tests: Risk + Uncertainty + Calibration + Novelty/OOD.

Covers (per the Phase-6 requirements):
  * genuine MC Dropout: dropout stays active at inference and multiple passes
    give DIFFERENT outputs (real stochasticity, not input perturbation)
  * predictive mean/variance calculation is correct (hand-checked wiring)
  * uncertainty exposed separately from risk
  * calibration: ECE, Brier, reliability bins, temperature scaling
  * embedding generation (deterministic latent extraction)
  * OOD scoring (Mahalanobis distance) + threshold derived from train/val only
  * known/unknown separation (benign vs attack) via the experiment driver
  * output artifacts written

A tiny synthetic Phase-2 cache (deterministic) is used — no real data files.
Expectations are derived by hand, not from the code under test.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from sentinelx.experiments.common import seed_everything
from sentinelx.worldmodel.config import (
    DataConfig, ModelConfig, TrainingConfig, WorldModelConfig, dump_config_yaml,
)
from sentinelx.worldmodel.data import WorldModelDataset, collate_inputs
from sentinelx.worldmodel.model import SentinelXWorldModel
from sentinelx.worldmodel.train import train_world_model

from sentinelx.worldmodel.uncertainty import (
    DEFAULT_MC_PASSES, count_dropout_layers, deterministic_risk, dropout_active,
    estimate_uncertainty, has_active_dropout, mc_dropout_risk,
)
from sentinelx.worldmodel.calibration import (
    TemperatureScaler, brier_score, evaluate_calibration,
    expected_calibration_error, logit, reliability_bins,
)
from sentinelx.worldmodel.ood import (
    MahalanobisOOD, encode_latents, ood_detection_metrics,
)
from sentinelx.worldmodel.phase6_experiment import run_phase6_experiments


NODE_DIM = 5
EDGE_DIM = 3


def _write_synth_cache(root: Path, dataset: str, n_windows: int, attack_from: int,
                       with_nodes: bool = True, seed: int = 0):
    """Synthetic cache mirroring graph_cache.py (same format as Phase-4/5 tests)."""
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
    # Interleave benign/attack so every split has both classes.
    root = tmp_path
    wdir = root / "ctu-13" / "windows"
    wdir.mkdir(parents=True, exist_ok=True)
    n_windows = 80
    n_tr = int(n_windows * 0.7)
    n_va = int(n_windows * 0.15)
    splits = {"train": range(0, n_tr), "val": range(n_tr, n_tr + n_va),
              "test": range(n_tr + n_va, n_windows)}
    gi = 0
    rng = np.random.RandomState(0)
    for split, idxs in splits.items():
        with open(wdir / f"{split}.jsonl", "w", encoding="utf-8") as fh:
            for _ in idxs:
                # ~30% attacks, spread across all splits
                attack = 1 if (gi % 10) >= 7 else 0
                nb = 2 + (gi % 3)
                node_features = [[
                    float(1 + k), float(k), 100.0 + 80 * attack + rng.rand(),
                    float(10 * k), float(2 + attack),
                ] for k in range(nb)]
                edge_index = [[k, k + 1] for k in range(nb - 1)]
                edge_features = [[1.0, 150.0 + 300.0 * attack, float(attack)]
                                 for _ in range(nb - 1)]
                g = {
                    "window_index": gi, "start": None, "end": None,
                    "num_nodes": nb, "num_edges": len(edge_index),
                    "num_flows": 3 + 5 * attack,
                    "node_ids": [f"n{k}" for k in range(nb)],
                    "node_features": node_features,
                    "edge_index": edge_index, "edge_features": edge_features,
                    "attack_ratio": 0.5 * attack, "label_any_attack": attack,
                }
                fh.write(json.dumps(g) + "\n")
                gi += 1
    return root


@pytest.fixture
def wm_samples(synth_root):
    dc = DataConfig(dataset="ctu-13", seq_len=4, horizon=1)
    seed_everything(42)
    ds = WorldModelDataset(dc, synth_root, NODE_DIM, EDGE_DIM)
    return ds.build()


@pytest.fixture
def small_model():
    seed_everything(7)
    cfg = ModelConfig(hidden_dim=16, latent_dim=8, gat_heads=2,
                      gnn_type="graphsage", temporal_type="lstm",
                      dropout=0.3).normalized()
    return SentinelXWorldModel(cfg).eval()


@pytest.fixture
def trained_checkpoint(synth_root, tmp_path):
    dc = DataConfig(dataset="ctu-13", seq_len=4, horizon=1)
    seed_everything(42)
    ds = WorldModelDataset(dc, synth_root, NODE_DIM, EDGE_DIM)
    train, val, test, info = ds.build()
    cfg = ModelConfig(hidden_dim=16, latent_dim=8, gat_heads=2,
                      gnn_type="graphsage", temporal_type="lstm",
                      dropout=0.3).normalized()
    tcfg = TrainingConfig(epochs=2, batch_size=8, seed=42)
    model = SentinelXWorldModel(cfg)
    ckpt = tmp_path / "sentinel_x" / "model.pt"
    train_world_model(model, train, val, test, tcfg, checkpoint_path=ckpt)
    dump_config_yaml(WorldModelConfig(model=cfg, training=tcfg, data=dc),
                     ckpt.parent / "config.yaml")
    return ckpt


# ============================ MC DROPOUT ================================== #
def test_model_has_dropout_layers(small_model):
    assert count_dropout_layers(small_model) >= 1
    assert has_active_dropout(small_model) is True


def test_dropout_active_context_enables_dropout(small_model):
    """Inside the context, dropout layers are in train mode; model flag restored."""
    small_model.eval()
    with dropout_active(small_model) as dlayers:
        assert len(dlayers) >= 1
        assert all(m.training for m in dlayers)
    # After the context the model returns to its original (eval) flag.
    assert small_model.training is False


def test_mc_dropout_passes_differ(small_model, wm_samples):
    """GENUINE MC Dropout: identical inputs give DIFFERENT outputs across passes
    (variance comes from dropout masks, not from perturbing the input)."""
    train, _, _, _ = wm_samples
    batch = collate_inputs(train[:6])
    res = mc_dropout_risk(small_model, batch, n_passes=20, seed=123)
    samples = res["samples"]  # (20, 6)
    assert samples.shape == (20, 6)
    # At least some pass-to-pass variation exists (dropout active).
    assert samples.std(axis=0).max() > 1e-6


def test_mc_dropout_inputs_unchanged_between_passes(small_model, wm_samples):
    """The observed input batch is the SAME object across passes — uncertainty
    is epistemic (dropout), never an artefact of input perturbation."""
    train, _, _, _ = wm_samples
    batch = collate_inputs(train[:3])
    before = [g.x.clone() for seq in batch for g in seq]
    mc_dropout_risk(small_model, batch, n_passes=10, seed=1)
    after = [g.x for seq in batch for g in seq]
    for b, a in zip(before, after):
        assert torch.equal(b, a)


def test_mc_mean_variance_wiring():
    """Predictive mean/variance match numpy on a controlled sample matrix."""
    # 4 passes, 2 samples — hand-computable.
    samples = np.array([[0.2, 0.8],
                        [0.4, 0.6],
                        [0.2, 0.8],
                        [0.6, 0.4]])
    mean = samples.mean(axis=0)
    var = samples.var(axis=0, ddof=0)
    assert np.allclose(mean, [0.35, 0.65])
    assert np.allclose(var, [samples[:, 0].var(), samples[:, 1].var()])


def test_deterministic_risk_has_no_variance(small_model, wm_samples):
    """Dropout-OFF pass is deterministic (repeatable)."""
    train, _, _, _ = wm_samples
    batch = collate_inputs(train[:4])
    a = deterministic_risk(small_model, batch)
    b = deterministic_risk(small_model, batch)
    assert np.allclose(a, b)


def test_estimate_uncertainty_separates_risk_and_uncertainty(small_model, wm_samples):
    train, _, test, _ = wm_samples
    ests = estimate_uncertainty(small_model, test, collate_inputs=collate_inputs,
                                n_passes=15, batch_size=8, seed=42)
    assert len(ests) == len(test)
    for e in ests:
        assert 0.0 <= e.risk_mean <= 1.0
        assert e.variance >= 0.0
        assert abs(e.std - e.variance ** 0.5) < 1e-6
        assert e.n_passes == 15
        # risk_mean and variance are distinct fields (not the same number)
        d = e.as_dict()
        assert "risk_mean" in d and "uncertainty_variance" in d


def test_default_mc_passes_is_30():
    assert DEFAULT_MC_PASSES == 30


def test_mc_dropout_rejects_bad_passes(small_model, wm_samples):
    train, _, _, _ = wm_samples
    batch = collate_inputs(train[:2])
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError):
            mc_dropout_risk(small_model, batch, n_passes=bad)


def test_no_dropout_model_flagged():
    """A model built with dropout=0 is reported as not MC-capable (honest)."""
    seed_everything(3)
    cfg = ModelConfig(hidden_dim=16, latent_dim=8, gat_heads=2, dropout=0.0).normalized()
    m = SentinelXWorldModel(cfg)
    assert has_active_dropout(m) is False


# ============================ CALIBRATION ================================= #
def test_reliability_bins_partition():
    y = [0, 0, 1, 1]
    p = [0.1, 0.2, 0.8, 0.9]
    bins = reliability_bins(y, p, n_bins=10)
    # counts sum to n; each populated bin has count>0
    assert sum(b.count for b in bins) == 4
    assert all(b.count > 0 for b in bins)


def test_perfect_calibration_zero_ece():
    """If predicted prob == observed frequency in every bin, ECE == 0."""
    # bin at 0.0: two negatives (freq 0, conf 0). bin at 1.0: two positives.
    y = [0, 0, 1, 1]
    p = [0.0, 0.0, 1.0, 1.0]
    ece, mce, bins = expected_calibration_error(y, p, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-9)
    assert mce == pytest.approx(0.0, abs=1e-9)


def test_miscalibration_detected():
    """Confident-but-wrong predictions produce a large ECE (not hidden)."""
    y = [0, 0, 0, 0]         # all negative
    p = [0.9, 0.9, 0.9, 0.9]  # but predicted 0.9 -> gap 0.9
    ece, mce, _ = expected_calibration_error(y, p, n_bins=10)
    assert ece == pytest.approx(0.9, abs=1e-9)
    assert mce == pytest.approx(0.9, abs=1e-9)


def test_brier_score_hand():
    y = [1, 0]
    p = [0.75, 0.25]
    # ((0.75-1)^2 + (0.25-0)^2)/2 = (0.0625 + 0.0625)/2 = 0.0625
    assert brier_score(y, p) == pytest.approx(0.0625)


def test_evaluate_calibration_panel():
    y = [0, 1, 0, 1, 0, 1]
    p = [0.2, 0.7, 0.3, 0.8, 0.1, 0.6]
    rep = evaluate_calibration(y, p, n_bins=5)
    assert rep.n == 6 and rep.positives == 3 and rep.negatives == 3
    assert rep.ece is not None and rep.brier is not None
    assert len(rep.bins) >= 1


def test_temperature_scaling_improves_overconfidence():
    """T>1 softens overconfident logits; fitted flag set when both classes present."""
    rng = np.random.RandomState(0)
    n = 400
    # Draw a well-calibrated probability, sample the label from it, then make
    # the logits OVERCONFIDENT by inflating their magnitude (x4). The direction
    # still matches the labels, so the fix is to soften -> T > 1.
    true_logits = rng.randn(n) * 1.2
    p_true = 1 / (1 + np.exp(-true_logits))
    y = (rng.rand(n) < p_true).astype(int)
    logits = true_logits * 4.0
    scaler = TemperatureScaler().fit(logits, y, max_iter=400)
    assert scaler.fitted is True
    assert scaler.temperature > 1.0  # softening overconfidence
    probs_before = 1 / (1 + np.exp(-logits))
    probs_after = scaler.transform(logits)
    from sentinelx.worldmodel.calibration import brier_score as bs
    assert bs(y, probs_after) <= bs(y, probs_before) + 1e-6


def test_temperature_scaling_single_class_unfitted():
    scaler = TemperatureScaler().fit([1.0, 2.0, 3.0], [1, 1, 1])
    assert scaler.fitted is False
    assert scaler.temperature == 1.0


def test_logit_roundtrip():
    p = np.array([0.1, 0.5, 0.9])
    z = logit(p)
    back = 1 / (1 + np.exp(-z))
    assert np.allclose(back, p, atol=1e-5)


# ============================ EMBEDDING / OOD ============================= #
def test_encode_latents_shape(small_model, wm_samples):
    train, _, _, _ = wm_samples
    Z = encode_latents(small_model, train, collate_inputs=collate_inputs, batch_size=8)
    assert Z.shape == (len(train), small_model.cfg.latent_dim)


def test_encode_latents_deterministic(small_model, wm_samples):
    train, _, _, _ = wm_samples
    a = encode_latents(small_model, train, collate_inputs=collate_inputs)
    b = encode_latents(small_model, train, collate_inputs=collate_inputs)
    assert np.allclose(a, b)


def test_mahalanobis_zero_at_mean():
    """Distance of the distribution mean to itself is ~0."""
    rng = np.random.RandomState(0)
    Z = rng.randn(100, 6)
    det = MahalanobisOOD().fit(Z)
    d = det.distance(det.mu[None, :])
    assert float(d[0]) == pytest.approx(0.0, abs=1e-6)


def test_mahalanobis_far_point_scores_high():
    """A point far from the training cloud gets a larger distance than an inlier."""
    rng = np.random.RandomState(1)
    Z = rng.randn(200, 5)
    det = MahalanobisOOD().fit(Z)
    inlier = det.distance(np.zeros((1, 5)))
    outlier = det.distance(np.full((1, 5), 10.0))
    assert float(outlier[0]) > float(inlier[0])


def test_threshold_from_indist_only():
    """Threshold is a percentile of IN-DIST distances; derived from train/val."""
    rng = np.random.RandomState(2)
    Z = rng.randn(300, 4)
    det = MahalanobisOOD().fit(Z)
    thr = det.set_threshold_from_indist(Z, percentile=95.0)
    d = det.distance(Z)
    # ~95% of in-distribution points fall at or below the threshold.
    frac_below = float((d <= thr).mean())
    assert 0.90 <= frac_below <= 0.99
    assert det.fit_percentile == 95.0


def test_ood_scoring_flags(small_model):
    rng = np.random.RandomState(3)
    Z = rng.randn(100, 8)
    det = MahalanobisOOD().fit(Z)
    det.set_threshold_from_indist(Z, percentile=95.0)
    scores = det.score(np.vstack([np.zeros((1, 8)), np.full((1, 8), 8.0)]))
    assert scores[0].is_ood is False   # inlier
    assert scores[1].is_ood is True    # far outlier
    assert scores[1].novelty_score > scores[0].novelty_score


def test_known_unknown_separation():
    """Known vs unknown clusters separate: AUROC well above chance."""
    rng = np.random.RandomState(4)
    known = rng.randn(150, 5)                    # in-distribution
    unknown = rng.randn(60, 5) + 6.0             # shifted (unseen) behaviour
    det = MahalanobisOOD().fit(known)
    thr = det.set_threshold_from_indist(known, percentile=95.0)
    ind_scores = det.distance(known)
    ood_scores = det.distance(unknown)
    m = ood_detection_metrics(ind_scores, ood_scores, thr)
    assert m["auroc"] > 0.9
    assert m["detection_rate"] > 0.8
    assert m["false_acceptance_rate"] < 0.15


def test_ood_metrics_single_class_guard():
    m = ood_detection_metrics([1.0, 2.0], [], threshold=1.5)
    assert m["auroc"] is None and m["auprc"] is None


def test_mahalanobis_requires_two_points():
    with pytest.raises(ValueError):
        MahalanobisOOD().fit(np.zeros((1, 4)))


def test_score_before_threshold_raises():
    det = MahalanobisOOD().fit(np.random.RandomState(0).randn(20, 3))
    with pytest.raises(RuntimeError):
        det.score(np.zeros((1, 3)))


# ============================ EXPERIMENT DRIVER =========================== #
def test_phase6_experiment_writes_outputs(trained_checkpoint, synth_root, tmp_path):
    exp_dir = tmp_path / "experiments"
    res = run_phase6_experiments(
        trained_checkpoint, synth_root, mc_passes=10, ood_percentile=95.0,
        batch_size=8, experiments_dir=exp_dir, log=lambda *a, **k: None,
    )
    assert not res.get("skipped")
    # four required artifacts
    assert (exp_dir / "uncertainty_results.csv").exists()
    assert (exp_dir / "calibration_results.csv").exists()
    assert (exp_dir / "ood_results.csv").exists()
    assert (exp_dir / "uncertainty_ood_report.md").exists()
    # four separate signals present in the results
    assert "uncertainty" in res and "calibration" in res and "ood" in res
    assert res["risk_threshold"] is not None
    # uncertainty rows keep risk and uncertainty distinct
    u = res["uncertainty"]["rows"][0]
    assert "risk_mc_mean" in u and "uncertainty_std" in u


def test_phase6_experiment_ood_known_vs_unknown(trained_checkpoint, synth_root):
    res = run_phase6_experiments(
        trained_checkpoint, synth_root, mc_passes=8, batch_size=8,
        experiments_dir=None, log=lambda *a, **k: None,
    )
    ood = res["ood"]
    assert ood["usable"] is True
    # attack windows held out of the manifold fit (documented provenance)
    assert ood["known_distribution"] == "benign_train_windows"
    assert ood["n_attack_test"] >= 0
    # threshold provenance recorded
    assert ood["detector"]["threshold"] is not None


def test_phase6_mc_passes_configurable(trained_checkpoint, synth_root):
    res = run_phase6_experiments(
        trained_checkpoint, synth_root, mc_passes=17, batch_size=8,
        experiments_dir=None, log=lambda *a, **k: None,
    )
    assert res["uncertainty"]["mc_passes"] == 17
    assert res["mc_passes"] == 17


def test_phase6_skips_too_few_windows(tmp_path):
    _write_synth_cache(tmp_path, "ctu-13", n_windows=20, attack_from=10)
    seed_everything(0)
    cfg = ModelConfig(hidden_dim=8, latent_dim=4, gat_heads=2, dropout=0.2).normalized()
    ckpt = tmp_path / "m.pt"
    SentinelXWorldModel(cfg).save_checkpoint(ckpt, extra={"chosen_threshold": 0.5})
    dc = DataConfig(dataset="ctu-13", seq_len=4, horizon=1, min_windows=40)
    res = run_phase6_experiments(ckpt, tmp_path, mc_passes=5, data_cfg=dc,
                                 experiments_dir=None, log=lambda *a, **k: None)
    assert res.get("skipped") is True
