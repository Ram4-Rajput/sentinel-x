"""Phase-7 tests: attack trajectory + high-level MITRE ATT&CK mapping.

Covers the spec's test matrix:
  * observed vs forecast distinction (a forecast is never marked observed)
  * temporal ordering (observed precedes forecast; horizons monotonic)
  * missing stages (empty observed / empty forecast handled gracefully)
  * uncertain mappings (weak evidence -> coarse stage, not a precise technique)
  * unsupported techniques (no precise ATT&CK technique IDs are ever emitted)
  * confidence handling (in [0,1]; observed=evidence, forecast=model risk)

A tiny synthetic Phase-2 cache (deterministic) is used — no real data files.
Expectations are hand-derived, not read from the code under test.
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
from sentinelx.worldmodel.kstep_data import KStepDataset
from sentinelx.worldmodel.trajectory import (
    AttackTrajectory, TrajectoryStage, MITRE_STAGES, STAGE_BENIGN, STAGE_C2,
    STAGE_LATERAL, STAGE_RECON, STAGE_SUSPICIOUS, STAGE_INITIAL_ACCESS,
    STATUS_FORECAST, STATUS_OBSERVED, build_forecast_stages,
    build_observed_stages, build_trajectory, map_behaviour_to_stage,
)
from sentinelx.worldmodel.trajectory_experiment import run_trajectory_experiment


NODE_DIM = 5
EDGE_DIM = 3


def _write_synth_cache(root: Path, dataset: str, n_windows: int, attack_from: int,
                       with_nodes: bool = True, seed: int = 0):
    """Synthetic cache mirroring graph_cache.py format (as in Phase-4/5 tests)."""
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
    _write_synth_cache(tmp_path, "ctu-13", n_windows=80, attack_from=40)
    return tmp_path


@pytest.fixture
def trained_checkpoint(synth_root, tmp_path):
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
    dump_config_yaml(WorldModelConfig(model=cfg, training=tcfg, data=dc),
                     ckpt.parent / "config.yaml")
    return ckpt


# ------------------------- behaviour -> stage mapping ---------------------- #
def _benign_signals():
    return {"attack_ratio": 0.0, "edge_attack_fraction": 0.0, "num_flows": 2.0,
            "num_nodes": 2.0, "num_edges": 1.0, "density": 0.5,
            "mean_node_out_degree": 1.0, "total_edge_bytes": 100.0,
            "label_any_attack": 0.0}


def _c2_signals():
    return {"attack_ratio": 0.6, "edge_attack_fraction": 0.6, "num_flows": 20.0,
            "num_nodes": 8.0, "num_edges": 20.0, "density": 0.4,
            "mean_node_out_degree": 3.0, "total_edge_bytes": 5e4,
            "label_any_attack": 1.0}


def test_benign_maps_to_benign_stage():
    m = map_behaviour_to_stage(_benign_signals(), risk=0.1)
    assert m["stage"] == STAGE_BENIGN
    assert 0.0 <= m["confidence"] <= 1.0


def test_strong_corroborated_signals_map_to_c2():
    m = map_behaviour_to_stage(_c2_signals(), risk=0.9)
    assert m["stage"] == STAGE_C2
    assert m["confidence"] > 0.5
    assert m["evidence"]  # non-empty evidence


def test_weak_signal_backs_off_to_coarse_stage():
    """Uncertain mapping: a faint, uncorroborated signal must NOT jump to a
    specific late-stage tactic. It should return a coarse stage."""
    faint = {"attack_ratio": 0.0, "edge_attack_fraction": 0.0, "num_flows": 3.0,
             "num_nodes": 6.0, "num_edges": 5.0, "density": 0.2,
             "mean_node_out_degree": 1.0, "total_edge_bytes": 500.0,
             "label_any_attack": 0.0}
    m = map_behaviour_to_stage(faint, risk=0.55)
    assert m["stage"] in (STAGE_RECON, STAGE_SUSPICIOUS)
    assert m["confidence"] <= 0.5  # coarse stages capped low


def test_mapping_never_returns_precise_technique():
    """Unsupported techniques: only high-level stages are ever emitted."""
    allowed = set(MITRE_STAGES) | {STAGE_BENIGN, STAGE_SUSPICIOUS}
    for sig, risk in [(_benign_signals(), 0.1), (_c2_signals(), 0.9)]:
        m = map_behaviour_to_stage(sig, risk=risk)
        assert m["stage"] in allowed
        # No ATT&CK technique IDs (e.g. "T1059") anywhere in the stage string.
        assert "T1" not in m["stage"]


def test_confidence_always_in_unit_interval():
    for risk in [None, 0.0, 0.3, 0.5, 0.8, 1.0]:
        for sig in [_benign_signals(), _c2_signals()]:
            m = map_behaviour_to_stage(sig, risk=risk)
            assert 0.0 <= m["confidence"] <= 1.0


# ------------------------- observed vs forecast ---------------------------- #
def _observed_windows(n=4, attack=False):
    ws = []
    for i in range(n):
        nb = 8 if attack else 2
        node_features = [[float(1 + k), float(k), 100.0, float(10 * k),
                          float(2 + int(attack))] for k in range(nb)]
        edge_index = [[k, k + 1] for k in range(nb - 1)]
        ef = [[2.0, 5e4 if attack else 100.0, 1.0 if attack else 0.0]
              for _ in range(nb - 1)]
        ws.append({
            "window_index": i, "start": f"2011-08-18T00:0{i}:00",
            "num_nodes": nb, "num_edges": len(edge_index),
            "num_flows": 20 if attack else 2,
            "node_features": node_features, "edge_index": edge_index,
            "edge_features": ef,
            "attack_ratio": 0.6 if attack else 0.0,
            "label_any_attack": 1 if attack else 0,
        })
    return ws


def test_observed_stages_are_all_observed():
    stages = build_observed_stages(_observed_windows(4, attack=True))
    assert len(stages) == 4
    assert all(s.status == STATUS_OBSERVED for s in stages)
    assert all(s.source == "observed-state" for s in stages)


def test_forecast_stages_are_all_forecast_never_observed():
    stages = build_forecast_stages([0.9, 0.8, 0.6], t_index=10)
    assert len(stages) == 3
    assert all(s.status == STATUS_FORECAST for s in stages)
    # A forecast must never be labelled observed.
    assert not any(s.status == STATUS_OBSERVED for s in stages)
    # horizons are t+1, t+2, t+3
    assert [s.horizon for s in stages] == [1, 2, 3]
    assert [s.window_index for s in stages] == [11, 12, 13]


def test_forecast_confidence_equals_model_risk():
    risks = [0.91, 0.42, 0.10]
    stages = build_forecast_stages(risks, t_index=5)
    for s, r in zip(stages, risks):
        assert abs(s.confidence - r) < 1e-6


# ------------------------- trajectory assembly ----------------------------- #
def test_build_trajectory_orders_observed_before_forecast():
    traj = build_trajectory(
        observed_windows=_observed_windows(4, attack=True),
        forecast_risks=[0.8, 0.7, 0.6, 0.5, 0.4],
        t_index=20,
    )
    statuses = [s.status for s in traj.stages]
    # every observed index must be before every forecast index
    first_forecast = statuses.index(STATUS_FORECAST)
    assert all(st == STATUS_OBSERVED for st in statuses[:first_forecast])
    assert all(st == STATUS_FORECAST for st in statuses[first_forecast:])
    assert len(traj.observed) == 4
    assert len(traj.forecast) == 5


def test_build_trajectory_horizons_monotonic():
    traj = build_trajectory(
        observed_windows=_observed_windows(4),
        forecast_risks=[0.6, 0.5, 0.4],
        t_index=15,
    )
    horizons = [s.horizon for s in traj.stages]
    assert horizons == sorted(horizons)
    # observed horizons are <= 0, forecast strictly > 0
    assert traj.observed[-1].horizon == 0
    assert all(s.horizon > 0 for s in traj.forecast)


def test_trajectory_does_not_claim_forecast_occurred():
    """Core invariant: forecast stages carry status='forecast' and a future
    (no observed timestamp), so they are never presented as accomplished."""
    traj = build_trajectory(
        observed_windows=_observed_windows(3, attack=True),
        forecast_risks=[0.95, 0.9],
        t_index=30,
    )
    for s in traj.forecast:
        assert s.status == STATUS_FORECAST
        assert s.timestamp is None       # future has no observed timestamp
        assert s.source == "forecast-rollout"


# ------------------------- missing stages ---------------------------------- #
def test_missing_forecast_segment():
    """K=0 forecast risks -> trajectory has only the observed segment."""
    traj = build_trajectory(_observed_windows(4), forecast_risks=[], t_index=8)
    assert len(traj.forecast) == 0
    assert len(traj.observed) == 4


def test_missing_observed_segment():
    """No observed windows -> only forecast stages, ordering still valid."""
    traj = build_trajectory([], forecast_risks=[0.7, 0.6], t_index=3)
    assert len(traj.observed) == 0
    assert len(traj.forecast) == 2
    assert all(s.status == STATUS_FORECAST for s in traj.stages)


def test_empty_trajectory_is_valid():
    traj = build_trajectory([], forecast_risks=[], t_index=0)
    assert traj.stages == []


# ------------------------- ordering guard ---------------------------------- #
def test_ordering_guard_rejects_forecast_before_observed():
    """The invariant assertion must fire if a trajectory is mis-assembled."""
    from sentinelx.worldmodel.trajectory import _assert_temporal_order
    bad = AttackTrajectory(t_index=0, stages=[
        TrajectoryStage(stage=STAGE_C2, status=STATUS_FORECAST, confidence=0.5,
                        horizon=1),
        TrajectoryStage(stage=STAGE_RECON, status=STATUS_OBSERVED, confidence=0.3,
                        horizon=0),
    ])
    with pytest.raises(ValueError):
        _assert_temporal_order(bad)


def test_ordering_guard_rejects_non_monotonic_horizon():
    from sentinelx.worldmodel.trajectory import _assert_temporal_order
    bad = AttackTrajectory(t_index=0, stages=[
        TrajectoryStage(stage=STAGE_RECON, status=STATUS_FORECAST, confidence=0.5,
                        horizon=3),
        TrajectoryStage(stage=STAGE_C2, status=STATUS_FORECAST, confidence=0.5,
                        horizon=1),
    ])
    with pytest.raises(ValueError):
        _assert_temporal_order(bad)


# ------------------------- stage dict contract ----------------------------- #
def test_stage_dict_matches_spec_contract():
    s = TrajectoryStage(stage=STAGE_LATERAL, status=STATUS_FORECAST,
                        confidence=0.73, evidence=["spread across hosts"],
                        horizon=2, window_index=42)
    d = s.as_dict()
    # spec requires at least these keys with these semantics
    assert d["stage"] == STAGE_LATERAL
    assert d["status"] in (STATUS_OBSERVED, STATUS_FORECAST)
    assert 0.0 <= d["confidence"] <= 1.0
    assert isinstance(d["evidence"], list)


# ------------------------- end-to-end experiment --------------------------- #
def test_experiment_writes_outputs(trained_checkpoint, synth_root, tmp_path):
    exp_dir = tmp_path / "experiments"
    res = run_trajectory_experiment(
        trained_checkpoint, synth_root, K=5, batch_size=8,
        experiments_dir=exp_dir, log=lambda *a, **k: None,
    )
    assert not res.get("skipped")
    assert (exp_dir / "trajectory_results.csv").exists()
    assert (exp_dir / "trajectory_report.md").exists()
    assert (exp_dir / "trajectory_metrics.json").exists()
    assert res["summary"]["n_trajectories"] > 0
    # every trajectory keeps observed<forecast and forecast never observed
    for traj in res["trajectories"]:
        statuses = [s["status"] for s in traj["stages"]]
        if STATUS_FORECAST in statuses:
            fi = statuses.index(STATUS_FORECAST)
            assert all(st == STATUS_OBSERVED for st in statuses[:fi])
        for s in traj["stages"]:
            assert 0.0 <= s["confidence"] <= 1.0


def test_experiment_forecast_stages_have_k_horizons(trained_checkpoint, synth_root, tmp_path):
    res = run_trajectory_experiment(
        trained_checkpoint, synth_root, K=3, max_trajectories=2, batch_size=8,
        experiments_dir=None, log=lambda *a, **k: None,
    )
    assert res["K"] == 3
    for traj in res["trajectories"]:
        forecast = [s for s in traj["stages"] if s["status"] == STATUS_FORECAST]
        assert len(forecast) == 3
        assert [s["horizon"] for s in forecast] == [1, 2, 3]


def test_experiment_skips_too_few_windows(tmp_path):
    _write_synth_cache(tmp_path, "ctu-13", n_windows=20, attack_from=10)
    seed_everything(0)
    cfg = ModelConfig(hidden_dim=8, latent_dim=4, gat_heads=2).normalized()
    ckpt = tmp_path / "m.pt"
    SentinelXWorldModel(cfg).save_checkpoint(ckpt, extra={"chosen_threshold": 0.5})
    dc = DataConfig(dataset="ctu-13", seq_len=4, horizon=1, min_windows=40)
    res = run_trajectory_experiment(ckpt, tmp_path, K=5, data_cfg=dc,
                                    experiments_dir=None, log=lambda *a, **k: None)
    assert res.get("skipped") is True
