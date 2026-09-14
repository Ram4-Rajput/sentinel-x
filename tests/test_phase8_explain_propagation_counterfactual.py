"""Phase-8 tests: explainability, propagation, counterfactual, stability.

Covers the spec's test matrix:
  * explainability output (contract + determinism + attention discipline)
  * node/edge attribution (ranked, magnitudes non-negative, sensible)
  * propagation calculations (affected/newly/velocity/direction/branching/
    persistence/growth) with hand-derived expectations
  * graph intervention (clone isolation — real inputs never mutated)
  * forecast rerun + baseline/intervention comparison
  * stability (documented metric; deterministic; NOT MC-Dropout)

A tiny synthetic Phase-2 cache (deterministic) is reused — no real data files.
Expectations for the pure graph-analytics (propagation) are hand-derived; the
model-dependent parts assert structural invariants rather than exact values.
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
from sentinelx.worldmodel.model import GraphTensors, SentinelXWorldModel
from sentinelx.worldmodel.kstep_data import KStepDataset

from sentinelx.worldmodel.explain import (
    ATTENTION_DISCLAIMER, Explanation, explain_forecast,
)
from sentinelx.worldmodel.propagation import (
    SUSPICION_LABEL, SUSPICION_MODEL, analyze_propagation,
)
from sentinelx.worldmodel.counterfactual import (
    ISOLATE_NODE, REMOVE_EDGE, SIMULATION_LABEL, SUPPRESS_PATH,
    apply_intervention, clone_sequence, evaluate_stability, isolate_node,
    remove_edge, simulate_intervention, suppress_path,
)
from sentinelx.worldmodel.phase8_experiment import run_phase8_experiment


NODE_DIM = 5
EDGE_DIM = 3


# --------------------------------------------------------------------------- #
# Synthetic cache + trained checkpoint fixtures (mirror Phase-7 tests).
# --------------------------------------------------------------------------- #
def _write_synth_cache(root: Path, dataset: str, n_windows: int, attack_from: int,
                       with_nodes: bool = True, seed: int = 0):
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


def _build_model(gnn_type="graphsage"):
    cfg = ModelConfig(hidden_dim=16, latent_dim=8, gat_heads=2,
                      gnn_type=gnn_type, temporal_type="lstm").normalized()
    return SentinelXWorldModel(cfg)


@pytest.fixture
def trained_checkpoint(synth_root, tmp_path):
    from sentinelx.worldmodel.data import WorldModelDataset
    from sentinelx.worldmodel.train import train_world_model
    dc = DataConfig(dataset="ctu-13", seq_len=4, horizon=1)
    seed_everything(42)
    ds = WorldModelDataset(dc, synth_root, NODE_DIM, EDGE_DIM)
    train, val, test, info = ds.build()
    model = _build_model("graphsage")
    tcfg = TrainingConfig(epochs=2, batch_size=8, seed=42)
    ckpt = tmp_path / "sentinel_x" / "model.pt"
    train_world_model(model, train, val, test, tcfg, checkpoint_path=ckpt)
    dump_config_yaml(WorldModelConfig(model=model.cfg, training=tcfg, data=dc),
                     ckpt.parent / "config.yaml")
    return ckpt


def _sample_input_seq(synth_root, gnn_type="graphsage"):
    """Grab one real KStepSample input_seq from the synthetic cache."""
    dc = DataConfig(dataset="ctu-13", seq_len=4, horizon=1)
    ds = KStepDataset(dc, synth_root, NODE_DIM, EDGE_DIM, K=3)
    train, val, test, info = ds.build()
    assert info["usable"]
    return test[0].input_seq


# --------------------------------------------------------------------------- #
# Hand-built graph sequences for pure-analytics tests.
# --------------------------------------------------------------------------- #
def _mk_graph(node_feats, edges, edge_feats, node_ids=None):
    n = len(node_feats)
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges \
        else torch.empty((2, 0), dtype=torch.long)
    ea = torch.tensor(edge_feats, dtype=torch.float32) if edge_feats \
        else torch.empty((0, EDGE_DIM), dtype=torch.float32)
    x = torch.tensor(node_feats, dtype=torch.float32) if n \
        else torch.empty((0, NODE_DIM), dtype=torch.float32)
    return GraphTensors(x=x, edge_index=ei, edge_attr=ea, num_nodes=n)


def _mk_window(n_nodes, edges, attack_flags, node_ids=None, window_index=0):
    """Build a cached-style window dict. attack_flags: per-edge 0/1."""
    node_features = [[1.0, 0.0, 100.0, 0.0, 2.0] for _ in range(n_nodes)]
    edge_features = [[1.0, 150.0, float(f)] for f in attack_flags]
    return {
        "window_index": window_index,
        "num_nodes": n_nodes,
        "num_edges": len(edges),
        "node_ids": node_ids or [f"n{i}" for i in range(n_nodes)],
        "node_features": node_features,
        "edge_index": [list(e) for e in edges],
        "edge_features": edge_features,
    }


# =========================================================================== #
# 1. EXPLAINABILITY
# =========================================================================== #
def test_explanation_contract_keys(synth_root):
    model = _build_model("graphsage")
    seq = _sample_input_seq(synth_root)
    expl = explain_forecast(model, seq, target="state")
    assert isinstance(expl, Explanation)
    d = expl.as_dict()
    for key in ("top_features", "important_nodes", "important_edges",
                "temporal_evidence"):
        assert key in d
    # temporal evidence has exactly one entry per observed timestep
    assert len(d["temporal_evidence"]) == len(seq)


def test_explanation_is_deterministic(synth_root):
    model = _build_model("graphsage")
    seq = _sample_input_seq(synth_root)
    a = explain_forecast(model, seq, target="state").as_dict()
    b = explain_forecast(model, seq, target="state").as_dict()
    assert a["target_value"] == b["target_value"]
    assert a["top_features"] == b["top_features"]


def test_explanation_does_not_mutate_inputs(synth_root):
    model = _build_model("graphsage")
    seq = _sample_input_seq(synth_root)
    before = [g.x.clone() for g in seq]
    explain_forecast(model, seq, target="state")
    for g, b in zip(seq, before):
        assert torch.equal(g.x, b)
        assert g.x.grad is None  # caller tensors never accrue grad


def test_node_edge_attribution_ranked_and_nonneg(synth_root):
    model = _build_model("graphsage")
    seq = _sample_input_seq(synth_root)
    expl = explain_forecast(model, seq, target="state")
    # feature attributions ranked by |attribution| descending
    mags = [f.abs_attribution for f in expl.top_features]
    assert mags == sorted(mags, reverse=True)
    assert all(m >= 0 for m in mags)
    # node importances ranked descending, non-negative
    nimp = [n.importance for n in expl.important_nodes]
    assert nimp == sorted(nimp, reverse=True)
    assert all(v >= 0 for v in nimp)
    # edge importances ranked descending, non-negative
    eimp = [e.importance for e in expl.important_edges]
    assert eimp == sorted(eimp, reverse=True)
    assert all(v >= 0 for v in eimp)


def test_graphsage_has_no_attention(synth_root):
    """GraphSAGE has no attention channel; explanation must say so."""
    model = _build_model("graphsage")
    seq = _sample_input_seq(synth_root)
    expl = explain_forecast(model, seq, target="state")
    assert expl.uses_attention is False
    assert expl.attention_note == ""
    for e in expl.important_edges:
        assert e.attention is None


def test_gat_exposes_attention_but_not_causal(synth_root):
    """GAT exposes attention as SUPPORTING evidence, flagged non-causal."""
    seed_everything(0)
    model = _build_model("gat")
    seq = _sample_input_seq(synth_root)
    expl = explain_forecast(model, seq, target="state")
    assert expl.uses_attention is True
    # discipline: attention is explicitly documented as non-causal
    assert expl.attention_note == ATTENTION_DISCLAIMER
    assert "not" in expl.attention_note.lower()
    assert "causal" in expl.attention_note.lower()


def test_explain_risk_target(synth_root):
    model = _build_model("graphsage")
    seq = _sample_input_seq(synth_root)
    expl = explain_forecast(model, seq, target="risk")
    assert expl.target == "risk"
    assert isinstance(expl.target_value, float)


def test_explain_rejects_bad_target(synth_root):
    model = _build_model("graphsage")
    seq = _sample_input_seq(synth_root)
    with pytest.raises(ValueError):
        explain_forecast(model, seq, target="nonsense")


# =========================================================================== #
# 2. PROPAGATION (hand-derived expectations)
# =========================================================================== #
def test_propagation_affected_and_newly_affected():
    # w0: edge 0->1 attack        -> affected {n0,n1}, newly {n0,n1}
    # w1: edge 1->2 attack        -> affected {n1,n2}, newly {n2}
    # w2: edge 0->1 NOT attack    -> affected {}, newly {}
    windows = [
        _mk_window(3, [(0, 1), (1, 2)], [1, 0], window_index=0),
        _mk_window(3, [(1, 2), (0, 1)], [1, 0], window_index=1),
        _mk_window(3, [(0, 1)], [0], window_index=2),
    ]
    a = analyze_propagation(windows, suspicion_source=SUSPICION_LABEL)
    pw = a.per_window
    assert [len(p.affected) for p in pw] == [2, 2, 0]
    assert set(pw[0].affected) == {"n0", "n1"}
    assert set(pw[0].newly_affected) == {"n0", "n1"}
    assert set(pw[1].affected) == {"n1", "n2"}
    assert set(pw[1].newly_affected) == {"n2"}         # n1 already seen
    assert len(pw[2].affected) == 0
    # cumulative affected across all windows = {n0,n1,n2}
    assert a.total_affected == 3
    # velocity = mean newly-affected per window = (2 + 1 + 0)/3
    assert abs(a.velocity - 1.0) < 1e-9
    assert a.peak_velocity == 2


def test_propagation_growth_and_spreading():
    # Monotonically growing suspicious set -> positive growth, spreading True.
    windows = [
        _mk_window(4, [(0, 1)], [1], window_index=0),
        _mk_window(4, [(1, 2)], [1], window_index=1),
        _mk_window(4, [(2, 3)], [1], window_index=2),
    ]
    a = analyze_propagation(windows, suspicion_source=SUSPICION_LABEL)
    assert a.total_affected == 4
    assert a.growth > 0.0
    assert a.is_spreading is True


def test_propagation_no_suspicious_edges_is_quiet():
    windows = [
        _mk_window(3, [(0, 1), (1, 2)], [0, 0], window_index=0),
        _mk_window(3, [(0, 1)], [0], window_index=1),
    ]
    a = analyze_propagation(windows, suspicion_source=SUSPICION_LABEL)
    assert a.total_affected == 0
    assert a.velocity == 0.0
    assert a.is_spreading is False
    assert a.direction == "none"


def test_propagation_direction_fan_out():
    # One source (n0) reaching many distinct sinks -> fan-out.
    windows = [
        _mk_window(4, [(0, 1), (0, 2), (0, 3)], [1, 1, 1], window_index=0),
    ]
    a = analyze_propagation(windows, suspicion_source=SUSPICION_LABEL)
    # sources={n0}, sinks={n1,n2,n3} -> ratio = sinks/(sources+sinks) = 3/4
    # -> 0.75 > 0.55 -> fan-out (one host reaching many destinations).
    assert abs(a.direction_ratio - 0.75) < 1e-9
    assert a.direction == "fan-out"


def test_propagation_branching_mean_outdegree():
    # n0 originates 2 flagged edges in one window -> branching = 2.0
    windows = [_mk_window(3, [(0, 1), (0, 2)], [1, 1], window_index=0)]
    a = analyze_propagation(windows, suspicion_source=SUSPICION_LABEL)
    assert abs(a.per_window[0].branching - 2.0) < 1e-9
    assert abs(a.mean_branching - 2.0) < 1e-9


def test_propagation_persistence():
    # n0,n1 affected in BOTH windows -> persistence for them = 1.0.
    windows = [
        _mk_window(2, [(0, 1)], [1], window_index=0),
        _mk_window(2, [(0, 1)], [1], window_index=1),
    ]
    a = analyze_propagation(windows, suspicion_source=SUSPICION_LABEL)
    assert abs(a.persistence - 1.0) < 1e-9


def test_propagation_model_source():
    # Same graph, but model flags a different edge than the label.
    windows = [_mk_window(3, [(0, 1), (1, 2)], [0, 0], window_index=0)]
    a = analyze_propagation(
        windows, suspicion_source=SUSPICION_MODEL,
        model_suspicious_edges=[{(1, 2)}])
    assert set(a.per_window[0].affected) == {"n1", "n2"}
    assert a.total_affected == 2


def test_propagation_empty_is_valid():
    a = analyze_propagation([], suspicion_source=SUSPICION_LABEL)
    assert a.total_affected == 0
    assert a.per_window == []
    assert a.direction == "none"


def test_propagation_rejects_bad_source():
    with pytest.raises(ValueError):
        analyze_propagation([_mk_window(2, [(0, 1)], [1])], suspicion_source="bogus")


# =========================================================================== #
# 3. COUNTERFACTUAL — graph intervention + rerun + comparison
# =========================================================================== #
def test_clone_sequence_is_independent():
    seq = [_mk_graph([[1, 0, 100, 0, 2], [0, 1, 0, 50, 2]], [[0, 1]], [[1, 200, 1]])]
    clone = clone_sequence(seq)
    clone[0].x[0, 0] = 999.0
    assert seq[0].x[0, 0] == 1.0    # original untouched


def test_isolate_node_removes_incident_edges():
    seq = [_mk_graph(
        [[1, 0, 100, 0, 2], [0, 1, 0, 50, 2], [1, 1, 10, 10, 2]],
        [[0, 1], [1, 2]], [[1, 200, 1], [1, 200, 1]])]
    clone = clone_sequence(seq)
    out = isolate_node(clone, node_index=1)
    # both edges touch node 1 -> both removed
    assert out[0].edge_index.shape[1] == 0
    # node 1 features zeroed
    assert torch.all(out[0].x[1] == 0.0)
    # original sequence untouched
    assert seq[0].edge_index.shape[1] == 2


def test_remove_edge_only_targets_that_edge():
    seq = [_mk_graph(
        [[1, 0, 100, 0, 2], [0, 1, 0, 50, 2], [1, 1, 10, 10, 2]],
        [[0, 1], [1, 2]], [[1, 200, 1], [1, 200, 1]])]
    clone = clone_sequence(seq)
    out = remove_edge(clone, src=0, dst=1)
    assert out[0].edge_index.shape[1] == 1
    assert out[0].edge_index[:, 0].tolist() == [1, 2]   # only 1->2 remains


def test_suppress_path_removes_chain():
    seq = [_mk_graph(
        [[1, 0, 100, 0, 2]] * 3,
        [[0, 1], [1, 2], [2, 0]], [[1, 200, 1]] * 3)]
    clone = clone_sequence(seq)
    out = suppress_path(clone, [0, 1, 2])   # removes 0->1 and 1->2
    remaining = out[0].edge_index.t().tolist()
    assert [0, 1] not in remaining
    assert [1, 2] not in remaining
    assert [2, 0] in remaining              # reverse/other edges kept


def test_apply_intervention_dispatch_and_validation():
    seq = [_mk_graph([[1, 0, 100, 0, 2], [0, 1, 0, 50, 2]], [[0, 1]], [[1, 200, 1]])]
    with pytest.raises(ValueError):
        apply_intervention(clone_sequence(seq), "teleport")


def test_counterfactual_compares_baseline_vs_intervention(synth_root):
    model = _build_model("graphsage")
    seq = _sample_input_seq(synth_root)
    res = simulate_intervention(model, seq, ISOLATE_NODE, K=3, node_index=0)
    d = res.as_dict()
    assert d["label"] == SIMULATION_LABEL
    assert len(d["baseline_risk"]) == 3
    assert len(d["intervention_risk"]) == 3
    assert len(d["risk_delta"]) == 3
    assert len(d["latent_shift"]) == 3
    # risk_delta == intervention - baseline (per horizon)
    for k in range(3):
        assert abs(d["risk_delta"][k]
                   - (d["intervention_risk"][k] - d["baseline_risk"][k])) < 1e-6


def test_counterfactual_does_not_mutate_real_inputs(synth_root):
    model = _build_model("graphsage")
    seq = _sample_input_seq(synth_root)
    before_edges = [g.edge_index.shape[1] for g in seq]
    simulate_intervention(model, seq, ISOLATE_NODE, K=2, node_index=0)
    after_edges = [g.edge_index.shape[1] for g in seq]
    assert before_edges == after_edges      # real graph never modified


def test_counterfactual_intervention_changes_forecast(synth_root):
    """Removing an edge should generally move the modelled forecast (some
    horizon shows a non-zero latent shift). This is a modelled effect."""
    model = _build_model("graphsage")
    seq = _sample_input_seq(synth_root)
    res = simulate_intervention(model, seq, REMOVE_EDGE, K=3, src=0, dst=1)
    assert any(s > 0 for s in res.latent_shift)


def test_counterfactual_rejects_unknown_intervention(synth_root):
    model = _build_model("graphsage")
    seq = _sample_input_seq(synth_root)
    with pytest.raises(ValueError):
        simulate_intervention(model, seq, "nuke", K=2)


# =========================================================================== #
# 4. STABILITY (documented metric; deterministic; NOT MC-Dropout)
# =========================================================================== #
def test_stability_contract_and_ranges(synth_root):
    model = _build_model("graphsage")
    seq = _sample_input_seq(synth_root)
    res = evaluate_stability(model, seq, K=3, n_trials=8, epsilon=0.05, seed=1)
    d = res.as_dict()
    assert 0.0 < d["stability_score"] <= 1.0
    assert d["mean_latent_drift"] >= 0.0
    assert d["mean_risk_drift"] >= 0.0
    assert len(d["per_horizon_risk_drift"]) == 3
    # explicitly documented as NOT MC-Dropout
    assert "MC-Dropout" in d["note"]


def test_stability_zero_epsilon_is_perfectly_stable(synth_root):
    model = _build_model("graphsage")
    seq = _sample_input_seq(synth_root)
    res = evaluate_stability(model, seq, K=3, n_trials=4, epsilon=0.0, seed=1)
    # no perturbation -> no drift -> score exactly 1.0
    assert res.mean_latent_drift == pytest.approx(0.0, abs=1e-6)
    assert res.stability_score == pytest.approx(1.0, abs=1e-6)


def test_stability_larger_epsilon_not_more_stable(synth_root):
    """Bigger valid perturbations should not INCREASE stability (monotone-ish).
    We assert the small-epsilon run is at least as stable as the large one."""
    model = _build_model("graphsage")
    seq = _sample_input_seq(synth_root)
    small = evaluate_stability(model, seq, K=3, n_trials=8, epsilon=0.02, seed=7)
    large = evaluate_stability(model, seq, K=3, n_trials=8, epsilon=0.20, seed=7)
    assert small.stability_score >= large.stability_score - 1e-6


def test_stability_is_deterministic(synth_root):
    model = _build_model("graphsage")
    seq = _sample_input_seq(synth_root)
    a = evaluate_stability(model, seq, K=3, n_trials=8, epsilon=0.05, seed=123)
    b = evaluate_stability(model, seq, K=3, n_trials=8, epsilon=0.05, seed=123)
    assert a.mean_latent_drift == pytest.approx(b.mean_latent_drift, abs=1e-9)
    assert a.stability_score == pytest.approx(b.stability_score, abs=1e-9)


def test_stability_rejects_bad_params(synth_root):
    model = _build_model("graphsage")
    seq = _sample_input_seq(synth_root)
    with pytest.raises(ValueError):
        evaluate_stability(model, seq, K=3, n_trials=0)
    with pytest.raises(ValueError):
        evaluate_stability(model, seq, K=3, epsilon=-0.1)


# =========================================================================== #
# END-TO-END EXPERIMENT
# =========================================================================== #
def test_phase8_experiment_writes_outputs(trained_checkpoint, synth_root, tmp_path):
    exp_dir = tmp_path / "experiments"
    res = run_phase8_experiment(
        trained_checkpoint, synth_root, K=3, max_anchors=4,
        stability_trials=4, stability_epsilon=0.05,
        experiments_dir=exp_dir, log=lambda *a, **k: None,
    )
    assert not res.get("skipped")
    assert (exp_dir / "phase8_results.csv").exists()
    assert (exp_dir / "phase8_report.md").exists()
    assert (exp_dir / "phase8_metrics.json").exists()
    assert res["summary"]["n_anchors"] > 0
    assert res["simulation_label"] == SIMULATION_LABEL
    for a in res["anchors"]:
        expl = a["explanation"]
        assert 0.0 <= a["stability"]["stability_score"] <= 1.0
        # every anchor produces the four explainability channels
        for key in ("top_features", "important_nodes", "important_edges",
                    "temporal_evidence"):
            assert key in expl
        if a["counterfactual"] is not None:
            assert a["counterfactual"]["label"] == SIMULATION_LABEL


def test_phase8_experiment_skips_too_few_windows(tmp_path):
    _write_synth_cache(tmp_path, "ctu-13", n_windows=20, attack_from=10)
    seed_everything(0)
    model = _build_model("graphsage")
    ckpt = tmp_path / "m.pt"
    model.save_checkpoint(ckpt, extra={"chosen_threshold": 0.5})
    dc = DataConfig(dataset="ctu-13", seq_len=4, horizon=1, min_windows=40)
    res = run_phase8_experiment(ckpt, tmp_path, K=3, data_cfg=dc,
                                experiments_dir=None, log=lambda *a, **k: None)
    assert res.get("skipped") is True
