"""Phase-10 API tests for the Sentinel-X serving layer.

These tests exercise the FastAPI app end-to-end against the REAL trained
checkpoint (models/sentinel_x/model.pt) and the REAL Phase-2 cache
(data/processed/ctu-13/). Nothing is retrained or rebuilt — the serving runtime
loads the existing artifacts exactly as production would.

Design of the tests:
  * A single shared runtime + TestClient (the model loads once, lazily).
  * If the checkpoint or a usable data cache is missing, the data-dependent
    tests skip cleanly rather than fail (the environment, not the code, is the
    gap) — but /health and schema/validation behaviour are always asserted.
  * Expectations (shapes, invariants, contracts) are derived by hand from the
    Phase-10 spec, never read back from the code under test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastapi.testclient import TestClient  # noqa: E402

from sentinelx.serving.app import create_app  # noqa: E402
from sentinelx.serving.runtime import SentinelRuntime, set_runtime  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def runtime() -> SentinelRuntime:
    rt = SentinelRuntime(K=5)
    set_runtime(rt)
    yield rt
    set_runtime(None)


@pytest.fixture(scope="module")
def client(runtime) -> TestClient:
    return TestClient(create_app())


@pytest.fixture(scope="module")
def data_ready(runtime) -> bool:
    """True when the checkpoint + a usable ctu-13 cache are present."""
    if not runtime.checkpoint_available:
        return False
    try:
        return bool(runtime.data_usable and runtime.samples.test)
    except Exception:
        return False


@pytest.fixture(scope="module")
def anchor_t(runtime, data_ready) -> int:
    if not data_ready:
        pytest.skip("no trained checkpoint / usable cache in this environment")
    latest = runtime.latest_sample()
    return int(latest.t_index)


def _skip_if_no_data(data_ready: bool) -> None:
    if not data_ready:
        pytest.skip("no trained checkpoint / usable cache in this environment")


# --------------------------------------------------------------------------- #
# Health (always available, even without a checkpoint)
# --------------------------------------------------------------------------- #
def test_health_always_responds(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert set(["model_loaded", "data_usable", "checkpoint_available"]).issubset(body)
    assert isinstance(body["checkpoint_available"], bool)


def test_health_reports_model_when_available(client, data_ready):
    _skip_if_no_data(data_ready)
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["combo"]  # the trained combo name (e.g. graphsage_lstm)
    assert body["serving_k"] == 5
    assert body["num_test_anchors"] >= 1
    assert 0.0 <= body["risk_threshold"] <= 1.0


# --------------------------------------------------------------------------- #
# Network state / history
# --------------------------------------------------------------------------- #
def test_network_state(client, data_ready, anchor_t):
    _skip_if_no_data(data_ready)
    r = client.get("/network/state", params={"t_index": anchor_t})
    assert r.status_code == 200
    body = r.json()
    assert body["t_index"] == anchor_t
    assert body["seq_len"] >= 1
    assert len(body["observed_graphs"]) == body["seq_len"]
    # graph contract: nodes carry the 5 cache features, edges the attack flag.
    g = body["latest_graph"]
    assert g["num_nodes"] == len(g["nodes"])
    assert g["num_edges"] == len(g["edges"])
    for n in g["nodes"]:
        assert {"id", "index", "out_degree", "in_degree", "bytes_sent",
                "bytes_received", "flow_count"}.issubset(n)
    for e in g["edges"]:
        assert isinstance(e["contains_attack"], bool)
    assert 0.0 <= body["risk_now"] <= 1.0


def test_network_state_latest_when_no_t_index(client, data_ready):
    _skip_if_no_data(data_ready)
    r = client.get("/network/state")
    assert r.status_code == 200
    assert "t_index" in r.json()


def test_network_state_unknown_anchor_404(client, data_ready):
    _skip_if_no_data(data_ready)
    r = client.get("/network/state", params={"t_index": 10_000_000})
    assert r.status_code == 404


def test_network_history(client, data_ready):
    _skip_if_no_data(data_ready)
    r = client.get("/network/history")
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == len(body["points"])
    assert body["n"] >= 1
    ts = [p["t_index"] for p in body["points"]]
    assert ts == sorted(ts)  # chronological
    for p in body["points"]:
        assert 0.0 <= p["risk"] <= 1.0


def test_network_history_limit(client, data_ready):
    _skip_if_no_data(data_ready)
    r = client.get("/network/history", params={"limit": 2})
    assert r.status_code == 200
    assert r.json()["n"] <= 2


# --------------------------------------------------------------------------- #
# Forecast trajectory + risk + horizons
# --------------------------------------------------------------------------- #
def test_forecast_trajectory(client, data_ready, anchor_t):
    _skip_if_no_data(data_ready)
    r = client.get("/forecast/trajectory",
                   params={"t_index": anchor_t, "horizons": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["K"] == 3
    assert len(body["steps"]) == 3
    # horizons are 1..K, target indices monotonically increasing from t.
    assert [s["horizon"] for s in body["steps"]] == [1, 2, 3]
    assert [s["target_index"] for s in body["steps"]] == [anchor_t + 1,
                                                          anchor_t + 2,
                                                          anchor_t + 3]
    for s in body["steps"]:
        assert 0.0 <= s["risk"] <= 1.0
        assert len(s["latent"]) == 64  # latent_dim from the trained config
    assert len(body["current_latent"]) == 64


def test_forecast_trajectory_horizons_bound(client, data_ready, anchor_t):
    _skip_if_no_data(data_ready)
    # serving K is 5; asking for 6 must be rejected (422 from Query le=10 is not
    # hit, but the service caps at serving K -> 400).
    r = client.get("/forecast/trajectory",
                   params={"t_index": anchor_t, "horizons": 6})
    assert r.status_code == 400


def test_risk(client, data_ready, anchor_t):
    _skip_if_no_data(data_ready)
    r = client.get("/risk", params={"t_index": anchor_t})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["risk_now"] <= 1.0
    assert body["alert"] == (body["risk_now"] >= body["risk_threshold"])
    assert body["horizons"] == list(range(1, 6))
    assert len(body["forecast_risk"]) == 5


# --------------------------------------------------------------------------- #
# Uncertainty / novelty
# --------------------------------------------------------------------------- #
def test_uncertainty_distinct_from_risk(client, data_ready, anchor_t):
    _skip_if_no_data(data_ready)
    r = client.get("/uncertainty", params={"t_index": anchor_t, "mc_passes": 8})
    assert r.status_code == 200
    body = r.json()
    assert body["method"] == "mc-dropout"
    assert body["n_passes"] == 8
    assert body["uncertainty_variance"] >= 0.0
    assert body["uncertainty_std"] >= 0.0
    assert 0.0 <= body["risk_mean"] <= 1.0


def test_novelty(client, data_ready, anchor_t):
    _skip_if_no_data(data_ready)
    r = client.get("/novelty", params={"t_index": anchor_t})
    # novelty needs >=2 train latents; if unavailable the service returns 503.
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        body = r.json()
        assert body["method"] == "mahalanobis"
        assert body["novelty_score"] >= 0.0
        assert isinstance(body["is_novel"], bool)


# --------------------------------------------------------------------------- #
# MITRE / trajectory / propagation / explainability / stability
# --------------------------------------------------------------------------- #
def test_mitre_separates_observed_and_forecast(client, data_ready, anchor_t):
    _skip_if_no_data(data_ready)
    r = client.get("/mitre", params={"t_index": anchor_t, "horizons": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["n_forecast"] == 3
    assert body["n_observed"] >= 1
    statuses = {s["status"] for s in body["stages"]}
    assert statuses.issubset({"observed", "forecast"})
    # forecast stages must never be reported as observed facts
    forecast_stages = [s for s in body["stages"] if s["status"] == "forecast"]
    assert len(forecast_stages) == 3


def test_propagation(client, data_ready, anchor_t):
    _skip_if_no_data(data_ready)
    r = client.get("/propagation", params={"t_index": anchor_t})
    assert r.status_code == 200
    body = r.json()
    assert body["t_index"] == anchor_t
    assert "is_spreading" in body
    assert isinstance(body["per_window"], list)


def test_explainability(client, data_ready, anchor_t):
    _skip_if_no_data(data_ready)
    r = client.get("/explainability", params={"t_index": anchor_t})
    assert r.status_code == 200
    body = r.json()
    assert body["target"] == "state"
    assert "top_features" in body
    assert isinstance(body["uses_attention"], bool)


def test_explainability_rejects_bad_target(client):
    r = client.get("/explainability", params={"target": "nonsense"})
    assert r.status_code == 422  # Query pattern validation


def test_stability(client, data_ready, anchor_t):
    _skip_if_no_data(data_ready)
    r = client.get("/stability", params={"t_index": anchor_t, "horizons": 3})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 < body["stability_score"] <= 1.0
    assert "MC-Dropout" in body["note"]  # explicitly distinguished from uncertainty


# --------------------------------------------------------------------------- #
# Counterfactual (POST)
# --------------------------------------------------------------------------- #
def test_counterfactual_isolate_node(client, data_ready, anchor_t):
    _skip_if_no_data(data_ready)
    r = client.post("/counterfactual", json={
        "t_index": anchor_t, "intervention": "isolate_node",
        "node_index": 0, "horizons": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["intervention"] == "isolate_node"
    assert body["K"] == 3
    assert len(body["baseline_risk"]) == 3
    assert len(body["intervention_risk"]) == 3
    assert body["label"] == "Modelled / simulated outcome."


def test_counterfactual_bad_intervention(client, data_ready, anchor_t):
    _skip_if_no_data(data_ready)
    r = client.post("/counterfactual", json={
        "t_index": anchor_t, "intervention": "nuke_everything"})
    assert r.status_code == 400


def test_counterfactual_missing_params(client, data_ready, anchor_t):
    _skip_if_no_data(data_ready)
    r = client.post("/counterfactual", json={
        "t_index": anchor_t, "intervention": "remove_edge"})
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# The aggregate POST /forecast — everything the frontend needs
# --------------------------------------------------------------------------- #
def test_full_forecast_contains_everything(client, data_ready, anchor_t):
    _skip_if_no_data(data_ready)
    r = client.post("/forecast", json={"t_index": anchor_t, "horizons": 3,
                                       "mc_passes": 8})
    assert r.status_code == 200
    body = r.json()
    # The Phase-10 spec: /forecast must provide EVERYTHING the frontend needs.
    required = [
        "network_state", "graph_nodes", "graph_edges", "forecast", "horizons",
        "risk", "uncertainty", "novelty", "attack_trajectory", "mitre",
        "propagation", "explainability", "stability",
    ]
    for key in required:
        assert key in body, f"missing forecast field: {key}"
    assert body["t_index"] == anchor_t
    assert body["K"] == 3
    assert body["horizons"] == [1, 2, 3]
    assert len(body["forecast"]["steps"]) == 3
    # graph nodes/edges mirror the current network state's latest graph
    assert body["graph_nodes"] == body["network_state"]["latest_graph"]["nodes"]
    assert body["attack_trajectory"]["n_forecast"] == 3


def test_full_forecast_can_exclude_layers(client, data_ready, anchor_t):
    _skip_if_no_data(data_ready)
    r = client.post("/forecast", json={
        "t_index": anchor_t, "horizons": 2,
        "include_uncertainty": False, "include_novelty": False,
        "include_explainability": False, "include_propagation": False,
        "include_stability": False})
    assert r.status_code == 200
    body = r.json()
    assert body["uncertainty"] is None
    assert body["novelty"] is None
    assert body["explainability"] is None
    assert body["propagation"] is None
    assert body["stability"] is None
    # core forecast is always present
    assert body["forecast"]["K"] == 2
    assert body["risk"]["horizons"] == [1, 2]


# --------------------------------------------------------------------------- #
# Ingest (POST) — caller-supplied window sequence
# --------------------------------------------------------------------------- #
def test_ingest_reconstructs_and_forecasts(client, data_ready, anchor_t, runtime):
    _skip_if_no_data(data_ready)
    # Build a window payload from the anchor sample's own observed tensors so we
    # feed the API a valid, in-distribution sequence.
    sample = runtime.find_sample(anchor_t)
    windows = []
    for g in sample.input_seq:
        n = int(g.num_nodes)
        windows.append({
            "num_nodes": n,
            "node_features": g.x.cpu().tolist() if n else [],
            "edge_index": (g.edge_index.t().cpu().tolist()
                           if g.edge_index.numel() else []),
            "edge_features": (g.edge_attr.cpu().tolist()
                              if g.edge_attr.numel() else []),
        })
    r = client.post("/ingest", json={"windows": windows, "horizons": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["accepted"] is True
    assert body["K"] == 3
    assert len(body["forecast_risk"]) == 3
    assert len(body["steps"]) == 3
    assert 0.0 <= body["risk_now"] <= 1.0


def test_ingest_requires_windows(client):
    r = client.post("/ingest", json={"windows": []})
    assert r.status_code == 422  # min_length=1 on the schema


# --------------------------------------------------------------------------- #
# Experiments registry
# --------------------------------------------------------------------------- #
def test_experiments_registry(client):
    r = client.get("/experiments")
    assert r.status_code == 200
    body = r.json()
    # Phase-9 registry names must be exposed.
    for name in ["known_attacks", "kstep", "early_warning", "ood_detection",
                 "baseline_comparison"]:
        assert name in body["available_experiments"]
    assert "pr_auc" in body["comparison_columns"]
    assert isinstance(body["comparison"], list)


# --------------------------------------------------------------------------- #
# Request-schema validation (extra fields forbidden)
# --------------------------------------------------------------------------- #
def test_forecast_rejects_unknown_field(client):
    r = client.post("/forecast", json={"t_index": 1, "bogus_field": True})
    assert r.status_code == 422
