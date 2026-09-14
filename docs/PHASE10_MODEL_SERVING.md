# Sentinel-X — Phase 10: Model Serving & FastAPI

**Goal.** Expose the trained Sentinel-X pipeline (Phase 4 world model + the
Phases 5–9 analytical layers) through a typed FastAPI application, so a frontend
can consume forecasts, risk, uncertainty, novelty, MITRE trajectory,
propagation, explainability, counterfactuals and stability over HTTP.

Nothing is retrained and no cache is rebuilt: the API loads the existing
`models/sentinel_x/model.pt` checkpoint and reuses the Phase-2 window cache via
the leakage-safe Phase-5 `KStepDataset`.

## Design: clean layers, no ML in routes

```
HTTP request
  -> app.py (route)         validate input (Pydantic), call a service, serialise
  -> services.py            ALL model interaction lives here (thin adapter)
  -> runtime.py             loads model + samples ONCE, thread-safe, cached
  -> worldmodel / research  the already-built Phase 4–9 code (unchanged)
```

* **Routes** (`sentinelx/serving/app.py`) never touch the model or tensors. Each
  handler validates the typed request, calls one service function, and returns a
  typed Pydantic response. `ServiceError` is mapped to the right HTTP status.
* **Services** (`sentinelx/serving/services.py`) are the only place ML runs
  (rollout, MC-Dropout, Mahalanobis OOD, trajectory, explain, propagation,
  counterfactual, stability). They convert tensors to plain Python before
  returning.
* **Runtime** (`sentinelx/serving/runtime.py`) is a process-wide singleton that
  lazily loads the checkpoint and rebuilds the K-step samples exactly once, then
  serves every request from cache. `set_runtime()` lets tests inject a runtime.
* **Schemas** (`sentinelx/serving/schemas.py`) are Pydantic v2 models — the only
  shapes that cross the boundary. Request bodies use `extra="forbid"`.

**Internal tensors are never exposed.** Latent vectors appear solely as
`list[float]` on the forecast trajectory (the frontend renders them); raw
`torch.Tensor` objects never leave the service layer.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/health` | Liveness/readiness; degrades gracefully with no checkpoint |
| POST | `/ingest` | Forecast over a caller-supplied observed window sequence |
| POST | `/forecast` | **Aggregate** — everything the frontend needs (see below) |
| GET  | `/network/state` | Current observed network state + graph (nodes/edges) |
| GET  | `/network/history` | Per-anchor risk history (chronological) |
| GET  | `/forecast/trajectory` | Future latent states + risk per horizon |
| GET  | `/risk` | Risk now + forecast risk per horizon + alert flag |
| GET  | `/uncertainty` | MC-Dropout predictive variance (distinct from risk) |
| GET  | `/novelty` | Mahalanobis novelty/OOD score + decision |
| GET  | `/propagation` | How flagged suspicious behaviour spreads |
| GET  | `/explainability` | Gradient×input attribution (+ GAT attention support) |
| POST | `/counterfactual` | Simulation-only intervention comparison |
| GET  | `/mitre` | High-level ATT&CK observed→forecast trajectory |
| GET  | `/experiments` | Phase-9 registry + `model_comparison.csv` |
| GET  | `/trajectory` | Alias of `/mitre` (attack trajectory) |
| GET  | `/stability` | Forecast stability under controlled perturbations |

### `POST /forecast` — the aggregate the frontend consumes

Returns, in one payload: `network_state`, `graph_nodes`, `graph_edges`,
`forecast` (future graph/latent states), `horizons`, `risk`, `uncertainty`,
`novelty`, `attack_trajectory`, `mitre`, `propagation`, `explainability`,
`stability`, plus `K` and `risk_threshold`. Optional layers can be toggled off
per request (`include_*` flags); a single failing/unavailable layer is returned
as `null` rather than sinking the whole response.

Anchor selection: `t_index` picks a specific test anchor; omit it to use the
latest (most recent) anchor as the "current" network state.

## Running

```
pip install -e .[serve]        # fastapi + uvicorn
python scripts/serve.py        # http://127.0.0.1:8000  (docs at /docs)
python scripts/serve.py --host 0.0.0.0 --port 8080
```

The model + samples load lazily on the first request. On Windows run the server
in your own terminal (it is long-running).

## Tests

`tests/test_serving_api.py` drives the app end-to-end with FastAPI's
`TestClient` against the **real** checkpoint and cache. Data-dependent tests
skip cleanly if the checkpoint / usable cache is absent; `/health`, schema
validation, and the experiments registry are always asserted. It verifies:
horizon/target-index invariants, latent dimensionality (64), observed-vs-forecast
separation in MITRE, the aggregate `/forecast` contract, request validation
(`extra="forbid"`, bad intervention/target, empty ingest), and graceful 404/503.

Run: `python -m pytest tests/test_serving_api.py -q` (or the full suite
`python -m pytest -q`).

## Discipline preserved

* Threshold is the checkpoint's validation-tuned value; never re-tuned on test.
* OOD detector is fit on in-distribution (train) latents; threshold from
  train+val only — identical to Phase 6.
* Forecast stages are always flagged `status="forecast"` and never presented as
  observed facts; counterfactuals carry the "Modelled / simulated outcome."
  label; stability is explicitly distinguished from MC-Dropout uncertainty.
