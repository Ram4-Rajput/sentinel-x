# Sentinel-X — Developer Documentation

> A complete, ground-truth reference for the Sentinel-X codebase. Written from
> the actual source (not the marketing summary), for engineers who need to
> extend, debug, operate, or evaluate the system.

**Version:** 0.1.0 &nbsp;•&nbsp; **Status:** research prototype (Phases 1–11 built)
&nbsp;•&nbsp; **Primary dataset:** CTU-13 scenario 11 &nbsp;•&nbsp; **Runtime:** Python 3.14, torch 2.14 (CPU), torch-geometric 2.8

---

## Table of contents

1. [What Sentinel-X is (and is not)](#1-what-sentinel-x-is-and-is-not)
2. [System architecture at a glance](#2-system-architecture-at-a-glance)
3. [Repository layout](#3-repository-layout)
4. [Core concepts & vocabulary](#4-core-concepts--vocabulary)
5. [The unified data layer (`data/`)](#5-the-unified-data-layer-data)
6. [The processing pipeline (`pipeline/`)](#6-the-processing-pipeline-pipeline)
7. [Baselines & the fairness protocol (`experiments/`)](#7-baselines--the-fairness-protocol-experiments)
8. [The world model (`worldmodel/`)](#8-the-world-model-worldmodel)
9. [Research infrastructure (`research/`)](#9-research-infrastructure-research)
10. [Model serving (`serving/`)](#10-model-serving-serving)
11. [The frontend (`frontend/`)](#11-the-frontend-frontend)
12. [Command-line tools (`scripts/`)](#12-command-line-tools-scripts)
13. [End-to-end workflows](#13-end-to-end-workflows)
14. [Configuration reference](#14-configuration-reference)
15. [Testing](#15-testing)
16. [Design principles & invariants](#16-design-principles--invariants)
17. [Known limitations & honest results](#17-known-limitations--honest-results)
18. [Extending Sentinel-X](#18-extending-sentinel-x)
19. [Troubleshooting](#19-troubleshooting)

---

## 1. What Sentinel-X is (and is not)

Sentinel-X is a **network world model**. Instead of classifying a single
window of traffic as attack/benign, it learns how the *state of a network*
evolves over time and forecasts what that state will look like several steps
into the future. On top of that latent forecast it layers a set of analytical
capabilities: risk scoring, predictive uncertainty, novelty/out-of-distribution
detection, MITRE ATT&CK trajectory interpretation, explainability, propagation
analysis, counterfactual simulation, and forecast stability.

The core research question:

> *Does modelling the temporal evolution of network state (as a dynamic graph)
> let us forecast future attack activity, quantify how confident that forecast
> is, and flag behaviour the model has never seen — better than conventional
> single-window classifiers?*

**What it is:**
- A complete, reproducible, leakage-safe research pipeline from raw capture
  files to an HTTP API and a live visualization frontend.
- Honest about provenance: a feature a dataset does not provide is `None`, never
  fabricated. A metric that cannot be measured is blank, never invented.

**What it is not:**
- A validated production intrusion-detection system. On the one graph-capable
  dataset used for evaluation (CTU-13 scenario 11, a small split), the
  forecasting/uncertainty/OOD results are weak-to-failing and the world model
  does not clearly beat the simple baselines. See
  [§17](#17-known-limitations--honest-results) and `docs/final_system_audit.md`.

---

## 2. System architecture at a glance

```
                          RAW CAPTURES (external, immutable)
                          CTU-13 .binetflow / UNSW-NB15 / CIC-IDS2018 / CICIoT2023
                                        │
              ┌─────────────────────────┴──────────────────────────┐
              │  data/  — unified layer (stdlib-only)               │
              │  adapters → UnifiedFlowRecord → preprocessing        │
              │  temporal split + dynamic-graph construction G_t     │
              └─────────────────────────┬──────────────────────────┘
                                        │
              ┌─────────────────────────┴──────────────────────────┐
              │  pipeline/  — Phase 2                               │
              │  stream → chronological split → windows → graphs     │
              │  → GraphCache (JSONL + optional PyG .pt) + manifest  │
              └─────────────────────────┬──────────────────────────┘
                                        │ data/processed/<dataset>/
        ┌───────────────────────────────┼───────────────────────────────┐
        │                               │                               │
┌───────┴────────┐          ┌───────────┴─────────────┐      ┌──────────┴─────────┐
│ experiments/   │          │ worldmodel/  — Phases 4–8 │      │ research/ — Phase 9│
│ Phase 3        │          │ GNN → temporal → latent   │      │ named experiments  │
│ LogReg/GRU/GNN │◄────────►│ → forecast head (+risk)   │      │ + comparison table │
│ (baseline bar) │  same    │ → K-step rollout          │      │ + reproducibility  │
└────────────────┘ protocol │ → uncertainty / OOD /     │      └──────────┬─────────┘
                            │   calibration / trajectory │                 │
                            │   / explain / propagation  │                 │
                            │   / counterfactual         │                 │
                            └───────────┬───────────────┘                 │
                                        │  models/sentinel_x/model.pt      │
                            ┌───────────┴──────────────────────────────────┴──┐
                            │ serving/  — Phase 10 (FastAPI)                    │
                            │ app (routes) → services (ML) → runtime (cache)    │
                            │ typed Pydantic schemas; no tensors escape         │
                            └───────────┬──────────────────────────────────────┘
                                        │  HTTP /api
                            ┌───────────┴──────────────────────────────────────┐
                            │ frontend/  — Phase 11 (React + Vite + D3 + R3F)   │
                            │ live model via /health probe, schema-faithful mock │
                            └───────────────────────────────────────────────────┘
```

The build phases and their status:

| Phase | Deliverable | Package |
|-------|-------------|---------|
| 1 | Dataset audit (4 datasets) | `data/metadata/` |
| 2 | Unified data layer + streaming pipeline + graph cache | `data/`, `pipeline/` |
| 3 | Scientific baselines (LogReg, LSTM/GRU, Temporal GNN) | `experiments/` |
| 4 | World-model core | `worldmodel/model.py`, `train.py`, `experiment.py` |
| 5 | K-step future-state forecasting (autoregressive rollout) | `worldmodel/rollout.py`, `kstep_*.py` |
| 6 | Risk / uncertainty / calibration / novelty-OOD | `worldmodel/uncertainty.py`, `ood.py`, `calibration.py` |
| 7 | Attack trajectory + MITRE ATT&CK interpretation | `worldmodel/trajectory*.py` |
| 8 | Explainability + propagation + counterfactual + stability | `worldmodel/explain.py`, `propagation.py`, `counterfactual.py` |
| 9 | Research experiment infra (reproducibility, registry, comparison) | `research/` |
| 10 | Model serving (FastAPI) | `serving/` |
| 11 | Frontend command center | `frontend/` |

---

## 3. Repository layout

```
sentinel/
├── configs/
│   └── world_model.yaml            # default world-model config (CLI-overridable)
├── data/
│   ├── raw/                        # (empty) raw data is external & immutable
│   ├── processed/<dataset>/        # Phase-2 cache (git-ignored, reproducible)
│   │   ├── windows/<split>.jsonl   # portable graph sequences
│   │   ├── graphs/<split>.pt       # optional PyG export
│   │   └── metadata/manifest.json  # reproduction metadata
│   ├── unified/                    # reserved for unified exports
│   └── metadata/                   # audit + config + reports (checked in)
├── docs/                           # phase docs + this file + final audit
├── experiments/                    # metrics CSV/JSON/MD, mlflow.db, phase9/, runs/
├── models/sentinel_x/              # model.pt + config.yaml + metadata.json
├── scripts/                        # CLIs (one per phase + serve + inspect)
├── src/sentinelx/                  # the Python package
│   ├── data/                       # unified layer + adapters
│   ├── pipeline/                   # streaming build → graph cache
│   ├── experiments/                # Phase-3 baselines + fairness protocol
│   ├── worldmodel/                 # Phases 4–8 core
│   ├── research/                   # Phase-9 reproducibility + experiments
│   └── serving/                    # Phase-10 FastAPI
├── frontend/                       # Phase-11 React/Vite app
├── tests/                          # 284 tests (pytest)
└── pyproject.toml
```

Every `scripts/*.py` prepends `../src` to `sys.path`, so the CLIs run without an
editable install. For serving and tests you still want `pip install -e .[serve,test]`.

---

## 4. Core concepts & vocabulary

- **Unified flow record** (`UnifiedFlowRecord`): one network flow mapped onto a
  canonical 19-numeric-feature + 1-categorical schema. Every numeric field is
  `Optional[float]`; `None` means "the source dataset genuinely lacks this."
- **Feature availability** (`FeatureAvailability`): an honest per-dataset record
  classifying every canonical feature as `present`, `derived`, or `unavailable`,
  with free-text transform notes. Travels with the adapter so audits can prove
  nothing was fabricated.
- **Network entity** (`NetworkEntity`): a host/node in the dynamic graph, keyed
  by IP, with aggregate byte/degree/flow stats.
- **Communication edge** (`CommunicationEdge`): a directed src→dst relationship
  within a window, with flow count, byte total, and an `contains_attack` flag.
- **Temporal network window** (`TemporalNetworkWindow`): the graph `G_t` for a
  time slice `[start, end)` — the flows plus the entities and edges they induce.
- **Latent network state** (`z_t`): the learned vector the world model encodes
  from a sequence of windows. This is the object the model forecasts.
- **Anchor** (`t_index`): the last observed window in a forecasting sample. The
  model sees history up to and including `t` and predicts `t+1 … t+K`.
- **Horizon** (`H` / `K`): how many windows ahead. `H` is the training target
  offset (default 1); `K` is the autoregressive rollout depth at inference.

---

## 5. The unified data layer (`data/`)

The unified layer converts heterogeneous raw datasets into one internal
representation using **only the Python standard library** (no pandas/numpy). It
is deliberately dependency-light and heavily tested.

### 5.1 `schema.py` — the canonical representation

Defines the shapes everything else consumes:

- `UNIFIED_NUMERIC_FEATURES` — the ordered 19-feature tuple (flow_duration,
  fwd/bwd packets & bytes, byte/packet rates, IAT means, TCP flag counts, TCP
  window, TTLs, header length, src/dst port).
- `UNIFIED_CATEGORICAL_FEATURES` — `("protocol",)`.
- `BinaryLabel` — `BENIGN | ATTACK | UNKNOWN`. `UNKNOWN` is used for CTU-13
  *Background* flows, which the audit flags as not-confirmed-benign.
- `FeatureAvailability` — `present` / `derived` / `unavailable` sets + notes.
  `assert_consistent()` guarantees every canonical feature is classified exactly
  once (no gaps, no double-classification).
- `UnifiedFlowRecord` — the flow dataclass. Helpers: `feature_vector(order)`,
  `has_entities()`, `has_time()`.
- `PacketFeatureRecord` — packet-level features; **only** created from a real
  PCAP source (there is no path that synthesizes packets from flows).
- `NetworkEntity`, `CommunicationEdge`, `TemporalNetworkWindow` — the graph
  primitives (see [§4](#4-core-concepts--vocabulary)).

### 5.2 `preprocessing.py` — train-only preprocessing

`FlowPreprocessor` learns all statistics from the **training split only**, then
applies them unchanged. Leakage-safety is structural: calling `transform` before
`fit` raises.

- Numeric imputation: per-feature median (train), plus an optional
  `<feat>_was_missing` indicator (informative missingness).
- Numeric scaling: standardization with a `std==0 → 1.0` guard.
- Categorical encoding: a protocol vocabulary fit on train, with a reserved
  `<unk>` index (0) so unseen test categories never crash.

It never clips, drops rows, or fabricates values. A feature a dataset never
provided arrives as `None` and is imputed like any other missing value — but
`FeatureAvailability` still records it as unavailable-at-source.

### 5.3 `temporal.py` — time, splitting, windowing, graph construction

- `normalize_timestamp` — coerces to UTC (naive assumed UTC).
- `chronological_split(records, train_frac, val_frac)` — sorts by time and cuts
  by time. **Never shuffles.** Raises on datasets without timestamps.
- `assert_chronological_disjoint(train, val, test)` — hard check that the future
  never lands in an earlier split.
- `build_graph(flows, …)` — constructs entities + directed edges. Flows without
  IPs are counted in the window but skipped from the graph.
- `generate_windows(records, window_seconds, …)` — contiguous non-overlapping
  windows anchored at the first flow's timestamp.

### 5.4 `leakage.py` — executable leakage checks

`run_leakage_checks(...)` returns a `LeakageReport` of findings (each with a
severity), rather than raising, so callers decide policy:

| Check | What it verifies | Default severity |
|-------|------------------|------------------|
| `temporal_order` | train max-time ≤ val/test min-time | high (hard gate) |
| `entity_overlap` | hosts shared across train/test | medium (warn-only) |
| `duplicate_rows` | identical feature vectors across splits | high |
| `preprocessor_fit` | preprocessor was fit (train-only) | high (hard gate) |

The pipeline treats `temporal_order` and `preprocessor_fit` as hard gates;
`entity_overlap`/`duplicate_rows` are recorded as dataset-property limitations
for testbed captures (CTU-13/UNSW reuse hosts and repeat Background flows).

### 5.5 `adapters/` — one adapter per dataset

`BaseDatasetAdapter` (`adapters/base.py`) declares the contract: `dataset_name`,
`required_columns`, `availability()`, `map_row(row, index)`. The base provides
`validate_schema`, `iter_records` (with a `_assert_no_fabrication` guard that
raises if an "unavailable" numeric feature ever gets populated), and label
helpers. `to_float`/`to_int` normalize cells and return `None` for Inf/NaN so the
preprocessor treats them as missing rather than letting Inf poison scaling.

| Adapter | `dataset_name` | Source | Graph-capable? | Notable handling |
|---------|----------------|--------|----------------|------------------|
| `CTU13Adapter` | `ctu-13` | 15-col `.binetflow` | ✅ has IPs + time | `total_bwd_bytes = TotBytes − SrcBytes` (derived); rates ÷Dur; syn flag from `State`; `TotPkts` left unavailable (it's a total, not a fwd split); Label: Botnet→attack, Normal→benign, Background→UNKNOWN |
| `UNSWNB15Adapter` | `unsw-nb15` | 49-feature Argus/Bro | ✅ (raw variant) | Only source with TTL/TCP-window/retransmission; header-less raw files use `RAW_COLUMN_ORDER` + `attach_raw_header`; tolerant `_get` for raw-vs-partition names; `case_insensitive_schema=True` |
| `CICIDS2018Adapter` | `cic-ids2018` | 80-col CICFlowMeter | ⚠️ no IPs | Flow Duration µs→s; Inf/NaN rates→None; header_length = Fwd+Bwd; windows form but graphs have 0 nodes |
| `CICIoT2023Adapter` | `ciciot2023` | 47-col engineered | ❌ no time, no IPs | Single `IAT`→flow_iat_mean (approx); Rate/Srate proxies; cannot be windowed at all |

The registry (`adapters/__init__.py`) exposes `list_adapters()` and
`get_adapter(name)`.

---

## 6. The processing pipeline (`pipeline/`)

Phase 2 turns raw files into a reusable, portable cache of temporal graph
sequences. It streams (bounded memory), splits chronologically, fits the
preprocessor on train only, builds per-window graphs, and writes the cache with
full reproduction metadata.

### 6.1 `config.py` — path resolution & config

Raw paths are **never hardcoded**. `resolve_dataset(dataset, …)` resolves in
order: explicit `--raw-path` → env var `SENTINELX_<DATASET>_RAW` → the flat YAML
`data/metadata/data_paths.yaml` → raises with a clear message. `_FILE_SPEC` maps
each dataset (and the `unsw-nb15-raw` variant) to its glob, binetflow flag, and
headerless flag. `PipelineConfig` carries windowing/streaming knobs
(`window_size`, `stride`, fractions, `chunk_size`, `max_records`, `build_graphs`).

### 6.2 `loaders.py` — streaming ingestion

`iter_unified_records(...)` streams chunks (default 100k rows), validates the
header once via the adapter, routes rows through the adapter, and tracks
`LoaderStats` (rows read, malformed, skipped, records yielded). Malformed rows
are counted and skipped, never crash the run.

### 6.3 `windows.py` — strided windows

`generate_windows_strided(records, window_size, stride, …)` produces window `i`
over `[t0 + i·stride, +window_size)`. `stride == size` → contiguous;
`stride < size` → overlap. Reuses `build_graph`; raises without timestamps.

### 6.4 `graph_cache.py` — serialization & reload

Each window becomes a portable graph dict via `window_to_graph_dict`:

```jsonc
{
  "window_index": 12, "start": "…", "end": "…",
  "num_nodes": 40, "num_edges": 55, "num_flows": 812,
  "node_ids": ["10.0.0.1", …],
  "node_features": [[out_deg, in_deg, bytes_sent, bytes_recv, flow_count], …],
  "edge_index": [[src_idx, dst_idx], …],
  "edge_features": [[flow_count, total_bytes, contains_attack], …],
  "attack_ratio": 0.03, "label_any_attack": 1
}
```

- Node feature order (`NODE_FEATURE_NAMES`): `out_degree, in_degree,
  total_bytes_sent, total_bytes_received, flow_count`.
- Edge feature order (`EDGE_FEATURE_NAMES`): `flow_count, total_bytes,
  contains_attack`.

`GraphCache` writes `windows/<split>.jsonl` (source of truth, dependency-free),
a best-effort `graphs/<split>.pt` PyG export when torch is installed, and
`metadata/manifest.json` (dataset/version/window-config/split stats/feature
names/class distribution). Node features come only from aggregates the graph
already tracks — nothing packet-level is fabricated.

### 6.5 `build.py` — orchestration

`build_dataset(config, …) → BuildReport` runs the full flow: stream+collect →
temporal-capability check → chronological split → `assert_chronological_disjoint`
→ `FlowPreprocessor.fit(train)` → `run_leakage_checks` (hard gates enforced) →
per-split windows+graphs → `GraphCache.write_split` + manifest. Datasets without
timestamps (CICIoT2023) skip windowing and record the limitation rather than
faking an order. `BuildReport` captures records, windows, nodes/edges, class
distribution, temporal range, timing, peak memory, cache location/size, and
limitations.

### 6.6 `pcap.py` — optional packet features

`iter_packet_features(pcap_path, …)` streams packet-level features via Scapy.
It is never imported by the core path and raises `ImportError` if Scapy is
absent. PCAP features are optional; the flow-based graph never depends on them.

---

## 7. Baselines & the fairness protocol (`experiments/`)

Phase 3 answers whether temporal/topological modelling beats simpler approaches.
Its real value is the **shared protocol** (`common.py`) that every model —
baselines *and* the world model — consumes, so comparisons are apples-to-apples.

### 7.1 `common.py` — the shared protocol

- Loads cached windows in time order (`_load_ordered_windows`) — no raw re-read.
- Builds forecasting samples: for each anchor `t` with a full history of
  `seq_len` windows and a future target at `t+horizon`, the sample carries the
  window's aggregate features, the sequence of aggregates, the graph sequence,
  and `y = label_any_attack(t+horizon)`.
- **Target:** given information up to and including window `t`, predict whether
  window `t+H` contains attack activity. Inputs never include `t+H` → no leakage.
- Chronological split by `t_index` with leakage assertions; `seed_everything`
  fixes python/numpy/torch RNGs.
- `WINDOW_FEATURE_NAMES` — 11 aggregate features (counts, attack ratio, mean
  node stats, edge byte totals, edge attack fraction, density).

### 7.2 The three baselines

| Baseline | File | Model | Imbalance handling |
|----------|------|-------|--------------------|
| Logistic Regression | `baseline_logreg.py` | StandardScaler + LogReg on window aggregates | `class_weight="balanced"` |
| Sequence (LSTM/GRU) | `baseline_sequence.py` | GRU/LSTM over the feature sequence + linear head | weighted BCE, train `pos_weight` |
| Temporal GNN | `baseline_gnn.py` | SAGEConv/GATConv + global mean pool per graph, temporal mean, linear head | weighted BCE |

All three tune the decision threshold on the **validation** PR curve, apply it to
test, and flag datasets with 0-node graphs as not graph-capable.

### 7.3 `runner.py`, `metrics.py`, `tracking.py`

`run_all_baselines(cfg, …)` trains all three, logs to MLflow, and returns nested
results. `write_outputs(...)` produces `baseline_results.csv`,
`baseline_metrics.json`, and `baseline_report.md` (with the dataset-applicability
table). `metrics.py` provides `BinaryMetrics` + `compute_binary_metrics` with
single-class guards (PR-AUC → NaN, ROC-AUC → None). `MlflowTracker` degrades to a
no-op when MLflow is unavailable.

### 7.4 The baseline bar to beat (CTU-13 scenario 11, forecast attack@t+1)

| Model | PR-AUC |
|-------|--------|
| Logistic Regression | 0.917 |
| GRU | 0.955 |
| GraphSAGE | 0.736 |

---

## 8. The world model (`worldmodel/`)

This is the scientific core (Phases 4–8). It learns `P(S_{t+1} | S_t)` over the
dynamic network state.

### 8.1 Architecture

```
G_{t-L+1..t}  ──GraphEncoder (GAT/GraphSAGE + pooling)──►  h_{t-L+1..t}
              ──TemporalEncoder (GRU/LSTM → Linear → LayerNorm → Tanh)──►  z_t   (latent state)
              ──forecast_head (latent→latent)──►  ẑ_{t+1}   (PRIMARY objective)
              ──risk_head (latent→1)──►  risk_logit @ t+H    (AUXILIARY objective)
```

- `graph_encoder.py` — stacked GAT (edge-feature aware, multi-head; `hidden_dim`
  must divide by `gat_heads`) or GraphSAGE (topology only) + LayerNorm/ReLU/
  dropout, then mean/max/sum pooling. Empty graphs pool to a zero state.
- `temporal_encoder.py` — GRU/LSTM over the sequence of pooled graph states,
  projected to `latent_dim` and passed through Tanh (direction-encoding latent).
- `model.py` — `SentinelXWorldModel`. `_encode_sequence_batch` builds one PyG
  mini-batch per timestep for efficient message passing, then regroups into
  `(batch, seq_len, hidden)`. Public API:
  - `encode_state(batch_seq) → z_t` (exposed for OOD/explain/trajectory reuse)
  - `forward(batch_seq) → WorldModelOutput(z_t, z_next_pred, risk_logit)`
  - `encode_target_state(target_seq)` — the `no_grad` stop-grad target for the
    future-state loss
  - `save_checkpoint` / `load_checkpoint` — checkpoint is
    `{state_dict, model_config, extra}`.

### 8.2 The loss (`losses.py`)

`L = λ_state · L_state + λ_risk · L_risk`, and this is deliberate, not arbitrary:

- **`L_state` (primary, self-supervised):** `1 − cos(ẑ_{t+1}, sg(z_{t+1}))`. The
  target is the encoder's own encoding of the *actual* future window sequence,
  taken with stop-gradient (`sg`) to prevent representation collapse (BYOL-style
  safeguard). Cosine distance suits the Tanh/LayerNorm latent where direction
  matters more than magnitude.
- **`L_risk` (auxiliary, supervised):** class-weighted BCE against `attack@t+H`,
  so the latent stays useful for risk forecasting and directly comparable to the
  Phase-3 baselines. `pos_weight` is computed on train.

Setting `λ_risk = 0` recovers a pure self-supervised world model; setting
`λ_state = 0` is discouraged (it degrades the model into a classifier).

### 8.3 Training (`train.py`)

Adam + gradient clipping (recurrent/message-passing stability), early stopping on
the **selection metric** (default `state` — the primary future-state objective),
best-epoch checkpointing, and threshold tuning on validation (lowest threshold
reaching target precision, else F1-optimal — never blind 0.5, never tuned on
test). Reports VAL + TEST metrics for the auxiliary risk head.

### 8.4 Phase-4 driver (`experiment.py`)

Trains the four architecture combinations and selects the best by primary
validation state loss:

```
COMBOS = [(gat, gru), (gat, lstm), (graphsage, gru), (graphsage, lstm)]
```

Writes the winner to `models/sentinel_x/{model.pt, config.yaml, metadata.json}`
and the comparison to `experiments/world_model_{results.csv, report.md,
metrics.json}`. The shipped checkpoint's best combo is **graphsage_lstm**
(185,729 params) on CTU-13.

### 8.5 K-step forecasting (`rollout.py`, `kstep_*.py`) — Phase 5

Instead of training K classifiers, the single learned one-step transition is
rolled forward autoregressively:

```
ẑ_{t+1} = f(z_t);   ẑ_{t+2} = f(ẑ_{t+1});   …;   ẑ_{t+K} = f(ẑ_{t+K-1})
```

where `f = forecast_head`. At each horizon the risk head reads off the predicted
latent. `rollout_latents(model, batch_seq, K)` is deterministic (eval + no_grad).
Ground-truth for evaluating rollout error is the encoder's own encoding of the
real future window sequence (`encode_future_states`) — the same quantity Phase-4
trained against, so the error metric is self-consistent and never leaks future
inputs into the prediction path. `kstep_eval.py` reports per-horizon state error
(cosine/L2) and risk metrics; `kstep_experiment.py` writes `k_step_*`. Supported
`K ∈ {1, 3, 5, 10}`.

### 8.6 Uncertainty, OOD, calibration (`uncertainty.py`, `ood.py`, `calibration.py`) — Phase 6

- **MC-Dropout:** `mc_dropout_risk` re-enables dropout at inference and runs N
  passes over identical inputs, returning risk mean + predictive variance/std.
  Risk (point estimate) and uncertainty (spread) are reported **separately**.
- **Novelty/OOD:** `MahalanobisOOD` fits a ridge-regularized covariance on
  in-distribution (train) latents; the threshold comes from a train+val
  percentile (default 95), **never test**. `score` returns `OODScore(novelty,
  is_ood)`.
- **Calibration:** reliability bins, ECE/MCE, Brier score, and an optional
  `TemperatureScaler` (fit on validation).

### 8.7 Trajectory & MITRE (`trajectory.py`, `trajectory_experiment.py`) — Phase 7

`build_trajectory(observed_windows, forecast_risks, t_index)` produces an
`AttackTrajectory` of stages, each marked `observed` (from labels) or `forecast`
(from rollout risk), with confidence and evidence. `_assert_temporal_order`
enforces observed-before-forecast and monotonic horizons. Mapping is an
evidence-gated heuristic to **coarse ATT&CK tactics only** — it never asserts
precise technique IDs (honestly scoped; see [§17](#17-known-limitations--honest-results)).

### 8.8 Explainability, propagation, counterfactual, stability (`explain.py`, `propagation.py`, `counterfactual.py`, `phase8_experiment.py`) — Phase 8

- **Explainability:** `explain_forecast(model, input_seq, target)` computes
  gradient×input attributions over node/edge/temporal inputs (`target ∈
  {state, risk}`). GAT attention is captured via a temporary hook as
  supporting-only (association, not causation — disclaimed).
- **Propagation:** `analyze_propagation(windows, …)` is a descriptive analysis of
  how flagged suspicious behaviour spreads (affected hosts, velocity, direction,
  branching, persistence, growth). It does not forecast.
- **Counterfactual:** `simulate_intervention(model, input_seq, kind, K, …)` clones
  the sequence (never mutates the caller), applies `isolate_node` / `remove_edge`
  / `suppress_path`, re-rolls, and diffs baseline vs intervention risk. Labelled
  "modelled/simulated outcome."
- **Stability:** `evaluate_stability(model, input_seq, K, n_trials, epsilon)`
  perturbs inputs with multiplicative jitter (structure preserved, dropout OFF —
  explicitly *not* MC-Dropout) and measures latent/risk drift. `stability_score =
  1 / (1 + mean_latent_drift)`.

---

## 9. Research infrastructure (`research/`)

Phase 9 makes the science reproducible and comparable. Nothing is retrained or
rebuilt here — every experiment reuses the frozen Phase-4 checkpoint and the
Phase-2 window cache, on the same chronological, leakage-safe protocol. Where a
measurement is impossible (single-class split, non-graph dataset), the result is
reported as skipped/`n/a` rather than invented.

### 9.1 `reproducibility.py`

Captures the environment (tracked package versions), the git commit
(`null` here because the workspace is not a git repo), a config snapshot, and a
`RunManifest` (command, experiment, seed, config, environment, argv, outputs,
timestamps, status). `new_manifest(...)` / `finalize_manifest(...)` bracket a run
and write to `experiments/runs/`.

### 9.2 `experiments.py` — the named experiment registry

Ten experiments, dispatched by `run_experiment(name, **kwargs)`
(`ALL_EXPERIMENTS` lists them all):

| Name | What it measures | Reuses |
|------|------------------|--------|
| `known_attacks` | Risk head on the standard test split at the checkpoint's val-tuned threshold | model + cache |
| `unseen_attacks` | Whether genuinely-unseen attack windows are flagged novel | Phase-6 OOD |
| `kstep` | Autoregressive rollout at K ∈ {1,3,5,10} | Phase-5 |
| `early_warning` | Lead time (windows) between forecast crossing threshold and the first true attack | Phase-5 rollout |
| `missing_telemetry` | PR-AUC/recall degradation as node features are zeroed | model + cache |
| `uncertainty_error` | Uncertainty↔error correlation + ECE | Phase-6 |
| `ood_detection` | Mahalanobis OOD AUROC | Phase-6 |
| `forecast_stability` | Mean stability score under perturbation | Phase-8 |
| `cross_dataset` | Frozen model applied to other graph-capable datasets | model + cache |
| `baseline_comparison` | Re-run Phase-3 baselines for a head-to-head | Phase-3 |

### 9.3 `comparison.py` — the machine-readable comparison

`build_and_write(...)` assembles `experiments/model_comparison.csv` with columns:

```
model, dataset, precision, recall, f1, pr_auc, false_positive_rate,
forecast_error, lead_time, ece, ood_auroc, stability
```

Only genuinely-measured cells are populated. Baselines legitimately have no
`forecast_error`/`lead_time`/`ece`/`ood_auroc`/`stability` (they lack the world
model's latent forecasting/novelty/uncertainty machinery), so those cells are
left blank — never faked. Sources: baseline & world-model result CSVs, the t+1
state cosine distance from `k_step_results.csv`, the ECE/OOD from
`uncertainty_ood_metrics.json`, stability from `phase8_metrics.json`, and lead
time from the early-warning result.

---

## 10. Model serving (`serving/`)

Phase 10 exposes the whole pipeline over HTTP with clean layers and **no ML logic
in the routes**.

```
HTTP request
  → app.py       validate (Pydantic), call one service, return typed response
  → services.py  the ONLY place ML runs (rollout, MC-Dropout, OOD, trajectory, …)
  → runtime.py   loads model + K-step samples ONCE, thread-safe, cached
  → worldmodel / research   the already-built Phase 4–9 code (unchanged)
```

### 10.1 `runtime.py` — the cached singleton

`SentinelRuntime` lazily loads `models/sentinel_x/model.pt` and rebuilds the
K-step samples exactly once, then serves every request from cache. It exposes
`model`, `config`, `combo`, `threshold`, `samples`, `data_usable`, and
sample-resolution helpers (`resolve_sample`, `latest_sample`, `find_sample`).
Defaults: `DEFAULT_SERVING_K = 5`, `MAX_SERVING_K = 10`. `get_runtime()` /
`set_runtime()` manage the process singleton (tests inject their own).

### 10.2 `services.py` — the ML adapter

The only module that touches the model. Converts tensors to plain Python before
returning; **raw `torch.Tensor` objects never escape**, and latents appear only
as `list[float]` on the forecast trajectory. Each analytical layer maps to a
function (`network_state`, `forecast_trajectory`, `risk`, `uncertainty`,
`novelty`, `attack_trajectory`/`mitre`, `propagation`, `explainability`,
`counterfactual`, `stability`, `ingest`, `experiments`). `full_forecast(...)`
assembles the aggregate payload and guards each optional layer with `_safe` so
one failing layer never sinks the whole response. `ServiceError` carries an HTTP
status code.

### 10.3 `app.py` — routes

`create_app()` wires thin handlers; the module-level `app` is what
`uvicorn sentinelx.serving.app:app` serves. `_handle` maps `ServiceError` →
HTTPException and `FileNotFoundError` → 503.

| Method | Path | Response model | Notes |
|--------|------|----------------|-------|
| GET | `/health` | `HealthResponse` | degrades gracefully with no checkpoint |
| POST | `/ingest` | dict | forecast over a caller-supplied window sequence |
| POST | `/forecast` | `ForecastResponse` | **aggregate** — everything the frontend needs |
| GET | `/network/state` | `NetworkState` | `t_index` (latest if omitted) |
| GET | `/network/history` | `NetworkHistoryResponse` | `limit ≥ 1` |
| GET | `/forecast/trajectory` | `ForecastTrajectory` | `t_index`, `horizons 1–10` |
| GET | `/risk` | `RiskResponse` | risk now + forecast + alert flag |
| GET | `/uncertainty` | `UncertaintyResponse` | `mc_passes 1–200` |
| GET | `/novelty` | `NoveltyResponse` | Mahalanobis |
| GET | `/propagation` | `PropagationResponse` | |
| GET | `/explainability` | `ExplainabilityResponse` | `target ∈ {state, risk}` |
| POST | `/counterfactual` | `CounterfactualResponse` | simulation-only |
| GET | `/mitre` | `MitreResponse` | == trajectory |
| GET | `/trajectory` | `TrajectoryResponse` | alias of mitre |
| GET | `/stability` | `StabilityResponse` | controlled perturbation |
| GET | `/experiments` | `ExperimentsResponse` | Phase-9 registry + comparison |

### 10.4 `schemas.py` — the boundary contract

Pydantic v2 models are the only shapes that cross the API boundary. Request
bodies use `extra="forbid"` (reject unknown fields); deeply-nested analytical
payloads mirror the worldmodel dataclasses' `as_dict()`. `HealthResponse`
disables Pydantic's protected namespace so `model_loaded` doesn't warn.

### 10.5 Running the server

```
pip install -e .[serve]
python scripts/serve.py --host 127.0.0.1 --port 8000
# interactive docs at http://127.0.0.1:8000/docs
```

---

## 11. The frontend (`frontend/`)

Phase 11 is a React + TypeScript + Vite command center where the network graph
is the protagonist. It auto-detects the Phase-10 backend via `GET /health` (over
the `/api` dev proxy) and drives every visual from the live model; if the backend
is unreachable it falls back to a **schema-faithful mock** (`src/api/scenario.ts`)
that conforms 1:1 to `serving/schemas.py`.

> ⚠️ The mock fabricates every metric client-side. It exists so the UI is
> demonstrable without a backend, but any number shown while the mock is active
> is synthetic. Treat a running backend as the only source of real values.

### 11.1 Layout → the questions each panel answers

| Region | Component | Answers |
|--------|-----------|---------|
| Top | CommandBar (PAST/NOW/FUTURE lens) | temporal navigation |
| Left | NetworkState | WHAT / WHERE |
| Center | GraphStage (D3 canvas network) | WHAT / WHERE / WHAT NEXT (morphs to forecast) |
| Right | ThreatForecast + ThreatOrbit (R3F) | HOW CERTAIN (risk · uncertainty · novelty, kept separate) |
| Bottom | RiskTimeline (D3) | WHEN |
| Bottom | MitreTrajectory | WHAT NEXT |
| Bottom | Explainability | WHY |
| Bottom | NodeIntel + WhatIf | WHERE detail + WHAT IF |

### 11.2 API mapping & structure

`src/api/types.ts` mirrors the Pydantic schemas exactly; `src/api/client.ts`
calls `/health`, `/forecast`, `/network/history`, and `/counterfactual`. State
lives in a single Zustand store (`src/store.ts`); canvas/WebGL render loops run
outside React reconciliation to handle 100–300 nodes without rerenders.

Source tree:

```
frontend/src/
├── api/         client.ts, scenario.ts (mock), types.ts
├── components/  graph/ · panels/ · threat/ · ui/
├── lib/         format.ts
├── App.tsx  main.tsx  store.ts  index.css
```

### 11.3 Running the frontend

```
cd sentinel/frontend
npm install
npm run dev        # http://localhost:5173
npm run build      # type-check + production bundle
```

Libraries: React 18, Vite 5, Tailwind, D3 (canvas force graph, timelines,
bars), React Three Fiber + Drei + Postprocessing (3D threat orbit), Framer
Motion (state-driven transitions), Zustand (store).

---

## 12. Command-line tools (`scripts/`)

Each script prepends `../src` to `sys.path`. Windows note: `cmd`/PowerShell strip
`$` from inline commands — put shell logic in a temp `.ps1`/`.py` file.

| Script | Phase | Purpose | Key args |
|--------|-------|---------|----------|
| `build_dataset.py` | 2 | Build the graph cache | `--dataset` (req), `--raw-path`, `--variant`, `--window-size` (10.0), `--stride`, `--train-frac`, `--val-frac`, `--chunk-size`, `--max-records`, `--no-graphs`, `--no-cache`, `--report` |
| `run_baselines.py` | 3 | Train LogReg/GRU/GNN | `--dataset` (repeatable), `--seq-len`, `--horizon`, `--seed`, `--epochs`, `--temporal-type {gru,lstm}`, `--gnn-type {sage,gat}`, `--mlflow-uri`, `--out-dir` |
| `train_world_model.py` | 4 | Train the world model over combos | `--config`, `--dataset`, `--seq-len`, `--horizon`, `--epochs`, `--batch-size`, `--learning-rate`, `--hidden-dim`, `--latent-dim`, `--combos`, `--models-dir`, `--experiments-dir` |
| `run_kstep_forecast.py` | 5 | K-step rollout eval | `--checkpoint`, `--ks` (1,3,5,10), `--batch-size`, `--seed`, `--experiments-dir` |
| `run_phase6.py` | 6 | Uncertainty/OOD/calibration | `--checkpoint`, `--mc-passes`, `--calibration-bins`, `--ood-percentile`, `--no-temperature` |
| `run_trajectory.py` | 7 | Build trajectories | `--checkpoint`, `--k` (5), `--max-trajectories` |
| `run_phase8.py` | 8 | Explain/propagation/counterfactual/stability | `--checkpoint`, `--k`, `--max-anchors`, `--stability-trials`, `--stability-epsilon` |
| `train.py` | 9 | Reproducible training + manifest | as `train_world_model` + snapshot to `experiments/runs/` |
| `evaluate.py` | 9 | Reproducible evaluation | `--checkpoint`, `--dataset`, `--ks`, `--comparison` |
| `run_experiment.py` | 9 | Run named experiment(s) | `--experiment` (repeatable or `all`), `--list`, `--ks`, `--k`, `--no-comparison` |
| `serve.py` | 10 | Launch FastAPI (needs uvicorn) | `--host`, `--port`, `--reload` |
| `inspect_state.py` | — | Report on-disk raw/cache state | (no args) |

---

## 13. End-to-end workflows

### 13.1 Reproduce the whole pipeline from scratch

> The shipped checkpoint and cache already exist — you normally do **not** need
> to rebuild. This is for a clean reproduction.

```bat
:: 1. Build the Phase-2 cache for the primary dataset
python scripts/build_dataset.py --dataset ctu-13 --report

:: 2. Baselines (sets the bar)
python scripts/run_baselines.py --dataset ctu-13 --epochs 30

:: 3. Train the world model (writes models/sentinel_x/)
python scripts/train_world_model.py --config configs/world_model.yaml

:: 4-8. Analytical layers off the checkpoint
python scripts/run_kstep_forecast.py --ks 1,3,5,10
python scripts/run_phase6.py
python scripts/run_trajectory.py --k 5
python scripts/run_phase8.py --k 5 --max-anchors 25

:: 9. Named experiments + comparison table
python scripts/run_experiment.py --experiment all

:: 10. Serve
python scripts/serve.py
```

### 13.2 Just serve the existing model

```bat
pip install -e .[serve]
python scripts/serve.py --host 127.0.0.1 --port 8000
```

Then, in `frontend/`, `npm install && npm run dev`. The app auto-detects the
backend and drives visuals from the live model.

### 13.3 Add a new dataset

1. Point `data/metadata/data_paths.yaml` (or `SENTINELX_<DATASET>_RAW`) at the
   raw files.
2. Write an adapter in `data/adapters/` (subclass `BaseDatasetAdapter`,
   implement `availability()` + `map_row()`), register it in
   `adapters/__init__.py`, add a `_FILE_SPEC` entry in `pipeline/config.py`.
3. `python scripts/build_dataset.py --dataset <name> --report`.
4. `python scripts/run_baselines.py --dataset <name>` to check graph-capability
   and set a bar.

---

## 14. Configuration reference

`configs/world_model.yaml` is the default; every value is CLI-overridable and
nothing is hardcoded in model/training code. `models/sentinel_x/config.yaml` is
the *winning* config saved with the checkpoint.

```yaml
model:
  gnn_type: graphsage      # gat | graphsage
  temporal_type: lstm      # gru | lstm
  hidden_dim: 128
  graph_layers: 2
  temporal_layers: 1
  dropout: 0.2
  latent_dim: 64           # dimension of the learned network state z_t
  gat_heads: 4             # hidden_dim must divide by gat_heads (GAT only)
  pooling: mean            # mean | max | sum
  node_feature_dim: 5      # from the Phase-2 cache
  edge_feature_dim: 3
  use_edge_features: true
training:
  learning_rate: 0.001
  batch_size: 32
  epochs: 50
  seed: 42
  weight_decay: 0.0
  grad_clip_norm: 1.0
  early_stopping_patience: 8
  early_stopping_min_delta: 0.0001
  lambda_state: 1.0        # primary future-state weight
  lambda_risk: 1.0         # auxiliary risk weight
  selection_metric: state  # state (primary) | total
data:
  dataset: ctu-13
  seq_len: 4
  horizon: 1
  train_frac: 0.7
  val_frac: 0.15
  min_windows: 12
```

The config loader (`worldmodel/config.py`) uses PyYAML if installed, otherwise a
small stdlib nested-YAML reader. `ModelConfig.normalized()` validates and
canonicalizes values (`sage` → `graphsage`).

### 14.1 Real data locations (external, immutable)

See `data/metadata/data_paths.yaml`.

| Dataset | Location | Capability |
|---------|----------|------------|
| cic-ids2018 | `D:\DATA\CSE-CIC-IDS2018\processed` | no IPs → no graph |
| ciciot2023 | `D:\DATA\CSE-CIC-IDS2018` | no timestamps → no windows/graph |
| unsw-nb15 | `D:\DATA\UNSW-NB15` | raw variant has IPs + time → graph-capable |
| ctu-13 | `…\CTU-13-Dataset\CTU-13-Dataset` | time + IPs + PCAPs; **scenario 11** is attack-dense |

---

## 15. Testing

```bat
pip install -e .[test]
python -m pytest -q
```

284 tests should pass (258 prior + 26 Phase-10 API). The suite covers adapters,
schema, preprocessing, temporal splitting, leakage, pipeline (config/loaders/
windows/cache/build/integration), baselines (common/metrics/runner), the world
model, K-step forecasting, Phase-6 uncertainty/OOD, Phase-8 explain/propagation/
counterfactual, trajectory/MITRE, research infra, and the serving API (via
FastAPI's TestClient, which is why `httpx` is a test dependency). Don't break
these.

---

## 16. Design principles & invariants

These are enforced in code and tests; preserve them when extending.

1. **No fabrication.** A feature a dataset lacks is `None` and recorded as
   unavailable. The adapter base class raises if an unavailable numeric feature
   is ever populated. A metric that cannot be measured is blank/`n/a`.
2. **Leakage-safety is structural.** Splits are chronological (never shuffled);
   the preprocessor fits on train only and raises if used before fit; thresholds
   are tuned on validation and never on test; hard leakage gates
   (`temporal_order`, `preprocessor_fit`) fail the build.
3. **Reuse, don't rebuild.** Phases 5–11 consume the frozen Phase-4 checkpoint
   and the Phase-2 cache. No phase retrains or re-audits a prior phase.
4. **Same protocol for everyone.** Baselines and the world model share
   `experiments/common.py` (windows, seq_len, horizon, split, seed), so
   comparisons are apples-to-apples.
5. **The primary objective is future-state modelling.** The risk head is
   auxiliary; model selection tracks the state loss by default. Never reduce the
   model to `z_t → attack yes/no`.
6. **Clean serving layers.** No ML in routes; tensors never leave the service
   layer; latents cross the boundary only as `list[float]`.
7. **Reproducibility.** Fixed seeds, deterministic RNGs, captured environment,
   run manifests, config snapshots.
8. **Honesty about scope.** MITRE mapping is coarse tactics only; attention is
   association-only; counterfactuals are labelled "modelled outcome"; the
   frontend mock is clearly a mock.

---

## 17. Known limitations & honest results

From `docs/final_system_audit.md` (2026-09-05), validated against the real
checkpoint and cached CTU-13 data:

- **The machinery is real and wired.** GNN→temporal→latent→forecast head,
  autoregressive rollout, genuine MC-Dropout, real Mahalanobis OOD, trajectory
  builder, gradient×input explainability, propagation, counterfactual, and
  stability all exist, run, and are tested.
- **The scientific claim of effectiveness is NOT established.** On the only
  graph-capable dataset used (CTU-13 scenario 11, a very small split):
  - K-step *state* prediction is strong (cosine similarity ≈ 0.99).
  - Forecast *risk* is weak/erratic (ROC-AUC ≈ 0.41 at h1; early-warning lead
    time ≈ 0 windows).
  - Uncertainty↔error correlation ≈ −0.20; ECE ≈ 0.21 (poorly calibrated).
  - OOD AUROC ≈ 0.30 (worse than chance on this data).
  - The world model does **not** clearly beat the simple baselines.
- **MITRE mapping** is a hand-written heuristic over aggregate signals, not a
  learned technique-level classifier (scoped to coarse tactics).
- **Research infra** captures seed + environment, but `git_commit` is `null`
  because the workspace is not a git repository.
- **The frontend** builds cleanly but ships a client-side mock that fabricates
  every metric when the backend is unreachable.

Verdict: Sentinel-X is a **complete, honest research prototype**, not a validated
detection system. Treat the numbers accordingly and evaluate on more/larger
graph-capable datasets before drawing conclusions.

---

## 18. Extending Sentinel-X

- **New GNN or temporal cell:** add the branch in `graph_encoder.py` /
  `temporal_encoder.py`, extend `ModelConfig.normalized()` validation, and add a
  combo to `experiment.py`.
- **New analytical layer:** implement it in `worldmodel/`, expose it through a
  `services.py` function, add a Pydantic schema and a thin route in `app.py`, and
  mirror the type in `frontend/src/api/types.ts`.
- **New named experiment:** add a `run_*` callable to `research/experiments.py`,
  register it in `EXPERIMENT_REGISTRY`, and (if it yields a comparable metric)
  wire it into `comparison.py`.
- **New dataset:** see [§13.3](#133-add-a-new-dataset). The `availability()`
  contract and the `_assert_no_fabrication` guard will keep you honest.

When adding features, respect the invariants in [§16](#16-design-principles--invariants):
keep splits chronological, fit on train only, tune thresholds on validation, and
never fabricate an unavailable feature or an unmeasured metric.

---

## 19. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `Could not resolve raw path for '<dataset>'` | No `--raw-path`, env var, or `data_paths.yaml` entry | Provide one of the three (see `pipeline/config.py`) |
| `No Phase-2 cache for '<dataset>'` | Cache not built | `python scripts/build_dataset.py --dataset <name>` |
| Dataset "SKIPPED — unusable" | Too few windows, or no timestamps/IPs | Check dataset capability table ([§14.1](#141-real-data-locations-external-immutable)); CICIoT2023 can't be windowed |
| GNN baseline flagged not graph-capable | 0-node graphs (no IPs, e.g. CIC-IDS2018) | Expected; use a graph-capable dataset for GNN comparisons |
| `/health` returns `degraded` | Checkpoint missing at `models/sentinel_x/model.pt` | Train Phase 4 or restore the checkpoint |
| 503 from serving endpoints | `data_usable` is false for the dataset | Ensure the Phase-2 cache exists and is usable |
| PyG `.pt` not written | torch/torch-geometric not installed | Optional; JSONL is the source of truth. `pip install -e .[graph]` |
| Frontend shows data with no backend | Mock fallback is active | Start `scripts/serve.py`; the app auto-detects it via `/health` |
| Inline `$` stripped in commands | Windows `cmd`/PowerShell | Put shell logic in a temp `.ps1`/`.py` file |

---

*This document reflects the code as of version 0.1.0. When you change a public
interface, update the relevant section here and the matching phase doc in
`docs/`.*
