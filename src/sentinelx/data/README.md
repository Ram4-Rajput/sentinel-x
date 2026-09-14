# Sentinel-X Unified Data Layer

Converts each supported dataset into a common internal representation **without
modifying raw data, without merging datasets, and without fabricating
unavailable features.** Grounded in the Phase-1 audit (`data/metadata/`).

## Modules

| Module | Responsibility |
|---|---|
| `schema.py` | The 5 core representations + `FeatureAvailability` |
| `adapters/` | Per-dataset raw→unified mapping + schema validation |
| `preprocessing.py` | Train-only imputation / scaling / categorical encoding |
| `temporal.py` | Timestamp normalization, chronological split, windows, graph `G_t` |
| `leakage.py` | Executable leakage checks |

## The five core representations (`schema.py`)

- **UnifiedFlowRecord** — one flow mapped to the canonical schema. Every numeric
  feature is `Optional`: `None` means the source genuinely lacks it (recorded in
  `FeatureAvailability`), never a fabricated constant.
- **PacketFeatureRecord** — packet-level features; only created from a real PCAP
  source. No code path synthesizes packet features from flow statistics.
- **NetworkEntity** — a host/node in the dynamic graph `G_t`.
- **CommunicationEdge** — a directed communication between two entities.
- **TemporalNetworkWindow** — a time-bounded slice: flows + entities + edges,
  the unit consumed by the Temporal GNN + GRU/LSTM in later phases.

`FeatureAvailability` classifies every canonical feature as **present** (mapped
from a real column), **derived** (computed from present columns, documented), or
**unavailable** (absent at source → value stays `None`). `assert_consistent()`
fails fast if the contract is malformed, and the base adapter's
`_assert_no_fabrication()` raises if an "unavailable" feature is ever populated.

## Per-dataset transformations (documented)

### CIC-IDS2018 (`cic_ids2018.py`) — 80-col CICFlowMeter, has time, NO IPs
| Unified | Source | Transform |
|---|---|---|
| flow_duration | `Flow Duration` | microseconds → seconds (`/1e6`) |
| flow_bytes_per_s | `Flow Byts/s` | Inf/NaN → `None` (missing), **not clipped** |
| flow_pkts_per_s | `Flow Pkts/s` | Inf/NaN → `None` |
| header_length | `Fwd Header Len`+`Bwd Header Len` | summed |
| dst_port, protocol | `Dst Port`, `Protocol` | direct |
| timestamp | `Timestamp` | parsed `dd/MM/yyyy HH:mm:ss` |
| label_binary | `Label` | `Benign`→benign else attack |
| **unavailable** | — | `src_ip`, `dst_ip`, `src_port`, `ttl_src`, `ttl_dst` |

### CICIoT2023 (`ciciot2023.py`) — 47-col, NO time, NO IPs
| Unified | Source | Transform |
|---|---|---|
| flow_duration | `flow_duration` | direct |
| flow_iat_mean | `IAT` | single aggregate IAT (approx; no fwd/bwd split) |
| flow_bytes_per_s / flow_pkts_per_s | `Rate` / `Srate` | rate proxies (units approx) |
| total_fwd_packets / total_fwd_bytes | `Number` / `Tot sum` | count/byte proxies |
| header_length | `Header_Length` | direct |
| protocol | `Protocol Type` | numeric proto id (string-cast) |
| label_binary | `label` | `BenignTraffic`→benign else attack |
| **unavailable** | — | timestamp, IPs, ports, TTL, bwd splits, fwd_iat |

### CTU-13 (`ctu13.py`) — 15-col binetflow, full entities + time + PCAPs
| Unified | Source | Transform |
|---|---|---|
| src_ip/dst_ip/src_port/dst_port | `SrcAddr/DstAddr/Sport/Dport` | direct (graph-capable) |
| timestamp | `StartTime` | parsed `YYYY/MM/DD HH:MM:SS.ffffff` |
| flow_duration | `Dur` | direct (s) |
| total_fwd_bytes | `SrcBytes` | direct |
| total_bwd_bytes | `TotBytes - SrcBytes` | **derived** (clamp ≥ 0) |
| flow_bytes_per_s | `TotBytes / Dur` | **derived** (`Dur≤0` → `None`) |
| flow_pkts_per_s | `TotPkts / Dur` | **derived** |
| syn_flag_count | `State` | **derived**: `1.0` if `S` in state else `0.0` |
| label | `Label` free-text `flow=...` | Botnet→attack, Normal→benign, Background→**unknown** |
| **unavailable** | — | fwd/bwd packet split (TotPkts is a total, not mislabeled), ack/fin/rst flags, IAT, TTL, window, header_length |

### UNSW-NB15 (`unsw_nb15.py`) — 49-feature, full entities, **only source with TTL**
| Unified | Source | Transform |
|---|---|---|
| src_ip/dst_ip/src_port/dst_port | `srcip/dstip/sport/dsport` | direct (raw files; absent in partition CSVs) |
| timestamp | `Stime` | epoch seconds → UTC datetime |
| flow_duration | `dur` | direct |
| total_fwd/bwd_packets | `Spkts` / `Dpkts` | direct |
| total_fwd/bwd_bytes | `sbytes` / `dbytes` | direct |
| flow_bytes_per_s | `Sload` | bits/s → bytes/s (`/8`) proxy |
| fwd_iat_mean | `Sintpkt` | source interpacket time |
| init_win_bytes_fwd | `swin` | TCP window |
| ttl_src / ttl_dst | `sttl` / `dttl` | direct (**unique to UNSW-NB15**) |
| label | `attack_cat` + `Label` | multiclass + binary (0/1; Normal→benign) |
| **unavailable** | — | single flow IAT, per-flag counts, header_length |

Raw `UNSW-NB15_1..4.csv` are **header-less**: use `attach_raw_header()` with
`RAW_COLUMN_ORDER` (49 names) before mapping. Partition CSVs are headered with
lowercase names; the adapter's tolerant `_get()` and `case_insensitive_schema`
accept both.

## Preprocessing (`preprocessing.py`) — train-only

`FlowPreprocessor.fit(train)` learns **from training records only**:
- numeric **median** (imputation) + **mean/std** (standardization); `std==0`→`1.0` guard;
- categorical **vocabulary** of `protocol` (index 0 = `<unk>` for unseen test values).

`transform()` applies those fitted stats. Calling `transform` before `fit`
raises (leakage guard). Missing values (including source-unavailable features)
are imputed with the train median and flagged by a `<feat>_was_missing`
indicator (informative missingness). No clipping, no row dropping, no fabrication.

## Temporal (`temporal.py`)

- `normalize_timestamp` → tz-aware UTC (naive assumed UTC).
- `chronological_split(train_frac, val_frac)` → **sorts by time and cuts by
  time; never shuffles.** Datasets without timestamps (CICIoT2023) raise
  `TemporalError` rather than faking an order.
- `assert_chronological_disjoint` → verifies train ≤ val ≤ test in time.
- `generate_windows(window_seconds)` → contiguous non-overlapping windows; each
  optionally builds its `G_t`.
- `build_graph` → entities + directed edges from flows (skips flows without IPs;
  aggregates bytes, degrees, protocols, and an attack flag per edge).

## Leakage checks (`leakage.py`)

`run_leakage_checks(train, val, test, preprocessor=...)` returns a `LeakageReport`:
1. **temporal_order** — train max-time ≤ val/test min-time (skipped if no time).
2. **entity_overlap** — hosts shared between train/test (medium warning →
   suggests GroupKFold by host; can hard-fail via `entity_overlap_warn_only=False`).
3. **duplicate_rows** — identical feature keys spanning splits.
4. **preprocessor_fit** — the preprocessor is fitted (train-only) before use.

`report.raise_if_failed()` is provided for CI/tests.

## Tests

`tests/` (run `python -m pytest -q`, 53 tests) cover schema/availability,
each adapter's mapping + no-fabrication guard, train-only preprocessing,
temporal split/window/graph, leakage checks, and an end-to-end pipeline.
Tests use hand-written synthetic rows; a separate `smoke_real_data.py`
(not in the suite) validates adapters against the real local files.

## What is intentionally NOT done here
- No raw file is read/modified by the library at import; adapters take rows.
- No datasets are merged and no label taxonomies are unified (per-dataset spaces
  preserved for OOD/novelty work; a coarse `label_binary` view is provided).
- No packet-level feature is synthesized. PacketFeatureRecord awaits a real PCAP
  parser (Phase 2) — CTU-13 and UNSW-NB15 PCAPs are the only packet sources.
