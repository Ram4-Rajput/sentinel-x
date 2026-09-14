# Sentinel-X — Phase 2 Data Processing Report

**Scope:** Full-scale streaming data loading + temporal graph sequence pipeline,
built ON TOP of the existing unified data layer (adapters/schema/preprocessing/
temporal split/windows/graph builder/leakage checks were reused, not rebuilt).

**Raw data immutability:** No file under any raw location was modified. All
outputs go to `data/processed/<dataset>/`. Raw lives outside the workspace
(`D:\DATA`, `D:\sentinel\data`) and is read-only-streamed.

---

## Runs (representative real-data builds)

| Dataset | Variant | Records | Rows read | Windows | Graphs | Nodes | Edges | Time (s) | Peak MB | Cache | Temporal? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ctu-13 | binetflow | 300,000 | 300,000 | 91 | 91 | 158,952 | 162,223 | 52.8 | 335.6 | 12.2 MB | yes |
| cic-ids2018 | CICFlowMeter | 200,000 | 200,000 | 722 | 722 | 0 | 0 | 41.4 | 660.0 | 0.2 MB | yes (no graph — no IPs) |
| unsw-nb15 | raw 1–4 | 150,000 | 150,000 | 20 | 20 | 817 | 3,218 | 21.0 | 612.2 | 0.15 MB | yes |
| ciciot2023 | 47-col | 5,000 | 5,000 | 0 | 0 | 0 | 0 | 4.8 | 362.3 | ~1 KB | **no (no timestamps)** |

(Values are from bounded `--max-records` runs used for smoke/medium validation;
full runs scale linearly — memory stays bounded by chunk size, never O(file).)

### Class distribution (binary view, from the runs above)
- **ctu-13:** unknown(Background)=297,047, benign=2,953 (Botnet is a tiny fraction; extreme imbalance — matches audit).
- **cic-ids2018:** attack=162,906, benign=37,094 (day-file dependent; single-family dominance).
- **unsw-nb15:** benign=131,169, attack=18,831.
- **ciciot2023:** rare BenignTraffic, DDoS/DoS dominate (streamed only; no windows).

---

## What each stage does (pipeline)

```
raw files (immutable)
  -> loaders.iter_unified_records   streaming, chunked (csv module), adapter-routed,
                                     malformed-row-safe, progress-logged
  -> chronological_split (existing)  TRAIN < VAL < TEST by time, never shuffled
  -> assert_chronological_disjoint   explicit temporal-leakage assertion (hard gate)
  -> FlowPreprocessor.fit(train)     train-only stats (existing; leakage-safe)
  -> windows.generate_windows_strided  configurable window_size + stride (overlap OK);
                                        reuses data.build_graph for each G_t
  -> graph_cache.GraphCache          JSONL graph sequences + manifest (+ optional PyG .pt)
  -> BuildReport                     records/windows/graphs/nodes/edges/time/mem/limits
```

### Windowing (documented, per acceptance criteria)
- Window `i` covers `[t0 + i*stride, t0 + i*stride + window_size)`.
- `stride == window_size` → contiguous non-overlapping (default).
- `stride < window_size` → overlapping sliding windows.
- Time normalized to UTC; records sorted by time; **no shuffle**; timestamps preserved.
- Datasets without timestamps cannot be windowed → the builder records a
  limitation instead of fabricating order (CICIoT2023).

### Graph G_t = (V_t, E_t, X_t)
- **Nodes (V):** source + destination entities (host IPs) tracked by the existing
  `NetworkEntity`.
- **Edges (E):** directed communication relationships (`CommunicationEdge`),
  aggregating flow_count, total_bytes, attack flag, protocols.
- **Node features (X):** `out_degree, in_degree, total_bytes_sent,
  total_bytes_received, flow_count` — all derived from tracked flow aggregates.
  **No packet-level feature is fabricated.**

### Cache & reproducibility
- Layout: `data/processed/<dataset>/{windows/*.jsonl, graphs/*.pt (optional), metadata/manifest.json, metadata/build_report.json}`.
- Manifest stores: dataset, schema/preprocessing/pipeline versions, window config,
  split fractions, per-split stats, timestamp ranges, class distribution,
  preprocessor feature names, raw file list — enough to reproduce the transform.
- Reload verified: graphs reload from JSONL with **no raw CSV re-read**.
- Optional PyG `.pt` written when torch + torch_geometric are installed
  (JSONL remains the portable source of truth).

---

## Memory / scale evidence
- Peak memory 335–660 MB on 150k–300k record runs — bounded by `chunk_size`
  (default 100k rows), **not** by file size. The 3.8 GB CIC day file and 71.7 GB
  CTU-13 PCAPs are never loaded into RAM by the flow pipeline.
- Smoke (20k) → medium (150k–300k) both succeed before any full run.

## Errors / skipped
- All representative runs: `malformed_rows=0`, `skipped_records=0` on the sampled
  prefixes. The loader counts and skips malformed rows without aborting (tested).

## PCAP
- Optional, non-blocking Scapy interface (`pipeline/pcap.py`). CTU-13 ships 13
  PCAPs (71.7 GB); UNSW-NB15 PCAPs are not downloaded. Extraction is opt-in and
  never imported by the core build path. Only defensible packet features
  (length, TTL, TCP flags, payload length, timestamp) — no fabricated payloads.

## Reproducible CLI
```
python scripts/build_dataset.py --dataset cic-ids2018
python scripts/build_dataset.py --dataset ctu-13 --window-size 60 --stride 60 --max-records 300000
python scripts/build_dataset.py --dataset unsw-nb15 --variant unsw-nb15-raw --window-size 300 --stride 300
python scripts/inspect_state.py
```
No paths hardcoded (resolved via `--raw-path`, env `SENTINELX_<DATASET>_RAW`, or
`data/metadata/data_paths.yaml`).
