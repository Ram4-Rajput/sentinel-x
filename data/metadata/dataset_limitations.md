# Sentinel-X — Per-Dataset Limitations (Phase 2)

The four datasets do NOT carry identical information. This is enforced honestly
by the `FeatureAvailability` contract and the pipeline's behaviour. Summary:

| Aspect | CIC-IDS2018 | CICIoT2023 | CTU-13 | UNSW-NB15 |
|---|---|---|---|---|
| Timestamps | ✅ (parsed) | ❌ none | ✅ (StartTime) | ✅ (Stime epoch) |
| Src/Dst IP (entities) | ❌ dropped in ML CSV | ❌ none | ✅ full | ✅ (raw files) |
| Temporal windows | ✅ | ❌ (no time) | ✅ | ✅ |
| Dynamic graph G_t | ❌ (no entities) | ❌ | ✅ reliable | ✅ (raw variant) |
| Packet-level (PCAP) | ❌ | ❌ | ✅ 13 PCAPs on disk | ⚠️ upstream only, not downloaded |
| Trajectory forecasting suitability | partial (time only) | ❌ | ✅ best | ✅ good |

## CIC-IDS2018
- **Available:** rich flow stats, timestamps, flags, IAT.
- **Unavailable:** Src/Dst IP, Src port, TTL → **no graph nodes/edges** (windows
  form from time, but G_t is empty). Confirmed in Phase-2 run (nodes=0, edges=0).
- **Inferred:** `header_length` = Fwd+Bwd header len; duration µs→s.
- **Omitted:** the mislabeled-folder relationship (the CICIoT2023 train/test/
  validation CSVs living in the same folder are excluded by file filter).
- **Property:** high duplicate-flow rate (≈85% on sampled prefix) — a known
  CIC-IDS2018 issue, recorded as a dataset-property, NOT a split defect.

## CICIoT2023
- **Available:** 47 engineered flow stats, flags, single aggregate IAT.
- **Unavailable:** timestamps, IPs, ports, TTL, directional (bwd) splits.
- **Consequence:** **cannot be windowed or graph-built** — the pipeline streams
  + records it and writes a `temporal_capable: false` manifest instead of
  fabricating an order. Suitable for IID classification / OOD, not trajectories.
- **Property:** extreme class imbalance; arrived pre-split (unknown provenance).

## CTU-13
- **Available:** full entities (SrcAddr/DstAddr/Sport/Dport), StartTime, NetFlow
  volumes; **PCAPs present** (13 files, 71.7 GB) for optional packet features.
- **Unavailable:** per-flag counts (only `State`), fwd/bwd packet split (TotPkts
  is a total, not mislabeled), TTL/window/header_length.
- **Derived:** total_bwd_bytes = TotBytes−SrcBytes; rates = Tot/Dur; syn from State.
- **Property:** Background label is *unknown* (not confirmed benign) → mapped to
  `unknown`, not `benign`. Many near-identical Background flows → duplicate-key
  overlap across splits (dataset property, not leakage). Host reuse across
  scenarios → consider GroupKFold by host for modeling.
- **Suitability:** best temporal-graph + trajectory-forecasting source.

## UNSW-NB15
- **Available:** full entities (raw files), timestamps, and the ONLY local TTL
  (`sttl/dttl`), TCP window (`swin/dwin`), retransmission (`sloss/dloss`).
- **Unavailable:** single flow IAT (only per-direction Sintpkt/Dintpkt), per-flag
  counts, header_length.
- **Note:** raw `UNSW-NB15_1..4.csv` are HEADER-LESS → use `--variant
  unsw-nb15-raw` (loader attaches the 49 canonical names). The partition CSVs
  (`UNSW_NB15_*-set.csv`) are the default headered variant but lack Src/Dst IP,
  so they window without graph entities.
- **PCAP:** exists upstream (~100 GB) but is NOT downloaded → packet features
  remain unavailable for UNSW until obtained. Never fabricated.
- **Property:** testbed uses few synthetic IPs → host-identity can leak; provided
  train/test split is IID (not temporal).

## Cross-cutting rules honoured
- No packet-level feature is synthesized from flow stats anywhere.
- Features a dataset lacks stay `None` and are documented `unavailable`; the
  train-only preprocessor imputes them like any other missing value but the
  availability record preserves the distinction.
- Temporal ordering + preprocessor-fit are HARD leakage gates (build aborts on
  failure). Duplicate-rows and host-overlap are reported as dataset properties.
