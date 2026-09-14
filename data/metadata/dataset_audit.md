# Sentinel-X Dataset Audit — Phase 1

**Skill applied:** `sentinel-x-dataset-research`
**Date:** 2026-09-04
**Scope:** Locate and inspect the four named datasets. Audit only — **no raw file was moved, renamed, deleted, merged, preprocessed, or modified.** No model was trained.

---

## 0. Headline findings (read first)

The filesystem does **not** match the assumed layout, and **filenames are misleading**. Verifying by *schema and labels* (not names) revealed:

1. **There is no `data/raw/` in the workspace.** The workspace root `D:\sentinel\sentinel\` contains only `.kiro`. The raw data lives in a **sibling** location: `D:\sentinel\data\` and `D:\DATA\`.

2. **The folder `D:\DATA\CSE-CIC-IDS2018\` is a mixed bag of three different datasets:**
   - `processed/*.csv` → **genuine CIC-IDS2018** (80-column CICFlowMeter, day-stamped Feb–Mar 2018). ✅
   - root `train.csv` / `test.csv` / `validation.csv` → **actually CICIoT2023** (47-column schema, Mirai/DDoS-*/Recon-* labels), *mislabeled* under the IDS2018 folder. ✅ identity confirmed by schema + labels.
   - `CTU-13-Dataset.tar.bz2` → **CTU-13** botnet dataset (compressed, not extracted). ✅

3. **UNSW-NB15 was missing — now RESOLVED (downloaded 2026-09-04).** The original artifact (`OneDrive_2026-09-03.zip`, 1.2 MB) held **only `*_Error.txt` placeholder stubs** — debris of a failed download whose **root cause is Avast "Web/Mail Shield" TLS interception** (it swaps site certificates, breaking HTTPS transfers). It has been **re-downloaded from the Kaggle mirror** (`mrwellsdavid/unsw-nb15`) into `D:\DATA\UNSW-NB15\` and **verified as real data**: full 49-feature schema (incl. `srcip/dstip/sport/dsport`, `sttl/dttl`, `swin/dwin`, `Stime/Ltime`, `attack_cat/Label`), **2,540,044** raw records + the provided train/test partition, and **9 attack categories**. Status: **PRESENT+VERIFIED**.

4. **No packet-level (PCAP) data is present** for CIC-IDS2018 or CICIoT2023 — both are flow-level only. CTU-13 *may* contain PCAPs inside its tarball (verify on extraction).

> Consequence: two of the four "expected" datasets are not where/what their names imply, and one (UNSW-NB15) is unusable. Recommendations below are based on **what is actually on disk**, not popularity.

---

## 1. Where each dataset actually is

| Dataset | Real location | On-disk form | Size |
|---|---|---|---|
| CIC-IDS2018 | `D:\DATA\CSE-CIC-IDS2018\processed\*.csv` | 10 day CSVs (CICFlowMeter, 80 cols) | ~6.57 GB |
| CICIoT2023 | `D:\DATA\CSE-CIC-IDS2018\{train,test,validation}.csv` | 3 pre-split CSVs (47 cols) | ~2.21 GB |
| CTU-13 | **extracted** at `D:\sentinel\data\CSE-CIC-IDS2018\CSE-CIC-IDS2018\CTU-13-Dataset\CTU-13-Dataset\` (tarball untouched) | 13 scenarios: `.binetflow` + `.pcap` each | ~2.6 GB binetflow (+ PCAPs) ✅ |
| UNSW-NB15 | `D:\DATA\UNSW-NB15\` (downloaded 2026-09-04 via Kaggle) | 9 CSVs (raw 4 + partition 2 + features/GT/events) | ~1.4 GB ✅ |

Also present: `D:\sentinel\data\CSE-CIC-IDS2018.zip` and `D:\DATA\CSE-CIC-IDS2018.zip` (~3.93 GB each) — archived copies; treat as redundant/unused if `processed/` is the authoritative extraction (verify before deleting — and we are not deleting anything in Phase 1).

---

## 2. Per-dataset audit

### 2.1 CIC-IDS2018 — *genuine* (flow-level, temporal)
- **Files / types:** 10 `.csv` (CICFlowMeter), 0 PCAP. One per capture day (14 Feb → 02 Mar 2018).
- **Schema:** 80 columns, `Label` last.
- **Timestamp:** `Timestamp` (`dd/MM/yyyy HH:mm:ss`) — strong, usable.
- **Source/dest:** `Dst Port` + `Protocol` only. **Src/Dst IP are NOT in the ML CSVs** → cannot build graphs from these files alone.
- **Flow features:** rich (fwd/bwd packet & byte stats, flow bytes/pkts per sec).
- **Packet-level:** none.
- **TCP flags:** flag **counts** (FIN/SYN/RST/PSH/ACK/URG/CWE/ECE Flag Cnt, Fwd/Bwd PSH/URG Flags). No per-packet flag sequence.
- **TTL/window/frag/retransmit:** TTL ✗; window = `Init Fwd/Bwd Win Byts` only; fragmentation ✗; retransmission ✗.
- **IAT:** rich — `Flow IAT`, `Fwd IAT`, `Bwd IAT`, `Active`/`Idle`.
- **Attack families (sampled):** FTP-BruteForce, SSH-Bruteforce, DoS-Hulk, DoS-SlowHTTPTest (full taxonomy also includes DDoS-LOIC/HOIC, Web attacks, Infiltration, Bot).
- **Benign:** present (`Benign`), proportion varies per day.
- **Missing values:** known `Flow Byts/s` / `Flow Pkts/s` Infinity/NaN — clean before use.
- **Duplicates:** known duplicate-flow issue — dedup on **train split only**.
- **Class imbalance:** severe and **per-day single-family dominance** (e.g. Wed-14-02 ≈ 90% FTP-BruteForce; Fri-16-02 ≈ DoS SlowHTTPTest+Hulk).
- **Temporal coverage:** 10 weekday captures over ~2.5 weeks.
- **Scenario/day boundaries:** one file = one day = one scenario. **Do not merge across days** before schema+temporal analysis.
- **Leakage risks:** per-day family dominance (time↔label), Inf/NaN correlated with attacks, timestamp-as-feature, cross-day shuffle.
- **Temporal graph?** Partial — great time+IAT, but **no entity IDs** blocks `G_t` node identity.
- **Required files:** all 10 day CSVs. **Unused:** root `CSE-CIC-IDS2018.zip` (redundant archive).

### 2.2 CICIoT2023 — *mislabeled as IDS2018 train/test/validation* (flow-level, NON-temporal)
- **Files / types:** 3 `.csv` (pre-split), 0 PCAP.
- **Schema:** 47 columns, `label` last (34-class taxonomy).
- **Timestamp:** **none** → cannot form temporal windows or ordered sequences.
- **Source/dest:** **no IPs**, no explicit port column; only protocol-presence flags (HTTP/HTTPS/DNS/… + TCP/UDP/ICMP/ARP).
- **Flow features:** rich engineered stats (`flow_duration`, `Rate`, `Srate`, `Drate`, `Magnitue`, `Radius`, `Covariance`, `Variance`, `Weight`, single `IAT`).
- **Packet-level:** none.
- **TCP flags:** per-flow indicators + counts (`*_flag_number`, `*_count`).
- **TTL/window/frag/retransmit:** all ✗ (fragmentation only implied by label names).
- **IAT:** single aggregate `IAT` (not fwd/bwd split).
- **Attack families (sampled 200k):** DDoS-ICMP/UDP/TCP/PSHACK/SYN/RSTFIN/SynonymousIP_Flood, DoS-UDP/TCP/SYN_Flood, Mirai-greeth/udpplain/greip, MITM-ArpSpoofing, DNS_Spoofing, Recon-Host/OS/Port, VulnerabilityScan, DDoS-/DoS-HTTP_Flood.
- **Benign:** `BenignTraffic` present but **rare (~2.3%)**.
- **Missing values:** protocol-absence zeros are structural, not missing; still verify.
- **Duplicates:** high risk (flood attacks → many near-identical flows).
- **Class imbalance:** **extreme** — DDoS/DoS dominate; benign & Recon/Web classes <1%.
- **Temporal coverage:** unknown (no timestamp).
- **Scenario boundaries:** arrived **pre-split** with **unknown methodology** → possible split leakage.
- **Leakage risks:** *(critical)* unknown-provenance pre-split may share correlated flows across train/val/test; no time separation; accuracy meaningless under this imbalance.
- **Temporal graph?** No (no time, no entities).
- **Required files:** all 3 (but re-verify split independence). **Unused:** none.

### 2.3 CTU-13 — *EXTRACTED & VERIFIED (2026-09-04)* (flow-level + entities + time + PCAPs → graph-capable)
- **Location:** `D:\sentinel\data\CSE-CIC-IDS2018\CSE-CIC-IDS2018\CTU-13-Dataset\CTU-13-Dataset\` (scenarios `1`–`13`). Original tarball at `D:\DATA\CSE-CIC-IDS2018\CTU-13-Dataset.tar.bz2` left untouched.
- **Files / types:** **all 13 scenarios present**, each with a `.binetflow` **and** its `.pcap` (scenario 1 also `.html`; scenario 7 also an `.exe` malware sample). ~2.6 GB of `.binetflow` total.
- **Schema (CONFIRMED, 15 cols):** `StartTime, Dur, Proto, SrcAddr, Sport, Dir, DstAddr, Dport, State, sTos, dTos, TotPkts, TotBytes, SrcBytes, Label`.
- **Timestamp:** `StartTime` — strong, ordered (confirmed e.g. `2011/08/10 09:46:59`).
- **Source/dest:** **full** `SrcAddr`/`DstAddr`/`Sport`/`Dport` → **graph-constructable**.
- **Flow features:** bidirectional NetFlow volumes; **packet-level: YES — per-scenario PCAPs are present** (the only local packet-level data).
- **TCP flags:** via `State` (e.g. `S_RA`), not per-flag columns.
- **TTL/window/frag/retransmit:** not native (has `sTos`/`dTos` type-of-service).
- **IAT:** derivable from `StartTime`+`Dur` per host pair.
- **Attack families:** 13 real botnet scenarios (Neris, Rbot, Virut, Menti, Sogou, Murlo, NSIS.ay, …).
- **Benign:** `Normal` + `Background` (Background = *unknown*, not confirmed benign). **Label is a free-text `flow=...` string** — must be parsed into Background/Normal/Botnet.
- **Class imbalance:** extreme (confirmed) — scenario 9 sample: 294,179 Background / 5,770 Normal / **51 Botnet** per 300k rows.
- **Temporal coverage:** 13 continuous timed captures, **10–19 Aug 2011**.
- **Scenario boundaries:** 13 explicit scenarios → natural scenario-disjoint splits.
- **Leakage risks:** host reuse across scenarios (use scenario/host `GroupKFold`); treat `Background` cautiously; sort by `StartTime`, never shuffle before split.
- **Temporal graph?** **Yes — the primary dataset for dynamic host-communication graphs `G_t`.**
- **Required files:** the 13 `.binetflow` files (present). **PCAPs:** keep — they're the only local packet-level source.

### 2.4 UNSW-NB15 — *downloaded & verified (2026-09-04)*
- **Location:** `D:\DATA\UNSW-NB15\` (Kaggle mirror `mrwellsdavid/unsw-nb15`; official SharePoint is browser-only).
- **Files / types:** `UNSW-NB15_1..4.csv` (raw, header-less, 700k/700k/700k/440,044 = **2,540,044** records), `UNSW_NB15_training-set.csv` (~175,341) + `UNSW_NB15_testing-set.csv` (~82,332) (labeled, with headers), `NUSW-NB15_features.csv` (49-feature dictionary), `NUSW-NB15_GT.csv`, `UNSW-NB15_LIST_EVENTS.csv`. (A `Payload_data_CICIDS2017.csv.zip` also shipped in the bundle — belongs to CICIDS2017, not UNSW-NB15; ignore.) 0 PCAP.
- **Schema:** 49 features. **Src/Dst IP + ports** (`srcip/sport/dstip/dsport`) ✅; **TTL** (`sttl/dttl`) ✅ *(unique among local datasets)*; **TCP window** (`swin/dwin`) ✅; **retransmission** (`sloss/dloss`) ✅; timestamps `Stime/Ltime` ✅; `attack_cat` (9-class) + binary `Label`.
- **Attack families:** Normal, Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms.
- **Graph/temporal?** **Yes** — full entities + timestamps make it graph-capable (second such dataset alongside CTU-13).
- **Key cautions:** raw CSVs are **header-less** (attach names from `NUSW-NB15_features.csv` before use); the provided train/test split is **IID, not temporal** (don't present it as a forecasting split); testbed uses few synthetic IPs (host identity can leak — consider `GroupKFold` by host).
- **Root cause of the earlier broken copy:** Avast "Web/Mail Shield" TLS interception (`Issuer: CN=Avast Web/Mail Shield Root`) broke HTTPS downloads; worked around by routing Python TLS validation through the Windows trust store (`truststore`).

#### 2.4-OLD (superseded) — original missing/broken finding
- Only artifact is `OneDrive_2026-09-03.zip` → entirely `*_Error.txt` stubs (failed download).
- **No usable data.** Do **not** fabricate its TTL/window features.
- **Action required:** re-download the UNSW-NB15 flow CSVs (49-feature schema with `srcip/sport/dstip/dsport/proto/sttl/dttl/swin/dwin/…` and `attack_cat/Label`) before it can serve as a generalization dataset.

---

## 3. Compatibility matrix (summary)

Full machine-readable version: `dataset_compatibility.csv`. Summary:

| Axis | CIC-IDS2018 | CICIoT2023 | CTU-13 | UNSW-NB15 |
|---|---|---|---|---|
| Present on disk | ✅ | ✅ (mislabeled) | ✅ (tarball) | ❌ broken |
| Temporal info | strong (per-flow ts + IAT) | **none** (no ts) | strong (StartTime) | — |
| Src/Dst entities | ❌ (IP dropped) | ❌ | ✅ full | — |
| Flow features | rich (80) | rich (47) | NetFlow | — |
| Packet features | ❌ | ❌ | maybe (PCAP?) | — |
| Attack labels | multi-class families | 34-class modern | botnet families | — |
| Attack progression | medium | low | **high** | — |
| Graph construction | low (as-is) | none | **high** | — |
| Unseen/OOD experiments | medium | **strong** | medium | — |
| Cross-dataset eval | medium | medium | medium | — |
| Compute cost | high (6.57GB) | medium (2.21GB) | med-high (decompress) | — |
| Leakage risk | med-high (per-day dominance) | **high** (pre-split, no time) | medium (host reuse) | — |

---

## 4. Recommendations (based on actual files, not popularity)

**A. Primary training dataset → CIC-IDS2018 (`processed/*.csv`).**
Richest flow-level features with genuine per-flow timestamps and clear per-day scenario boundaries — the best base for learning normal→attack flow behaviour and short-horizon forecasting. Split **chronologically by day**; dedup and fit preprocessing on train days only; clean `Flow Byts/s`/`Flow Pkts/s` Inf/NaN.

**B. Secondary training / generalization dataset → UNSW-NB15 (`D:\DATA\UNSW-NB15\`).** *(Updated 2026-09-04 — now that it's downloaded.)*
Distinct enterprise attack taxonomy (Exploits, Fuzzers, Generic, Backdoor, Worms, …) and the **only** local source carrying TTL (`sttl/dttl`), TCP window (`swin/dwin`), and retransmission (`sloss/dloss`) — ideal for testing cross-dataset generalization and for enriching the feature space. It also has full entities + timestamps, so it doubles as a supplementary graph/temporal source. **Cautions:** attach column names from `NUSW-NB15_features.csv` to the header-less raw files; treat the provided train/test partition as **IID, not temporal**; use host-grouped splits (testbed IPs). This restores the ideal 4-way separation and frees **CICIoT2023 to be the pure OOD source (D)**.

**C. Temporal trajectory dataset → CTU-13.**
The **only** dataset on disk with full entity identifiers (`SrcAddr`/`DstAddr`/`Sport`/`Dport`) **plus** ordered `StartTime` across 13 timed scenarios. This is what enables the Sentinel-X core: dynamic network graphs `G_t`, botnet propagation over time, and future-state / attack-trajectory forecasting. Extract it (Phase 2), split by scenario, and sort by `StartTime`.

**D. Unseen / OOD evaluation dataset → CICIoT2023.**
Its attack families (Mirai variants, IoT-specific floods, ARP/DNS spoofing, IoT recon) are largely disjoint from CIC-IDS2018's enterprise families, making it a strong **novelty/OOD** probe for representation-space detection (per `sentinel-x-ml-research`: OOD must use learned representations). Using it for both (B) and (D) is acceptable **only** with strictly separated splits; if a cleaner OOD source is needed, reserve specific CTU-13 botnet scenarios as held-out unseen families instead.

> Update (2026-09-04): **UNSW-NB15 has been downloaded and verified**, so the ideal 4-way separation is now achievable — **A** CIC-IDS2018 (primary), **B** UNSW-NB15 (generalization), **C** CTU-13 (temporal trajectory / graph), **D** CICIoT2023 (unseen/OOD). No dataset needs to pull double duty anymore. Still outstanding: extract CTU-13; PCAPs remain undownloaded for all datasets (packet-level features stay `unavailable` until then).

---

## 5. Constraints honored in this phase

- ✅ Did **not** merge datasets. ✅ Did **not** normalize together. ✅ Did **not** train a model.
- ✅ Did **not** randomly sample away temporal structure (label sampling was head-of-file only, for inspection).
- ✅ Did **not** delete/move/rename/modify any raw file. ✅ Did **not** invent missing features (absences recorded as `unavailable`).
- ✅ Identified datasets by **schema + labels**, not by trusting filenames.

## 6. Recommended next steps (Phase 2 — not executed here)
1. ~~Extract `CTU-13-Dataset.tar.bz2`~~ ✅ DONE (2026-09-04). 13 scenarios verified, 15-col schema confirmed, per-scenario **PCAPs are included** (only local packet-level data). Remaining CTU-13 step: write a parser for the free-text `flow=...` Label into Background/Normal/Botnet + family before modeling.
2. ~~Re-download UNSW-NB15~~ ✅ DONE (2026-09-04, Kaggle CSVs verified in `D:\DATA\UNSW-NB15\`). Remaining UNSW step: attach headers to raw `UNSW-NB15_1..4.csv` from `NUSW-NB15_features.csv` before modeling. Optional: obtain UNSW-NB15 PCAPs (~100 GB, official SharePoint) if packet-level features are wanted.
3. Rename/relocate the mislabeled CICIoT2023 CSVs out of the `CSE-CIC-IDS2018` folder (with a copy, not a move, until confirmed) to remove the naming hazard.
4. Confirm whether `processed/` or the `.zip` archives are the authoritative CIC-IDS2018 copy before any cleanup.
5. Verify CICIoT2023 train/val/test split independence, or re-derive splits.

## 7. Generated artifacts
- `data/metadata/datasets.yaml` — per-dataset provenance, schema, labels, risks.
- `data/metadata/feature_mapping.yaml` — mapping into the Sentinel-X unified schema (with honest `unavailable` markers).
- `data/metadata/dataset_compatibility.csv` — full comparison matrix.
- `data/metadata/dataset_audit.md` — this report.

**STOP — audit complete. ML model not implemented.**
