# Sentinel-X — Final System Audit

**Date:** 2026-09-05
**Scope:** Complete end-to-end validation of the research claim, ran against the
existing code, checkpoint, and cached data (nothing was retrained or rebuilt).
**Method:** static code review of every subsystem + live execution: full test
suite (284 tests), and every FastAPI endpoint exercised against the **real**
trained checkpoint (`models/sentinel_x/model.pt`) and the **real** cached
CTU-13 data, plus review of the on-disk Phase-9 experiment artifacts.

---

## 0. The research claim, judged against evidence

> *Sentinel-X learns network-state evolution and forecasts future attack
> trajectories while estimating uncertainty and detecting unfamiliar behaviour.*

| Claim sub-part | Mechanism exists & runs | Empirically validated as *effective* |
|---|---|---|
| Learns network-state evolution | **YES** — GNN→temporal→latent, real forecast head; K-step state cosine-sim ≈ 0.99 | **PARTIAL** — state is predicted well; but on 1 tiny dataset |
| Forecasts future attack trajectories | **YES** — autoregressive rollout + risk head + trajectory builder | **WEAK** — forecast risk ROC-AUC is low/erratic (0.41 @ h1); "lead time" ≈ 0 windows |
| Estimates uncertainty | **YES** — genuine MC-Dropout (30 passes, real variance) | **WEAK** — uncertainty↔error correlation = −0.20; ECE = 0.21 (poorly calibrated) |
| Detects unfamiliar behaviour (OOD) | **YES** — real Mahalanobis detector, leakage-safe threshold | **FAILS on this data** — OOD AUROC = 0.30 (worse than chance) |

**Verdict:** the *architecture and machinery* for the full claim are genuinely
implemented, wired together, and reproducible. The *scientific claim of
effectiveness is NOT established* — on the only graph-capable dataset used
(CTU-13 scenario 11, a very small split), the forecasting/uncertainty/OOD
results are weak-to-failing and the world model does **not** beat the simple
baselines. Sentinel-X is a **complete, honest research prototype**, not a
validated detection system.

---

## 1. Component status

Legend: **IMPLEMENTED** (real, tested, runs) · **PARTIAL** · **PLACEHOLDER** · **NOT IMPLEMENTED**

| # | Component | Status | Notes |
|---|---|---|---|
| 1 | Data pipeline (loaders, preprocessing, windows) | IMPLEMENTED | stdlib-only unified layer; real CTU-13 binetflow parsed (107,251 rows) |
| 2 | Real dataset loading | IMPLEMENTED | CTU-13 scenario 11 from disk; cache present for 4 datasets |
| 3 | Temporal graph construction | IMPLEMENTED | real directed comm graphs from src/dst IP; 195 windows / 62,106 nodes / 59,369 edges |
| 4 | Baselines (LogReg, GRU/LSTM, Temporal GNN) | IMPLEMENTED | measured, on identical split/protocol |
| 5 | Sentinel-X world model core | IMPLEMENTED | GraphSAGE/GAT → GRU/LSTM → latent → forecast head + aux risk head |
| 6 | K-step forecasting (autoregressive rollout) | IMPLEMENTED | real recursion `z ← forecast_head(z)`; K∈{1,3,5,10} evaluated |
| 7 | Risk head | IMPLEMENTED | auxiliary sigmoid risk on latent |
| 8 | MC-Dropout uncertainty | IMPLEMENTED | dropout re-enabled at inference, fixed input, 30 passes, real variance |
| 9 | Calibration (ECE/MCE/Brier, temperature scaling) | IMPLEMENTED | real binning; ECE measured = 0.206 |
| 10 | OOD / novelty | IMPLEMENTED (mechanism) / PARTIAL (efficacy) | real Mahalanobis; AUROC 0.30 on test |
| 11 | Attack trajectory | IMPLEMENTED | observed (labels) + forecast (rollout) stages, temporal order enforced |
| 12 | MITRE ATT&CK mapping | PARTIAL | **hand-written heuristic rules** on aggregate signals, not a learned/technique-level classifier (honestly scoped to coarse tactics) |
| 13 | Explainability | IMPLEMENTED | real gradient×input attribution + GAT attention (association-only, disclaimed) |
| 14 | Propagation analysis | IMPLEMENTED | descriptive graph spread analytics over real edges |
| 15 | Counterfactual simulation | IMPLEMENTED | real clone→intervene→re-rollout→diff; labelled "modelled outcome" |
| 16 | Forecast stability | IMPLEMENTED | controlled input-perturbation sensitivity (explicitly not MC-Dropout) |
| 17 | Research infra (reproducibility/experiments/comparison) | IMPLEMENTED / PARTIAL | seed+env captured; **git commit = null** (workspace is not a git repo) |
| 18 | FastAPI serving | IMPLEMENTED | thin routes → service layer → real ML; no tensors/hardcoded metrics escape |
| 19 | Frontend (React/Vite) | IMPLEMENTED (UI) / **contains FAKE fallback** | builds cleanly; but ships a full client-side mock that fabricates every metric when the backend is unreachable |
| 20 | Full test suite | IMPLEMENTED | **284 passed** in ~38s |

---

## 2. Specific integrity checks requested

| Risk checked | Result | Evidence |
|---|---|---|
| **Temporal leakage** | **CLEAN** | Splits are chronological by `t_index`/timestamp with hard asserts `train[-1] < val[0] < test[0]` (`experiments/common.py`, `data/temporal.py`); `leakage.py::_check_temporal` tested and passing |
| **Feature leakage** | **CLEAN** | Window features are per-window aggregates; preprocessor fit is train-only and guarded (`_check_preprocessor`); no scaler shared across splits in the world-model path |
| **Test-set contamination** | **CLEAN (split) / DISCLOSED (data)** | No split contamination. **BUT** build report discloses `duplicate_rows: 2952 test rows (18.35%) share an identical feature key with train` — inherent to the capture, not a split defect. Reported, not hidden. |
| **Fabricated metrics (Python)** | **NONE** | Every served number derives from the model at call time; verified live via all endpoints |
| **Fake uncertainty** | **NONE** | MC-Dropout re-enables only dropout modules, resamples masks, identical fixed input; live variance = 2.9e-4 |
| **Fake OOD** | **NONE** (mechanism real) | Real Mahalanobis distance + in-distribution threshold; live: score 6.93 > threshold 6.76 → novel. Efficacy is poor (AUROC 0.30) but real. |
| **Fake counterfactuals** | **NONE** | Real graph edit + re-rollout; live isolate_node produced genuine (small) risk deltas |
| **Hardcoded UI metrics** | **FOUND in frontend mock** | `frontend/src/api/scenario.ts` synthesises risk/graphs/uncertainty/OOD/MITRE via a `mulberry32` PRNG + scripted phases when backend is down. Live mode fetches the real API. |
| **Broken API contracts** | **NONE** | All 14 endpoints returned HTTP 200 with schema-valid bodies against the real runtime |
| **Inconsistent preprocessing** | **NONE** | One `graph_cache`/`ExperimentConfig` path shared by baselines, world model, K-step, serving |
| **Inconsistent feature schemas** | **MINOR** | Node/edge feature names consistent (5 node / 3 edge). Frontend mock uses different SEQ_LEN(8)/K(6)/LATENT_DIM(16) than the real model (4/5/64) — mock only, not the live contract. |
| **Missing checkpoints** | **NONE** | `models/sentinel_x/model.pt` present, loads, 185,729 params |
| **Non-reproducible experiments** | **PARTIAL** | Seed=42, full env + package versions + run manifests captured; **git commit null** (not a git repo) weakens exact reproducibility |

---

## 3. Datasets actually used

- **CTU-13 scenario 11** — `.../CTU-13-Dataset/11/capture20110818-2.binetflow`
  (the attack-dense scenario). This is the **only** dataset any measured result
  in this project comes from. 107,251 flow records → 195 five-second windows.
  Class distribution: unknown 96,369 / benign 2,718 / attack 8,164.
- Cache directories exist for `cic-ids2018`, `ciciot2023`, `unsw-nb15` but no
  model/experiment results were produced from them in this validation. The
  `cross_dataset` Phase-9 experiment result should be treated as unverified
  here (single-dataset run).

**Split sizes vary by experiment framing** (all chronological, all leakage-safe):
- World-model / baselines: 133 train / 28 val / 30 test windows (seq_len 4, horizon 1); test = 19 positives / 11 negatives.
- K-step (K=10): 127 / 27 / 28 (fewer usable samples because K future windows must exist).
- Graph-cache manifest: 143 / 34 / 18 (raw windowing before seq/horizon trimming).
This is a **very small evaluation set** — the single biggest limitation on every claim below.

---

## 4. Experiments actually run + real measured results

All values below are read from on-disk artifacts produced by
`scripts/run_experiment.py` (run manifest `run_experiment_20260905T114600Z.json`,
seed 42, completed) and confirmed live.

### 4.1 Forecast-attack@t+1 comparison (`experiments/model_comparison.csv`)
| Model | PR-AUC | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|
| logistic_regression | **0.9172** | 0.8636 | 1.0 | **0.9268** | 0.2727 |
| sequence_gru | **0.9546** | 0.7917 | 1.0 | 0.8837 | 0.4545 |
| temporal_gnn_sage | 0.7358 | 0.8750 | 0.3684 | 0.5185 | 0.0909 |
| **sentinel_x_world_model** | 0.8957 | 0.6552 | 1.0 | 0.7917 | **0.9091** |

> **The world model does not beat the baselines** on this dataset. Its recall is
> 1.0 but at FPR 0.91 (it flags 10 of 11 benign windows). The GRU baseline has
> the best PR-AUC (0.955); LogReg the best F1 (0.927).

### 4.2 K-step forecasting (`phase9/kstep_result.json`)
- State prediction is strong and stable: cosine-sim 0.997 (h1) → 0.978 (h10).
- Risk PR-AUC degrades with horizon: 0.864 (h1) → 0.871 (h5) → 0.537 (h10).
- Risk ROC-AUC is low and erratic (0.41 h1, 0.55 h5, 0.25 h10) — the risk head
  ranks poorly even where PR-AUC looks high (driven by high positive prevalence).

### 4.3 Early warning / lead time (`phase9/early_warning_result.json`)
- Coverage 1.0 (27/27 anchors with a future attack were warned), **but mean lead
  time = 0.148 windows, median = 0, max = 2.** In practice it warns at the same
  window the attack appears — it is **not** demonstrating meaningful look-ahead.

### 4.4 Uncertainty & calibration (`phase9/uncertainty_error_result.json`)
- MC-Dropout mean std = 0.0238 (30 passes) — real, non-zero.
- **uncertainty↔error correlation = −0.20** (weak, wrong sign — higher
  uncertainty does not reliably indicate higher error).
- **ECE = 0.206, Brier = 0.274** — poorly calibrated.

### 4.5 OOD / novelty (`phase9/ood_detection_result.json`)
- Detector real (Mahalanobis, dim 64, fit on 126 train latents, 95th-pct threshold).
- **AUROC = 0.301, AUPRC = 0.534, detection 0.42, false-acceptance 0.73** —
  on this data the novelty score does **not** separate known from unseen
  behaviour (below chance). Mechanism is sound; result is negative.

### 4.6 Live serving spot-check (this audit)
Every endpoint returned real model output against `t_index=189`:
- `/health`: model_loaded true, combo `graphsage_lstm`, 29 test anchors, threshold 0.4269.
- `/risk`: risk_now 0.427, 5-step forecast rising 0.444→0.480.
- `/uncertainty`: variance 2.89e-4 over 30 passes.
- `/novelty`: score 6.93 > threshold 6.76 → is_novel true.
- `/counterfactual` (isolate_node 0): genuine per-horizon risk deltas (~−5e-5).
- `/stability`: score 0.99999 (forecast barely moves under 5% input jitter).
- `/explainability`: gradient×input ranks `node.total_bytes_received` top.
- `/trajectory` & `/mitre`: 4 observed + 5 forecast stages, temporal order enforced.

---

## 5. Known limitations (not hidden)

1. **Single small dataset.** Every result rests on CTU-13 scenario 11 with a
   30-window test set (19 pos / 11 neg). Statistical power is minimal; metrics
   should be read as indicative, not conclusive. Multi-dataset / multi-seed
   evaluation has not been done.
2. **World model underperforms baselines** on the headline forecast task
   (Section 4.1). The research contribution is currently the *capability suite*,
   not superior detection accuracy.
3. **Poor calibration & unhelpful uncertainty** (ECE 0.21; unc↔error corr −0.20).
   The uncertainty is genuinely computed but not yet shown to be useful.
4. **OOD does not work on this data** (AUROC 0.30). Real method, negative result.
5. **"Early warning" lead time ≈ 0.** Forecasts don't yet buy meaningful time.
6. **MITRE mapping is heuristic**, not learned; coarse tactic level only.
7. **Dataset-level train/test duplicate feature keys (18.35%)** — inherent to
   the capture; inflates apparent baseline scores. Disclosed, not a split bug.
8. **Frontend mock fallback fabricates all metrics** when the backend is
   unreachable, and its mock `/health` reports `model_loaded: true`. A viewer
   cannot visually distinguish live model output from the scripted demo (only
   `version: "0.1.0-mock"` betrays it). This is a demo affordance, but it is a
   real risk of presenting fabricated numbers as model output.
9. **Reproducibility gap:** the workspace is not a git repo, so run manifests
   record `git_commit: null`. Exact code-state pinning is therefore absent.
10. **Cross-dataset experiment unverified** in this validation (single-dataset run).

---

## 6. Reproducibility instructions

Environment (verified): Python 3.14.3, numpy 2.5.1, scipy 1.18.1, scikit-learn
1.9.0, torch 2.14.0+cpu, torch-geometric 2.8.0.post1, mlflow 3.16.0, FastAPI
0.141.1. Windows (PowerShell). Raw data path resolved via
`data/metadata/data_paths.yaml` (`ctu-13` → the in-repo CTU-13 folder).

```powershell
# from d:\sentinel\sentinel
pip install -e ".[test,graph,serve]"

# 1. (already cached) rebuild the temporal graph cache if ever needed
python scripts/build_dataset.py --dataset ctu-13

# 2. baselines (reuses cache)
python scripts/run_baselines.py

# 3. world model (reuses cache) — produces models/sentinel_x/model.pt
python scripts/train_world_model.py

# 4. all research experiments (reuses checkpoint + cache; seed 42)
python scripts/run_experiment.py --experiment all
#    -> experiments/phase9/*_result.json, experiments/model_comparison.csv
#    -> run manifest in experiments/runs/

# 5. full test suite (expect: 284 passed)
python -m pytest -q

# 6. serve the real API
python scripts/serve.py            # sentinelx.serving.app:app

# 7. frontend (LIVE mode requires the API up on the /api proxy)
cd frontend; npm ci; npm run build     # tsc + vite build both pass
```

To guarantee the frontend shows **real** data, run the backend first; otherwise
it silently serves the fabricated scenario from `scenario.ts`.

**To make experiments fully reproducible:** initialise git in the workspace so
`git_commit` is captured in future run manifests.

---

## 7. Remaining risks

- **Presentation risk (highest):** the frontend can display fabricated metrics
  indistinguishably from live model output. Recommend a visible "MOCK / DEMO"
  banner whenever `mode === "mock"`, and make mock `/health` report
  `model_loaded: false`.
- **Overclaiming risk:** any statement that Sentinel-X "detects attacks
  early / detects novelty / is well-calibrated" is **not supported** by the
  current measurements. Claims must be limited to "implements and integrates"
  these capabilities.
- **Generalisation risk:** one small dataset; results may not transfer.
- **Reproducibility risk:** no git pinning; results reproducible by seed+env but
  not by exact commit.

---

## 8. Bottom line

**Sentinel-X is a genuinely and completely *implemented* research prototype:**
the full pipeline (data → temporal graphs → world model → K-step rollout →
risk/uncertainty/calibration/OOD → trajectory/MITRE → explain/propagation/
counterfactual/stability → FastAPI → frontend) is real, wired together, runs
end-to-end against real data, and passes 284 tests with no fabricated numbers in
the Python/serving stack.

**It is NOT a validated system.** On the single small dataset used, the world
model does not beat simple baselines, uncertainty is poorly calibrated and
weakly correlated with error, OOD performs below chance, and "early warning"
buys ~0 lead time. Two integrity items need attention: the frontend's
fabricated-metric fallback and the missing git-commit provenance.

Per the instruction: **Sentinel-X is NOT claimed complete.** The MITRE mapping
is PARTIAL (heuristic), the frontend contains a FAKE data fallback, research
reproducibility is PARTIAL (no git commit), and OOD/uncertainty/early-warning
are IMPLEMENTED-but-empirically-WEAK. Everything else is IMPLEMENTED and verified.
