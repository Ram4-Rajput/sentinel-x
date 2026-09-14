# Sentinel-X — Project Kickoff (paste this into a FRESH chat)

This is the **stable, phase-agnostic** kickoff note. Its name never changes.
Before pasting a new phase prompt, give a fresh chat session this file first —
it tells the new session where the project stands so it doesn't re-do work.
**Do NOT list `.kiro/skills/` (1,000+ dirs — huge token cost).**

## Phase status
| Phase | What it delivered | Status |
|-------|-------------------|--------|
| 1 | Dataset audit (4 datasets) | ✅ done |
| 2 | Unified data layer + streaming pipeline + temporal graph cache | ✅ done |
| 3 | Scientific baselines (LogReg, LSTM/GRU, Temporal GNN) | ✅ done |
| 4 | Sentinel-X world-model core | ✅ done |
| 5 | K-step future-state forecasting (autoregressive rollout) | ✅ done |
| 6 | Risk / uncertainty / calibration / novelty-OOD | ✅ done |
| 7 | Attack trajectory + high-level MITRE ATT&CK interpretation | ✅ done |
| 8 | Explainability + propagation + counterfactual + forecast stability | ✅ done |
| 9 | Research experiment infrastructure (reproducibility + named experiments + comparison) | ✅ done |
| 10 | Model serving & FastAPI (typed API over Phases 4–9; clean service layers) | ✅ done |
| 11+ | Frontend | ⏳ later (confirm scope) |

## Where things stand
- **Phase 1** — `data/metadata/dataset_audit.md`, `datasets.yaml`,
  `feature_mapping.yaml`, `dataset_compatibility.csv`, `dataset_limitations.md`.
- **Phase 2** — code in `src/sentinelx/data/` and `src/sentinelx/pipeline/`.
  Report: `data/metadata/phase2_processing_report.md`. Build CLI:
  `scripts/build_dataset.py`. Cache lands in `data/processed/<dataset>/`.
- **Phase 3** — code in `src/sentinelx/experiments/`. Run CLI:
  `scripts/run_baselines.py`. Results:
  `experiments/baseline_results.csv|baseline_metrics.json|baseline_report.md`.
  MLflow → `experiments/mlflow.db` (SQLite).
- **Phase 9** — code in `src/sentinelx/research/` (reproducibility, experiments
  registry, comparison). Reproducible CLIs: `scripts/train.py`,
  `scripts/evaluate.py`, `scripts/run_experiment.py`. Final artifact:
  `experiments/model_comparison.csv`. Per-experiment results in
  `experiments/phase9/`; run manifests in `experiments/runs/`. Doc:
  `docs/PHASE9_RESEARCH_INFRA.md`. Reuses the Phase-4 checkpoint + Phase-2 cache
  (no retraining/rebuilding).
- **Phase 10** — code in `src/sentinelx/serving/` (`app.py` routes, `services.py`
  ML adapter, `runtime.py` cached model+samples singleton, `schemas.py` Pydantic
  types). Launch CLI: `scripts/serve.py` (needs `pip install -e .[serve]`; FastAPI
  app object at `sentinelx.serving.app:app`). Doc: `docs/PHASE10_MODEL_SERVING.md`.
  No ML logic in routes; tensors never exposed. Reuses the Phase-4 checkpoint +
  Phase-2 cache (no retraining/rebuilding).
- **Tests:** `python -m pytest -q` → 284 passing (258 prior + 26 Phase-10 API).
  Do not break these.

## Verified environment
Python 3.14, numpy, scikit-learn 1.9, torch 2.14 (CPU), torch-geometric 2.8,
mlflow 3.16. Windows shell strips `$` in inline commands → put shell logic in a
temp `.ps1`/`.py` file instead.

## Real data locations (external, immutable — see data/metadata/data_paths.yaml)
- cic-ids2018: `D:\DATA\CSE-CIC-IDS2018\processed` (no IPs → no graph)
- ciciot2023: `D:\DATA\CSE-CIC-IDS2018` (no timestamps → no windows/graph)
- unsw-nb15: `D:\DATA\UNSW-NB15` (raw variant has IPs+time → graph-capable)
- ctu-13: `...\CTU-13-Dataset\CTU-13-Dataset` (time + IPs + PCAPs). **Scenario 11
  is the attack-dense one** used for Phase-3 (scenario 1's first 300k rows had 0
  attack windows). PCAPs (~72 GB) exist but packet features are optional.

## Baseline bar to beat (CTU-13 scenario 11, forecast attack@t+1)
- LogReg PR-AUC 0.917 | GRU PR-AUC 0.955 | GraphSAGE PR-AUC 0.736.

## Current target — Phase 4 (the actual next step)
Implement the **Sentinel-X world-model core** per `sentinel-x-ml-research`:
`GAT/GraphSAGE → graph pooling/state encoder → GRU/LSTM → temporal latent →
future-state forecasting head`. Evaluate on the SAME protocol as Phase 3
(same windows/splits/target/seed, `src/sentinelx/experiments/common.py`) so the
comparison against the baselines is apples-to-apples. Reuse the existing data
layer + cache; do NOT rebuild them. Keep it leakage-safe; keep risk/uncertainty/
novelty conceptually separate (those come in a later phase — confirm scope first).

## Rules that keep the fresh session cheap
1. Never list `.kiro/skills/`. Read specific files only.
2. Resume from the files above; don't re-audit or re-run prior phases.
3. Batch instructions; use temp script files for shell logic.
