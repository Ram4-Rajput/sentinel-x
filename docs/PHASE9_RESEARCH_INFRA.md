# Sentinel-X — Phase 9: Research Experiment Infrastructure

Phase 9 **consolidates** the ML experimentation system that Phases 3–8 already
built into one reproducible, auditable surface. It introduces **no new model**
and **no new data protocol**: it reuses the trained Phase-4 checkpoint
(`models/sentinel_x/model.pt`), the Phase-2 window cache
(`data/processed/<dataset>/`), and the existing phase drivers. Nothing is
retrained or rebuilt unless explicitly requested.

Code lives in `src/sentinelx/research/`:

| Module | Responsibility |
|--------|----------------|
| `reproducibility.py` | deterministic seeding, dependency/version recording, config snapshots, per-run manifests |
| `experiments.py` | registry of named research experiments (dispatch + implementations) |
| `comparison.py` | assembles `experiments/model_comparison.csv` from genuinely measured results only |

Tests: `tests/test_research_infra.py`.

## Reproducible commands

Three thin, documented CLIs over the package. Each writes a machine-readable
run manifest to `experiments/runs/` (seed, exact config, dependency versions,
platform, git commit if present, argv, output artifacts, timestamps, status).

```
python scripts/train.py --dataset ctu-13            # (re)train the world model
python scripts/evaluate.py --ks 1,3,5,10 --comparison  # evaluate a checkpoint
python scripts/run_experiment.py --experiment all   # run named experiments + comparison
```

- `train.py` wraps the Phase-4 trainer. Produces the checkpoint, its config
  snapshot (`models/sentinel_x/config.yaml`), metadata, training logs, and the
  world-model experiment CSV/report/JSON.
- `evaluate.py` scores a trained checkpoint on the standard chronological test
  split (known attacks), optionally runs K-step forecasting, and optionally
  rebuilds the comparison. Uses the checkpoint's **validation-tuned threshold**
  — thresholds are never re-tuned on the final test set.
- `run_experiment.py` runs one, several, or `all` named experiments and (by
  default) assembles the comparison.

## Reproducibility guarantees

- **Seeds** — `seed_everything` (re-exported from `experiments.common`) fixes
  python / numpy / torch RNGs.
- **Dependency/version recording** — `capture_environment()` records python +
  numpy/scipy/sklearn/torch/torch-geometric/mlflow/yaml versions + platform.
- **Configuration snapshots** — `snapshot_config()` writes the exact run config
  to JSON alongside the outputs.
- **MLflow tracking** — reused from Phase 3 (`experiments/mlflow.db`).
- **Checkpoints / training logs** — produced by the Phase-4 trainer.
- **Run manifests** — `RunManifest` records everything needed to reproduce a run.

## Named experiments

Each reuses existing code / checkpoint / cache; unmeasurable cases are reported
as skipped or `n/a`, never fabricated.

| Experiment | What it measures | Reuses |
|------------|------------------|--------|
| `known_attacks` | risk head on the standard chronological test split | world model + Phase-3 split |
| `unseen_attacks` | detection of held-out (unseen) attack behaviour via novelty | Phase-6 Mahalanobis OOD (benign-train fit) |
| `kstep` | K = 1/3/5/10 autoregressive rollout state error + risk | Phase-5 driver |
| `early_warning` | lead time (windows) between first threshold crossing and first true attack | Phase-5 rollout |
| `missing_telemetry` | PR-AUC/recall degradation as node features are dropped | frozen checkpoint |
| `uncertainty_error` | MC-Dropout uncertainty ↔ error correlation, ECE | Phase-6 |
| `ood_detection` | Mahalanobis novelty AUROC/detection rate | Phase-6 |
| `forecast_stability` | forecast sensitivity to controlled perturbations | Phase-8 |
| `cross_dataset` | frozen model applied to other graph-capable datasets | frozen checkpoint |
| `baseline_comparison` | LogReg / GRU / GraphSAGE on the identical protocol | Phase-3 runner |

### Discipline

- **Chronological splits only** (inherited from `experiments.common`).
- **Thresholds never tuned on the final test set** — the checkpoint's
  validation-tuned threshold is reused everywhere.
- **No fabricated results** — single-class splits, non-graph datasets, and
  unmeasurable cases are reported honestly (`n/a` / skipped).

## Machine-readable comparison

`experiments/model_comparison.csv` has exactly these columns:

```
model dataset precision recall f1 pr_auc false_positive_rate
forecast_error lead_time ece ood_auroc stability
```

Only genuinely measured fields are populated. Baselines are single-step
classifiers without the world model's latent forecasting / novelty /
uncertainty machinery, so `forecast_error`, `lead_time`, `ece`, `ood_auroc`,
and `stability` are intentionally **blank** for them (never fabricated). Sources:

| Column | Source |
|--------|--------|
| precision/recall/f1/pr_auc/false_positive_rate | `baseline_results.csv` (baselines), `world_model_results.csv` best combo (world model) |
| forecast_error | `k_step_results.csv`, one-step (t+1) state cosine distance |
| lead_time | `early_warning` experiment (mean lead time in windows) |
| ece | `uncertainty_ood_metrics.json` → `calibration.before.ece` |
| ood_auroc | `uncertainty_ood_metrics.json` → `ood.metrics.auroc` |
| stability | `phase8_metrics.json` → `summary.mean_stability_score` |

## Outputs

```
experiments/model_comparison.csv                  # the final comparison
experiments/phase9/<experiment>_result.json       # per-experiment result
experiments/phase9/<experiment>/...               # scoped sub-artifacts
experiments/runs/<command>_<timestamp>.json       # reproducibility manifests
experiments/runs/<command>_<timestamp>.config.json# config snapshots (train)
```
