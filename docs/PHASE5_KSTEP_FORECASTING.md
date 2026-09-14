# Sentinel-X — Phase 5: K-Step Future-State Forecasting

Phase 5 turns the Phase-4 world model into a **multi-step forecaster**. It
predicts the network-state trajectory

```
S_t → S_{t+1} → S_{t+2} → … → S_{t+K}
```

for configurable `K` (1, 3, 5, 10 supported and reported). Scope note:
uncertainty/OOD, MITRE, counterfactuals, and the frontend remain out of scope.

## Method — autoregressive rollout (NOT K independent classifiers)

The Phase-4 model learns a single one-step latent transition `f: z_t → z_{t+1}`
(`forecast_head`) plus an auxiliary risk head. Phase 5 rolls that **one learned
transition** forward, feeding each prediction back in:

```
z_t          = encode_state(G_{t-L+1..t})   # observed history -> latent (once)
z_hat_{t+1}  = f(z_t)
z_hat_{t+2}  = f(z_hat_{t+1})
...
z_hat_{t+K}  = f(z_hat_{t+K-1})
risk_{t+k}   = sigmoid(risk_head(z_hat_{t+k}))
```

No new classifier is trained per horizon; the trajectory and its risk profile
come from recursively applying the same world model. See
`src/sentinelx/worldmodel/rollout.py`.

For each future step the API returns (`rollout_steps` /
`StepPrediction`): the predicted latent/state, the predicted risk (flagged when
a ground-truth label exists), the horizon and target window index, and
prediction metadata (combo, latent dim, deterministic, autoregressive).

## Evaluation — error as a function of horizon

`src/sentinelx/worldmodel/kstep_eval.py` scores the rollout on the SAME
protocol/split/seed as Phase-3/Phase-4 (`kstep_data.py` reuses
`experiments.common`). At each horizon `k`:

- **State prediction error** — cosine distance (matches the training loss) and
  L2 between the rolled-out latent `z_hat_{t+k}` and the encoder's own encoding
  of the ACTUAL future window sequence ending at `t+k`. The observed input path
  never includes future windows (leakage-safe); future encodings are used only
  as evaluation targets.
- **Future-risk performance** — PR-AUC / ROC-AUC / F1 / precision / recall of the
  risk head read off the predicted latent vs. `attack@(t+k)`. Single-class
  horizons report `n/a` rather than a misleading number.
- **Degradation with horizon** — state error at the deepest horizon vs. `t+1`
  (absolute and ratio), summarising how autoregressive error compounds.
- **Computational cost** — rollout wall-clock and µs / sample-step (one encode +
  K cheap latent transitions; linear in K).

Every K is scored on one shared test population (built at the deepest K so all
horizons have real targets), so K=1/3/5/10 are apples-to-apples.

## How to run

```
python scripts/run_kstep_forecast.py                 # K in 1,3,5,10
python scripts/run_kstep_forecast.py --ks 5
python scripts/run_kstep_forecast.py --checkpoint models/sentinel_x/model.pt --ks 1,3,5,10
```

Loads `models/sentinel_x/model.pt` (Phase-4 best combo `graphsage_lstm`) and its
`config.yaml` (dataset / seq_len / split fractions), so evaluation matches
training exactly.

## Outputs

- `experiments/k_step_results.csv`  — one row per `(K, horizon)`.
- `experiments/k_step_report.md`    — comparison + interpretation.
- `experiments/k_step_metrics.json` — full nested results.

## Results (CTU-13 scenario 11, graphsage_lstm checkpoint)

On the 28-sample test split the rollout behaves as an autoregressive forecaster
should:

- **State error compounds with horizon.** Cosine distance grows ≈0.0026 (t+1) →
  ≈0.0225 (t+10); L2 ≈0.98 → ≈3.86; degradation ≈8.8× at K=10, ≈2.5× at K=5,
  ≈1.2× at K=3. Near-term forecasts are accurate; error accumulates as
  predictions feed back into themselves.
- **Future-risk performance degrades with depth.** Risk PR-AUC ≈0.86 at t+1
  (consistent with the Phase-4 regime) declining to ≈0.54 at t+10 as the latent
  drifts. Reported honestly, not tuned.
- **Cost is linear in K.** One encode plus K latent-space transitions per
  sample; per-sample-step cost falls as the fixed encode overhead amortises.

Exact numbers live in `experiments/k_step_results.csv` /
`experiments/k_step_report.md`.

## Tests

`tests/test_kstep_forecasting.py` (32 tests): rollout shapes for K=1/3/5/10,
invalid-K rejection (0, negative, float, bool, non-int), rollout determinism,
autoregressive composition (step k+1 = `f`(step k)), checkpoint loading,
per-step contract, the K-step data layer (future targets + labels per horizon,
chronological/leakage-safe), evaluation metrics + degradation + cost, and the
experiment driver writing all three artifacts. All previous tests remain green
(164 total).
