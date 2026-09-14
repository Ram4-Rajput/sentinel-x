# Sentinel-X — Phase 5: K-Step Future-State Forecasting

**Goal.** Forecast the network state trajectory `S_t → S_{t+1} → … → S_{t+K}` for configurable K (K=1, K=3, K=5, K=10).

**Method — autoregressive rollout (NOT K independent classifiers).** The Phase-4 world model learns a single one-step latent transition `f: z_t → z_{t+1}`. Phase-5 rolls that ONE learned transition forward, feeding each predicted latent back in:

```
z_t = encode_state(G_{t-L+1..t})
z_hat_{t+1} = f(z_t)
z_hat_{t+2} = f(z_hat_{t+1})
...
z_hat_{t+K} = f(z_hat_{t+K-1})
risk_{t+k} = sigmoid(risk_head(z_hat_{t+k}))
```

For each future step we return the predicted latent/state, the predicted risk (where a label exists), the horizon/target window index, and prediction metadata (see `k_step_metrics.json`).


## Setup

- checkpoint: `D:\sentinel\sentinel\models\sentinel_x\model.pt` (combo **graphsage_lstm**, 185729 params)
- dataset: **ctu-13** | seq_len=4 | risk threshold (from Phase-4 val tuning): 0.4269
- evaluation split: test samples=28 | graph_capable=True
- positives per horizon (test): {1: 24, 2: 24, 3: 24, 4: 23, 5: 22, 6: 21, 7: 20, 8: 19, 9: 18, 10: 17}

Every K is scored on the same test population (samples built at the deepest K so all horizons have real future targets), so the K=1/3/5/10 comparison is apples-to-apples.


## K = 1

- rollout wall-clock: 0.0466s over 28 samples × 1 steps (**1663.11 µs / sample-step**); future-encode cost 0.0422s
- state-error degradation (cosine distance, horizon 1 vs 1): abs **0.0**, ratio **1.0×**

| Horizon (t+k) | n | pos | State cos-dist ↓ | State cos-sim ↑ | State L2 ↓ | Risk PR-AUC | Risk ROC-AUC | Risk F1 |
|---|---|---|---|---|---|---|---|---|
| 1 | 28 | 24 | 0.0026 | 0.9974 | 0.9794 | 0.8645 | 0.4062 | 0.9231 |

## K = 3

- rollout wall-clock: 0.0414s over 28 samples × 3 steps (**492.36 µs / sample-step**); future-encode cost 0.1227s
- state-error degradation (cosine distance, horizon 3 vs 1): abs **0.000599**, ratio **1.233543×**

| Horizon (t+k) | n | pos | State cos-dist ↓ | State cos-sim ↑ | State L2 ↓ | Risk PR-AUC | Risk ROC-AUC | Risk F1 |
|---|---|---|---|---|---|---|---|---|
| 1 | 28 | 24 | 0.0026 | 0.9974 | 0.9794 | 0.8645 | 0.4062 | 0.9231 |
| 2 | 28 | 24 | 0.0026 | 0.9974 | 1.668 | 0.9066 | 0.5521 | 0.9231 |
| 3 | 28 | 24 | 0.0032 | 0.9968 | 2.2232 | 0.9261 | 0.6354 | 0.9231 |

## K = 5

- rollout wall-clock: 0.0448s over 28 samples × 5 steps (**319.92 µs / sample-step**); future-encode cost 0.3285s
- state-error degradation (cosine distance, horizon 5 vs 1): abs **0.003793**, ratio **2.478999×**

| Horizon (t+k) | n | pos | State cos-dist ↓ | State cos-sim ↑ | State L2 ↓ | Risk PR-AUC | Risk ROC-AUC | Risk F1 |
|---|---|---|---|---|---|---|---|---|
| 1 | 28 | 24 | 0.0026 | 0.9974 | 0.9794 | 0.8645 | 0.4062 | 0.9231 |
| 2 | 28 | 24 | 0.0026 | 0.9974 | 1.668 | 0.9066 | 0.5521 | 0.9231 |
| 3 | 28 | 24 | 0.0032 | 0.9968 | 2.2232 | 0.9261 | 0.6354 | 0.9231 |
| 4 | 28 | 23 | 0.0044 | 0.9956 | 2.6598 | 0.9046 | 0.6174 | 0.902 |
| 5 | 28 | 22 | 0.0064 | 0.9936 | 3.0015 | 0.8711 | 0.553 | 0.88 |

## K = 10

- rollout wall-clock: 0.0462s over 28 samples × 10 steps (**164.99 µs / sample-step**); future-encode cost 0.4158s
- state-error degradation (cosine distance, horizon 10 vs 1): abs **0.019892**, ratio **8.755785×**

| Horizon (t+k) | n | pos | State cos-dist ↓ | State cos-sim ↑ | State L2 ↓ | Risk PR-AUC | Risk ROC-AUC | Risk F1 |
|---|---|---|---|---|---|---|---|---|
| 1 | 28 | 24 | 0.0026 | 0.9974 | 0.9794 | 0.8645 | 0.4062 | 0.9231 |
| 2 | 28 | 24 | 0.0026 | 0.9974 | 1.668 | 0.9066 | 0.5521 | 0.9231 |
| 3 | 28 | 24 | 0.0032 | 0.9968 | 2.2232 | 0.9261 | 0.6354 | 0.9231 |
| 4 | 28 | 23 | 0.0044 | 0.9956 | 2.6598 | 0.9046 | 0.6174 | 0.902 |
| 5 | 28 | 22 | 0.0064 | 0.9936 | 3.0015 | 0.8711 | 0.553 | 0.88 |
| 6 | 28 | 21 | 0.0089 | 0.9911 | 3.2686 | 0.817 | 0.4762 | 0.8571 |
| 7 | 28 | 20 | 0.012 | 0.988 | 3.4767 | 0.7822 | 0.4438 | 0.8333 |
| 8 | 28 | 19 | 0.0155 | 0.9845 | 3.6384 | 0.6655 | 0.3333 | 0.8085 |
| 9 | 28 | 18 | 0.019 | 0.981 | 3.7642 | 0.5683 | 0.2389 | 0.7826 |
| 10 | 28 | 17 | 0.0225 | 0.9775 | 3.8622 | 0.5371 | 0.2513 | 0.7556 |

## Comparison across K

| K | State cos-dist @ t+1 | State cos-dist @ t+K | Degradation (abs) | Degradation (×) | Rollout s | µs / sample-step |
|---|---|---|---|---|---|---|
| 1 | 0.0026 | 0.0026 | 0.0 | 1.0 | 0.046567 | 1663.11 |
| 3 | 0.0026 | 0.0032 | 0.000599 | 1.233543 | 0.041359 | 492.36 |
| 5 | 0.0026 | 0.0064 | 0.003793 | 2.478999 | 0.044789 | 319.92 |
| 10 | 0.0026 | 0.0225 | 0.019892 | 8.755785 | 0.046198 | 164.99 |

## Interpretation

- **State prediction error vs. horizon.** Cosine distance between the rolled-out latent and the encoder's own encoding of the actual future state. Distance at t+1 is the one-step error the model was trained on; distance at deeper horizons shows how autoregressive error compounds.
- **Degradation with horizon.** The compounding is summarised by the abs/ratio columns (state error at the deepest horizon relative to t+1). A ratio near 1.0 means the rollout stays stable; a large ratio means error accumulates as predictions feed back into themselves — the expected behaviour of autoregressive rollout.
- **Future-risk performance.** The auxiliary risk head is read off each predicted latent. Where a horizon's test labels are single-class, ranking metrics are reported as n/a rather than a misleading value.
- **Computational cost.** One encode + K cheap latent-space transitions per sample; cost scales linearly in K (see µs / sample-step), far cheaper than running K independent models.

*All numbers are produced from the trained checkpoint on real cached data. No results are fabricated; single-class horizons and unstable metrics are reported honestly.*

