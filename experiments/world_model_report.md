# Sentinel-X — Phase 4 World Model Report

**Objective.** Learn `P(S_{t+1} | S_t)` over a *dynamic network state* — how network behaviour evolves — rather than classifying the current window. Architecture:

```
dynamic graph G_t -> GAT/GraphSAGE -> node embeddings -> graph pooling
 -> graph state h_t -> GRU/LSTM -> temporal latent z_t
 -> forecasting head -> predicted next state z_{t+1}
 (+ auxiliary risk head -> attack@t+H)
```

**Loss (documented, multi-objective).** `L = λ_state · (1 − cos(ẑ_{t+1}, sg(z_{t+1}))) + λ_risk · wBCE(risk, attack@t+H)`. The primary term is self-supervised future-state prediction (stop-grad target prevents collapse); the auxiliary term is a class-weighted risk BCE so the model stays comparable to the Phase-3 baselines on the SAME target. The model is never reduced to `z_t → attack yes/no`.

**Protocol.** Identical windows / seq_len / horizon / chronological split / seed / leakage policy as Phase-3 (`sentinelx.experiments.common`), so results are apples-to-apples with the baselines.


## Dataset: ctu-13

- windows: 195 | forecasting samples: 191 | seq_len=4 horizon=1
- split counts: {'train': 133, 'val': 28, 'test': 30} | positives per split: {'train': 7, 'val': 5, 'test': 19} | graph_capable=True
- best combo (by PRIMARY future-state validation loss): **graphsage_lstm**


### Architecture comparison

*Selection column is the PRIMARY future-state validation loss (lower = better forecasting of the next network state). The remaining columns are the AUXILIARY risk head on TEST.*

| Combo | Params | Best epoch | Val state loss | Val total loss | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR |
|---|---|---|---|---|---|---|---|---|---|---|
| gat_gru | 137217 | 16 | 0.003 | 3.1487 | 0.6333 | 1.0 | 0.7755 | 0.8653 | 0.7895 | 1.0 |
| gat_lstm | 170241 | 17 | 0.0064 | 3.2398 | 0.6333 | 1.0 | 0.7755 | 0.8686 | 0.7943 | 1.0 |
| graphsage_gru | 152705 | 20 | 0.0048 | 3.0948 | 0.6552 | 1.0 | 0.7917 | 0.9008 | 0.8565 | 0.9091 |
| graphsage_lstm | 185729 | 16 | 0.0026 | 3.1938 | 0.6552 | 1.0 | 0.7917 | 0.8957 | 0.8469 | 0.9091 |

### Interpreting these numbers

- **Val state loss** reflects the PRIMARY future-state objective (1 − cosine similarity between the predicted and actual next latent state) and is what model selection uses. The risk-head columns are the AUXILIARY metrics, shown so the world model can be compared to the Phase-3 baseline bar (CTU-13 scenario 11: LogReg PR-AUC 0.917, GRU 0.955, GraphSAGE 0.736).
- A world model that forecasts future state well AND matches/beats the baselines on the shared risk target is the desired outcome; either result is reported honestly rather than tuned for the classifier alone.

### Reproducibility

- Fixed seed 42, deterministic RNGs, gradient clipping, early stopping on validation objective, best-epoch checkpointing. Config is saved to `models/sentinel_x/config.yaml`.
