# Sentinel-X — Phase 4: Core Network World Model

Phase 4 implements the **Sentinel-X world model** — the core research component.
It learns `P(S_{t+1} | S_t)` over a *dynamic network state* (how network
behaviour evolves) rather than classifying the current traffic window.

Scope note: uncertainty/OOD, MITRE, counterfactuals, and the frontend are
intentionally **not** implemented in this phase.

## Architecture

```
dynamic graph G_t
   -> GAT / GraphSAGE        (PyTorch Geometric message passing; node embeddings)
   -> graph pooling          (mean/max/sum -> graph state h_t)
   -> GRU / LSTM             (temporal latent z_t over h_{t-n..t})
   -> forecasting head       (predict next latent state z_{t+1})   [PRIMARY]
   -> auxiliary risk head    (attack risk @ t+H)                   [AUXILIARY]
```

- **Graph encoder** (`worldmodel/graph_encoder.py`): stacked `GATConv`
  (consumes edge features via `edge_dim`) or `SAGEConv` (uses topology). The
  graph is never flattened before the GNN. Empty graphs (datasets without
  entities) fall back to a zero graph state and are flagged not graph-capable.
- **Temporal encoder** (`worldmodel/temporal_encoder.py`): configurable GRU/LSTM
  over the sequence of graph states, projected to the latent state `z_t`.
- **Model** (`worldmodel/model.py`): ties it together, exposes `encode_state`
  (returns `z_t` for downstream reuse: forecasting, novelty/OOD, explainability,
  trajectory analysis), a forecasting head (`z_t -> z_{t+1}`), and an auxiliary
  risk head. Checkpoint save/load included.

Everything is configurable (`worldmodel/config.py`, `configs/world_model.yaml`);
no hyperparameters are hardcoded.

## Loss (documented, multi-objective)

`L = λ_state · (1 − cos(ẑ_{t+1}, sg(z_{t+1}))) + λ_risk · wBCE(risk, attack@t+H)`

- **Future-state prediction (primary, self-supervised).** Predict the next
  latent state and compare (cosine distance) against the encoder's own encoding
  of the actual future window sequence, computed with **stop-gradient** to
  prevent representation collapse. This directly targets `P(S_{t+1}|S_t)`.
- **Malicious-risk (auxiliary, supervised).** Class-weighted BCE on `attack@t+H`
  where labels exist, so the world model stays comparable to the Phase-3
  baselines on the same target. The model is never reduced to `z_t → attack
  yes/no`. See `worldmodel/losses.py` for the full rationale.

`λ_state` / `λ_risk` are configurable (default 1.0 / 1.0).

## Training

`worldmodel/train.py`: PyTorch train/val loops, early stopping on the **primary
future-state objective** (`selection_metric="state"`), best-epoch checkpointing,
reproducible seeds, gradient clipping (recurrent + message-passing stability),
configurable LR / batch size / epochs. Threshold for the auxiliary risk head is
tuned on validation (target precision, F1-optimal fallback) and applied to test.

## Protocol (apples-to-apples with Phase 3)

`worldmodel/data.py` reuses `sentinelx.experiments.common`: identical windows,
`seq_len`, `horizon`, chronological split, seed, and leakage policy as the
Phase-3 baselines. The future-state target is the graph sequence ending at
`t+H`; inputs never include `t+H`, so no future leaks.

## How to run

```
# build the CTU-13 cache first (Phase 2) if not present, then:
python scripts/train_world_model.py --config configs/world_model.yaml
# or override on the CLI:
python scripts/train_world_model.py --dataset ctu-13 --epochs 50 \
    --combos gat_gru,gat_lstm,graphsage_gru,graphsage_lstm
```

## Outputs

- `models/sentinel_x/model.pt` — best-combo checkpoint (state dict + config).
- `models/sentinel_x/config.yaml` — winning configuration.
- `models/sentinel_x/metadata.json` — metrics, params, provenance.
- `experiments/world_model_results.csv` — one row per architecture combo.
- `experiments/world_model_report.md` — comparison + interpretation.
- `experiments/world_model_metrics.json` — full nested results.

## Results (CTU-13 scenario 11, forecast @ t+1)

All four combos ran on real data. The primary future-state validation loss
converges to ≈0.003–0.006 (predicted next latent ≈ actual next latent). The
auxiliary risk head reaches PR-AUC ≈0.87–0.90, in the range of the Phase-3
baseline bar (LogReg 0.917, GRU 0.955, GraphSAGE 0.736). Best combo by the
primary objective: `graphsage_lstm`. Exact numbers live in
`experiments/world_model_results.csv` / `world_model_report.md`.

The validation split is small and highly imbalanced (5 positives), so the risk
head's tuned threshold yields a high FPR; the threshold-free ranking metrics
(PR-AUC / ROC-AUC) are the honest comparison points. This is reported, not
hidden.

## Tests

`tests/test_worldmodel.py` (26 tests): graph encoder (GAT + GraphSAGE, edge
features and topology actually consumed), temporal encoder (GRU + LSTM), tensor
dimensions, forward/backward passes, checkpoint save/load, deterministic
inference, variable graph sizes, config round-trip, loss behaviour, training
loop + reproducibility, and the experiment driver. All previous tests remain
passing (132 total).
