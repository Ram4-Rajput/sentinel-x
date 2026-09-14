# Sentinel-X — Phase 3 Baseline Report

**Question:** does modelling temporal structure and network topology provide measurable benefit over simpler approaches?

**Target (all baselines, identical):** given information up to and including window *t*, predict whether window *t+H* contains attack activity (`label_any_attack`). Inputs never include *t+H* → no future leakage. Same chronological split, seed, imbalance handling, and train-only preprocessing across all three baselines.

**Baselines:** (1) Logistic Regression on aggregated window features, (2) LSTM/GRU on the feature sequence, (3) Temporal GNN (GraphSAGE/GAT) on the graph sequence. The Sentinel-X world model is intentionally NOT included in this phase.


## Dataset: ctu-13

- windows: 195 | forecasting samples: 191 | seq_len=4 horizon=1
- split counts: {'train': 133, 'val': 28, 'test': 30} | positives per split: {'train': 7, 'val': 5, 'test': 19}

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|---|
| logistic_regression | 0.8636 | 1.0 | 0.9268 | 0.9172 | 0.9234 | 0.2727 | 19 | 3 | 8 | 0 |
| sequence_gru | 0.7917 | 1.0 | 0.8837 | 0.9546 | 0.933 | 0.4545 | 19 | 5 | 6 | 0 |
| temporal_gnn_sage | 0.875 | 0.3684 | 0.5185 | 0.7358 | 0.6627 | 0.0909 | 7 | 1 | 10 | 12 |

## Fairness & reproducibility

- Identical temporal split policy, leakage policy, target definition, seed, and train-only preprocessing for every baseline.
- Class imbalance handled uniformly (LogReg `class_weight=balanced`; sequence/GNN weighted BCE with train `pos_weight`).
- Decision threshold tuned on VALIDATION PR curve, applied to TEST (never tuned on test).
- No baseline intentionally weakened; no cherry-picking (all runs + seeds recorded to MLflow and this report).

## Interpretation guidance

- Compare each model's PR-AUC / recall at matched FPR. If temporal (LSTM/GRU) and topological (GNN) baselines do NOT beat Logistic Regression, that is a legitimate finding to report — it sets the bar the Sentinel-X world model must clear in a later phase.
- Datasets flagged `graph_capable=false` (no entities, e.g. CIC-IDS2018) cannot fairly support the GNN baseline; this is documented, not hidden.

## Dataset applicability to the three baselines

Not every dataset can fairly support every baseline (from the Phase-1 audit + Phase-2 build):

| Dataset | LogReg | LSTM/GRU | Temporal GNN | Reason |
|---|---|---|---|---|
| CTU-13 | ✅ | ✅ | ✅ | Has timestamps AND src/dst entities → windows + graphs. Fair for all three. |
| UNSW-NB15 (raw) | ✅ | ✅ | ✅ | Raw files have timestamps + IPs → graph-capable (partition CSVs lack IPs). |
| CIC-IDS2018 | ✅ | ✅ | ⚠️ | Has timestamps but NO src/dst IPs → windows form but graphs have 0 nodes; GNN not fair. |
| CICIoT2023 | ⚠️ | ❌ | ❌ | NO timestamps → cannot build temporal windows or graph sequences at all; only IID classification is possible. |

CTU-13 is therefore the primary dataset for the head-to-head comparison of all three baselines; the others are reported with their documented limitations rather than forced into an unfair comparison.
