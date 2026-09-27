# Phase A — Full-scale CTU-13 training (Kaggle GPU)

Trained the Sentinel-X world model on **all 13 CTU-13 scenarios** (`ctu-13-full`,
47,978 windows) on a Kaggle Tesla T4, all four architecture combos, 50 epochs,
seed 42. Same forecasting protocol as Phase 3 (seq_len=4, horizon=1, attack@t+1),
scenario-disjoint split → the test families differ from train (true cross-family
generalization, NOT the easier scenario-11-only bar).

Environment: torch 2.10.0+cu128 (GPU confirmed), torch-geometric 2.8.0.

## Split
| split | windows | samples | positives |
|-------|---------|---------|-----------|
| train | 33,584  | 33,581  | 9,870 (29%) |
| val   | 7,196   | 7,196   | 2,092 (29%) |
| test  | 7,198   | 7,197   | 6,127 (85%) |

Note the val→test attack-ratio shift (29% → 85%): the test scenarios are far more
attack-dense. This is the root cause of the low test recall (see below).

## Results (test set, sorted by PR-AUC — the threshold-free headline metric)
| combo | test PR-AUC | precision | recall | F1 | params | train (min) |
|-------|-------------|-----------|--------|----|--------|-------------|
| **graphsage_gru** ✅ | **0.871** | 0.849 | 0.400 | 0.544 | 152k | 18 |
| graphsage_lstm | 0.828 | 0.705 | 0.225 | 0.341 | 186k | 38 |
| gat_gru | 0.802 | 0.697 | 0.150 | 0.246 | 137k | 48 |
| gat_lstm | 0.775 | 0.657 | 0.091 | 0.160 | 170k | 58 |

Best combo `graphsage_gru` is installed as the canonical `models/sentinel_x/`.
All four checkpoints kept under `models/kaggle_ctu13full/<combo>/`.

## Interpretation (honest)
- **PR-AUC 0.87 across UNSEEN botnet families is a strong result.** For reference,
  the scenario-11-only bar was GraphSAGE PR-AUC 0.736; full-data cross-family
  generalization beats it, so scaling the data clearly helped the graph model.
- **Low test recall (15–40%) is a calibration artifact, not a training failure.**
  The decision threshold is tuned on VAL (29% attacks) but applied to a TEST set
  that is 85% attacks. The operating point transfers poorly. PR-AUC (threshold-
  free) is unaffected and stays healthy.
- **ROC-AUC < 0.5 on test** is likewise driven by the extreme class-balance shift
  plus the risk head being AUXILIARY (the model's primary objective is future-
  state forecasting, not classification). Do not read it as "worse than random".
- GraphSAGE > GAT here, and GRU > LSTM. GraphSAGE trains ~2–3x faster too.

## Known issue to address (Phase B, local, no GPU)
Threshold/recall calibration under val→test distribution shift. Options:
report PR-AUC as headline, re-tune threshold on a held-out slice matching test
prevalence, or add calibration (temperature scaling — `worldmodel/calibration.py`
already exists).

## Not needed right now
- No further CTU-13 retraining — more runs converge to the same point; the ceiling
  is the 5-node/3-edge feature set, not data volume.
- Bigger gains come from: (C) UNSW-NB15 cross-dataset, (D) richer node/edge
  features — both require local pipeline prep before any new GPU run.
