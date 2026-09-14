# Sentinel-X — Phase 6: Risk + Uncertainty + Calibration + Novelty/OOD

Three (four) **explicitly separate** signals. They are NEVER collapsed into a single "confidence" number:

- **Risk** — the trained risk head's estimated likelihood of malicious progression at the prediction target (model output, not a dashboard formula).
- **Uncertainty** — genuine MC-Dropout predictive variance (dropout kept active at inference; multiple stochastic passes).
- **Calibration** — ECE / Brier / reliability, plus optional temperature scaling as a separate stage.
- **Novelty / OOD** — Mahalanobis distance in the learned latent space, threshold derived from the training distribution.

## Setup

- checkpoint: `D:\sentinel\sentinel\models\sentinel_x\model.pt` (combo **graphsage_lstm**, 185729 params)
- dataset: **ctu-13** | seq_len=4
- test samples: 30 | dropout layers: 3 (active p>0: True)
- MC passes: **30** | OOD percentile: 95.0 | calibration bins: 10

### Signals stay separate (example)

For test window t=176 (label=1):

- Risk = **0.434**  (deterministic 0.4273)
- Uncertainty (std) = **0.0262**
- Novelty = **6.1734** (is_ood=False)

These are three independent numbers; the interpretation below is documented, not a universal rule.


## 1-2. Risk & Uncertainty (MC Dropout)

Risk and uncertainty are reported as **distinct** quantities. The MC-Dropout mean is an uncertainty-aware risk estimate; the variance/std is the uncertainty itself. Inputs are identical across passes — only dropout masks change (no input perturbation).

- mean predictive std (uncertainty): **0.0238**
- mean predictive variance: 0.000575
- uncertainty↔error correlation: **-0.2048** (positive ⇒ the model is more uncertain where it errs more).

Per-sample values in `uncertainty_results.csv`.

## 3. Calibration

| Stage | ECE ↓ | MCE ↓ | Brier ↓ | Temperature |
|---|---|---|---|---|
| before | 0.2057 | 0.2057 | 0.2743 | 1.0 |
| after (T-scaled) | 0.4538 | 0.4538 | 0.4374 | 0.1918 |

Reliability bins (confidence vs observed frequency) are in `calibration_results.csv`. Poor calibration is reported, not hidden — on this small, imbalanced split ECE/Brier should be read with the sample count in mind.

## 4. Novelty / OOD (Mahalanobis in latent space)

Known-training distribution = **benign train windows only**; the held-out unseen behaviour = **attack windows**, which never enter the manifold fit or the threshold. The threshold is the 95.0th percentile of in-distribution Mahalanobis distances (train/val), NOT tuned on test.

- detector: Mahalanobis, latent dim 64, fit on 126 benign latents, threshold **6.4059**
- test: 11 benign vs 19 attack windows

| Metric | Value |
|---|---|
| OOD AUROC | 0.3014 |
| OOD AUPRC | 0.5344 |
| Detection rate (recall on unseen) | 0.4211 |
| False acceptance rate | 0.7273 |

Per-sample `novelty_score` and `is_ood` in `ood_results.csv`.

## Keeping the signals distinct

Risk, uncertainty, and novelty answer different questions and can disagree: a sample can be high-risk with low uncertainty (a confident attack call), or low-risk with high novelty (benign but unfamiliar behaviour). We expose the RAW values plus a documented derived status and deliberately avoid hardcoding any single interpretation as universal truth.

*All numbers come from the trained checkpoint on real cached data. Single-class splits and unstable metrics are reported as n/a, not fabricated.*

