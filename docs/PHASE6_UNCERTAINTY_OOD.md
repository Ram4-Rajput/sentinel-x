# Sentinel-X — Phase 6: Risk + Uncertainty + Calibration + Novelty/OOD

Phase 6 adds three (four) **explicitly separate** signals on top of the Phase-4
world model. They are never collapsed into a single "confidence" number — each
answers a different question and they are allowed to disagree.

| Signal | Question it answers | Source |
|---|---|---|
| **Risk** | How likely is malicious progression at the target? | trained risk head output |
| **Uncertainty** | How sure is the model about that risk? | MC-Dropout predictive variance |
| **Calibration** | Do the probabilities mean what they say? | ECE / Brier / reliability (+ optional T-scaling) |
| **Novelty / OOD** | Is this behaviour unlike anything seen in training? | Mahalanobis distance in latent space |

Scope note: MITRE, counterfactuals, and the frontend remain out of scope.

## 1. Risk

Risk is the trained model's estimated likelihood of future malicious
progression at the prediction target (`attack@t+H`). It is read directly from
the auxiliary risk head (`sigmoid(risk_head(z_t))`) — the same output Phase-4/5
produced. **No arbitrary dashboard formula** is invented.

## 2. Uncertainty — genuine MC Dropout

`src/sentinelx/worldmodel/uncertainty.py`. The model already contains
`nn.Dropout` layers (graph encoder, forecast head, risk head). At inference we:

1. put the model in eval mode (LayerNorm/BatchNorm stats fixed), then
2. **re-enable only the dropout layers** (`dropout_active` context manager),
3. run **N stochastic forward passes** (default **30**, configurable via
   `--mc-passes`), each sampling a fresh dropout mask on the SAME fixed input,
4. compute the predictive **mean** (uncertainty-aware risk estimate) and the
   predictive **variance/std** (the uncertainty signal).

Key guarantees:
- Uncertainty comes only from dropout stochasticity. **Inputs are identical
  across passes** — we do NOT perturb inputs to fabricate variance (tested).
- `UncertaintyEstimate` exposes `risk_mean` and `uncertainty_variance`/`_std` as
  **distinct** fields; they are never merged.
- Models built with `dropout=0` are flagged (`has_active_dropout`) rather than
  silently returning zero variance.

## 3. Calibration

`src/sentinelx/worldmodel/calibration.py`:
- **Expected Calibration Error (ECE)** + **MCE** over equal-width confidence bins.
- **Brier score** (proper scoring rule).
- **Reliability analysis** — the per-bin confidence-vs-observed-frequency table
  (exposed as data, no plotting dependency).
- **Temperature scaling** (optional, **separate** post-hoc stage): fit a single
  scalar `T` on the **validation** logits by minimising NLL, then apply
  `sigmoid(logit / T)`. Calibration is reported **before and after** so any
  change is visible. Poor calibration is never hidden.

## 4. Novelty / OOD — Mahalanobis in latent space

`src/sentinelx/worldmodel/ood.py`. Uses the learned latent `z_t`:

```
G_t --encoder--> z_t --> Mahalanobis distance to the training distribution
```

- Fit mean `mu` and (ridge-regularised) covariance on **in-distribution**
  latents; score `d_M(z) = sqrt((z-mu)^T Sigma^{-1} (z-mu))`.
- Outputs `novelty_score` (raw distance) and `is_ood` (distance > threshold).
- **Threshold discipline:** the threshold is a high percentile (default 95th) of
  **in-distribution** (train/val) Mahalanobis distances. It is **never** tuned on
  the test/unseen set.

### Held-out unseen behaviour (no leakage)

The known-training distribution is fit on **benign train windows only**. The
held-out unseen behaviour is the **attack windows**, which never enter the
manifold fit or the threshold. This guarantees the OOD experiment does not leak
the held-out behaviour into the "known" distribution.

## Keeping the signals distinct

Every artifact reports the raw values plus a documented derived status. A sample
can be high-risk / low-uncertainty (confident attack call) or low-risk /
high-novelty (benign but unfamiliar). The interpretation is documented, not
hardcoded as universal truth.

## How to run

```
python scripts/run_phase6.py
python scripts/run_phase6.py --mc-passes 50 --ood-percentile 97.5
python scripts/run_phase6.py --no-temperature
```

Loads `models/sentinel_x/model.pt` and its `config.yaml` so evaluation matches
training exactly.

## Outputs (`experiments/`)

- `uncertainty_results.csv` — per-sample risk (deterministic + MC mean) and
  uncertainty (variance/std), kept distinct.
- `calibration_results.csv` — reliability bins + ECE/MCE/Brier, before and after
  temperature scaling.
- `ood_results.csv` — per-sample `novelty_score` and `is_ood`.
- `uncertainty_ood_report.md` — the combined narrative report.
- `uncertainty_ood_metrics.json` — full nested results.

## Experiments evaluated

- **Risk prediction** — the trained risk head on the same target as Phase-3/4/5.
- **Calibration** — ECE, Brier, reliability (before/after T-scaling).
- **Uncertainty vs prediction error** — correlation of MC-Dropout std with the
  absolute error of the MC-mean risk.
- **OOD known vs unseen** — Mahalanobis AUROC/AUPRC, detection rate and false
  acceptance rate for benign (known) vs attack (unseen) windows, with the
  held-out attack behaviour excluded from the fit.

## Results (CTU-13 scenario 11, graphsage_lstm checkpoint)

The test split is small (30 samples) and imbalanced, so absolute numbers are
reported honestly rather than tuned:

- **Uncertainty is real.** 30 MC-Dropout passes on fixed inputs produce non-zero
  predictive variance (mean std ≈0.024) purely from dropout masks.
- **Calibration is imperfect and shown.** ECE ≈0.21 / Brier ≈0.27 before
  scaling. On this tiny, imbalanced validation split temperature scaling does
  not help (it can worsen ECE) — reported before/after, not hidden.
- **OOD threshold is train/val-derived.** The 95th-percentile in-distribution
  threshold is applied to test; benign-vs-attack separation on this short tail
  split is weak and reported as-is.

Exact numbers live in the four `experiments/` artifacts.

## Tests

`tests/test_phase6_uncertainty_ood.py`: genuine MC Dropout (passes differ,
inputs unchanged, default 30, bad-pass rejection, no-dropout flagged), predictive
mean/variance wiring, risk-vs-uncertainty separation, calibration (reliability
bins, perfect/mis-calibration, Brier, temperature scaling incl. single-class
guard, logit round-trip), embedding generation (shape + determinism), OOD
scoring (Mahalanobis zero-at-mean, far-point, threshold from in-dist only,
flags, known/unknown separation, guards), and the experiment driver (four
artifacts, known-vs-unknown OOD, configurable passes, too-few-windows skip). All
previous tests remain green.
