# Sentinel-X — Phase 8: Explainability, Propagation, Counterfactual & Stability

Phase 8 layers four analytical capabilities **around** the trained Phase-4
world model, **without modifying the core forecasting architecture**. Every
quantity is derived from the existing checkpoint applied to real cached windows
(the same leakage-safe Phase-3/5 protocol). No retraining, nothing fabricated.

All code lives in `src/sentinelx/worldmodel/`:

| Layer | Module | Question answered |
|-------|--------|-------------------|
| Explainability | `explain.py` | Why did the model produce this forecast? |
| Propagation | `propagation.py` | How does suspicious behaviour spread on the graph? |
| Counterfactual | `counterfactual.py` | What if we intervened (simulation only)? |
| Stability | `counterfactual.py` | How sensitive is the forecast to valid perturbations? |

Driver: `worldmodel/phase8_experiment.py`. CLI: `scripts/run_phase8.py`.
Tests: `tests/test_phase8_explain_propagation_counterfactual.py`.

## 1. Explainability

`explain_forecast(model, input_seq, target=...)` returns model-attributed
evidence for one forecast:

- **top_features** — feature-level attribution via **gradient x input**
  sensitivity of the forecast objective w.r.t. node and edge features
  (`node.*` / `edge.*`, ranked by magnitude).
- **important_nodes / important_edges** — the same attribution rolled up per
  node and per edge (graph-structure attribution), ranked.
- **temporal_evidence** — sensitivity per observed timestep, showing which past
  windows mattered most.

The default `target="state"` explains the **primary** objective — the movement
(L2 norm) of the predicted next latent `z_hat_{t+1}`. `target="risk"` explains
the auxiliary risk logit instead.

**Attention discipline.** When the encoder is a GAT, learned attention weights
are additionally exposed as *supporting* evidence via temporary forward hooks
(the core module is not modified). Attention is reported as *association / where
the model looked* and is **explicitly not causal** (`ATTENTION_DISCLAIMER`).
GraphSAGE encoders have no attention, so that channel is simply absent and
attribution falls back to gradients only. The shipped checkpoint is GraphSAGE,
so `uses_attention=False` there — by design, not omission.

Attribution is deterministic (dropout off) and never mutates the caller's
tensors.

## 2. Propagation

`analyze_propagation(windows, suspicion_source=...)` is a **supporting
analytical layer** describing how flagged suspicious behaviour spreads across
the observed window graphs. It does **not** forecast and does **not** replace
the world model with graph-reachability rules.

A node is *affected* in a window when it participates in a suspicious edge.
Suspicion source is explicit:

- `"label"` — the cached edge `contains_attack` flag (ground truth in the cache).
- `"model"` — an externally supplied per-window set of suspicious edges (e.g.
  from the explainer's edge attribution), so propagation can be model-driven.

Measured: affected / newly-affected nodes, cumulative footprint,
**velocity** (new affected per window), **direction** (fan-out vs fan-in from
source/sink breadth), **branching** (mean out-degree in the suspicious
subgraph), **persistence** (fraction of windows a node stays affected), and
**growth** (least-squares slope of the cumulative affected set).

## 3. Counterfactual (simulation-only)

`simulate_intervention(model, input_seq, intervention, K=...)` follows the
workflow:

```
Current G_t -> clone graph -> apply simulated intervention
            -> run trained world model -> K-step forecast
            -> compare baseline vs intervention
```

Interventions (applied to a **clone**, never the real inputs):

- `isolate_node` — drop all incident edges + zero the node's activity features.
- `remove_edge` — drop a specific directed communication edge.
- `suppress_path` — drop a chain of edges forming a communication path.

The result reports per-horizon `baseline_risk`, `intervention_risk`,
`risk_delta`, and `latent_shift`. **Every result is labelled
`"Modelled / simulated outcome."`** No real network infrastructure is ever
modified, and no causal certainty is claimed — the reported difference is the
modelled effect under the learned dynamics.

## 4. Forecast stability

`evaluate_stability(model, input_seq, K=..., n_trials=..., epsilon=...)`
measures sensitivity of the forecast to **controlled, valid** input
perturbations: small multiplicative feature jitter (`f -> f*(1+eps*u)`, clamped
non-negative) that preserves graph structure. It compares each perturbed
forecast against the unperturbed baseline and returns a documented metric:

```
stability_score = 1 / (1 + mean_relative_latent_drift)   in (0, 1]
```

Higher = more stable. `epsilon=0` gives a perfect score of 1.0 (a sanity
anchor). **This is NOT MC-Dropout uncertainty** — dropout is OFF throughout, so
this measures deterministic *input-sensitivity*, a different question from the
Phase-6 model uncertainty. The two are kept conceptually separate.

## Running it

```
python scripts/run_phase8.py --k 5 --max-anchors 20 --stability-trials 12
```

Writes `experiments/{phase8_results.csv, phase8_report.md, phase8_metrics.json}`.

## Discipline summary

- Explainability is model-attributed (gradients), not hand-written rules;
  attention is supporting association only, never causal.
- Propagation is a descriptive supporting layer over the real graphs; it does
  not forecast and does not replace the world model.
- Counterfactuals are simulation-only on graph clones, always labelled a
  modelled/simulated outcome, with no causal-certainty claim.
- Stability is deterministic input-sensitivity, explicitly distinct from
  MC-Dropout model uncertainty.
