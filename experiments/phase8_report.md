# Sentinel-X — Phase 8: Explainability, Propagation, Counterfactual & Stability

**Goal.** Layer four analytical capabilities around the trained world model *without* changing the core forecasting architecture: (1) explain why the model produced a forecast, (2) analyse how suspicious behaviour propagates on the graph, (3) run simulation-only counterfactual interventions, and (4) measure forecast stability under controlled perturbations.

## Setup

- checkpoint: `D:\sentinel\sentinel\models\sentinel_x\model.pt` (combo **graphsage_lstm**, gnn=**graphsage**)
- dataset: **ctu-13** | observed L=4 | forecast depth K=5
- attention available (GAT only): **False**
- stability: 12 trials, epsilon=0.05
- anchors analysed: 20

## 1. Explainability

Attribution uses **gradient x input** sensitivity of the primary forecast (predicted next-latent movement) w.r.t. node/edge/temporal inputs. For GAT encoders, attention weights are additionally exposed as *supporting* evidence.

> **Attention is not causal.** It indicates where the model attended (association), reported as supporting evidence only.

Most-frequent top-attributed feature across anchors:

| Feature | Times ranked #1 |
|---|---|
| node.total_bytes_received | 20 |

## 2. Propagation

Supporting analytical layer describing how label-flagged suspicious behaviour spreads across the observed window graphs (affected/newly-affected nodes, velocity, direction, branching, persistence, growth). This does **not** replace the world model with reachability rules.

- anchors whose observed suspicious footprint is spreading (growth>0 & new nodes appear): **20** / 20

## 3. Counterfactual (simulation-only)

For each anchor we clone the graph, apply a **simulated** intervention (isolate the model's most-important node, or remove its top edge), rerun the trained world model, and compare the baseline vs intervention K-step forecast. Every such result is labelled: *"Modelled / simulated outcome."*

> No real network infrastructure is ever modified, and no causal certainty is claimed — the reported difference is the modelled effect under the learned dynamics.

- counterfactuals produced: **20** (mean modelled risk delta = -0.0001)

## 4. Forecast stability

Sensitivity of the forecast to controlled, valid input perturbations (small multiplicative feature jitter, structure preserved). Stability score = 1/(1+mean relative latent drift) in (0,1]; higher = more stable.

> **Not MC-Dropout.** Dropout is OFF throughout; this measures input-sensitivity, a different question from model uncertainty.

- mean stability score: **1.0** (min 1.0, max 1.0)

## Example anchors


**Anchor t=161**

```
explain target : state (value 4.5997)
top features   : node.total_bytes_received=0.002, node.total_bytes_sent=-0.001, node.flow_count=-0.000
top node       : idx 2 @ts3 (importance 0.001)
attention used : False
propagation    : affected=4 velocity=1.00 dir=balanced spreading=True
counterfactual : isolate_node -> mean risk delta -0.0001 (max|Δ| 0.0001)  [Modelled / simulated outcome.]
stability      : score 1.0000 (mean latent drift 0.0000)
```

**Anchor t=162**

```
explain target : state (value 4.6043)
top features   : node.total_bytes_received=0.002, node.total_bytes_sent=-0.002, node.flow_count=-0.000
top node       : idx 12 @ts3 (importance 0.001)
attention used : False
propagation    : affected=6 velocity=1.50 dir=balanced spreading=True
counterfactual : isolate_node -> mean risk delta -0.0001 (max|Δ| 0.0002)  [Modelled / simulated outcome.]
stability      : score 1.0000 (mean latent drift 0.0000)
```

**Anchor t=163**

```
explain target : state (value 4.6125)
top features   : node.total_bytes_received=0.002, node.total_bytes_sent=-0.002, node.flow_count=-0.000
top node       : idx 16 @ts0 (importance 0.001)
attention used : False
propagation    : affected=9 velocity=2.25 dir=fan-in spreading=True
counterfactual : isolate_node -> mean risk delta -0.0000 (max|Δ| 0.0000)  [Modelled / simulated outcome.]
stability      : score 1.0000 (mean latent drift 0.0000)
```

## Interpretation & discipline

- **Explainability** is model-attributed (gradients), not hand-written rules; attention is supporting association only, never causal.
- **Propagation** is a descriptive supporting layer over the real graphs; it does not forecast and does not replace the world model.
- **Counterfactuals** are simulation-only, applied to graph clones, and always labelled a modelled/simulated outcome with no causal-certainty claim.
- **Stability** is deterministic input-sensitivity, explicitly distinct from MC-Dropout model uncertainty (Phase 6).

*All quantities are derived from the trained checkpoint applied to real cached windows. Nothing is fabricated.*

