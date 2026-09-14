# Sentinel-X — Phase 7: Attack Trajectory & MITRE ATT&CK Interpretation

Phase 7 turns the **existing** temporal forecasts (Phase-4 world model +
Phase-5 autoregressive rollout) into an **interpretable attack trajectory**
that strictly separates what the system has **OBSERVED** from what it merely
**FORECASTS**. No retraining, no new model outputs — every number comes from the
Phase-4 checkpoint and the Phase-5 rollout on real cached windows.

```
Observed:  Reconnaissance
Observed:  Command & Control
Forecast:  Initial Access
Forecast:  Lateral Movement
Forecast:  Command & Control
```

A forecasted stage is **never** presented as having already happened.

## Trajectory

`src/sentinelx/worldmodel/trajectory.py` builds a structured
`AttackTrajectory` anchored at a test window `t`:

- **Observed segment** — the real cached window states `[t-L+1 .. t]` (the same
  sequence fed to the world model). Each becomes an `observed` stage. Confidence
  reflects **evidence strength in the real data**, not a model forecast.
- **Forecast segment** — the Phase-5 rolled-out risk for horizons `t+1 .. t+K`.
  Each becomes a `forecast` stage. Confidence is the **model's risk** at that
  horizon (honestly decays as the rollout compounds error — see Phase-5).

Each `TrajectoryStage` carries the spec contract plus provenance:

```json
{
  "stage": "...",                 // high-level ATT&CK stage
  "status": "observed|forecast",  // NEVER claim a forecast occurred
  "confidence": 0.0,              // [0,1]
  "evidence": [ ... ],            // supporting signals
  "horizon": 0,                   // <=0 observed (0 = anchor t), +k forecast t+k
  "window_index": 163,
  "timestamp": "...",             // observed only; forecast = null (future)
  "source": "observed-state|forecast-rollout"
}
```

`build_trajectory` asserts the core invariant: **observed stages precede
forecast stages and horizons are monotonic** (`_assert_temporal_order`).

## MITRE mapping (high-level only)

`map_behaviour_to_stage` maps interpretable behaviour signals — reused from the
same `experiments.common._aggregate_window` the baselines/world-model consume
(attack ratio, malicious-edge fraction, flow volume, host fan-out, density) — to
**high-level ATT&CK stages (tactics)**:

    Reconnaissance · Initial Access · Execution · Persistence ·
    Privilege Escalation · Lateral Movement · Command & Control · Exfiltration

Discipline (per spec):

- **Evidence-gated.** Strong, corroborated signals (labelled attack + high
  attack ratio + malicious edges + a spread/volume shape) map to a specific
  high-level stage (e.g. Command & Control, Lateral Movement).
- **Back-off.** A single weak/uncorroborated signal maps to a coarse stage
  (`Reconnaissance` or the generic `Suspicious Activity`) rather than a precise
  technique. Confidence for coarse stages is capped low.
- **No precise techniques.** ATT&CK **technique IDs are never emitted** — only
  tactic-level stages. This satisfies "do not blindly assign precise
  techniques; if evidence is insufficient, return a higher-level stage."

Forecast stages are inferred from the model's forecast risk plus the observed
present state's behaviour shape (a prior for plausible progression). No future
window features are used — that would be leakage.

## How to run

```
python scripts/run_trajectory.py                       # K=5, all test anchors
python scripts/run_trajectory.py --k 10 --max-trajectories 5
python scripts/run_trajectory.py --checkpoint models/sentinel_x/model.pt --k 5
```

Loads `models/sentinel_x/model.pt` (Phase-4 `graphsage_lstm`) and its
`config.yaml`, rebuilds the SAME K-step samples as Phase-5, and rolls the
learned one-step transition forward for the forecast risk.

## Outputs (`experiments/`)

- `trajectory_results.csv`   — one row per stage (t_index, order, stage, status,
  confidence, horizon, window_index, timestamp, source, evidence).
- `trajectory_report.md`     — narrative + observed→forecast example trajectories.
- `trajectory_metrics.json`  — full nested trajectories.

## Results (CTU-13 scenario 11, graphsage_lstm checkpoint)

On the short test tail the trajectories behave as designed: observed windows
that carry real attack behaviour map to **Command & Control** (grounded in the
labelled state), while the forecast segment sits just below the Phase-4 decision
threshold and honestly backs off to **Reconnaissance** rather than confidently
claiming escalation. Observed and forecast are never conflated. Exact per-stage
values live in the three `experiments/trajectory_*` artifacts.

## Tests

`tests/test_trajectory_mitre.py` (20 tests): observed-vs-forecast distinction
(forecast never marked observed, no observed timestamp on forecasts), temporal
ordering (observed before forecast, monotonic horizons, guard assertions),
missing stages (empty observed / empty forecast / empty trajectory), uncertain
mappings (weak evidence → coarse stage), unsupported techniques (only tactic-
level stages, never a `T####` id), confidence handling (always `[0,1]`; observed
= evidence strength, forecast = model risk), the per-stage dict contract, and the
experiment driver (three artifacts, K horizons per forecast, too-few-windows
skip). All previous tests remain green.

STOP — Phase 7 complete.
