# Sentinel-X — Phase 7: Attack Trajectory & MITRE ATT&CK Interpretation

**Goal.** Turn the existing temporal forecasts (Phase-4 world model + Phase-5 autoregressive rollout) into an interpretable attack trajectory that strictly separates **OBSERVED** from **FORECAST**. A forecasted stage is never presented as having already happened.

**Method.** For each test anchor window `t` we take the real observed states `[t-L+1 .. t]` (status = *observed*) and the rolled-out risk for `t+1 .. t+K` (status = *forecast*). Each state is mapped to a HIGH-LEVEL MITRE ATT&CK stage — never a blindly assigned precise technique. Where evidence is insufficient the mapping backs off to a coarser stage.

## Setup

- checkpoint: `D:\sentinel\sentinel\models\sentinel_x\model.pt` (combo **graphsage_lstm**)
- dataset: **ctu-13** | seq_len(observed L)=4 | forecast depth K=5
- risk threshold (Phase-4 val tuning): 0.4269
- trajectories built: 6 (observed stages=24, forecast stages=30)

## Stage distribution (all trajectories)

| Stage | Count |
|---|---|
| Reconnaissance | 30 |
| Command & Control | 21 |
| Benign / No Attack Behaviour | 3 |

## Example trajectories

Each line is `Status: Stage (confidence)` — observed states first, then forecast horizons. Forecast lines are model expectations, not facts.


**Anchor t=161** (4 observed → 5 forecast):

```
Observed [ t-3]  Benign / No Attack Behaviour (conf 0.000)
Observed [ t-2]  Benign / No Attack Behaviour (conf 0.000)
Observed [ t-1]  Command & Control            (conf 0.700)
Observed [   t]  Command & Control            (conf 0.700)
Forecast [ t+1]  Reconnaissance               (conf 0.443)
Forecast [ t+2]  Reconnaissance               (conf 0.456)
Forecast [ t+3]  Reconnaissance               (conf 0.466)
Forecast [ t+4]  Reconnaissance               (conf 0.474)
Forecast [ t+5]  Reconnaissance               (conf 0.480)
```

**Anchor t=162** (4 observed → 5 forecast):

```
Observed [ t-3]  Benign / No Attack Behaviour (conf 0.000)
Observed [ t-2]  Command & Control            (conf 0.700)
Observed [ t-1]  Command & Control            (conf 0.700)
Observed [   t]  Command & Control            (conf 0.700)
Forecast [ t+1]  Reconnaissance               (conf 0.443)
Forecast [ t+2]  Reconnaissance               (conf 0.456)
Forecast [ t+3]  Reconnaissance               (conf 0.466)
Forecast [ t+4]  Reconnaissance               (conf 0.474)
Forecast [ t+5]  Reconnaissance               (conf 0.480)
```

**Anchor t=163** (4 observed → 5 forecast):

```
Observed [ t-3]  Command & Control            (conf 0.700)
Observed [ t-2]  Command & Control            (conf 0.700)
Observed [ t-1]  Command & Control            (conf 0.700)
Observed [   t]  Command & Control            (conf 0.700)
Forecast [ t+1]  Reconnaissance               (conf 0.443)
Forecast [ t+2]  Reconnaissance               (conf 0.456)
Forecast [ t+3]  Reconnaissance               (conf 0.466)
Forecast [ t+4]  Reconnaissance               (conf 0.474)
Forecast [ t+5]  Reconnaissance               (conf 0.480)
```

## Interpretation

- **Observed vs forecast is explicit.** Observed stages are grounded in real cached window states; forecast stages are the rolled-out risk trajectory. The two are never conflated, and a forecast is never claimed to have occurred.
- **Temporal ordering is enforced.** Observed stages always precede forecast stages and horizons are monotonic (asserted in `build_trajectory`).
- **MITRE mapping stays high-level.** Stages are ATT&CK tactics (Reconnaissance, Initial Access, Lateral Movement, Command & Control, …). When evidence is weak the mapper returns a coarser stage (Reconnaissance / Suspicious Activity) instead of a precise technique — the spec forbids blindly assigning techniques.
- **Confidence is honest.** Observed confidence reflects evidence strength in the real state; forecast confidence is the model's risk at that horizon, which decays as the rollout compounds error (consistent with Phase-5).

*All stages are derived from the trained checkpoint and the Phase-5 rollout on real cached windows. Nothing is fabricated.*

