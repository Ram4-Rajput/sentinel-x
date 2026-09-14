# Sentinel-X — Production Frontend (Phase 11)

A cyber-intelligence command center for the Sentinel-X network world model.
The **network graph is the protagonist**; every panel answers one of the
acceptance questions and every animation is bound to real application state.

## Run

```bash
cd sentinel/frontend
npm install
npm run dev        # http://localhost:5173
```

- If the Phase-10 FastAPI backend is running on `http://127.0.0.1:8000`
  (`python scripts/serve.py`), the app auto-detects it (via `GET /health`) and
  drives every visual from the **live model** through the `/api` dev proxy.
- Otherwise it transparently falls back to a **schema-faithful mock**
  (`src/api/scenario.ts`) that conforms 1:1 to `serving/schemas.py`. No fake
  results are hardcoded into components — all values flow from the API layer.

`npm run build` type-checks and produces a production bundle.

## How it maps to the API

`src/api/types.ts` mirrors the Pydantic schemas exactly. `src/api/client.ts`
calls `/health`, `/forecast`, `/network/history`, and `/counterfactual`.

## Layout → acceptance questions

| Region | Component | Answers |
|--------|-----------|---------|
| Top | CommandBar (PAST/NOW/FUTURE lens) | temporal navigation |
| Left | NetworkState | WHAT / WHERE |
| Center | GraphStage (D3 canvas network) | WHAT / WHERE / WHAT NEXT (morphs to forecast) |
| Right | ThreatForecast + ThreatOrbit (R3F) | HOW CERTAIN (risk · uncertainty · novelty, kept separate) |
| Bottom | RiskTimeline (D3) | WHEN |
| Bottom | MitreTrajectory | WHAT NEXT |
| Bottom | Explainability | WHY |
| Bottom | NodeIntel + WhatIf | WHERE detail + WHAT IF |

## Visual language (state-bound animation)

- node pulse → activity/risk (`flow_count` + `risk_now`)
- edge dash-flow → communication (offset speed ∝ `flow_count`)
- red dashed edge → flagged/attack communication (`contains_attack`)
- ghost node/edge → predicted future host (FUTURE lens, scaled by forecast risk)
- graph morph → temporal state transition (observed ⇄ forecast)
- uncertainty halo → predictive std (MC-dropout)
- novelty ring ignites → Mahalanobis OOD over threshold
- risk orbit markers → forecast risk per horizon t+1..t+K

## Libraries / techniques

- **React + TypeScript + Vite**, **Tailwind** (shadcn-style panel primitives)
- **D3** — network graph (canvas force sim), risk timeline, what-if delta,
  explainability bars, propagation
- **React Three Fiber + Drei + React Postprocessing** — 3D threat orbit
  (risk core, forecast orbit, uncertainty halo, novelty ring, bloom)
- **Framer Motion** — state-driven transitions (temporal pill, trajectory
  reveal, forecast badge)
- **Zustand** — single application store; canvas/WebGL render loops run outside
  React reconciliation to avoid rerenders (handles 100–300 nodes).
