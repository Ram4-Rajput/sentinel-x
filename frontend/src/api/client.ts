// Sentinel-X API client.
//
// Talks to the real Phase-10 FastAPI backend (via the /api dev proxy) when it
// is reachable; otherwise transparently serves schema-faithful scenario data
// from scenario.ts. The component tree never knows the difference — it always
// receives objects shaped exactly like the Pydantic responses.

import type {
  CounterfactualRequest,
  CounterfactualResponse,
  ForecastResponse,
  HealthResponse,
  NetworkHistoryResponse,
} from "./types";
import { makeCounterfactual, makeForecast, makeHistory, SCENARIO_META } from "./scenario";

const API_BASE = "/api";

// A single /forecast is CPU-heavy on the backend (rollout + MC-Dropout + every
// analytical layer), and the FIRST call after boot is the slowest (cold model +
// lazy sample build). Without a ceiling, a plain fetch can hang indefinitely,
// which strands the app on "INITIALISING WORLD MODEL…" because init() never
// reaches `ready: true`. Every live fetch therefore has a timeout; a timeout
// rejects like any other error and the store handles it gracefully.
const HEALTH_TIMEOUT_MS = 1200;
const REQUEST_TIMEOUT_MS = 20000;

export type Mode = "live" | "mock";

let resolvedMode: Mode | null = null;

/** fetch with an AbortController timeout. Rejects (never hangs) past `timeoutMs`. */
async function fetchWithTimeout(
  input: string,
  init: RequestInit = {},
  timeoutMs: number = REQUEST_TIMEOUT_MS,
): Promise<Response> {
  const ctrl = new AbortController();
  const to = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    return await fetch(input, { ...init, signal: ctrl.signal });
  } finally {
    clearTimeout(to);
  }
}

async function probeLive(): Promise<boolean> {
  try {
    const res = await fetchWithTimeout(`${API_BASE}/health`, {}, HEALTH_TIMEOUT_MS);
    if (!res.ok) return false;
    const body = (await res.json()) as HealthResponse;
    return Boolean(body.model_loaded && body.data_usable);
  } catch {
    return false;
  }
}

export async function detectMode(): Promise<Mode> {
  if (resolvedMode) return resolvedMode;
  resolvedMode = (await probeLive()) ? "live" : "mock";
  return resolvedMode;
}

export function currentMode(): Mode {
  return resolvedMode ?? "mock";
}

export async function getHealth(): Promise<HealthResponse> {
  const mode = await detectMode();
  if (mode === "live") {
    const res = await fetchWithTimeout(`${API_BASE}/health`, {}, HEALTH_TIMEOUT_MS);
    return res.json();
  }
  return {
    status: "ok",
    model_loaded: true,
    data_usable: true,
    checkpoint_available: true,
    combo: "gat+gru",
    dataset: SCENARIO_META.DATASET,
    seq_len: SCENARIO_META.SEQ_LEN,
    horizon: 1,
    serving_k: SCENARIO_META.K,
    num_test_anchors: SCENARIO_META.N_ANCHORS,
    risk_threshold: SCENARIO_META.RISK_THRESHOLD,
    version: "0.1.0-mock",
  };
}

export async function getForecast(tIndex: number): Promise<ForecastResponse> {
  const mode = await detectMode();
  if (mode === "live") {
    const res = await fetchWithTimeout(`${API_BASE}/forecast`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      // Lighter payload for responsive stepping: 15 MC-Dropout passes (still a
      // genuine Monte-Carlo estimate, just slightly noisier) instead of 30.
      // All analytical layers stay ON so every panel keeps real data.
      body: JSON.stringify({ t_index: tIndex, mc_passes: 15 }),
    });
    if (!res.ok) throw new Error(`forecast ${res.status}`);
    return res.json();
  }
  // small async gap so loading states are exercised realistically
  await new Promise((r) => setTimeout(r, 60));
  return makeForecast(tIndex);
}

export async function getHistory(upToTIndex: number): Promise<NetworkHistoryResponse> {
  const mode = await detectMode();
  if (mode === "live") {
    const res = await fetchWithTimeout(`${API_BASE}/network/history`);
    if (!res.ok) throw new Error(`history ${res.status}`);
    return res.json();
  }
  return makeHistory(upToTIndex);
}

export async function postCounterfactual(
  req: CounterfactualRequest,
): Promise<CounterfactualResponse> {
  const mode = await detectMode();
  if (mode === "live") {
    const res = await fetchWithTimeout(`${API_BASE}/counterfactual`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!res.ok) throw new Error(`counterfactual ${res.status}`);
    return res.json();
  }
  await new Promise((r) => setTimeout(r, 80));
  return makeCounterfactual(req.t_index ?? 0, req.intervention, req.node_index ?? null);
}

export { SCENARIO_META };
