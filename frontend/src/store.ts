// Central application state. Every visual in the app derives from this store,
// so animations always correspond to real state (spec: no decorative motion).

import { create } from "zustand";
import type {
  CounterfactualResponse,
  ForecastResponse,
  NetworkHistoryResponse,
} from "./api/types";
import {
  getForecast,
  getHistory,
  postCounterfactual,
  detectMode,
  type Mode,
} from "./api/client";
import { SCENARIO_META } from "./api/scenario";

// Temporal lens over the world model: what the graph is currently showing.
export type TemporalView = "past" | "now" | "future";

interface SentinelState {
  mode: Mode;
  ready: boolean;

  tIndex: number;
  minAnchor: number;
  maxAnchor: number;
  playing: boolean;

  temporalView: TemporalView;
  // 0..K — which forecast horizon the "future" lens is focused on
  forecastHorizon: number;

  forecast: ForecastResponse | null;
  history: NetworkHistoryResponse | null;
  loading: boolean;

  selectedNode: string | null;
  counterfactual: CounterfactualResponse | null;
  whatIfNode: number | null;

  // Internal: monotonically increasing id of the latest setAnchor request, used
  // to discard out-of-order responses. Not read by the UI.
  _reqSeq: number;

  init: () => Promise<void>;
  setAnchor: (t: number) => Promise<boolean>;
  stepAnchor: (delta: number) => void;
  togglePlay: () => void;
  setTemporalView: (v: TemporalView) => void;
  setForecastHorizon: (h: number) => void;
  selectNode: (id: string | null) => void;
  runWhatIf: (nodeIndex: number) => Promise<void>;
  clearWhatIf: () => void;
}

export const useSentinel = create<SentinelState>((set, get) => ({
  mode: "mock",
  ready: false,
  tIndex: 0,
  minAnchor: 0,
  maxAnchor: SCENARIO_META.N_ANCHORS - 1,
  playing: false,
  temporalView: "now",
  forecastHorizon: 1,
  forecast: null,
  history: null,
  loading: false,
  selectedNode: null,
  counterfactual: null,
  whatIfNode: null,
  _reqSeq: 0,

  init: async () => {
    const mode = await detectMode();
    set({ mode });

    // Anchor indices differ between live and mock. The live backend only has a
    // handful of real test anchors (e.g. t_index 161..189), while the mock uses
    // 0..N_ANCHORS-1. Derive the valid anchor space from the actual history so
    // bootstrap starts on an anchor that really exists (otherwise /forecast 404s
    // and the app hangs on "INITIALISING WORLD MODEL…").
    let history: NetworkHistoryResponse | null = null;
    try {
      history = await getHistory(SCENARIO_META.N_ANCHORS);
    } catch {
      history = null;
    }

    const anchors = history?.points?.map((p) => p.t_index) ?? [];
    if (anchors.length > 0) {
      const minAnchor = Math.min(...anchors);
      const maxAnchor = Math.max(...anchors);
      // Start ~15% into the available range so there is history to the left.
      const start =
        anchors[Math.floor((anchors.length - 1) * 0.15)] ?? minAnchor;
      set({ minAnchor, maxAnchor, history });
      // A single bootstrap forecast. setAnchor has its own per-request timeout
      // and out-of-order guard, so a transient blip surfaces as a failed call
      // (spinner clears, app becomes usable) rather than a second blocking
      // await that could double the time on the loading screen.
      await get().setAnchor(start);
    } else {
      // No history available — fall back to the mock's 0..N-1 anchor space.
      const start = Math.floor(SCENARIO_META.N_ANCHORS * 0.15);
      set({ minAnchor: 0, maxAnchor: SCENARIO_META.N_ANCHORS - 1 });
      await get().setAnchor(start);
    }

    // ALWAYS leave the loading screen, even if a call failed — a transient
    // network blip must never hang the app on "INITIALISING WORLD MODEL…".
    set({ ready: true });
  },

  setAnchor: async (t) => {
    const clamped = Math.max(get().minAnchor, Math.min(get().maxAnchor, t));
    // Sequence guard: rapid stepping fires several /forecast calls that can
    // resolve OUT OF ORDER, letting a stale (older-anchor) response overwrite a
    // newer one — the UI then "lags" or "sticks". We tag every request and only
    // apply the response if it is still the latest one issued.
    const reqId = get()._reqSeq + 1;
    set({
      _reqSeq: reqId,
      loading: true,
      tIndex: clamped,
      counterfactual: null,
      whatIfNode: null,
    });
    try {
      const [forecast, history] = await Promise.all([
        getForecast(clamped),
        getHistory(clamped),
      ]);
      // Only the most recent request is allowed to write results.
      if (get()._reqSeq !== reqId) return false;
      set({ forecast, history, loading: false });
      return true;
    } catch (err) {
      console.error("setAnchor failed for t=", clamped, err);
      // Only clear the spinner if this is still the active request.
      if (get()._reqSeq === reqId) set({ loading: false });
      return false;
    }
  },

  stepAnchor: (delta) => {
    void get().setAnchor(get().tIndex + delta);
  },

  togglePlay: () => set((s) => ({ playing: !s.playing })),

  setTemporalView: (v) => set({ temporalView: v }),
  setForecastHorizon: (h) =>
    set({ forecastHorizon: Math.max(0, Math.min(SCENARIO_META.K, h)) }),

  selectNode: (id) => set({ selectedNode: id }),

  runWhatIf: async (nodeIndex) => {
    const t = get().tIndex;
    const cf = await postCounterfactual({
      t_index: t,
      intervention: "isolate_node",
      node_index: nodeIndex,
    });
    set({ counterfactual: cf, whatIfNode: nodeIndex });
  },

  clearWhatIf: () => set({ counterfactual: null, whatIfNode: null }),
}));
