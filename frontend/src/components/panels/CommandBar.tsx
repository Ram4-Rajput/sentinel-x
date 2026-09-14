import { motion } from "framer-motion";
import { useSentinel, type TemporalView } from "@/store";
import { currentMode } from "@/api/client";
import { cn } from "@/lib/format";

// Top command/navigation bar. Owns the PAST -> NOW -> FUTURE temporal lens
// that morphs the central graph, plus live/mock mode + connection status.
const VIEWS: { key: TemporalView; label: string; hint: string }[] = [
  { key: "past", label: "PAST", hint: "observed history" },
  { key: "now", label: "NOW", hint: "current state" },
  { key: "future", label: "FUTURE", hint: "forecast rollout" },
];

export function CommandBar() {
  const temporalView = useSentinel((s) => s.temporalView);
  const setTemporalView = useSentinel((s) => s.setTemporalView);
  const forecast = useSentinel((s) => s.forecast);
  const mode = useSentinel((s) => s.mode);
  const loading = useSentinel((s) => s.loading);
  const tIndex = useSentinel((s) => s.tIndex);
  const minAnchor = useSentinel((s) => s.minAnchor);
  const maxAnchor = useSentinel((s) => s.maxAnchor);
  const playing = useSentinel((s) => s.playing);
  const stepAnchor = useSentinel((s) => s.stepAnchor);
  const togglePlay = useSentinel((s) => s.togglePlay);

  const alert = forecast?.risk.alert ?? false;

  return (
    <header className="panel flex items-center justify-between px-4 py-2">
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2">
          <div className="relative h-2.5 w-2.5">
            <span
              className={cn(
                "absolute inset-0 rounded-full",
                alert ? "bg-risk" : "bg-observed",
              )}
            />
            {alert && (
              <span
                className="absolute inset-0 rounded-full bg-risk"
                style={{ animation: "pulse-ring 1.4s ease-out infinite" }}
              />
            )}
          </div>
          <div className="leading-tight">
            <div className="mono text-sm font-semibold tracking-widest text-ink">
              SENTINEL·X
            </div>
            <div className="panel-title">network world-model command</div>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-1 rounded-lg bg-[#0e1526] p-1">
        {VIEWS.map((v) => {
          const active = temporalView === v.key;
          return (
            <button
              key={v.key}
              onClick={() => setTemporalView(v.key)}
              className="relative rounded-md px-4 py-1.5 text-xs"
              title={v.hint}
            >
              {active && (
                <motion.span
                  layoutId="temporal-pill"
                  className="absolute inset-0 rounded-md bg-observed/15 border border-observed/40"
                  transition={{ type: "spring", stiffness: 400, damping: 32 }}
                />
              )}
              <span
                className={cn(
                  "relative mono tracking-widest",
                  active ? "text-observed" : "text-inkdim",
                )}
              >
                {v.label}
              </span>
            </button>
          );
        })}
      </div>

      <div className="flex items-center gap-4">
        {/* Anchor transport: step precisely between real anchors + play/pause */}
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => stepAnchor(-1)}
            disabled={tIndex <= minAnchor}
            title="previous anchor"
            className="mono rounded-md border border-observed/30 px-2 py-1 text-xs text-observed disabled:opacity-30 hover:bg-observed/10"
          >
            ◀
          </button>
          <button
            onClick={() => togglePlay()}
            title={playing ? "pause playback" : "play through anchors"}
            className="mono rounded-md border border-observed/30 px-2 py-1 text-xs text-observed hover:bg-observed/10"
          >
            {playing ? "❚❚" : "▶"}
          </button>
          <button
            onClick={() => stepAnchor(1)}
            disabled={tIndex >= maxAnchor}
            title="next anchor"
            className="mono rounded-md border border-observed/30 px-2 py-1 text-xs text-observed disabled:opacity-30 hover:bg-observed/10"
          >
            ▶
          </button>
          <span className="mono text-[10px] text-inkdim tabular-nums">
            t={tIndex} / {maxAnchor}
          </span>
        </div>
        {loading && (
          <span className="mono text-[10px] text-observed animate-pulse">
            rolling out…
          </span>
        )}
        <div className="flex items-center gap-1.5">
          <span
            className={cn(
              "h-1.5 w-1.5 rounded-full",
              mode === "live" ? "bg-risklow" : "bg-novelty",
            )}
          />
          <span className="mono text-[10px] text-inkdim uppercase">
            {mode === "live" ? "live model" : "mock · api-schema"}
          </span>
        </div>
        <span className="mono text-[10px] text-inkdim">{currentMode()}</span>
      </div>
    </header>
  );
}
