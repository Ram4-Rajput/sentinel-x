import { motion, AnimatePresence } from "framer-motion";
import { NetworkGraph } from "./NetworkGraph";
import { useSentinel } from "@/store";
import { pct } from "@/lib/format";

// Center stage: the protagonist graph plus a HUD overlay that reflects the
// active temporal lens (PAST/NOW/FUTURE) and a legend for the visual language.
const VIEW_COPY: Record<string, { title: string; sub: string; color: string }> = {
  past: { title: "OBSERVED HISTORY", sub: "solid — what happened", color: "#38e1ff" },
  now: { title: "CURRENT STATE", sub: "solid — anchor t", color: "#38e1ff" },
  future: { title: "FORECAST ROLLOUT", sub: "ghosted — predicted t+1..t+K", color: "#6b7bd6" },
};

export function GraphStage() {
  const temporalView = useSentinel((s) => s.temporalView);
  const forecast = useSentinel((s) => s.forecast);
  const playing = useSentinel((s) => s.playing);
  const togglePlay = useSentinel((s) => s.togglePlay);
  const copy = VIEW_COPY[temporalView];

  const meanForecast = forecast
    ? forecast.risk.forecast_risk.reduce((a, b) => a + b, 0) /
      Math.max(1, forecast.risk.forecast_risk.length)
    : 0;

  return (
    <div className="panel bg-yantra relative h-full overflow-hidden">
      <NetworkGraph />

      {/* temporal lens label — morphs with the state */}
      <div className="pointer-events-none absolute left-4 top-3">
        <AnimatePresence mode="wait">
          <motion.div
            key={temporalView}
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 6 }}
          >
            <div className="mono text-xs tracking-widest" style={{ color: copy.color }}>
              {copy.title}
            </div>
            <div className="panel-title">{copy.sub}</div>
          </motion.div>
        </AnimatePresence>
      </div>

      {/* forecast summary badge (only in future lens) */}
      <AnimatePresence>
        {temporalView === "future" && forecast && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="pointer-events-none absolute right-4 top-3 rounded-md border border-forecast/40 bg-forecast/10 px-3 py-1.5"
          >
            <div className="panel-title text-forecast">predicted attack surface</div>
            <div className="mono text-sm text-forecast">
              mean forecast risk {pct(meanForecast)}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* legend */}
      <div className="pointer-events-none absolute bottom-3 left-4 flex flex-col gap-1">
        <LegendRow color="#38e1ff" label="observed node / benign flow" />
        <LegendRow color="#ff4d5e" label="flagged node / attack flow" dashed />
        <LegendRow color="#6b7bd6" label="predicted (ghost) node / edge" dashed />
      </div>

      {/* playback */}
      <div className="absolute bottom-3 right-4 flex items-center gap-2">
        <button
          onClick={togglePlay}
          className="mono text-[10px] px-3 py-1.5 rounded border border-observed/40 text-observed hover:bg-observed/10"
        >
          {playing ? "❚❚ pause" : "▶ play"}
        </button>
      </div>
    </div>
  );
}

function LegendRow({ color, label, dashed }: { color: string; label: string; dashed?: boolean }) {
  return (
    <div className="flex items-center gap-2">
      <span
        className="inline-block h-2 w-5"
        style={{
          borderTop: `2px ${dashed ? "dashed" : "solid"} ${color}`,
        }}
      />
      <span className="mono text-[9px] text-inkdim">{label}</span>
    </div>
  );
}
