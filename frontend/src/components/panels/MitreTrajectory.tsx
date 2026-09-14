import { motion } from "framer-motion";
import { Panel } from "@/components/ui/Panel";
import { useSentinel } from "@/store";
import { pct } from "@/lib/format";

// WHAT NEXT — observed -> forecast attack trajectory (high-level MITRE tactics).
// Observed stages are solid; forecast stages are ghosted/dashed, mirroring the
// graph's temporal language.
export function MitreTrajectory() {
  const forecast = useSentinel((s) => s.forecast);
  if (!forecast) return null;
  const stages = forecast.mitre.stages;

  return (
    <Panel
      title="Attack Trajectory · MITRE"
      className="h-full"
      right={
        <span className="mono text-[10px] text-inkdim">
          {forecast.mitre.n_observed} obs · {forecast.mitre.n_forecast} fcst
        </span>
      }
      bodyClassName="p-3"
    >
      <ol className="relative space-y-3 pl-4">
        <span className="absolute left-1 top-1 bottom-1 w-px bg-[#16203a]" />
        {stages.map((s, i) => {
          const isForecast = s.status === "forecast";
          const color = isForecast ? "#6b7bd6" : "#38e1ff";
          return (
            <motion.li
              key={`${s.stage}-${s.horizon}-${i}`}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.04 }}
              className="relative"
            >
              <span
                className="absolute -left-[13px] top-1 h-2 w-2 rounded-full"
                style={{
                  background: isForecast ? "transparent" : color,
                  border: `1.5px ${isForecast ? "dashed" : "solid"} ${color}`,
                }}
              />
              <div className="flex items-center justify-between">
                <span
                  className="text-xs font-medium"
                  style={{ color, fontStyle: isForecast ? "italic" : "normal" }}
                >
                  {s.stage}
                </span>
                <span className="mono text-[10px] text-inkdim">
                  {isForecast ? `t+${s.horizon}` : "now"} · {pct(s.confidence)}
                </span>
              </div>
              {s.evidence.length > 0 && (
                <ul className="mt-0.5 space-y-0.5">
                  {s.evidence.map((e, j) => (
                    <li key={j} className="text-[10px] text-inkdim leading-snug">
                      · {e}
                    </li>
                  ))}
                </ul>
              )}
            </motion.li>
          );
        })}
      </ol>
    </Panel>
  );
}
