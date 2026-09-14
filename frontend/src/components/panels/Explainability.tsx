import { Panel } from "@/components/ui/Panel";
import { useSentinel } from "@/store";

// WHY — feature attributions + temporal evidence behind the current forecast.
// Attributions are signed; bars grow from a centre baseline (left = negative,
// right = positive contribution). This is model explanation, not decoration.
export function Explainability() {
  const forecast = useSentinel((s) => s.forecast);
  const expl = forecast?.explainability;
  if (!expl) return null;

  const maxAbs = Math.max(0.001, ...expl.top_features.map((f) => f.abs_attribution));
  const maxTemporal = Math.max(0.001, ...expl.temporal_evidence.map((t) => t.importance));

  return (
    <Panel
      title="Explainability"
      className="h-full"
      right={<span className="mono text-[10px] text-inkdim">{expl.method}</span>}
      bodyClassName="p-3 space-y-3"
    >
      <div>
        <div className="panel-title mb-1.5">Feature attribution</div>
        <div className="space-y-1.5">
          {expl.top_features.map((f) => {
            const w = (f.abs_attribution / maxAbs) * 50;
            const positive = f.attribution >= 0;
            return (
              <div key={f.name} className="flex items-center gap-2">
                <span className="mono text-[10px] text-inkdim w-24 truncate">
                  {f.name}
                </span>
                <div className="relative flex-1 h-2 bg-[#0e1526] rounded overflow-hidden">
                  <span className="absolute left-1/2 top-0 h-full w-px bg-[#1d2b4d]" />
                  <span
                    className="absolute top-0 h-full"
                    style={{
                      background: positive ? "#ff4d5e" : "#38e1ff",
                      width: `${w}%`,
                      left: positive ? "50%" : `${50 - w}%`,
                    }}
                  />
                </div>
                <span className="mono text-[10px] text-ink w-10 text-right">
                  {f.attribution.toFixed(2)}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      <div>
        <div className="panel-title mb-1.5">Temporal evidence (which windows)</div>
        <div className="flex items-end gap-1 h-12">
          {expl.temporal_evidence.map((t) => (
            <div
              key={t.timestep}
              className="flex-1 flex flex-col items-center justify-end gap-0.5"
              title={`window t${t.relative_horizon} · ${t.importance.toFixed(3)}`}
            >
              <div
                className="w-full rounded-t"
                style={{
                  height: `${(t.importance / maxTemporal) * 100}%`,
                  background: t.relative_horizon === 0 ? "#38e1ff" : "#25325a",
                }}
              />
              <span className="mono text-[8px] text-inkdim">
                {t.relative_horizon === 0 ? "t" : t.relative_horizon}
              </span>
            </div>
          ))}
        </div>
      </div>

      {expl.uses_attention && (
        <p className="text-[9px] text-inkdim leading-snug italic">
          {expl.attention_note}
        </p>
      )}
    </Panel>
  );
}
