import { Panel } from "@/components/ui/Panel";
import { ThreatOrbit } from "./ThreatOrbit";
import { useSentinel } from "@/store";
import { pct, pct1, riskColor } from "@/lib/format";

// Three SEPARATE channels — risk, uncertainty, novelty — never a single gauge.
function Channel({
  label,
  value,
  detail,
  color,
  bar,
}: {
  label: string;
  value: string;
  detail: string;
  color: string;
  bar: number;
}) {
  return (
    <div className="px-3 py-2 border-t border-[#16203a] first:border-t-0">
      <div className="flex items-center justify-between">
        <span className="panel-title" style={{ color }}>
          {label}
        </span>
        <span className="mono text-sm font-semibold" style={{ color }}>
          {value}
        </span>
      </div>
      <div className="mt-1 h-1 w-full rounded bg-[#0e1526] overflow-hidden">
        <div
          className="h-full rounded transition-all duration-500"
          style={{ width: `${Math.min(100, bar * 100)}%`, background: color }}
        />
      </div>
      <p className="mt-1 text-[10px] leading-snug text-inkdim">{detail}</p>
    </div>
  );
}

export function ThreatForecast() {
  const forecast = useSentinel((s) => s.forecast);
  if (!forecast) return null;

  const { risk, uncertainty, novelty } = forecast;
  const forecastMax = Math.max(...risk.forecast_risk, 0);

  return (
    <Panel title="Threat Forecast" className="h-full" bodyClassName="flex flex-col">
      <div className="h-[210px] shrink-0">
        <ThreatOrbit />
      </div>

      <Channel
        label="Risk (point estimate)"
        value={pct1(risk.risk_now)}
        detail={`Forecast peak ${pct(forecastMax)} across t+1..t+${risk.horizons.length}. Threshold ${pct(
          risk.risk_threshold,
        )}. ${risk.alert ? "ALERT — over threshold." : "Below threshold."}`}
        color={riskColor(risk.risk_now)}
        bar={risk.risk_now}
      />

      {uncertainty && (
        <Channel
          label="Uncertainty (MC-dropout σ)"
          value={uncertainty.uncertainty_std.toFixed(3)}
          detail={`Predictive std over ${uncertainty.n_passes} passes (var ${uncertainty.uncertainty_variance.toFixed(
            4,
          )}). How much to trust the point estimate — separate from risk.`}
          color="#a970ff"
          bar={Math.min(1, uncertainty.uncertainty_std * 3)}
        />
      )}

      {novelty && (
        <Channel
          label="Novelty / OOD (Mahalanobis)"
          value={novelty.novelty_score.toFixed(2)}
          detail={`${
            novelty.is_novel ? "NOVEL — outside training distribution. " : "In-distribution. "
          }Threshold ${novelty.threshold?.toFixed(2)} @ p${novelty.fit_percentile?.toFixed(0)}. Distance in latent space.`}
          color={novelty.is_novel ? "#ffb020" : "#65748f"}
          bar={Math.min(1, novelty.novelty_score / ((novelty.threshold ?? 3) * 1.5))}
        />
      )}
    </Panel>
  );
}
