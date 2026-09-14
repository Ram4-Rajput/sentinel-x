import { Panel, Metric } from "@/components/ui/Panel";
import { useSentinel } from "@/store";
import { bytesFmt, pct1, riskColor } from "@/lib/format";

// Left column — WHAT / WHERE: the observed network state at anchor t.
export function NetworkStatePanel() {
  const forecast = useSentinel((s) => s.forecast);
  if (!forecast) return null;
  const ns = forecast.network_state;
  const g = ns.latest_graph;

  const attackEdges = g.edges.filter((e) => e.contains_attack).length;
  const topTalkers = [...g.nodes]
    .sort((a, b) => b.bytes_sent + b.bytes_received - (a.bytes_sent + a.bytes_received))
    .slice(0, 6);

  return (
    <Panel title="Network State" className="h-full" bodyClassName="p-3 space-y-4">
      <div className="grid grid-cols-2 gap-3">
        <Metric label="Anchor t" value={ns.t_index} sub={`seq_len ${ns.seq_len}`} />
        <Metric
          label="Risk now"
          value={pct1(ns.risk_now)}
          color={riskColor(ns.risk_now)}
          sub={ns.label_any_attack ? "labelled attack window" : "benign window"}
        />
        <Metric label="Nodes" value={g.num_nodes} />
        <Metric label="Edges" value={g.num_edges} sub={`${attackEdges} flagged`} />
      </div>

      <div>
        <div className="panel-title mb-1">Dataset</div>
        <div className="mono text-xs text-ink">{forecast.dataset}</div>
      </div>

      <div>
        <div className="panel-title mb-1">Top talkers (WHERE)</div>
        <ul className="space-y-1">
          {topTalkers.map((n) => {
            const total = n.bytes_sent + n.bytes_received;
            const flagged = g.edges.some(
              (e) => (e.src_id === n.id || e.dst_id === n.id) && e.contains_attack,
            );
            return (
              <li
                key={n.id}
                className="flex items-center justify-between text-[11px] mono"
              >
                <span className="flex items-center gap-1.5">
                  <span
                    className="inline-block h-1.5 w-1.5 rounded-full"
                    style={{ background: flagged ? "#ff4d5e" : "#38e1ff" }}
                  />
                  {n.id}
                </span>
                <span className="text-inkdim">{bytesFmt(total)}</span>
              </li>
            );
          })}
        </ul>
      </div>
    </Panel>
  );
}
