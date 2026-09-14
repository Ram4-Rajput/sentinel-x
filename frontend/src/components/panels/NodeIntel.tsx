import { Panel, Metric } from "@/components/ui/Panel";
import { useSentinel } from "@/store";
import { bytesFmt } from "@/lib/format";

// Node intelligence — details for the selected node (click a node in the graph).
// Also surfaces the propagation summary (WHERE it is spreading).
export function NodeIntel() {
  const forecast = useSentinel((s) => s.forecast);
  const selectedNode = useSentinel((s) => s.selectedNode);
  const runWhatIf = useSentinel((s) => s.runWhatIf);
  if (!forecast) return null;

  const g = forecast.network_state.latest_graph;
  const node = g.nodes.find((n) => n.id === selectedNode);
  const prop = forecast.propagation;

  const incident = node
    ? g.edges.filter((e) => e.src_id === node.id || e.dst_id === node.id)
    : [];
  const flaggedIncident = incident.filter((e) => e.contains_attack).length;

  return (
    <Panel title="Node Intelligence" className="h-full" bodyClassName="p-3 space-y-3">
      {node ? (
        <>
          <div className="flex items-center justify-between">
            <span className="mono text-sm text-observed">{node.id}</span>
            <button
              onClick={() => runWhatIf(node.index)}
              className="mono text-[10px] px-2 py-1 rounded border border-uncertainty/50 text-uncertainty hover:bg-uncertainty/10"
            >
              isolate (what-if)
            </button>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <Metric label="flows" value={node.flow_count} />
            <Metric label="degree" value={`${node.out_degree}/${node.in_degree}`} sub="out/in" />
            <Metric label="sent" value={bytesFmt(node.bytes_sent)} />
            <Metric label="recv" value={bytesFmt(node.bytes_received)} />
          </div>
          <div className="text-[10px] text-inkdim">
            {incident.length} incident edges · {flaggedIncident} flagged as attack
          </div>
        </>
      ) : (
        <p className="text-[11px] text-inkdim leading-relaxed">
          Click a node in the graph to inspect it. Double-click to simulate
          isolating it (what-if).
        </p>
      )}

      {prop && (
        <div className="pt-2 border-t border-[#16203a]">
          <div className="panel-title mb-1.5">Propagation</div>
          <div className="grid grid-cols-2 gap-2">
            <Metric
              label="affected"
              value={prop.total_affected}
              color={prop.is_spreading ? "#ff4d5e" : undefined}
            />
            <Metric label="velocity" value={prop.velocity.toFixed(2)} sub="new/window" />
            <Metric label="direction" value={prop.direction} />
            <Metric
              label="spreading"
              value={prop.is_spreading ? "YES" : "no"}
              color={prop.is_spreading ? "#ff4d5e" : "#3ddc97"}
            />
          </div>
        </div>
      )}
    </Panel>
  );
}
