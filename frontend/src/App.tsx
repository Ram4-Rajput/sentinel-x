import { useEffect } from "react";
import { useSentinel } from "./store";
import { CommandBar } from "./components/panels/CommandBar";
import { NetworkStatePanel } from "./components/panels/NetworkState";
import { GraphStage } from "./components/graph/GraphStage";
import { ThreatForecast } from "./components/threat/ThreatForecast";
import { RiskTimeline } from "./components/panels/RiskTimeline";
import { Explainability } from "./components/panels/Explainability";
import { NodeIntel } from "./components/panels/NodeIntel";
import { MitreTrajectory } from "./components/panels/MitreTrajectory";
import { WhatIf } from "./components/panels/WhatIf";

// Layout answers the acceptance questions:
//   WHAT / WHERE   -> Network State (left) + graph nodes
//   WHEN           -> Risk Timeline (bottom)
//   WHY            -> Explainability (bottom)
//   WHAT NEXT      -> MITRE trajectory (bottom) + forecast graph (center/future)
//   HOW CERTAIN    -> Threat Forecast (right): risk / uncertainty / novelty
//   WHAT IF        -> What-If counterfactual (bottom)
export default function App() {
  const init = useSentinel((s) => s.init);
  const ready = useSentinel((s) => s.ready);
  const playing = useSentinel((s) => s.playing);
  const tIndex = useSentinel((s) => s.tIndex);
  const maxAnchor = useSentinel((s) => s.maxAnchor);
  const stepAnchor = useSentinel((s) => s.stepAnchor);
  const togglePlay = useSentinel((s) => s.togglePlay);

  useEffect(() => {
    void init();
  }, [init]);

  // Playback loop — advances the anchor, morphing the whole world model. Each
  // /forecast can take ~1-2s, so we use a self-scheduling timeout that fires
  // AFTER a settle delay rather than a fixed interval — this prevents requests
  // from piling up out of order during playback. Stops at the end.
  useEffect(() => {
    if (!playing) return;
    if (tIndex >= maxAnchor) {
      togglePlay();
      return;
    }
    const id = setTimeout(() => stepAnchor(1), 1200);
    return () => clearTimeout(id);
  }, [playing, tIndex, maxAnchor, stepAnchor, togglePlay]);

  if (!ready) {
    return (
      <div className="flex h-full w-full items-center justify-center bg-void">
        <div className="mono text-observed animate-pulse tracking-widest">
          INITIALISING WORLD MODEL…
        </div>
      </div>
    );
  }

  return (
    <div className="h-full w-full p-2 flex flex-col gap-2 bg-void">
      <CommandBar />

      <div className="flex-1 min-h-0 grid grid-cols-[240px_1fr_300px] gap-2">
        <NetworkStatePanel />
        <GraphStage />
        <ThreatForecast />
      </div>

      <div className="h-[240px] shrink-0 grid grid-cols-[1.3fr_1fr_1fr_1fr] gap-2">
        <RiskTimeline />
        <MitreTrajectory />
        <div className="grid grid-rows-2 gap-2 min-h-0">
          <NodeIntel />
          <WhatIf />
        </div>
        <Explainability />
      </div>
    </div>
  );
}
