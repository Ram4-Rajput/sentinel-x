"""Training-friendly cache of temporal graph sequences (+ reload).

Goal: avoid rebuilding graphs from raw CSVs on every training run.

Representation:
  Each temporal window -> a serialisable graph dict:
      {
        window_index, start, end, num_nodes, num_edges, num_flows,
        node_ids            : [entity_id, ...]           (index = node id)
        node_features       : [[f0, f1, ...], ...]       (per-node numeric)
        edge_index          : [[src_idx, dst_idx], ...]  (directed)
        edge_features       : [[flow_count, total_bytes], ...]
        window_features     : [ ... ]                    (window-level summary)
        attack_ratio, label_any_attack
      }

Storage layout (per dataset):
  data/processed/<dataset>/
      metadata/manifest.json      - dataset/version/window cfg/split/stats
      windows/<split>.jsonl       - one JSON graph per line (portable fallback)
      graphs/<split>.pt           - torch_geometric Data list IF torch available

Node features are derived ONLY from entities the graph already tracks
(byte/degree/flow stats) — no fabricated packet features. The chosen JSONL
format is dependency-free and reloadable; a PyG ``.pt`` is additionally written
when torch + torch_geometric are installed (optional, best-effort).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..data.schema import BinaryLabel, TemporalNetworkWindow


@dataclass
class GraphSequenceMeta:
    dataset: str
    split: str
    dataset_version: str
    schema_version: str
    preprocessing_version: str
    pipeline_version: str
    window_size: float
    stride: float
    num_windows: int
    total_nodes: int
    total_edges: int
    total_flows: int
    timestamp_min: Optional[str]
    timestamp_max: Optional[str]
    node_feature_names: List[str] = field(default_factory=list)
    edge_feature_names: List[str] = field(default_factory=list)
    class_distribution: Dict[str, int] = field(default_factory=dict)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt is not None else None


NODE_FEATURE_NAMES = [
    "out_degree", "in_degree", "total_bytes_sent",
    "total_bytes_received", "flow_count",
]
EDGE_FEATURE_NAMES = ["flow_count", "total_bytes", "contains_attack"]


def window_to_graph_dict(window: TemporalNetworkWindow) -> Dict:
    """Serialise a TemporalNetworkWindow into a portable graph dict.

    Node features come from NetworkEntity aggregates the graph already tracks;
    nothing packet-level is fabricated.
    """
    node_ids = list(window.entities.keys())
    idx = {nid: i for i, nid in enumerate(node_ids)}

    node_features = []
    for nid in node_ids:
        e = window.entities[nid]
        node_features.append([
            float(e.out_degree), float(e.in_degree),
            float(e.total_bytes_sent), float(e.total_bytes_received),
            float(e.flow_count),
        ])

    edge_index = []
    edge_features = []
    for (src, dst), edge in window.edges.items():
        if src not in idx or dst not in idx:
            continue
        edge_index.append([idx[src], idx[dst]])
        edge_features.append([
            float(edge.flow_count), float(edge.total_bytes),
            1.0 if edge.contains_attack else 0.0,
        ])

    n_attack = sum(1 for f in window.flows if f.label_binary == BinaryLabel.ATTACK)
    return {
        "window_index": window.window_index,
        "start": _iso(window.start),
        "end": _iso(window.end),
        "num_nodes": len(node_ids),
        "num_edges": len(edge_index),
        "num_flows": window.num_flows,
        "node_ids": node_ids,
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "attack_ratio": window.attack_ratio(),
        "label_any_attack": 1 if n_attack > 0 else 0,
    }


class GraphCache:
    """Writes / reads cached graph sequences for one dataset."""

    def __init__(self, dataset: str, root: Path):
        self.dataset = dataset
        self.root = Path(root)
        self.windows_dir = self.root / "windows"
        self.graphs_dir = self.root / "graphs"
        self.metadata_dir = self.root / "metadata"

    def _ensure_dirs(self) -> None:
        for d in (self.windows_dir, self.graphs_dir, self.metadata_dir):
            d.mkdir(parents=True, exist_ok=True)

    # -------- write --------
    def write_split(self, split: str, windows: List[TemporalNetworkWindow]) -> Path:
        """Write one split's windows as JSONL. Returns the file path."""
        self._ensure_dirs()
        path = self.windows_dir / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for w in windows:
                fh.write(json.dumps(window_to_graph_dict(w)) + "\n")
        # best-effort PyG export (optional)
        self._try_write_pyg(split, path)
        return path

    def _try_write_pyg(self, split: str, jsonl_path: Path) -> Optional[Path]:
        try:
            import torch  # noqa: F401
            from torch_geometric.data import Data  # noqa: F401
        except Exception:
            return None  # torch/PyG not installed -> JSONL is the source of truth
        import torch
        from torch_geometric.data import Data

        data_list = []
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                g = json.loads(line)
                if g["num_nodes"] == 0:
                    continue
                x = torch.tensor(g["node_features"], dtype=torch.float)
                if g["edge_index"]:
                    ei = torch.tensor(g["edge_index"], dtype=torch.long).t().contiguous()
                    ea = torch.tensor(g["edge_features"], dtype=torch.float)
                else:
                    ei = torch.empty((2, 0), dtype=torch.long)
                    ea = torch.empty((0, len(EDGE_FEATURE_NAMES)), dtype=torch.float)
                data_list.append(Data(
                    x=x, edge_index=ei, edge_attr=ea,
                    y=torch.tensor([g["label_any_attack"]], dtype=torch.long),
                    window_index=g["window_index"],
                ))
        out = self.graphs_dir / f"{split}.pt"
        torch.save(data_list, out)
        return out

    def write_manifest(self, metas: List[GraphSequenceMeta], extra: Optional[Dict] = None) -> Path:
        self._ensure_dirs()
        manifest = {
            "dataset": self.dataset,
            "node_feature_names": NODE_FEATURE_NAMES,
            "edge_feature_names": EDGE_FEATURE_NAMES,
            "splits": {m.split: asdict(m) for m in metas},
        }
        if extra:
            manifest.update(extra)
        path = self.metadata_dir / "manifest.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        return path

    # -------- read --------
    def read_split(self, split: str) -> List[Dict]:
        """Reload a split's graph dicts from JSONL (no raw CSV re-read)."""
        path = self.windows_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"No cached split '{split}' at {path}")
        graphs = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    graphs.append(json.loads(line))
        return graphs

    def read_manifest(self) -> Dict:
        path = self.metadata_dir / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"No manifest at {path}")
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def cache_size_bytes(self) -> int:
        total = 0
        for d in (self.windows_dir, self.graphs_dir, self.metadata_dir):
            if d.exists():
                for f in d.rglob("*"):
                    if f.is_file():
                        total += f.stat().st_size
        return total


def summarize_windows(windows: List[TemporalNetworkWindow]) -> Dict[str, int]:
    total_nodes = sum(w.num_nodes for w in windows)
    total_edges = sum(w.num_edges for w in windows)
    total_flows = sum(w.num_flows for w in windows)
    return {"nodes": total_nodes, "edges": total_edges, "flows": total_flows}
