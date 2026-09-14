"""Graph encoder: GAT / GraphSAGE over a single window graph G_t.

Consumes the dynamic graph directly (node features + edge connectivity + edge
features where the operator supports it) using PyTorch Geometric message
passing. The graph is NEVER flattened into a plain feature vector before the
GNN — node embeddings are produced by real message passing, then pooled to a
graph-level state h_t.

Encoder output per graph:
    h_t : (hidden_dim,)  graph-level representation for the temporal module.

Empty graphs (0 nodes — e.g. datasets without src/dst entities) return a zero
vector of size hidden_dim so the model still runs; such datasets are flagged as
not graph-capable by the data layer (documented, not hidden).
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from .config import ModelConfig


def _pool_fn(name: str):
    return {"mean": global_mean_pool, "max": global_max_pool, "sum": global_add_pool}[name]


class GraphEncoder(nn.Module):
    """Stacked GAT or GraphSAGE layers + graph pooling -> graph state h_t.

    * GAT   : uses ``edge_dim`` so edge features participate in attention when
              ``use_edge_features`` is set and the graph has edges.
    * GraphSAGE : SAGEConv does not consume edge features; edge connectivity is
              still fully used. This limitation is documented rather than faked.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.gnn_type = cfg.gnn_type
        self.hidden_dim = cfg.hidden_dim
        self.use_edge_features = cfg.use_edge_features
        self.pool = _pool_fn(cfg.pooling)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_dim = cfg.node_feature_dim
        for _ in range(cfg.graph_layers):
            if cfg.gnn_type == "gat":
                # multi-head attention; concat heads then project back to hidden_dim.
                heads = cfg.gat_heads
                assert cfg.hidden_dim % heads == 0, (
                    "hidden_dim must be divisible by gat_heads")
                per_head = cfg.hidden_dim // heads
                conv = GATConv(
                    in_dim, per_head, heads=heads, dropout=cfg.dropout,
                    edge_dim=cfg.edge_feature_dim if cfg.use_edge_features else None,
                )
            else:  # graphsage
                conv = SAGEConv(in_dim, cfg.hidden_dim)
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(cfg.hidden_dim))
            in_dim = cfg.hidden_dim

        self.dropout = nn.Dropout(cfg.dropout)
        self.supports_edge_features = (cfg.gnn_type == "gat" and cfg.use_edge_features)

    def encode_nodes(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Message passing -> per-node embeddings (N, hidden_dim)."""
        h = x
        for conv, norm in zip(self.convs, self.norms):
            if self.gnn_type == "gat" and self.use_edge_features and edge_attr is not None \
                    and edge_index.numel() > 0:
                h = conv(h, edge_index, edge_attr=edge_attr)
            else:
                h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = self.dropout(h)
        return h

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode one or more graphs (batched) -> graph states (num_graphs, hidden_dim).

        ``batch`` maps each node to its graph id (PyG convention). For an empty
        batch (no nodes) returns a single zero graph state.
        """
        if x.numel() == 0:
            return torch.zeros((1, self.hidden_dim), device=x.device, dtype=torch.float32)
        h = self.encode_nodes(x, edge_index, edge_attr)
        return self.pool(h, batch)          # (num_graphs, hidden_dim)
