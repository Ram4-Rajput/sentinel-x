"""Sentinel-X world model: the Phase-4 core research component.

Learns P(S_{t+1} | S_t) over a dynamic network state:

    G_{t-n..t}  --GraphEncoder-->  h_{t-n..t}
                --TemporalEncoder--> z_t              (latent network state)
                --forecasting head-> z_hat_{t+1}      (PRIMARY objective)
                --risk head--------> risk_logit@t+H   (AUXILIARY objective)

The primary objective is *future-state modelling* (predict the next latent
network state), NOT attack yes/no. A risk head exists only as an auxiliary
prediction where labels support it.

The learned latent ``z_t`` is exposed via ``encode_state`` for later reuse
(forecasting, novelty/OOD, explainability, trajectory analysis).

Input format (see data.py): a batch is a list of samples, each a sequence of
``seq_len`` per-window PyG-style graph tensors. Because graphs vary in node
count across windows and samples, we encode graphs with a flattened PyG
mini-batch per timestep, then regroup into (batch, seq_len, hidden_dim).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .config import ModelConfig
from .graph_encoder import GraphEncoder
from .temporal_encoder import TemporalEncoder


@dataclass
class GraphTensors:
    """One window graph as tensors."""
    x: torch.Tensor            # (N, node_feature_dim)
    edge_index: torch.Tensor   # (2, E)
    edge_attr: torch.Tensor    # (E, edge_feature_dim)
    num_nodes: int


@dataclass
class WorldModelOutput:
    z_t: torch.Tensor          # (batch, latent_dim) current latent network state
    z_next_pred: torch.Tensor  # (batch, latent_dim) predicted next latent state
    risk_logit: torch.Tensor   # (batch,) auxiliary malicious-risk logit @ t+H


class SentinelXWorldModel(nn.Module):
    """GNN + temporal encoder + future-state forecasting head (+ aux risk head)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg = cfg.normalized()
        self.graph_encoder = GraphEncoder(cfg)
        self.temporal_encoder = TemporalEncoder(cfg)

        # PRIMARY: forecasting head predicts the next latent network state z_{t+1}.
        self.forecast_head = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
        )
        # AUXILIARY: malicious-risk head (binary risk of attack at t+H).
        self.risk_head = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.latent_dim // 2, 1),
        )

    # ------------------------------------------------------------------ #
    # Graph-sequence encoding
    # ------------------------------------------------------------------ #
    def _encode_sequence_batch(
        self, batch_seq: List[List[GraphTensors]]
    ) -> torch.Tensor:
        """Encode a batch of graph sequences -> (batch, seq_len, hidden_dim).

        batch_seq[i][s] is the graph at timestep s of sample i. All samples share
        the same seq_len. For each timestep we build ONE PyG mini-batch across
        the batch dimension (efficient message passing), then gather back.
        """
        device = next(self.parameters()).device
        batch_size = len(batch_seq)
        seq_len = len(batch_seq[0])
        H = self.cfg.hidden_dim
        states = torch.zeros((batch_size, seq_len, H), device=device)

        for s in range(seq_len):
            xs, eis, eas, batch_vec = [], [], [], []
            node_offset = 0
            graph_slot = []  # which output row each graph maps to
            present = 0
            for i in range(batch_size):
                g = batch_seq[i][s]
                if g.num_nodes == 0:
                    continue  # empty graph -> stays zero state
                # Dataset tensors live on CPU (we never move the whole dataset to
                # GPU). Move THIS graph's tensors to the model device per batch so
                # message passing runs on the same device as the parameters.
                gx = g.x.to(device)
                g_edge_index = g.edge_index.to(device)
                g_edge_attr = g.edge_attr.to(device)
                xs.append(gx)
                ei = g_edge_index + node_offset if g_edge_index.numel() else g_edge_index
                eis.append(ei)
                eas.append(g_edge_attr)
                batch_vec.append(torch.full((g.num_nodes,), present, dtype=torch.long,
                                            device=device))
                graph_slot.append(i)
                node_offset += g.num_nodes
                present += 1

            if present == 0:
                continue
            x = torch.cat(xs, dim=0)
            edge_index = torch.cat(eis, dim=1) if eis else torch.empty((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.cat(eas, dim=0) if eas else torch.empty((0, self.cfg.edge_feature_dim), device=device)
            bvec = torch.cat(batch_vec, dim=0)
            pooled = self.graph_encoder(x, edge_index, bvec, edge_attr)  # (present, H)
            for row, i in enumerate(graph_slot):
                states[i, s, :] = pooled[row]
        return states

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def encode_state(self, batch_seq: List[List[GraphTensors]]) -> torch.Tensor:
        """Return the learned latent network state z_t for each sample.

        Exposed for downstream reuse (novelty/OOD, explainability, trajectory
        analysis) in later phases.
        """
        seq_states = self._encode_sequence_batch(batch_seq)   # (B, L, H)
        return self.temporal_encoder(seq_states)              # (B, latent_dim)

    def forward(self, batch_seq: List[List[GraphTensors]]) -> WorldModelOutput:
        z_t = self.encode_state(batch_seq)                    # (B, latent_dim)
        z_next_pred = self.forecast_head(z_t)                 # (B, latent_dim)
        risk_logit = self.risk_head(z_t).squeeze(-1)          # (B,)
        return WorldModelOutput(z_t=z_t, z_next_pred=z_next_pred, risk_logit=risk_logit)

    def encode_target_state(self, target_seq: List[List[GraphTensors]]) -> torch.Tensor:
        """Encode the sequence ending at window t+H into a latent state.

        Used as the (detached) prediction target for the future-state loss: the
        model predicts z_{t+1} and we compare it against the encoding of the
        actual future window sequence. No gradient flows through the target.
        """
        with torch.no_grad():
            return self.encode_state(target_seq)

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, path, extra: Optional[Dict] = None) -> None:
        import json
        from dataclasses import asdict
        payload = {
            "state_dict": self.state_dict(),
            "model_config": asdict(self.cfg),
            "extra": extra or {},
        }
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(cls, path, map_location="cpu") -> Tuple["SentinelXWorldModel", Dict]:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        cfg = ModelConfig(**payload["model_config"])
        model = cls(cfg)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model, payload.get("extra", {})

    def num_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))
