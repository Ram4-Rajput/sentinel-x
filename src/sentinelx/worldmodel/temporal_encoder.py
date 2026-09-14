"""Temporal encoder: GRU / LSTM over graph-level states across time.

Receives the sequence of graph-level representations produced by the graph
encoder:

    h_{t-n}, ..., h_{t-1}, h_t     each of size hidden_dim

and produces the learned latent network state:

    z_t                            of size latent_dim

z_t summarises how the network has been evolving up to and including window t.
It is the reusable "network state" the rest of Sentinel-X consumes.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig


class TemporalEncoder(nn.Module):
    """GRU or LSTM mapping a sequence of graph states to a latent state z_t."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.temporal_type = cfg.temporal_type
        rnn_cls = nn.GRU if cfg.temporal_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=cfg.hidden_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.temporal_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.temporal_layers > 1 else 0.0,
        )
        # project final recurrent hidden state to the latent network state z_t
        self.to_latent = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
            nn.LayerNorm(cfg.latent_dim),
            nn.Tanh(),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (batch, seq_len, hidden_dim) -> z_t: (batch, latent_dim).

        Uses the last-timestep hidden state (state after observing window t).
        """
        out, _ = self.rnn(seq)          # (batch, seq_len, hidden_dim)
        last = out[:, -1, :]            # (batch, hidden_dim) — state at time t
        return self.to_latent(last)     # (batch, latent_dim)
