"""Configuration for the Sentinel-X world model (no hardcoded hyperparameters).

Everything the model and training loop need is declared here as dataclasses and
loadable from a YAML file so experiments are reproducible and configurable, per
the Phase-4 spec:

    model:
      gnn_type: gat
      temporal_type: gru
      hidden_dim: 128
      graph_layers: 2
      temporal_layers: 1
      dropout: 0.2
    training:
      learning_rate: 0.001
      batch_size: 32
      epochs: 50
      seed: 42

A tiny nested-YAML reader is included so the package stays light, but if PyYAML
is installed it is used instead (more robust).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

# Node/edge feature dimensions come from the Phase-2 cache (graph_cache.py).
# node: out_degree, in_degree, total_bytes_sent, total_bytes_received, flow_count
NODE_FEATURE_DIM = 5
# edge: flow_count, total_bytes, contains_attack
EDGE_FEATURE_DIM = 3


@dataclass
class ModelConfig:
    """Architecture hyperparameters. Configurable; nothing hardcoded downstream."""

    gnn_type: str = "gat"            # "gat" | "graphsage"
    temporal_type: str = "gru"       # "gru" | "lstm"
    hidden_dim: int = 128
    graph_layers: int = 2
    temporal_layers: int = 1
    dropout: float = 0.2
    latent_dim: int = 64             # dimension of the learned network state z_t
    gat_heads: int = 4               # attention heads (GAT only)
    pooling: str = "mean"            # graph pooling: "mean" | "max" | "sum"
    node_feature_dim: int = NODE_FEATURE_DIM
    edge_feature_dim: int = EDGE_FEATURE_DIM
    use_edge_features: bool = True   # feed edge features where the GNN supports it

    def normalized(self) -> "ModelConfig":
        gnn = self.gnn_type.lower()
        if gnn in ("sage", "graph_sage"):
            gnn = "graphsage"
        if gnn not in ("gat", "graphsage"):
            raise ValueError(f"gnn_type must be 'gat' or 'graphsage', got {self.gnn_type!r}")
        temporal = self.temporal_type.lower()
        if temporal not in ("gru", "lstm"):
            raise ValueError(f"temporal_type must be 'gru' or 'lstm', got {self.temporal_type!r}")
        pooling = self.pooling.lower()
        if pooling not in ("mean", "max", "sum"):
            raise ValueError(f"pooling must be 'mean'|'max'|'sum', got {self.pooling!r}")
        return ModelConfig(
            gnn_type=gnn, temporal_type=temporal, hidden_dim=int(self.hidden_dim),
            graph_layers=int(self.graph_layers), temporal_layers=int(self.temporal_layers),
            dropout=float(self.dropout), latent_dim=int(self.latent_dim),
            gat_heads=int(self.gat_heads), pooling=pooling,
            node_feature_dim=int(self.node_feature_dim),
            edge_feature_dim=int(self.edge_feature_dim),
            use_edge_features=bool(self.use_edge_features),
        )


@dataclass
class TrainingConfig:
    """Training-loop hyperparameters."""

    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 50
    seed: int = 42
    weight_decay: float = 0.0
    grad_clip_norm: Optional[float] = 1.0   # gradient clipping (recurrent stability)
    early_stopping_patience: int = 8        # epochs w/o val improvement before stop
    early_stopping_min_delta: float = 1e-4
    # multi-objective loss weights (documented in losses.py)
    lambda_state: float = 1.0               # future-state representation prediction
    lambda_risk: float = 1.0                # future malicious-risk prediction (aux)
    # model selection / early-stopping metric. "state" tracks the PRIMARY
    # future-state objective (default, per spec); "total" tracks state+risk.
    selection_metric: str = "state"


@dataclass
class DataConfig:
    """Which cached windows to consume and how to form forecasting samples.

    Mirrors the Phase-3 protocol (src/sentinelx/experiments/common.py) so the
    world model is evaluated apples-to-apples with the baselines.
    """

    dataset: str = "ctu-13"
    seq_len: int = 4
    horizon: int = 1
    train_frac: float = 0.7
    val_frac: float = 0.15
    min_windows: int = 12


@dataclass
class WorldModelConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {"model": asdict(self.model), "training": asdict(self.training),
                "data": asdict(self.data)}


# --------------------------------------------------------------------------- #
# YAML loading (PyYAML if available, else a small nested reader).
# --------------------------------------------------------------------------- #
def _read_yaml(path: Path) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        loaded = yaml.safe_load(text)
        return loaded or {}
    except Exception:
        return _read_nested_yaml(text)


def _coerce(val: str) -> Any:
    v = val.strip().strip('"').strip("'")
    if v.lower() in ("null", "none", "~", ""):
        return None
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    try:
        if any(c in v for c in ".eE") and not v.isalpha():
            return float(v)
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def _read_nested_yaml(text: str) -> Dict[str, Any]:
    """Minimal 2-level 'section:\\n  key: value' reader (stdlib-only fallback)."""
    out: Dict[str, Any] = {}
    current: Optional[str] = None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        line = raw.strip()
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        if indent == 0:
            if val.strip() == "":
                current = key
                out[current] = {}
            else:
                out[key] = _coerce(val)
                current = None
        else:
            if current is None:
                out[key] = _coerce(val)
            else:
                out[current][key] = _coerce(val)
    return out


def from_dict(d: Dict[str, Any]) -> WorldModelConfig:
    m = d.get("model", {}) or {}
    t = d.get("training", {}) or {}
    dt = d.get("data", {}) or {}
    model = ModelConfig(**{k: v for k, v in m.items() if k in ModelConfig().__dict__})
    training = TrainingConfig(**{k: v for k, v in t.items() if k in TrainingConfig().__dict__})
    data = DataConfig(**{k: v for k, v in dt.items() if k in DataConfig().__dict__})
    return WorldModelConfig(model=model.normalized(), training=training, data=data)


def load_config(path: Optional[Path] = None) -> WorldModelConfig:
    """Load a WorldModelConfig from YAML, falling back to defaults if no path."""
    if path is None:
        return WorldModelConfig(model=ModelConfig().normalized())
    return from_dict(_read_yaml(Path(path)))


def dump_config_yaml(cfg: WorldModelConfig, path: Path) -> Path:
    """Write a config back to YAML (best-effort PyYAML, else hand-rolled)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = cfg.to_dict()
    try:
        import yaml  # type: ignore
        path.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
        return path
    except Exception:
        lines = []
        for section, kv in d.items():
            lines.append(f"{section}:")
            for k, v in kv.items():
                out = "null" if v is None else v
                lines.append(f"  {k}: {out}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path
