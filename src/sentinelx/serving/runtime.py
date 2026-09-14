"""Serving runtime: load the trained world model + data ONCE and cache it.

This module is the single place that touches the trained checkpoint and the
Phase-2 cache. Everything else in the serving layer works against the
:class:`SentinelRuntime` object, so:

  * the (expensive) checkpoint load + leakage-safe sample build happen exactly
    once per process (lazily, on first request), and
  * services never re-read paths / re-tune thresholds — they reuse the runtime.

Nothing here retrains or rebuilds: the model comes from
``models/sentinel_x/model.pt`` and the samples come from the existing
``data/processed/<dataset>/`` cache via the Phase-5 ``KStepDataset`` (same
chronological split the whole project uses).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..pipeline.config import PROCESSED_ROOT, REPO_ROOT
from ..worldmodel.config import WorldModelConfig, load_config
from ..worldmodel.kstep_data import KStepDataset, KStepSample
from ..worldmodel.model import SentinelXWorldModel

DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "sentinel_x" / "model.pt"
# How many future steps the serving layer rolls out by default. Bounded so a
# single request can never trigger an unbounded rollout.
DEFAULT_SERVING_K = 5
MAX_SERVING_K = 10


@dataclass
class SplitSamples:
    """The three chronological splits of leakage-safe K-step samples."""

    train: List[KStepSample] = field(default_factory=list)
    val: List[KStepSample] = field(default_factory=list)
    test: List[KStepSample] = field(default_factory=list)
    info: Dict = field(default_factory=dict)


class SentinelRuntime:
    """Process-wide handle to the trained model + rebuilt serving samples."""

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        processed_root: Optional[Path] = None,
        *,
        K: int = DEFAULT_SERVING_K,
    ):
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINT)
        self.processed_root = Path(processed_root or PROCESSED_ROOT)
        self.K = int(K)
        if not (1 <= self.K <= MAX_SERVING_K):
            raise ValueError(f"serving K must be in [1, {MAX_SERVING_K}], got {K}")

        self._lock = threading.RLock()
        self._model: Optional[SentinelXWorldModel] = None
        self._extra: Dict = {}
        self._threshold: float = 0.5
        self._config: Optional[WorldModelConfig] = None
        self._samples: Optional[SplitSamples] = None

    # ------------------------------------------------------------------ #
    # Availability
    # ------------------------------------------------------------------ #
    @property
    def checkpoint_available(self) -> bool:
        return self.checkpoint_path.exists()

    # ------------------------------------------------------------------ #
    # Lazy loaders (thread-safe; each expensive step runs at most once)
    # ------------------------------------------------------------------ #
    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if not self.checkpoint_available:
                raise FileNotFoundError(
                    f"No trained checkpoint at {self.checkpoint_path}. "
                    "Train Phase-4 first (scripts/train_world_model.py).")
            model, extra = SentinelXWorldModel.load_checkpoint(self.checkpoint_path)
            model.eval()
            self._model = model
            self._extra = extra or {}
            self._threshold = float(self._extra.get("chosen_threshold", 0.5))
            cfg_path = self.checkpoint_path.parent / "config.yaml"
            self._config = (load_config(cfg_path) if cfg_path.exists()
                            else WorldModelConfig())

    def _ensure_samples(self) -> None:
        if self._samples is not None:
            return
        self._ensure_model()
        with self._lock:
            if self._samples is not None:
                return
            data_cfg = self._config.data
            ds = KStepDataset(
                data_cfg, self.processed_root,
                self._model.cfg.node_feature_dim,
                self._model.cfg.edge_feature_dim,
                K=self.K,
            )
            train, val, test, info = ds.build()
            self._samples = SplitSamples(
                train=list(train), val=list(val), test=list(test), info=info)

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #
    @property
    def model(self) -> SentinelXWorldModel:
        self._ensure_model()
        return self._model  # type: ignore[return-value]

    @property
    def extra(self) -> Dict:
        self._ensure_model()
        return self._extra

    @property
    def threshold(self) -> float:
        self._ensure_model()
        return self._threshold

    @property
    def config(self) -> WorldModelConfig:
        self._ensure_model()
        return self._config  # type: ignore[return-value]

    @property
    def combo(self) -> Optional[str]:
        return self.extra.get("combo")

    @property
    def samples(self) -> SplitSamples:
        self._ensure_samples()
        return self._samples  # type: ignore[return-value]

    @property
    def data_usable(self) -> bool:
        return bool(self.samples.info.get("usable"))

    # ------------------------------------------------------------------ #
    # Sample lookup helpers (used by state/history/anchor endpoints)
    # ------------------------------------------------------------------ #
    def all_test_samples(self) -> List[KStepSample]:
        return self.samples.test

    def find_sample(self, t_index: int) -> Optional[KStepSample]:
        """Return the test sample anchored at ``t_index`` (or None)."""
        for s in self.samples.test:
            if int(s.t_index) == int(t_index):
                return s
        return None

    def latest_sample(self) -> Optional[KStepSample]:
        """The most recent (largest t_index) test anchor = 'current' state."""
        test = self.samples.test
        if not test:
            return None
        return max(test, key=lambda s: int(s.t_index))

    def resolve_sample(self, t_index: Optional[int]) -> Optional[KStepSample]:
        """Resolve an optional anchor: explicit t_index, else the latest."""
        if t_index is None:
            return self.latest_sample()
        return self.find_sample(t_index)


# --------------------------------------------------------------------------- #
# Process-wide singleton (created lazily; overridable in tests)
# --------------------------------------------------------------------------- #
_RUNTIME: Optional[SentinelRuntime] = None
_RUNTIME_LOCK = threading.Lock()


def get_runtime() -> SentinelRuntime:
    """Return the process-wide runtime, creating it on first use."""
    global _RUNTIME
    if _RUNTIME is None:
        with _RUNTIME_LOCK:
            if _RUNTIME is None:
                _RUNTIME = SentinelRuntime()
    return _RUNTIME


def set_runtime(runtime: Optional[SentinelRuntime]) -> None:
    """Override (or reset) the process-wide runtime. Used by tests."""
    global _RUNTIME
    with _RUNTIME_LOCK:
        _RUNTIME = runtime
