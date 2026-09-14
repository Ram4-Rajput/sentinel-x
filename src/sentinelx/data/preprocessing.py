"""Train-only preprocessing for unified flow records.

Leakage-safety is structural here (per sentinel-x-ml-research + data-cleaning
skills): statistics (medians for imputation, mean/std for scaling, category
vocabularies) are learned in ``fit`` from the TRAINING records ONLY, then
applied unchanged in ``transform``. Calling ``transform`` before ``fit`` raises.

What it does:
  * numeric imputation: per-feature median (train only); missing -> median,
    with a companion ``<feat>_was_missing`` indicator (informative missingness).
  * numeric scaling: standardization (train mean/std; std==0 -> 1.0 guard).
  * categorical encoding: fit a vocabulary of protocol values from train;
    unseen categories at transform time -> a reserved "<unk>" index (never
    crashes on unseen test categories).

It deliberately does NOT clip, drop rows, or fabricate values for features a
dataset never provided (those arrive as None and are imputed exactly like any
other missing value, but the FeatureAvailability record still documents them as
unavailable-at-source).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .schema import (
    UNIFIED_CATEGORICAL_FEATURES,
    UNIFIED_NUMERIC_FEATURES,
    UnifiedFlowRecord,
)

_UNK = "<unk>"


@dataclass
class FlowPreprocessor:
    numeric_features: Tuple[str, ...] = UNIFIED_NUMERIC_FEATURES
    categorical_features: Tuple[str, ...] = UNIFIED_CATEGORICAL_FEATURES
    add_missing_indicators: bool = True

    # fitted state (populated by fit)
    _medians: Dict[str, float] = field(default_factory=dict)
    _means: Dict[str, float] = field(default_factory=dict)
    _stds: Dict[str, float] = field(default_factory=dict)
    _vocab: Dict[str, Dict[str, int]] = field(default_factory=dict)
    _fitted: bool = False

    # ---------------------------------------------------------------
    def fit(self, train_records: Sequence[UnifiedFlowRecord]) -> "FlowPreprocessor":
        if not train_records:
            raise ValueError("fit requires at least one training record.")

        # numeric: median (impute) + mean/std (scale), computed on train only
        for feat in self.numeric_features:
            vals = [
                getattr(r, feat)
                for r in train_records
                if getattr(r, feat) is not None
            ]
            if vals:
                self._medians[feat] = float(statistics.median(vals))
                self._means[feat] = float(statistics.fmean(vals))
                self._stds[feat] = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
            else:
                # feature never observed in train (e.g. dataset lacks it) -> 0
                self._medians[feat] = 0.0
                self._means[feat] = 0.0
                self._stds[feat] = 0.0

        # categorical: vocabulary from train (index 0 reserved for <unk>)
        for feat in self.categorical_features:
            vocab: Dict[str, int] = {_UNK: 0}
            for r in train_records:
                v = getattr(r, feat)
                if v is None:
                    continue
                key = str(v)
                if key not in vocab:
                    vocab[key] = len(vocab)
            self._vocab[feat] = vocab

        self._fitted = True
        return self

    # ---------------------------------------------------------------
    def transform_record(self, rec: UnifiedFlowRecord) -> List[float]:
        if not self._fitted:
            raise RuntimeError("FlowPreprocessor.transform called before fit (leakage guard).")
        out: List[float] = []
        for feat in self.numeric_features:
            raw = getattr(rec, feat)
            missing = raw is None
            value = self._medians[feat] if missing else float(raw)
            std = self._stds[feat] if self._stds[feat] != 0.0 else 1.0
            out.append((value - self._means[feat]) / std)
            if self.add_missing_indicators:
                out.append(1.0 if missing else 0.0)
        for feat in self.categorical_features:
            vocab = self._vocab[feat]
            v = getattr(rec, feat)
            idx = vocab.get(str(v), vocab[_UNK]) if v is not None else vocab[_UNK]
            out.append(float(idx))
        return out

    def transform(self, records: Sequence[UnifiedFlowRecord]) -> List[List[float]]:
        return [self.transform_record(r) for r in records]

    def fit_transform(self, train_records: Sequence[UnifiedFlowRecord]) -> List[List[float]]:
        self.fit(train_records)
        return self.transform(train_records)

    # ---------------------------------------------------------------
    def feature_names(self) -> List[str]:
        """Ordered names of the produced vector (for documentation/debugging)."""
        names: List[str] = []
        for feat in self.numeric_features:
            names.append(feat)
            if self.add_missing_indicators:
                names.append(f"{feat}_was_missing")
        for feat in self.categorical_features:
            names.append(f"{feat}_idx")
        return names

    def vocabulary(self, feature: str) -> Dict[str, int]:
        if not self._fitted:
            raise RuntimeError("vocabulary requested before fit.")
        return dict(self._vocab[feature])

    @property
    def is_fitted(self) -> bool:
        return self._fitted
