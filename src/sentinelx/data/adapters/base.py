"""Base dataset adapter + schema validation.

An adapter maps raw source rows (dicts keyed by original column names) into
``UnifiedFlowRecord`` objects. Subclasses declare:

  * ``dataset_name``          - canonical key
  * ``required_columns``      - columns that MUST exist in the source (schema
                                validation fails loudly otherwise)
  * ``availability()``        - the FeatureAvailability contract for this source
  * ``map_row(row, index)``   - pure per-row mapping to UnifiedFlowRecord

The base class provides:
  * ``validate_schema(header)`` - checks required columns are present
  * ``iter_records(rows)``      - validated streaming conversion
  * documentation hooks so every transformation is captured in code.

No adapter fits statistics, splits data, or fabricates unavailable features.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

from ..schema import (
    UNIFIED_CATEGORICAL_FEATURES,
    UNIFIED_NUMERIC_FEATURES,
    BinaryLabel,
    FeatureAvailability,
    UnifiedFlowRecord,
)


class SchemaValidationError(ValueError):
    """Raised when a source file's columns do not match the adapter's contract."""


def to_float(value, *, default: Optional[float] = None) -> Optional[float]:
    """Parse a numeric cell. Returns ``default`` for blanks; returns None for
    non-finite (Inf/NaN) so the preprocessor can handle them as missing rather
    than letting Inf poison scaling. Documented behaviour: adapters do NOT clip
    or impute; they only normalise 'unparseable/non-finite' -> None."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        f = float(value)
    else:
        s = str(value).strip()
        if s == "" or s.lower() in {"nan", "na", "none", "-"}:
            return default
        try:
            f = float(s)
        except ValueError:
            return default
    if not math.isfinite(f):
        return None  # Inf / NaN -> treated as missing downstream
    return f


def to_int(value, *, default: Optional[int] = None) -> Optional[int]:
    f = to_float(value, default=None)
    if f is None:
        return default
    try:
        return int(f)
    except (ValueError, OverflowError):
        return default


class BaseDatasetAdapter(ABC):
    #: canonical dataset key
    dataset_name: str = "base"
    #: columns that must be present in the source header
    required_columns: Sequence[str] = ()
    #: when True, required-column matching ignores case (for datasets like
    #: UNSW-NB15 that ship both 'Label' (raw) and 'label' (partition) names).
    case_insensitive_schema: bool = False

    def __init__(self) -> None:
        av = self.availability()
        av.assert_consistent()  # fail fast if the contract is malformed
        self._availability = av

    # ---- contract each dataset must implement ----
    @abstractmethod
    def availability(self) -> FeatureAvailability:
        """Return the honest present/derived/unavailable classification."""

    @abstractmethod
    def map_row(self, row: Dict[str, object], index: int) -> UnifiedFlowRecord:
        """Map one raw row (dict) to a UnifiedFlowRecord. Pure; no side effects."""

    # ---- shared behaviour ----
    def get_availability(self) -> FeatureAvailability:
        return self._availability

    def validate_schema(self, header: Iterable[str]) -> None:
        """Ensure all required columns exist. Raises SchemaValidationError."""
        cols = {c.strip() for c in header}
        if self.case_insensitive_schema:
            cols_lower = {c.lower() for c in cols}
            missing = [c for c in self.required_columns if c.lower() not in cols_lower]
        else:
            missing = [c for c in self.required_columns if c not in cols]
        if missing:
            raise SchemaValidationError(
                f"{self.dataset_name}: source is missing required column(s): "
                f"{missing}. Present columns: {sorted(cols)[:15]}..."
            )

    def iter_records(
        self, rows: Iterable[Dict[str, object]], *, header: Optional[Iterable[str]] = None
    ) -> Iterator[UnifiedFlowRecord]:
        """Validate schema (if header given) then stream mapped records."""
        if header is not None:
            self.validate_schema(header)
        for i, row in enumerate(rows):
            rec = self.map_row(row, i)
            # post-condition: adapter must not have populated an 'unavailable'
            # numeric feature with a value (guards against accidental fabrication)
            self._assert_no_fabrication(rec)
            yield rec

    def _assert_no_fabrication(self, rec: UnifiedFlowRecord) -> None:
        for feat in self._availability.unavailable:
            if feat in UNIFIED_NUMERIC_FEATURES and getattr(rec, feat, None) is not None:
                raise AssertionError(
                    f"{self.dataset_name}: feature '{feat}' is declared "
                    f"unavailable but was populated (fabrication guard)."
                )

    # ---- label helpers ----
    @staticmethod
    def _binary_from_benign_token(label: Optional[str], benign_tokens: Sequence[str]) -> BinaryLabel:
        if label is None:
            return BinaryLabel.UNKNOWN
        low = str(label).strip().lower()
        if any(low == b.lower() for b in benign_tokens):
            return BinaryLabel.BENIGN
        return BinaryLabel.ATTACK
