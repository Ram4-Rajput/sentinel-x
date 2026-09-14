"""Dataset adapters: convert raw dataset rows into UnifiedFlowRecord streams.

Each adapter is responsible ONLY for faithful schema mapping + honest
availability reporting. It performs no train/test fitting and invents no
unavailable feature. Preprocessing (imputation/scaling/encoding) is a separate,
train-only stage.
"""

from .base import BaseDatasetAdapter, SchemaValidationError
from .cic_ids2018 import CICIDS2018Adapter
from .ciciot2023 import CICIoT2023Adapter
from .ctu13 import CTU13Adapter
from .unsw_nb15 import UNSWNB15Adapter

_ADAPTERS = {
    "cic-ids2018": CICIDS2018Adapter,
    "ciciot2023": CICIoT2023Adapter,
    "ctu-13": CTU13Adapter,
    "unsw-nb15": UNSWNB15Adapter,
}


def list_adapters():
    """Return the registered dataset keys."""
    return sorted(_ADAPTERS.keys())


def get_adapter(name: str, **kwargs) -> BaseDatasetAdapter:
    """Instantiate an adapter by canonical dataset key (case-insensitive)."""
    key = name.strip().lower()
    if key not in _ADAPTERS:
        raise KeyError(
            f"Unknown dataset '{name}'. Known: {list_adapters()}"
        )
    return _ADAPTERS[key](**kwargs)


__all__ = [
    "BaseDatasetAdapter",
    "SchemaValidationError",
    "CICIDS2018Adapter",
    "CICIoT2023Adapter",
    "CTU13Adapter",
    "UNSWNB15Adapter",
    "get_adapter",
    "list_adapters",
]
