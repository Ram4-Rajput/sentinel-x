"""Pipeline configuration + dataset path resolution.

Paths are NOT hardcoded. Resolution order for each dataset's raw location:
  1. explicit ``--raw-path`` / ``raw_path`` argument
  2. environment variable  SENTINELX_<DATASET>_RAW   (dataset upper, - -> _)
  3. config file           data/metadata/data_paths.yaml  (key = dataset)
  4. (no default guess)    -> raises with a clear message

Only a tiny hand-rolled YAML reader is used (flat "key: value" mapping) so the
pipeline stays stdlib-only, consistent with the existing data layer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Repository root = three parents up from this file (src/sentinelx/pipeline).
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "data" / "metadata" / "data_paths.yaml"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
UNIFIED_ROOT = REPO_ROOT / "data" / "unified"

# Filenames that must be ignored inside a dataset's raw directory (documented
# in the Phase-1 audit): the mislabeled CIC-IDS2018 processed CSVs living under
# the CICIoT2023 folder, and the unrelated CICIDS2017 payload file in UNSW.
_IGNORE_SUBSTRINGS = {
    "ciciot2023": ("cicflowmeter",),      # the 10 mislabeled CIC-IDS2018 day files
    "unsw-nb15": ("payload_data_cicids2017",),
}


def _read_flat_yaml(path: Path) -> Dict[str, str]:
    """Minimal flat 'key: value' YAML reader (no external dependency)."""
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or ":" not in s:
            continue
        key, _, val = s.partition(":")
        val = val.strip().strip('"').strip("'")
        if val:
            out[key.strip()] = val
    return out


@dataclass
class DatasetPaths:
    dataset: str
    raw_path: Path
    file_glob: str          # which files inside raw_path are this dataset
    is_binetflow: bool = False   # comma CSV with .binetflow ext (CTU-13)
    headerless: bool = False     # raw UNSW-NB15_1..4 have no header row

    def list_files(self) -> List[Path]:
        root = self.raw_path
        if root.is_file():
            return [root]
        files = sorted(root.glob(self.file_glob))
        ignore = _IGNORE_SUBSTRINGS.get(self.dataset, ())
        return [f for f in files if not any(sub in f.name.lower() for sub in ignore)]


# How each dataset's files are recognised inside its raw directory.
_FILE_SPEC = {
    "cic-ids2018": {"glob": "*_TrafficForML_CICFlowMeter.csv", "binetflow": False, "headerless": False},
    "ciciot2023": {"glob": "*.csv", "binetflow": False, "headerless": False},
    "ctu-13": {"glob": "**/*.binetflow", "binetflow": True, "headerless": False},
    # UNSW: default to the labeled partition CSVs (headered). Raw 1..4 are
    # header-less and handled via a dedicated glob when requested.
    "unsw-nb15": {"glob": "UNSW_NB15_*-set.csv", "binetflow": False, "headerless": False},
    "unsw-nb15-raw": {"glob": "UNSW-NB15_[1-4].csv", "binetflow": False, "headerless": True},
}


def resolve_dataset(
    dataset: str,
    *,
    raw_path: Optional[str] = None,
    config_path: Optional[Path] = None,
    variant: Optional[str] = None,
) -> DatasetPaths:
    """Resolve a dataset's raw location + file spec (no hardcoded default)."""
    key = dataset.strip().lower()
    spec_key = variant.lower() if variant else key
    if spec_key not in _FILE_SPEC:
        raise KeyError(f"No file-spec for dataset '{dataset}'. Known: {sorted(_FILE_SPEC)}")

    # 1. explicit
    resolved: Optional[str] = raw_path
    # 2. env var
    if not resolved:
        env_key = "SENTINELX_" + key.upper().replace("-", "_") + "_RAW"
        resolved = os.environ.get(env_key)
    # 3. config file
    if not resolved:
        cfg = _read_flat_yaml(config_path or DEFAULT_CONFIG)
        resolved = cfg.get(key)
    if not resolved:
        raise FileNotFoundError(
            f"Could not resolve raw path for '{dataset}'. Provide --raw-path, set "
            f"env SENTINELX_{key.upper().replace('-', '_')}_RAW, or add '{key}:' to "
            f"{DEFAULT_CONFIG}."
        )

    p = Path(resolved)
    if not p.exists():
        raise FileNotFoundError(f"Raw path for '{dataset}' does not exist: {p}")

    spec = _FILE_SPEC[spec_key]
    return DatasetPaths(
        dataset=key,
        raw_path=p,
        file_glob=spec["glob"],
        is_binetflow=spec["binetflow"],
        headerless=spec["headerless"],
    )


@dataclass
class PipelineConfig:
    """Windowing / processing configuration (CLI-overridable)."""

    dataset: str
    window_size: float = 10.0     # seconds (temporal window duration)
    stride: float = 10.0          # seconds between window starts (== size -> non-overlap)
    train_frac: float = 0.7
    val_frac: float = 0.15
    chunk_size: int = 100_000     # rows per streaming chunk
    max_records: Optional[int] = None  # cap for smoke/medium tests (None = full)
    build_graphs: bool = True
    processed_root: Path = field(default=PROCESSED_ROOT)

    def cache_dir(self) -> Path:
        return self.processed_root / self.dataset
