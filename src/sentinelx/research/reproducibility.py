"""Reproducibility utilities for the research experiment infrastructure.

Consolidates the reproducibility guarantees that were previously scattered
across the phase drivers into one place:

  * ``seed_everything``          — deterministic python/numpy/torch RNGs
    (re-exported from experiments.common so there is a single source of truth).
  * ``capture_environment``      — python + key dependency versions + platform.
  * ``snapshot_config``          — write a config dict/dataclass to JSON so the
    exact configuration of a run is preserved.
  * ``RunManifest``              — a machine-readable record of a single run
    (command, config, environment, git commit if available, seed, timestamp,
    output artifacts). Written next to the experiment outputs.

None of this fabricates anything: versions come from the installed packages,
git info from the repo if present, and timestamps from the system clock.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Single source of truth for deterministic seeding (do not re-implement).
from ..experiments.common import seed_everything  # noqa: F401  (re-exported)

# Dependencies whose versions materially affect reproducibility of results.
_TRACKED_PACKAGES = (
    "numpy",
    "scipy",
    "sklearn",
    "torch",
    "torch_geometric",
    "mlflow",
    "yaml",
)


def _pkg_version(name: str) -> Optional[str]:
    """Best-effort version lookup that never raises."""
    try:
        mod = __import__(name)
    except Exception:
        return None
    for attr in ("__version__", "version", "VERSION"):
        v = getattr(mod, attr, None)
        if isinstance(v, str):
            return v
    return "unknown"


def capture_environment() -> Dict[str, Any]:
    """Record python + dependency versions + platform for provenance."""
    packages: Dict[str, Optional[str]] = {}
    for pkg in _TRACKED_PACKAGES:
        packages[pkg] = _pkg_version(pkg)
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
    }


def git_commit(repo_root: Optional[Path] = None) -> Optional[str]:
    """Return the current git commit hash if this is a git repo, else None."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root) if repo_root else None,
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        return None
    return None


def _to_plain(obj: Any) -> Any:
    """Recursively convert dataclasses / Paths to JSON-serializable values."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_plain(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def snapshot_config(config: Any, path: Path) -> Path:
    """Write a config (dict or dataclass) to JSON, preserving the exact run config."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_plain(config), indent=2, default=str),
                    encoding="utf-8")
    return path


@dataclass
class RunManifest:
    """Machine-readable record of a single reproducible run."""

    command: str                       # e.g. "run_experiment"
    experiment: str                    # experiment / model name
    seed: int
    config: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    git_commit: Optional[str] = None
    argv: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    started_utc: str = ""
    finished_utc: str = ""
    status: str = "started"            # started | completed | skipped | error
    notes: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return _to_plain(asdict(self))

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2, default=str),
                        encoding="utf-8")
        return path


def new_manifest(command: str, experiment: str, seed: int, *,
                 config: Optional[Dict[str, Any]] = None,
                 repo_root: Optional[Path] = None) -> RunManifest:
    """Create a manifest pre-populated with environment + git + argv."""
    return RunManifest(
        command=command,
        experiment=experiment,
        seed=seed,
        config=_to_plain(config or {}),
        environment=capture_environment(),
        git_commit=git_commit(repo_root),
        argv=list(sys.argv),
        started_utc=datetime.now(timezone.utc).isoformat(),
    )


def finalize_manifest(manifest: RunManifest, *, status: str,
                      outputs: Optional[List[Path]] = None,
                      notes: Optional[str] = None) -> RunManifest:
    manifest.status = status
    manifest.finished_utc = datetime.now(timezone.utc).isoformat()
    if outputs:
        manifest.outputs = [str(p) for p in outputs]
    if notes:
        manifest.notes = notes
    return manifest
