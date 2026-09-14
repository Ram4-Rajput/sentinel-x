#!/usr/bin/env python
"""Report the exact on-disk state of the configured datasets (no assumptions).

    python scripts/inspect_state.py
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sentinelx.pipeline.config import resolve_dataset  # noqa: E402

DATASETS = ["cic-ids2018", "ciciot2023", "unsw-nb15", "ctu-13"]


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def main() -> int:
    for ds in DATASETS:
        print(f"\n=== {ds} ===")
        try:
            paths = resolve_dataset(ds)
        except Exception as e:
            print(f"   UNRESOLVED: {type(e).__name__}: {e}")
            continue
        files = paths.list_files()
        total = sum(f.stat().st_size for f in files if f.exists())
        print(f"   raw_path: {paths.raw_path}")
        print(f"   matched files: {len(files)}  total: {human(total)}")
        for f in files[:12]:
            print(f"     {human(f.stat().st_size):>10}  {f.name}")
        # PCAP presence (informational)
        pcaps = list(paths.raw_path.rglob("*.pcap")) if paths.raw_path.is_dir() else []
        if pcaps:
            psize = sum(p.stat().st_size for p in pcaps)
            print(f"   pcap available: {len(pcaps)} files ({human(psize)}) [optional pipeline]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
