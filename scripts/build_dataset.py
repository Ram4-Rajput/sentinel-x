#!/usr/bin/env python
"""Reproducible CLI: build cached temporal graph sequences for a dataset.

Examples:
    python scripts/build_dataset.py --dataset cic-ids2018
    python scripts/build_dataset.py --dataset cic-ids2018 --window-size 10 --stride 1
    python scripts/build_dataset.py --dataset ctu-13 --max-records 50000 --window-size 30 --stride 30
    python scripts/build_dataset.py --dataset unsw-nb15 --variant unsw-nb15-raw

No paths are hardcoded: raw locations resolve via --raw-path, env
SENTINELX_<DATASET>_RAW, or data/metadata/data_paths.yaml.
"""

import argparse
import json
import sys
from pathlib import Path

# Make src importable without installation.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sentinelx.pipeline import PipelineConfig, build_dataset  # noqa: E402
from sentinelx.pipeline.config import PROCESSED_ROOT  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build Sentinel-X temporal graph cache for a dataset.")
    ap.add_argument("--dataset", required=True,
                    help="cic-ids2018 | ciciot2023 | ctu-13 | unsw-nb15")
    ap.add_argument("--raw-path", default=None, help="Override raw data location.")
    ap.add_argument("--variant", default=None,
                    help="File-spec variant, e.g. 'unsw-nb15-raw' for header-less raw files.")
    ap.add_argument("--window-size", type=float, default=10.0, help="Window duration (seconds).")
    ap.add_argument("--stride", type=float, default=None,
                    help="Seconds between window starts (default = window-size, non-overlapping).")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--chunk-size", type=int, default=100_000)
    ap.add_argument("--max-records", type=int, default=None,
                    help="Cap records (for smoke/medium tests). Default: full dataset.")
    ap.add_argument("--no-graphs", action="store_true", help="Skip graph construction (windows only).")
    ap.add_argument("--no-cache", action="store_true", help="Do not write cache (dry run).")
    ap.add_argument("--report", default=None, help="Write JSON report to this path.")
    args = ap.parse_args(argv)

    stride = args.stride if args.stride is not None else args.window_size
    config = PipelineConfig(
        dataset=args.dataset.strip().lower(),
        window_size=args.window_size,
        stride=stride,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        chunk_size=args.chunk_size,
        max_records=args.max_records,
        build_graphs=not args.no_graphs,
    )

    report = build_dataset(
        config, raw_path=args.raw_path, variant=args.variant,
        write_cache=not args.no_cache,
    )

    # Persist a per-dataset report under metadata too.
    report_dict = report.as_dict()
    print("\n===== BUILD REPORT =====")
    print(json.dumps(report_dict, indent=2))

    out_path = args.report
    if out_path is None and not args.no_cache and report.cache_location:
        out_path = str(Path(report.cache_location) / "metadata" / "build_report.json")
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
        print(f"\n[report] written to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
