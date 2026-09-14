#!/usr/bin/env python
"""Package the minimal Sentinel-X bundle for a Kaggle training run (Phase A).

Produces TWO zips under dist/kaggle/:
  1. sentinelx_code.zip  -> src/, scripts/, configs/, pyproject.toml, README
  2. sentinelx_ctu13full_windows.zip -> data/processed/ctu-13-full/{windows,metadata}

The world-model trainer reads ONLY the windows/*.jsonl (see
worldmodel/data.py -> experiments/common._load_ordered_windows), so the large
graphs/*.pt files and PCAPs are intentionally excluded.

    python scripts/pack_kaggle.py
    python scripts/pack_kaggle.py --dataset ctu-13-full

Upload code zip + data zip as a single Kaggle Dataset. See docs/KAGGLE_TRAINING.md.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "dist" / "kaggle"

CODE_INCLUDE = ["src", "scripts", "configs", "pyproject.toml"]
DATA_SUBDIRS = ["windows", "metadata"]  # NOT graphs/ (.pt not used by trainer)


def _add_tree(zf: zipfile.ZipFile, base: Path, arc_prefix: str) -> int:
    count = 0
    if base.is_file():
        zf.write(base, arc_prefix)
        return 1
    for p in sorted(base.rglob("*")):
        if p.is_dir():
            continue
        if "__pycache__" in p.parts or p.suffix == ".pyc":
            continue
        rel = p.relative_to(base)
        zf.write(p, f"{arc_prefix}/{rel.as_posix()}")
        count += 1
    return count


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ctu-13-full")
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)

    # 1) code bundle
    code_zip = OUT / "sentinelx_code.zip"
    n_code = 0
    with zipfile.ZipFile(code_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in CODE_INCLUDE:
            src = REPO / item
            if not src.exists():
                print(f"[warn] missing {item}, skipping")
                continue
            n_code += _add_tree(zf, src, item)
    size_mb = code_zip.stat().st_size / 1e6
    print(f"[ok] {code_zip.name}: {n_code} files, {size_mb:.1f} MB")

    # 2) data bundle (windows + metadata only)
    ds_root = REPO / "data" / "processed" / args.dataset
    if not ds_root.exists():
        raise SystemExit(f"[err] dataset cache not found: {ds_root}")
    data_zip = OUT / f"sentinelx_{args.dataset.replace('-', '')}_windows.zip"
    n_data = 0
    with zipfile.ZipFile(data_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for sub in DATA_SUBDIRS:
            src = ds_root / sub
            if not src.exists():
                print(f"[warn] missing {args.dataset}/{sub}, skipping")
                continue
            n_data += _add_tree(
                zf, src, f"data/processed/{args.dataset}/{sub}"
            )
    size_mb = data_zip.stat().st_size / 1e6
    print(f"[ok] {data_zip.name}: {n_data} files, {size_mb:.1f} MB")

    print("\nNext: upload BOTH zips to one Kaggle Dataset, then follow "
          "docs/KAGGLE_TRAINING.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
