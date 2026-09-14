#!/usr/bin/env python
"""Tier 1: build ALL 13 CTU-13 scenarios at full scale + assemble combined and
family-holdout caches for the Sentinel-X world model.

Why
---
The prior CTU-13 cache used a SINGLE scenario (11) -> 195 windows, 30 test
samples. Metrics measured on 30 windows are statistical noise. CTU-13 ships 13
scenarios spanning several botnet FAMILIES. This script:

  1. Builds each scenario folder at FULL size (no --max-records) into its own
     cache  data/processed/ctu-13-s<N>/, reusing the existing streaming ->
     window -> graph -> cache pipeline verbatim (no rebuild of the data layer).
  2. Stamps every cached window with {scenario, family} so downstream code can
     do family-stratified / family-holdout evaluation.
  3. Assembles two derived caches the existing forecasting protocol can consume
     directly (they look like a normal single-dataset cache):
        data/processed/ctu-13-full/     - all scenarios, chronological-by-
                                          (scenario order, window_index), split
                                          train/val/test by fraction.
        data/processed/ctu-13-holdout/  - TRAIN+VAL come from a set of families,
                                          TEST is an UNSEEN family (OOD / novel-
                                          attack generalisation test).

Nothing is fabricated: windows/graphs come straight from the Phase-2 builder.
Only the {scenario, family} tags and the split assignment are added here.

Usage
-----
    python scripts/build_ctu13_all.py                      # full build (all 13)
    python scripts/build_ctu13_all.py --scenarios 1,2,3    # subset
    python scripts/build_ctu13_all.py --skip-existing      # reuse built scenarios
    python scripts/build_ctu13_all.py --window-size 10 --stride 10
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sentinelx.pipeline import PipelineConfig, build_dataset  # noqa: E402
from sentinelx.pipeline.config import PROCESSED_ROOT, resolve_dataset  # noqa: E402

# Canonical CTU-13 scenario -> botnet family (Garcia et al., 2014). The informal
# capture filenames (e.g. "fast-flux", "bot") differ from the malware family;
# both are recorded for honesty. Families are what we hold out for the OOD test.
SCENARIO_FAMILY: Dict[int, str] = {
    1: "Neris", 2: "Neris", 3: "Rbot", 4: "Rbot", 5: "Virut",
    6: "Menti", 7: "Sogou", 8: "Murlo", 9: "Neris", 10: "Rbot",
    11: "Rbot", 12: "NSIS.ay", 13: "Virut",
}

# Default family held out for the OOD / unseen-attack test. Sogou is a single,
# distinct scenario (7) that appears in no other capture -> a clean "family the
# model never saw during training".
DEFAULT_HOLDOUT_FAMILIES = ["Sogou", "Menti"]


def scenario_dataset_key(n: int) -> str:
    return f"ctu-13-s{n}"


def _ctu13_root() -> Path:
    """Resolve the CTU-13 raw root via the existing config resolution."""
    paths = resolve_dataset("ctu-13")
    return paths.raw_path


def build_one_scenario(n: int, *, window_size: float, stride: float,
                       skip_existing: bool) -> Optional[Path]:
    """Build one scenario folder into data/processed/ctu-13-s<N>/.

    Reuses build_dataset unchanged: we set dataset='ctu-13' (so the CTU-13
    adapter resolves) and point raw_path at the single scenario folder, and
    redirect the processed_root so the cache lands in a scenario-specific dir.
    """
    raw_root = _ctu13_root()
    scen_dir = raw_root / str(n)
    if not scen_dir.exists():
        print(f"[scenario {n}] MISSING folder {scen_dir}; skipping.")
        return None

    key = scenario_dataset_key(n)
    cache_dir = PROCESSED_ROOT / key
    if skip_existing and (cache_dir / "windows" / "train.jsonl").exists():
        print(f"[scenario {n}] cache exists at {cache_dir}; --skip-existing set, reusing.")
        return cache_dir

    # dataset='ctu-13' keeps the adapter key valid; processed_root override puts
    # the cache under ctu-13-s<N> (config.cache_dir() = processed_root/dataset).
    # We point processed_root at PROCESSED_ROOT and rename dataset dir after,
    # because config.cache_dir() uses config.dataset for the leaf folder.
    config = PipelineConfig(
        dataset="ctu-13",
        window_size=window_size,
        stride=stride,
        max_records=None,           # FULL scale
        build_graphs=True,
        processed_root=PROCESSED_ROOT / f"_tmp_{key}",
    )
    print(f"\n===== building scenario {n} ({SCENARIO_FAMILY.get(n, '?')}) FULL =====")
    report = build_dataset(config, raw_path=str(scen_dir), write_cache=True)

    tmp_cache = config.processed_root / "ctu-13"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp_cache), str(cache_dir))
    shutil.rmtree(config.processed_root, ignore_errors=True)

    # stamp scenario + family into every cached window (all splits)
    _stamp_scenario(cache_dir, n)
    print(f"[scenario {n}] records={report.records:,} windows={report.windows} "
          f"nodes={report.nodes:,} -> {cache_dir}")
    return cache_dir


def _stamp_scenario(cache_dir: Path, n: int) -> None:
    family = SCENARIO_FAMILY.get(n, "unknown")
    win_dir = cache_dir / "windows"
    for split in ("train", "val", "test"):
        p = win_dir / f"{split}.jsonl"
        if not p.exists():
            continue
        lines = []
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                g = json.loads(line)
                g["scenario"] = n
                g["family"] = family
                lines.append(json.dumps(g))
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))


def _load_scenario_windows(n: int) -> List[dict]:
    """Load a scenario's windows in chronological order (train+val+test)."""
    cache_dir = PROCESSED_ROOT / scenario_dataset_key(n)
    win_dir = cache_dir / "windows"
    out: List[dict] = []
    for split in ("train", "val", "test"):
        p = win_dir / f"{split}.jsonl"
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def _write_assembled(dataset_key: str, split_windows: Dict[str, List[dict]]) -> Path:
    """Write an assembled cache (train/val/test JSONL + manifest + PyG .pt)."""
    root = PROCESSED_ROOT / dataset_key
    win_dir = root / "windows"
    meta_dir = root / "metadata"
    graphs_dir = root / "graphs"
    for d in (win_dir, meta_dir, graphs_dir):
        d.mkdir(parents=True, exist_ok=True)

    splits_meta = {}
    for split, windows in split_windows.items():
        # global re-index so window_index is contiguous & chronological
        for i, g in enumerate(windows):
            g["window_index"] = i
        p = win_dir / f"{split}.jsonl"
        with open(p, "w", encoding="utf-8") as fh:
            for g in windows:
                fh.write(json.dumps(g) + "\n")
        n_pos = sum(int(g.get("label_any_attack", 0)) for g in windows)
        splits_meta[split] = {
            "num_windows": len(windows),
            "positives": n_pos,
            "families": sorted({g.get("family", "?") for g in windows}),
            "scenarios": sorted({g.get("scenario", -1) for g in windows}),
        }
        _write_pyg(graphs_dir / f"{split}.pt", windows)

    manifest = {
        "dataset": dataset_key,
        "assembled_from": "ctu-13 scenarios 1..13 (full scale)",
        "node_feature_names": ["out_degree", "in_degree", "total_bytes_sent",
                               "total_bytes_received", "flow_count"],
        "edge_feature_names": ["flow_count", "total_bytes", "contains_attack"],
        "scenario_family": SCENARIO_FAMILY,
        "splits": splits_meta,
    }
    (meta_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


def _write_pyg(out_path: Path, windows: List[dict]) -> None:
    """Best-effort PyG export mirroring graph_cache._try_write_pyg."""
    try:
        import torch
        from torch_geometric.data import Data
    except Exception:
        return
    data_list = []
    for g in windows:
        if g.get("num_nodes", 0) == 0:
            continue
        x = torch.tensor(g["node_features"], dtype=torch.float)
        if g.get("edge_index"):
            ei = torch.tensor(g["edge_index"], dtype=torch.long).t().contiguous()
            ea = torch.tensor(g["edge_features"], dtype=torch.float)
        else:
            ei = torch.empty((2, 0), dtype=torch.long)
            ea = torch.empty((0, 3), dtype=torch.float)
        data_list.append(Data(
            x=x, edge_index=ei, edge_attr=ea,
            y=torch.tensor([g["label_any_attack"]], dtype=torch.long),
            window_index=g["window_index"],
        ))
    torch.save(data_list, out_path)


def assemble_full(scenarios: List[int], train_frac: float, val_frac: float) -> Path:
    """ctu-13-full: concatenate all scenarios, split by fraction chronologically
    within the concatenated (scenario-ordered) stream."""
    all_windows: List[dict] = []
    for n in sorted(scenarios):
        all_windows.extend(_load_scenario_windows(n))
    n_tot = len(all_windows)
    n_train = int(n_tot * train_frac)
    n_val = int(n_tot * val_frac)
    split_windows = {
        "train": all_windows[:n_train],
        "val": all_windows[n_train:n_train + n_val],
        "test": all_windows[n_train + n_val:],
    }
    root = _write_assembled("ctu-13-full", split_windows)
    print(f"\n[assemble] ctu-13-full: {n_tot} windows "
          f"(train={n_train}, val={n_val}, test={n_tot - n_train - n_val}) -> {root}")
    return root


def assemble_holdout(scenarios: List[int], holdout_families: List[str],
                     val_frac_of_train: float = 0.15) -> Path:
    """ctu-13-holdout: TRAIN+VAL from families NOT in holdout; TEST = held-out
    families (an OOD / unseen-attack generalisation test)."""
    holdout = set(holdout_families)
    train_val: List[dict] = []
    test: List[dict] = []
    for n in sorted(scenarios):
        fam = SCENARIO_FAMILY.get(n, "unknown")
        wins = _load_scenario_windows(n)
        if fam in holdout:
            test.extend(wins)
        else:
            train_val.extend(wins)

    n_tv = len(train_val)
    n_val = int(n_tv * val_frac_of_train)
    n_train = n_tv - n_val
    split_windows = {
        "train": train_val[:n_train],
        "val": train_val[n_train:],
        "test": test,
    }
    root = _write_assembled("ctu-13-holdout", split_windows)
    print(f"\n[assemble] ctu-13-holdout: holdout families={sorted(holdout)} | "
          f"train={n_train} val={n_val} test(OOD)={len(test)} -> {root}")
    return root


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build all CTU-13 scenarios + assemble caches.")
    ap.add_argument("--scenarios", default="1,2,3,4,5,6,7,8,9,10,11,12,13",
                    help="Comma list of scenario numbers to build.")
    ap.add_argument("--window-size", type=float, default=10.0)
    ap.add_argument("--stride", type=float, default=None)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--holdout-families", default=",".join(DEFAULT_HOLDOUT_FAMILIES))
    ap.add_argument("--skip-existing", action="store_true",
                    help="Reuse already-built ctu-13-s<N> caches.")
    ap.add_argument("--no-assemble", action="store_true",
                    help="Only build per-scenario caches; skip full/holdout assembly.")
    args = ap.parse_args(argv)

    stride = args.stride if args.stride is not None else args.window_size
    scenarios = [int(s) for s in args.scenarios.split(",") if s.strip()]

    built: List[int] = []
    for n in scenarios:
        cd = build_one_scenario(n, window_size=args.window_size, stride=stride,
                                skip_existing=args.skip_existing)
        if cd is not None:
            built.append(n)

    if not built:
        print("[build] no scenarios built.")
        return 1

    if not args.no_assemble:
        assemble_full(built, args.train_frac, args.val_frac)
        holdout = [f.strip() for f in args.holdout_families.split(",") if f.strip()]
        assemble_holdout(built, holdout)

    print(f"\n[done] scenarios built: {built}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
