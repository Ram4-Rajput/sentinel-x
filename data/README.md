# Sentinel-X `data/` layout

```
data/
├── raw/         # (empty) raw data is IMMUTABLE and lives OUTSIDE the workspace
│                #   see data/metadata/data_paths.yaml for real locations
├── processed/   # Phase-2 output: cached temporal graph sequences per dataset
│   └── <dataset>/
│       ├── windows/<split>.jsonl     # portable graph sequences (reloadable)
│       ├── graphs/<split>.pt         # optional PyTorch Geometric export
│       └── metadata/manifest.json    # reproduction metadata + build_report.json
├── unified/     # (reserved) unified exports if needed by later phases
└── metadata/    # audit + config + reports (checked in)
    ├── datasets.yaml
    ├── feature_mapping.yaml
    ├── dataset_audit.md
    ├── dataset_compatibility.csv
    ├── data_paths.yaml               # dataset raw path resolution
    ├── phase2_processing_report.md
    └── dataset_limitations.md
```

**Raw immutability:** nothing writes to `data/raw/`. Raw datasets are streamed
read-only from their external locations. Build outputs only ever go to
`data/processed/` (and optionally `data/unified/`).

Generated caches under `data/processed/` are git-ignored (large, reproducible
via `scripts/build_dataset.py`).
