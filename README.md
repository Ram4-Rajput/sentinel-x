# Sentinel-X

Sentinel-X is an AI-powered network world model for forecasting how network behaviour evolves over time. It combines a unified flow-data layer, leakage-aware temporal experiments, graph and sequence models, uncertainty and novelty analysis, explainability, a FastAPI serving layer, and a React/Vite command-center frontend.

## Repository layout

- `src/sentinelx/` - Python package and model pipeline
- `scripts/` - dataset, training, evaluation, and serving CLIs
- `tests/` - Python test suite
- `frontend/` - React + TypeScript + Vite frontend
- `docs/` - phase reports and developer documentation
- `configs/` - model configuration

Raw datasets, generated caches, model weights, and local experiment stores are intentionally excluded from Git. See `data/README.md` and `data/metadata/` for data layout and provenance notes.

## Python setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test,serve,graph]"
python -m pytest -q
```

The core data layer is dependency-light. The `graph` extra installs the optional PyTorch/PyTorch Geometric model dependencies; the `serve` extra enables the API.

## Run the API

The serving layer needs a trained checkpoint and processed cache available locally:

```powershell
python scripts/serve.py
```

The API is available at `http://127.0.0.1:8000`, with interactive documentation at `/docs`.

## Run the frontend

```powershell
cd frontend
npm ci
npm run dev
```

Open `http://localhost:5173`. Without a running API, the frontend uses its schema-faithful local mock; with the API running, Vite proxies `/api` to FastAPI.

## Reproducibility

Start with `docs/DEVELOPER_DOCUMENTATION.md` and `docs/PROJECT_KICKOFF.md`. Dataset-specific setup and provenance remain in `data/metadata/`; datasets must be obtained separately and are not redistributed by this repository.

## Status

This is a research and engineering project. Reported metrics and limitations are preserved in `experiments/` and the phase reports under `docs/`.
