#!/usr/bin/env python
"""Phase-10 CLI: launch the Sentinel-X serving API.

    python scripts/serve.py
    python scripts/serve.py --host 0.0.0.0 --port 8080

Serves the trained Phase-4 world model + Phases 5-9 analytical layers through
FastAPI. The model + leakage-safe K-step samples are loaded once, lazily, on the
first request. Nothing is retrained and no cache is rebuilt.

Requires uvicorn (pip install uvicorn). On Windows, run this in your own
terminal — it is a long-running server.
"""

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Serve the Sentinel-X world model.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true",
                    help="Auto-reload on code changes (development only).")
    args = ap.parse_args(argv)

    try:
        import uvicorn
    except ImportError:
        print("[serve] uvicorn is not installed. Install it with: "
              "pip install uvicorn", file=sys.stderr)
        return 1

    uvicorn.run("sentinelx.serving.app:app", host=args.host, port=args.port,
                reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
