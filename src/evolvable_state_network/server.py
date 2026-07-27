"""Uvicorn entry point for the FastAPI application."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import uvicorn

from .api import app, create_app
from .storage import application_data_dir

__all__ = ["app", "create_app", "main"]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the State Network Lab FastAPI application.")
    parser.add_argument(
        "--host",
        default=os.environ.get("ESN_HOST", DEFAULT_HOST),
        help=f"interface to bind (default: {DEFAULT_HOST}; override with ESN_HOST)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("ESN_PORT", str(DEFAULT_PORT))),
        help=f"port to bind (default: {DEFAULT_PORT}; override with ESN_PORT)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="directory for generated runs (default: .\\.outputs; overrides ESN_DATA_DIR)",
    )
    parser.add_argument("--reload", action="store_true", help="reload when source files change")
    args = parser.parse_args(argv)
    host = args.host
    port = args.port
    if not 1 <= port <= 65_535:
        raise SystemExit("ESN_PORT must be between 1 and 65535")
    if args.data_dir is not None:
        os.environ["ESN_DATA_DIR"] = str(args.data_dir.expanduser().resolve())
    storage = application_data_dir()
    print(f"State Network Lab: http://{host}:{port}/")
    print(f"API documentation: http://{host}:{port}/docs")
    print(f"Run storage: {storage}")
    uvicorn.run(
        "evolvable_state_network.api:create_app",
        host=host,
        port=port,
        reload=args.reload,
        factory=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
