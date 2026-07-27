"""Uvicorn entry point for the FastAPI application."""

from __future__ import annotations

import argparse
import os
from typing import Sequence

import uvicorn

from .api import app, create_app

__all__ = ["app", "create_app", "main"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the State Network Lab FastAPI application.")
    parser.add_argument("--host", default=os.environ.get("ESN_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ESN_PORT", "8000")))
    parser.add_argument("--reload", action="store_true", help="reload when source files change")
    args = parser.parse_args(argv)
    host = args.host
    port = args.port
    if not 1 <= port <= 65_535:
        raise SystemExit("ESN_PORT must be between 1 and 65535")
    print(f"State Network Lab: http://{host}:{port}/")
    print(f"API documentation: http://{host}:{port}/docs")
    uvicorn.run(
        "evolvable_state_network.api:app",
        host=host,
        port=port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
