"""Local-only HTTP server for starting experiments from the dashboard."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence

from .dashboard import copy_dashboard_assets, dashboard_document
from .experiment import ExperimentRequest, run_experiment


def request_from_json(payload: dict[str, Any]) -> ExperimentRequest:
    """Validate a browser request against the intentionally small parameter API."""
    allowed = {"seed", "nodes", "mean_degree", "steps", "batch_size", "dt", "baseline"}
    unexpected = set(payload) - allowed
    if unexpected:
        raise ValueError(f"unsupported parameters: {', '.join(sorted(unexpected))}")
    defaults = ExperimentRequest()
    return ExperimentRequest(
        seed=int(payload.get("seed", defaults.seed)),
        nodes=int(payload.get("nodes", defaults.nodes)),
        mean_degree=float(payload.get("mean_degree", defaults.mean_degree)),
        steps=int(payload.get("steps", defaults.steps)),
        batch_size=int(payload.get("batch_size", defaults.batch_size)),
        dt=float(payload.get("dt", defaults.dt)),
        baseline=str(payload.get("baseline", defaults.baseline)),
    )


def make_handler(directory: Path) -> type[SimpleHTTPRequestHandler]:
    """Create a static-file handler with one local JSON experiment endpoint."""

    class DashboardHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, directory=str(directory), **kwargs)

        def do_POST(self) -> None:  # noqa: N802 - inherited HTTP handler name
            if self.path != "/api/experiment":
                self.send_error(HTTPStatus.NOT_FOUND, "unknown endpoint")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 32_768:
                    raise ValueError("request body must be between 1 and 32768 bytes")
                body = self.rfile.read(length)
                payload = json.loads(body)
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                experiment = run_experiment(request_from_json(payload))
                self._send_json(HTTPStatus.OK, dashboard_document(experiment.graph, experiment.runs, experiment.config))
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

        def _send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
            content = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    return DashboardHandler


def serve(directory: Path, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Serve a dashboard directory and accept only loopback requests by default."""
    directory.mkdir(parents=True, exist_ok=True)
    copy_dashboard_assets(directory)
    server = ThreadingHTTPServer((host, port), make_handler(directory))
    print(f"Dashboard server: http://{host}:{port}/dashboard/")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard server stopped.")
    finally:
        server.server_close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the local interactive state-network dashboard.")
    parser.add_argument("--output", type=Path, default=Path("experiment_output"), help="dashboard artifact directory")
    parser.add_argument("--host", default="127.0.0.1", help="bind address; default is loopback only")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65_535:
        raise SystemExit("--port must be between 1 and 65535")
    serve(args.output, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
