from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

from evolvable_state_network.server import make_handler, request_from_json


class DashboardServerTests(unittest.TestCase):
    def test_request_validation_and_local_experiment_endpoint(self) -> None:
        request = request_from_json({"seed": 12, "nodes": 5, "mean_degree": 2, "steps": 8, "batch_size": 1, "dt": 0.1, "baseline": "fixed_rnn"})
        self.assertEqual(request.nodes, 5)
        with self.assertRaises(ValueError):
            request_from_json({"nodes": 999})
        with tempfile.TemporaryDirectory() as directory:
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Path(directory)))
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                body = json.dumps({"seed": 3, "nodes": 5, "mean_degree": 2, "steps": 8, "batch_size": 1, "dt": 0.1, "baseline": "fixed_rnn"}).encode()
                response = urlopen(Request(f"http://127.0.0.1:{server.server_port}/api/experiment", body, {"Content-Type": "application/json"}), timeout=5)
                document = json.loads(response.read())
            finally:
                server.shutdown()
                worker.join(timeout=5)
                server.server_close()
        self.assertEqual(set(document["runs"]), {"fixed_rnn"})
        self.assertEqual(document["graph"]["nodes"], 5)
        self.assertEqual(document["simulation_config"]["dt"], 0.1)
