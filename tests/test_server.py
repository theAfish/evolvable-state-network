from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from dataclasses import asdict
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message="Using `httpx` with `starlette.testclient` is deprecated.*",
)

from fastapi.testclient import TestClient

from evolvable_state_network.api import create_app
from evolvable_state_network.candidate import RuleArchitecture


class FastAPIServerTests(unittest.TestCase):
    def test_frontend_root_health_docs_and_dashboard_redirect(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with TestClient(create_app(Path(directory))) as client:
                root = client.get("/")
                redirect = client.get("/dashboard/", follow_redirects=False)
                health = client.get("/api/health")
                docs = client.get("/docs")
        self.assertEqual(root.status_code, 200)
        self.assertIn("Every slot runs until death or graduation", root.text)
        self.assertEqual(redirect.status_code, 307)
        self.assertEqual(redirect.headers["location"], "/")
        self.assertEqual(health.json()["status"], "ok")
        self.assertEqual(docs.status_code, 200)

    def test_typed_experiment_validation_and_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with TestClient(create_app(Path(directory))) as client:
                invalid = client.post("/api/experiment", json={"nodes": 999})
                response = client.post(
                    "/api/experiment",
                    json={
                        "seed": 3, "nodes": 5, "mean_degree": 2, "steps": 8,
                        "batch_size": 1, "dt": .1, "baseline": "fixed_rnn",
                    },
                )
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(response.status_code, 200)
        document = response.json()
        self.assertEqual(set(document["runs"]), {"fixed_rnn"})
        self.assertEqual(document["graph"]["nodes"], 5)
        self.assertEqual(document["simulation_config"]["dt"], .1)

    def test_random_search_background_job(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with TestClient(create_app(Path(directory))) as client:
                started = client.post(
                    "/api/evolution/random-search", json={"seed": 9, "samples": 4}
                )
                status = client.get(f"/api/jobs/{started.json()['job_id']}")
                missing = client.get("/api/jobs/not-a-job")
        self.assertEqual(started.status_code, 200)
        self.assertEqual(status.json()["status"], "complete")
        self.assertEqual(len(status.json()["samples"]), 4)
        self.assertIn("smoke_report", status.json()["result"])
        self.assertEqual(missing.status_code, 404)

    def test_async_diagnostic_and_artifact_mount(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with TestClient(create_app(Path(directory))) as client:
                started = client.post("/api/async/diagnostic", json={"seed": 5})
                job = client.get(f"/api/jobs/{started.json()['job_id']}").json()
                latest = client.get("/api/async/latest").json()
                report = client.get(latest["artifacts"]["report"])
        self.assertEqual(job["status"], "complete")
        self.assertTrue(latest["available"])
        self.assertEqual(latest["report"]["mode"], "asynchronous_death_driven_joint_evolution")
        self.assertTrue(latest["candidates"])
        self.assertTrue(latest["slots"])
        self.assertEqual(report.status_code, 200)
        self.assertEqual(report.json()["mode"], "asynchronous_death_driven_joint_evolution")

    def test_live_run_uses_exported_parameters_on_a_fresh_graph(self) -> None:
        architecture = RuleArchitecture(state_width=1, hidden_width=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "evolution_runs" / "demo-best"
            model.mkdir(parents=True)
            (model / "best_genome.json").write_text(
                json.dumps(
                    {
                        "architecture": asdict(architecture),
                        "edge_architecture": None,
                        "target": "node",
                        "genome": [0.0] * architecture.parameter_count,
                        "validation": {"fitness": .2},
                        "test": {"fitness": .1},
                    }
                ),
                encoding="utf-8",
            )
            with TestClient(create_app(root)) as client:
                models = client.get("/api/live/models").json()
                initial = client.post(
                    "/api/live/sessions",
                    json={
                        "model_id": "demo-best", "seed": 7, "nodes": 5,
                        "mean_degree": 2, "batch_size": 1, "dt": .1,
                        "topology": "ring", "input_seed": 19,
                        "input_standard_deviation": .15,
                    },
                ).json()
                stepped = client.post(
                    f"/api/live/sessions/{initial['session_id']}/step",
                    json={"steps": 3},
                ).json()
        self.assertEqual(models["models"][0]["id"], "demo-best")
        self.assertEqual(initial["graph"]["nodes"], 5)
        self.assertEqual(initial["topology"], "ring")
        self.assertEqual(initial["step"], 0)
        self.assertEqual(stepped["step"], 3)


if __name__ == "__main__":
    unittest.main()
