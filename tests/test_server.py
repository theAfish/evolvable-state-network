from __future__ import annotations

import json
import os
import tempfile
import unittest
import warnings
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

warnings.filterwarnings(
    "ignore",
    message="Using `httpx` with `starlette.testclient` is deprecated.*",
)

from fastapi.testclient import TestClient

from evolvable_state_network.api import create_app
from evolvable_state_network.candidate import RuleArchitecture
from evolvable_state_network.server import main as server_main
from evolvable_state_network.storage import application_data_dir


class FastAPIServerTests(unittest.TestCase):
    def test_default_storage_is_project_local_outputs_directory(self) -> None:
        with patch.dict(os.environ):
            os.environ.pop("ESN_DATA_DIR", None)
            self.assertEqual(application_data_dir(), (Path.cwd() / ".outputs").resolve())

    def test_server_port_can_be_overridden_from_cli_or_environment(self) -> None:
        with patch("evolvable_state_network.server.uvicorn.run") as run:
            with patch.dict(os.environ):
                os.environ.pop("ESN_PORT", None)
                server_main([])
                self.assertEqual(run.call_args.kwargs["port"], 8000)
                self.assertTrue(run.call_args.kwargs["factory"])

            with patch.dict(os.environ, {"ESN_PORT": "8124"}):
                self.assertEqual(server_main([]), 0)
                self.assertEqual(run.call_args.kwargs["port"], 8124)

            server_main(["--port", "8125"])
            self.assertEqual(run.call_args.kwargs["port"], 8125)

            with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
                server_main(["--data-dir", directory])
                self.assertEqual(os.environ["ESN_DATA_DIR"], str(Path(directory).resolve()))

    def test_frontend_root_health_docs_and_dashboard_redirect(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with TestClient(create_app(Path(directory))) as client:
                root = client.get("/")
                redirect = client.get("/dashboard/", follow_redirects=False)
                health = client.get("/api/health")
                docs = client.get("/docs")
        self.assertEqual(root.status_code, 200)
        self.assertIn("Train local rules by how long they remain healthy", root.text)
        self.assertIn("Candidate lives", root.text)
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
                candidate = latest["candidates"][0]
                replica = candidate["replicas"][0]
                replay = client.get(replica["replay_url"])
        self.assertEqual(job["status"], "complete")
        self.assertTrue(latest["available"])
        self.assertEqual(latest["report"]["mode"], "asynchronous_death_driven_joint_evolution")
        self.assertTrue(latest["candidates"])
        self.assertTrue(latest["slots"])
        self.assertEqual(report.status_code, 200)
        self.assertEqual(report.json()["mode"], "asynchronous_death_driven_joint_evolution")
        self.assertEqual(replay.status_code, 200)
        replay_run = next(iter(replay.json()["runs"].values()))
        self.assertEqual(replay_run["trajectory"]["steps"][-1], replica["age"])
        self.assertEqual(replay.json()["graph"]["nodes"], 7)

    def test_configurable_survival_training_reports_episode_budget_and_stop_reason(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with TestClient(create_app(Path(directory))) as client:
                invalid = client.post(
                    "/api/async/train",
                    json={"stage_1_lifetime": 20, "stage_2_lifetime": 10},
                )
                started = client.post(
                    "/api/async/train",
                    json={
                        "seed": 12,
                        "candidate_budget": 8,
                        "max_ticks": 40,
                        "slots": 2,
                        "replicas": 1,
                        "optimizer_batch": 2,
                        "state_width": 3,
                        "stage_1_lifetime": 4,
                        "stage_2_lifetime": 8,
                        "stage_1_nodes": 4,
                        "stage_2_nodes": 5,
                        "mean_degree": 2,
                    },
                )
                job = client.get(f"/api/jobs/{started.json()['job_id']}").json()
                live_models = client.get("/api/live/models").json()
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(job["status"], "complete")
        self.assertEqual(job["kind"], "async_training")
        self.assertEqual(job["result"]["run_kind"], "training")
        report = job["result"]["report"]
        self.assertGreaterEqual(report["completed_candidates"], 8)
        self.assertEqual(report["candidate_budget"], 8)
        self.assertEqual(report["stop_reason"], "candidate_budget_reached")
        self.assertEqual(report["completed_replica_lives"], report["completed_candidates"])
        # This deliberately short run can only graduate candidates from the
        # first curriculum level.  Those records remain inspectable training
        # evidence, but must not be offered as Live deployments.
        self.assertFalse([model for model in live_models["models"] if model["source"] == "survival"])
        latest = live_models["latest_survival"]
        self.assertEqual(latest["run_id"], job["result"]["run_id"])
        self.assertTrue(latest["candidates"])
        self.assertTrue(all("live_eligible" in candidate for candidate in latest["candidates"]))

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
                        "model_id": models["models"][0]["id"], "seed": 7, "nodes": 5,
                        "mean_degree": 2, "batch_size": 1, "dt": .1,
                        "topology": "ring", "input_seed": 19,
                        "input_standard_deviation": .15,
                    },
                ).json()
                stepped = client.post(
                    f"/api/live/sessions/{initial['session_id']}/step",
                    json={"steps": 3},
                ).json()
        self.assertEqual(models["models"][0]["id"], "legacy:demo-best")
        self.assertEqual(models["models"][0]["source"], "legacy")
        self.assertEqual(initial["graph"]["nodes"], 5)
        self.assertEqual(initial["topology"], "ring")
        self.assertEqual(initial["step"], 0)
        self.assertEqual(stepped["step"], 3)

    def test_live_session_keeps_running_and_reports_safety_events(self) -> None:
        architecture = RuleArchitecture(state_width=1, hidden_width=1)
        # A positive output bias produces sustained growth.  Live replay must
        # surface whichever health failure occurs first, instead of continuing
        # with silently bounded state values.
        genome = [0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 2.0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "evolution_runs" / "runaway"
            model.mkdir(parents=True)
            (model / "best_genome.json").write_text(
                json.dumps(
                    {
                        "architecture": asdict(architecture),
                        "edge_architecture": None,
                        "target": "node",
                        "genome": genome,
                        "validation": {"fitness": .2},
                        "test": {"fitness": .1},
                    }
                ),
                encoding="utf-8",
            )
            with TestClient(create_app(root)) as client:
                model_id = client.get("/api/live/models").json()["models"][0]["id"]
                initial = client.post(
                    "/api/live/sessions",
                    json={
                        "model_id": model_id,
                        "seed": 7,
                        "nodes": 5,
                        "mean_degree": 2,
                        "input_standard_deviation": 1.0,
                    },
                ).json()
                observed_response = None
                for _ in range(10):
                    observed_response = client.post(
                        f"/api/live/sessions/{initial['session_id']}/step",
                        json={"steps": 8},
                    )
                    if (
                        observed_response.status_code == 200
                        and observed_response.json().get("last_safety_event")
                    ):
                        break
                continued_response = client.post(
                    f"/api/live/sessions/{initial['session_id']}/step",
                    json={"steps": 1},
                )
                assert observed_response is not None
                self.assertEqual(observed_response.status_code, 200, observed_response.text)
                self.assertEqual(continued_response.status_code, 200, continued_response.text)
                observed = observed_response.json()
                continued = continued_response.json()
        self.assertEqual(observed["status"], "running")
        safety = observed["last_safety_event"]
        self.assertEqual(safety["kind"], "node_state_clipped")
        self.assertEqual(safety["details"]["coordinate"], 0)
        self.assertGreater(abs(safety["details"]["after_delta_limit"]), safety["details"]["bound"])
        self.assertEqual(continued["status"], "running")
        self.assertEqual(continued["step"], observed["step"] + 1)


if __name__ == "__main__":
    unittest.main()
