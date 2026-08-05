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
from evolvable_state_network.application.runtime import ApplicationRuntime
from evolvable_state_network.evolution.candidate import EdgeArchitecture, RuleArchitecture
from evolvable_state_network.evolution.genome import GenomeCodec
from evolvable_state_network.server import main as server_main
from evolvable_state_network.storage import application_data_dir


class FastAPIServerTests(unittest.TestCase):
    @staticmethod
    def _write_survival_elite(
        root: Path, run_id: str, architecture: RuleArchitecture, genome: list[float]
    ) -> None:
        """Create the smallest current-format Live artifact for API tests."""
        run = root / "async_runs" / run_id
        run.mkdir(parents=True)
        config = {
            "architecture": asdict(architecture),
            "target": "node",
            "pathology": {},
            "levels": [{}],
            "elite_size": 4,
            "candidate_budget": 1,
        }
        record = {
            "candidate_id": 1,
            "genome": genome,
            "rank_key": [1.0, 1.0, 20.0],
            "status": "graduation",
            "level": 0,
            "age": 20,
            "live_eligible": True,
            "per_replica_results": [
                {
                    "normalized_pathology_burden": 0.0,
                    "responsiveness": 1.0,
                    "propagation": 1.0,
                    "distinguishability": 1.0,
                    "recovered": True,
                }
            ],
            "deployment_validation": {"passed": True, "autonomous": [], "perturbed": []},
        }
        (run / "diagnostic_config.json").write_text(json.dumps(config), encoding="utf-8")
        (run / "candidate_archive.json").write_text(json.dumps([record]), encoding="utf-8")

    @staticmethod
    def _write_embodied_run(root: Path, run_id: str) -> tuple[list[float], list[float]]:
        architecture = RuleArchitecture(state_width=2, hidden_width=2)
        edge_architecture = EdgeArchitecture(node_state_width=2, latent_width=2, hidden_width=2)
        dimension = GenomeCodec(architecture, edge_architecture, target="joint").dimension
        prey, predator = [0.0] * dimension, [0.1] * dimension
        report = {
            "ticks": 1, "architecture": asdict(architecture), "edge_architecture": asdict(edge_architecture),
            "prey_best_genome": prey, "predator_best_genome": predator,
            "prey": {"best_fitness": 0.0}, "predator": {"best_fitness": 0.0},
            "task_config": {"network": {"nodes": 12, "mean_degree": 0.0, "state_width": 2, "initial_state_scale": .12}, "environment": {}, "prey_count": 2, "predator_count": 1},
        }
        path = root / "embodied_runs" / run_id
        path.mkdir(parents=True)
        (path / "report.json").write_text(json.dumps(report), encoding="utf-8")
        return prey, predator

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
        self.assertIn("Evidence checkpoint", root.text)
        self.assertIn('id="embodied-algorithm"', root.text)
        self.assertIn('id="embodied-training-mode"', root.text)
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

    def test_embodied_training_can_continue_a_completed_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_embodied_run(root, "previousrun")
            with TestClient(create_app(root)) as client:
                invalid = client.post("/api/embodied/food-web/train", json={"model_id": "x", "continue_run_id": "previousrun"})
                started = client.post(
                    "/api/embodied/food-web/train",
                    json={"continue_run_id": "previousrun", "training_mode": "continuous", "algorithm": "genetic", "seed": 4, "population_size": 2, "ticks": 1, "prey_count": 1, "predator_count": 1, "nodes": 33, "mean_degree": 0},
                )
                job = client.get(f"/api/jobs/{started.json()['job_id']}").json()
                checkpoint = root / "embodied_runs" / started.json()["job_id"] / "checkpoint.json"
                checkpoint_exists = checkpoint.is_file()
                checkpoint_tick = json.loads(checkpoint.read_text(encoding="utf-8"))["checkpoint_tick"] if checkpoint_exists else None
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(started.status_code, 200)
        self.assertEqual(job["status"], "complete")
        self.assertEqual(job["result"]["algorithm"], "genetic")
        self.assertEqual(job["result"]["prey"]["algorithm"], "genetic")
        self.assertEqual(job["result"]["initialization"], {"kind": "embodied_run", "run_id": "previousrun"})
        self.assertTrue(checkpoint_exists)
        self.assertEqual(checkpoint_tick, 1)

    def test_embodied_batch_training_completes_comparable_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with TestClient(create_app(root)) as client:
                started = client.post(
                    "/api/embodied/food-web/train",
                    json={
                        "training_mode": "batch", "algorithm": "genetic", "seed": 4,
                        "population_size": 2, "batch_generations": 1,
                        "batch_episode_steps": 8, "batch_trials": 2, "batch_validation_trials": 1,
                        "batch_test_trials": 1, "batch_opponents": 1,
                        "enforce_survival_pressure": False,
                        "prey_count": 1, "predator_count": 1, "nodes": 33, "mean_degree": 0,
                    },
                )
                job = client.get(f"/api/jobs/{started.json()['job_id']}").json()
                checkpoint = root / "embodied_runs" / started.json()["job_id"] / "checkpoint.json"
                checkpoint_generation = json.loads(checkpoint.read_text(encoding="utf-8"))["checkpoint_generation"]
        self.assertEqual(started.status_code, 200, started.text)
        self.assertEqual(job["status"], "complete")
        self.assertEqual(job["result"]["training_mode"], "batch")
        self.assertEqual(job["result"]["prey"]["updates"], 1)
        self.assertGreater(job["result"]["prey"]["evaluations"], 0)
        self.assertIn("test_fitness", job["result"]["prey"])
        self.assertIn("vision_masked_fitness", job["result"]["prey"]["baselines"])
        self.assertTrue(job["result"]["task_config"]["diagnostics"]["final_test_touched_only_after_selection"])
        self.assertEqual(checkpoint_generation, 1)

    def test_embodied_demo_accepts_ecology_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_embodied_run(root, "demorun")
            with TestClient(create_app(root)) as client:
                response = client.post(
                    "/api/embodied/sessions",
                    json={"run_id": "demorun", "seed": 4, "prey_count": 4, "predator_count": 3, "initial_food": 11, "max_food": 30, "food_growth_rate": 5.5},
                )
        self.assertEqual(response.status_code, 200, response.text)
        state = response.json()["state"]
        self.assertEqual(state["population"], {"prey": 4, "predator": 3})
        self.assertEqual(len(state["plants"]), 11)
        self.assertEqual(state["plant_capacity"], 30)

    def test_embodied_demo_uses_current_checkpoint_before_run_completes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_embodied_run(root, "checkpointed")
            run = root / "embodied_runs" / "checkpointed"
            report = (run / "report.json").read_text(encoding="utf-8")
            (run / "checkpoint.json").write_text(report, encoding="utf-8")
            (run / "report.json").unlink()
            with TestClient(create_app(root)) as client:
                runs = client.get("/api/embodied/runs").json()["runs"]
                response = client.post(
                    "/api/embodied/sessions",
                    json={"run_id": "checkpointed", "seed": 4, "initial_food": 11, "max_food": 30},
                )
        selected = next(run for run in runs if run["id"] == "checkpointed")
        self.assertFalse(selected["complete"])
        self.assertEqual(selected["source"], "current_checkpoint")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["model_source"], "current_checkpoint")

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
                replay = client.get(replica["debug_replay_url"])
        self.assertEqual(job["status"], "complete")
        self.assertTrue(latest["available"])
        self.assertEqual(latest["report"]["mode"], "asynchronous_death_driven_joint_evolution")
        self.assertTrue(latest["candidates"])
        self.assertEqual(latest["report"]["stop_reason"], "stage_not_passed_tick_limit")
        self.assertGreater(latest["report"]["optimizer_updates"], 0)
        self.assertEqual(report.status_code, 200)
        self.assertEqual(report.json()["mode"], "asynchronous_death_driven_joint_evolution")
        self.assertEqual(replay.status_code, 200)
        replay_run = next(iter(replay.json()["runs"].values()))
        self.assertEqual(replay_run["trajectory"]["steps"][-1], replica["age"])
        self.assertEqual(replay.json()["graph"]["nodes"], 7)

    def test_configurable_survival_training_uses_budget_as_evidence_checkpoint(self) -> None:
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
                        "candidate_budget": 1,
                        "max_ticks": 40,
                        "slots": 2,
                        "replicas": 1,
                        "stable_population_size": 2,
                        "optimizer_batch": 2,
                        "state_width": 3,
                        "stage_1_lifetime": 100,
                        "stage_2_lifetime": 200,
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
        self.assertGreaterEqual(report["completed_candidates"], 1)
        self.assertEqual(report["candidate_budget"], 1)
        self.assertTrue(report["candidate_evidence_checkpoint_reached"])
        self.assertEqual(report["stop_reason"], "stage_not_passed_tick_limit")
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
            self._write_survival_elite(
                root, "demobest", architecture, [0.0] * architecture.parameter_count
            )
            with TestClient(create_app(root)) as client:
                models = client.get("/api/live/models").json()
                initial = client.post(
                    "/api/live/sessions",
                    json={
                        "model_id": models["models"][0]["id"], "seed": 7, "nodes": 5,
                        "mean_degree": 2, "batch_size": 1, "dt": .1,
                        "topology": "ring",
                    },
                ).json()
                repeat = client.post(
                    "/api/live/sessions",
                    json={
                        "model_id": models["models"][0]["id"], "seed": 7, "nodes": 5,
                        "mean_degree": 2, "batch_size": 1, "dt": .1,
                        "topology": "ring",
                    },
                ).json()
                stepped = client.post(
                    f"/api/live/sessions/{initial['session_id']}/step",
                    json={"steps": 3},
                ).json()
        self.assertEqual(models["models"][0]["id"], "survival:demobest:1")
        self.assertEqual(models["models"][0]["source"], "survival")
        self.assertEqual(initial["graph"]["nodes"], 5)
        self.assertEqual(initial["topology"], "ring")
        self.assertEqual(initial["step"], 0)
        self.assertEqual(initial["initial_state_scale"], .12)
        self.assertEqual(initial["node_state"], repeat["node_state"])
        self.assertEqual(len({node[0] for node in initial["node_state"][0]}), 5)
        self.assertNotEqual(initial["node_state"][0], [(0.0,)] * 5)
        self.assertEqual(stepped["step"], 3)

    def test_survival_live_models_require_fresh_graph_validation(self) -> None:
        record = {
            "candidate_id": 1,
            "genome": [1.0],
            "rank_key": [1.0, 1.0],
            "status": "graduation",
            "level": 0,
            "live_eligible": True,
            "per_replica_results": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "candidate_archive.json").write_text(json.dumps([record]), encoding="utf-8")
            self.assertEqual(ApplicationRuntime._survival_elites(run, {"levels": [{}]}), [])
            record["deployment_validation"] = {"passed": True, "autonomous": [], "perturbed": []}
            (run / "candidate_archive.json").write_text(json.dumps([record]), encoding="utf-8")
            self.assertEqual(
                [item["candidate_id"] for item in ApplicationRuntime._survival_elites(run, {"levels": [{}]})],
                [1],
            )

    def test_live_session_keeps_running_and_reports_safety_events(self) -> None:
        architecture = RuleArchitecture(state_width=1, hidden_width=1)
        # A positive output bias produces sustained growth.  Live replay must
        # surface whichever health failure occurs first, instead of continuing
        # with silently bounded state values.
        genome = [0.0, 0.0, 2.0, 0.0, 1.0, 2.0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_survival_elite(root, "runaway", architecture, genome)
            with TestClient(create_app(root)) as client:
                model_id = client.get("/api/live/models").json()["models"][0]["id"]
                initial = client.post(
                    "/api/live/sessions",
                    json={
                        "model_id": model_id,
                        "seed": 7,
                        "nodes": 5,
                        "mean_degree": 2,
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
