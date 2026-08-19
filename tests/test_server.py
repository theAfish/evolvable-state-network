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
            "prey": {"best_lifetime": 0.0}, "predator": {"best_lifetime": 0.0},
            "task_config": {"embodied_interface": "ray_image_v3_sparse_multichannel_v1", "network": {"nodes": 34, "mean_degree": 0.0, "state_width": 2, "initial_state_scale": .12, "vision_pixels": 9}, "environment": {}, "prey_count": 2, "predator_count": 1},
        }
        path = root / "embodied_runs" / run_id
        path.mkdir(parents=True)
        (path / "report.json").write_text(json.dumps(report), encoding="utf-8")
        return prey, predator

    def test_default_storage_is_project_local_outputs_directory(self) -> None:
        with patch.dict(os.environ):
            os.environ.pop("ESN_DATA_DIR", None)
            self.assertEqual(application_data_dir(), (Path.cwd() / ".outputs").resolve())

    def test_embodied_report_persists_a_bounded_representative_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = ApplicationRuntime(Path(directory))
            original = {"task": "batch_food_web_coevolution", "history": [
                {"generation": generation, "prey_best_lifetime": float(generation)}
                for generation in range(1, 401)
            ]}
            saved = runtime.write_embodied_report("longrun", original)
            path = Path(directory) / "embodied_runs" / "longrun" / "report.json"
            on_disk = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(len(original["history"]), 400)
        self.assertEqual(saved, on_disk)
        self.assertEqual(on_disk["history_total_records"], 400)
        self.assertEqual(on_disk["history_retained_records"], 128)
        self.assertEqual(on_disk["history_sampling"]["strategy"], "uniform_including_endpoints")
        self.assertEqual(on_disk["history"][0]["generation"], 1)
        self.assertEqual(on_disk["history"][-1]["generation"], 400)

    def test_embodied_checkpoint_is_used_when_report_is_corrupt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_embodied_run(root, "recoverable")
            path = root / "embodied_runs" / "recoverable"
            report = json.loads((path / "report.json").read_text(encoding="utf-8"))
            (path / "checkpoint.json").write_text(json.dumps(report), encoding="utf-8")
            (path / "report.json").write_bytes(b"\0" * 128)
            runtime = ApplicationRuntime(root)
            loaded = runtime.load_embodied_report("recoverable")
            runs = runtime.available_embodied_runs()

        self.assertEqual(loaded["ticks"], 1)
        run = next(item for item in runs if item["id"] == "recoverable")
        self.assertFalse(run["complete"])
        self.assertEqual(run["source"], "current_checkpoint")

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
        self.assertIn('id="embodied-state-width"', root.text)
        self.assertIn('id="embodied-terminate"', root.text)
        self.assertIn('data-view="diagnostics"', root.text)
        self.assertIn('id="diagnostics-load-server"', root.text)
        self.assertIn('id="diagnostics-report-file"', root.text)
        self.assertEqual(redirect.status_code, 307)
        self.assertEqual(redirect.headers["location"], "/")
        self.assertEqual(health.json()["status"], "ok")
        self.assertEqual(docs.status_code, 200)

    def test_frontend_disables_step_grid_validation_for_all_number_inputs(self) -> None:
        script = (
            Path(__file__).parents[1] / "src" / "evolvable_state_network" / "web" / "app.js"
        ).read_text(encoding="utf-8")
        self.assertIn("function allowArbitraryNumberPrecision", script)
        self.assertIn("input.step = 'any'", script)
        self.assertIn("new MutationObserver", script)

    def test_embodied_termination_endpoint_marks_only_running_embodied_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with TestClient(create_app(Path(directory))) as client:
                runtime = client.app.state.runtime
                job_id = runtime.new_job("embodied_food_web", 3, 50)
                response = client.post(f"/api/embodied/jobs/{job_id}/terminate")
                duplicate = client.post(f"/api/embodied/jobs/{job_id}/terminate")
                unrelated = runtime.new_job("async_training", 4, 50)
                wrong_kind = client.post(f"/api/embodied/jobs/{unrelated}/terminate")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"job_id": job_id, "termination_requested": True})
        self.assertTrue(runtime.job_termination_requested(job_id))
        self.assertEqual(duplicate.status_code, 200)
        self.assertEqual(wrong_kind.status_code, 404)

    def test_embodied_training_can_continue_a_completed_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_embodied_run(root, "previousrun")
            with TestClient(create_app(root)) as client:
                invalid = client.post("/api/embodied/food-web/train", json={"model_id": "x", "continue_run_id": "previousrun"})
                started = client.post(
                    "/api/embodied/food-web/train",
                    json={"continue_run_id": "previousrun", "training_mode": "continuous", "algorithm": "genetic", "seed": 4, "population_size": 2, "ticks": 1, "prey_count": 1, "predator_count": 1, "hidden_nodes": 1, "mean_degree": 0},
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
                        "prey_count": 1, "predator_count": 1, "hidden_nodes": 1, "state_width": 3, "mean_degree": 0,
                        "workers": 1,
                        "max_speed": 12.0, "max_turn": 4.0,
                        "network_dt": .05, "max_delta": .24, "edge_step_scale": .06,
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
        self.assertIn("test_lifetime", job["result"]["prey"])
        self.assertIn("vision_masked_lifetime", job["result"]["prey"]["baselines"])
        self.assertEqual(job["result"]["objective"], "restricted_mean_lifetime")
        self.assertFalse(job["result"]["task_config"]["reward_shaping"])
        self.assertFalse(job["result"]["task_config"]["environment"]["respawn_on_death"])
        self.assertEqual(job["result"]["task_config"]["environment"]["max_speed"], 12.0)
        self.assertEqual(job["result"]["task_config"]["environment"]["max_turn"], 4.0)
        self.assertEqual(job["result"]["task_config"]["network"]["dt"], .05)
        self.assertEqual(job["result"]["task_config"]["network"]["max_delta"], .24)
        self.assertEqual(job["result"]["task_config"]["network"]["edge_step_scale"], .06)
        self.assertEqual(job["result"]["task_config"]["network"]["state_width"], 3)
        self.assertEqual(job["result"]["task_config"]["network"]["nodes"], 31)
        self.assertEqual(job["result"]["task_config"]["network"]["body_inputs"], ["hunger"])
        self.assertEqual(job["result"]["task_config"]["diagnostics"]["hidden_nodes"], 1)
        self.assertTrue(job["result"]["task_config"]["diagnostics"]["final_test_touched_only_after_selection"])
        self.assertEqual(checkpoint_generation, 1)

    def test_embodied_batch_can_evolve_a_mixed_individual_population(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with TestClient(create_app(Path(directory))) as client:
                started = client.post(
                    "/api/embodied/food-web/train",
                    json={
                        "training_mode": "batch", "batch_population_mode": "mixed_individual_population",
                        "algorithm": "genetic", "seed": 5, "population_size": 2,
                        "prey_count": 2, "predator_count": 0, "batch_generations": 1,
                        "batch_episode_steps": 8, "batch_trials": 1,
                        "batch_validation_trials": 1, "batch_test_trials": 1,
                        "batch_opponents": 1, "enforce_survival_pressure": False,
                        "hidden_nodes": 1, "state_width": 3, "mean_degree": 0,
                        "workers": 1,
                    },
                )
                job = client.get(f"/api/jobs/{started.json()['job_id']}").json()
        self.assertEqual(started.status_code, 200, started.text)
        self.assertEqual(job["status"], "complete")
        report = job["result"]
        self.assertEqual(report["population_mode"], "mixed_individual_population")
        self.assertEqual(report["world_count"], 2)
        self.assertEqual(report["prey_genome_population_size"], 4)
        self.assertEqual(report["prey"]["evaluations"], 4)
        self.assertEqual(report["task_config"]["batch_population_mode"], "mixed_individual_population")

    def test_embodied_demo_accepts_ecology_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_embodied_run(root, "demorun")
            with TestClient(create_app(root)) as client:
                response = client.post(
                    "/api/embodied/sessions",
                    json={"run_id": "demorun", "seed": 4, "network_hidden_nodes": 42, "network_mean_degree": 7.5, "prey_count": 4, "predator_count": 3, "initial_food": 11, "max_food": 30, "food_growth_rate": 5.5},
                )
                session = response.json()
                individual_id = session["state"]["organisms"][0]["id"]
                network = client.get(
                    f"/api/embodied/sessions/{session['session_id']}/individuals/{individual_id}"
                ).json()["network"]
        self.assertEqual(response.status_code, 200, response.text)
        state = response.json()["state"]
        self.assertEqual(state["population"], {"prey": 4, "predator": 3})
        self.assertEqual(len(state["plants"]), 11)
        self.assertEqual(state["plant_capacity"], 30)
        self.assertEqual(network["nodes"], 42 + len(network["input_nodes"]) + len(network["action_nodes"]))

    def test_embodied_demo_exposes_selected_individual_network_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_embodied_run(root, "inspectrun")
            with TestClient(create_app(root)) as client:
                session = client.post(
                    "/api/embodied/sessions",
                    json={"run_id": "inspectrun", "seed": 4, "prey_count": 1, "predator_count": 0, "max_food": 30},
                ).json()
                individual_id = session["state"]["organisms"][0]["id"]
                response = client.get(
                    f"/api/embodied/sessions/{session['session_id']}/individuals/{individual_id}"
                )
        self.assertEqual(response.status_code, 200, response.text)
        snapshot = response.json()
        self.assertEqual(snapshot["individual"]["id"], individual_id)
        self.assertGreater(snapshot["network"]["nodes"], 0)
        self.assertEqual(len(snapshot["network"]["node_state"]), snapshot["network"]["nodes"])
        self.assertEqual(snapshot["network"]["state_width"], 2)
        self.assertGreaterEqual(snapshot["network"]["vision_pixels"], 1)
        # Hunger occupies state channel 1; the following ray-image inputs use
        # channel 0, matching FoodWebAgentAdapter's multichannel contract.
        self.assertEqual(snapshot["network"]["input_signal_channels"][:4], [1, 0, 0, 0])

    def test_post_run_random_graph_diagnostic_returns_individual_fitness_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_embodied_run(root, "diagnosticrun")
            with TestClient(create_app(root)) as client:
                started = client.post(
                    "/api/embodied/runs/diagnosticrun/diagnostics/random-graphs",
                    json={"sample_count": 2, "seed": 9},
                )
                job = client.get(f"/api/jobs/{started.json()['job_id']}").json()
        self.assertEqual(started.status_code, 200, started.text)
        self.assertEqual(job["status"], "complete")
        result = job["result"]
        self.assertEqual(result["mode"], "random_graph_random_state")
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["episode_seeds"], [9, 10016])
        self.assertEqual(len(result["prey"]["fitness_values"]), 2)
        self.assertIn("variance", result["prey"]["fitness"])

    def test_saved_runs_can_be_compared_with_common_raw_output_probes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_embodied_run(root, "first")
            self._write_embodied_run(root, "second")
            with TestClient(create_app(root)) as client:
                response = client.post(
                    "/api/embodied/diagnostics/compare",
                    json={"left_run_id": "first", "right_run_id": "second", "probe_count": 32, "seed": 9},
                )
        self.assertEqual(response.status_code, 200, response.text)
        comparison = response.json()
        self.assertEqual(comparison["probe_count"], 32)
        self.assertTrue(comparison["prey"]["joint_genome"]["compatible"])
        self.assertIn("abs_gt_3_fraction", comparison["prey"]["node_rule_raw_output"]["left"])
        self.assertIn("rms_distance", comparison["prey"]["edge_rule_raw_output"]["common_probe_response"])

    def test_checkpoint_evaluation_records_matched_behavior_and_dynamics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_embodied_run(root, "first")
            self._write_embodied_run(root, "second")
            with TestClient(create_app(root)) as client:
                started = client.post(
                    "/api/embodied/diagnostics/checkpoints/evaluate",
                    json={"left_run_id": "first", "right_run_id": "second", "evaluation_samples": 1, "parameter_scales": [1.0], "seed": 9},
                )
                job = client.get(f"/api/jobs/{started.json()['job_id']}").json()
                artifact = client.get(job["result"]["output_url"])
        self.assertEqual(started.status_code, 200, started.text)
        self.assertEqual(job["status"], "complete")
        result = job["result"]
        self.assertEqual(result["evaluation_seeds"], [9])
        self.assertIn("mean_absolute_action_difference", result["differences"])
        self.assertIn("abs_gt_10_fraction", result["checkpoint_a"]["node_rule_raw_output"])
        self.assertIn("near_zero_fraction", result["checkpoint_b"]["edge_update"])
        self.assertIn("1.0", result["parameter_scaling"]["results"])
        self.assertEqual(artifact.status_code, 200)

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
        self.assertEqual(selected["state_width"], 2)
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
        genome = [0.0, 0.0, 2.0, 1.0, 2.0]
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
