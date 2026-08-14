"""FastAPI route composition for the state-network research workspace."""

from __future__ import annotations

import json
import secrets
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from .application.configuration import build_async_training_config
from .application.models import (
    AsyncDiagnosticPayload,
    AsyncTrainingPayload,
    EmbodiedFoodWebTrainingPayload,
    EmbodiedDemoPayload,
    EmbodiedDemoStepPayload,
    EvolutionPayload,
    ExperimentPayload,
    LiveSessionPayload,
    LiveStepPayload,
)
from .application.runtime import ApplicationRuntime
from .evolution.asynchronous import run_async_experiment, run_diagnostic_experiment
from .evolution.candidate import EdgeArchitecture, RuleArchitecture
from .dashboard import dashboard_document
from .evolution.evaluation import CandidateEvaluator
from .evolution import EvolutionConfig, EvolutionRunner, random_search_smoke_test
from .experiment import ExperimentRequest, run_experiment
from .storage import application_data_dir
from .embodied import EmbodiedNetworkConfig, FoodWebAgentAdapter
from .environments import FoodWebConfig
from .tasks import (
    BatchFoodWebCoevolutionRunner,
    BatchFoodWebConfig,
    EmbodiedFoodWebTaskConfig,
    EmbodiedRuleEvolutionConfig,
    ContinuousFoodWebConfig,
    ContinuousFoodWebCoevolutionRunner,
    EvolutionTerminated,
    FoodWebCoevolutionEvaluator,
)


def _seed(value: int | None) -> int:
    return secrets.randbelow(2**32) if value is None else value


def create_app(data_dir: Path | None = None) -> FastAPI:
    """Create an isolated application instance rooted at ``data_dir``."""

    root = (data_dir or application_data_dir()).resolve()
    runtime = ApplicationRuntime(root)
    application = FastAPI(
        title="Evolvable State Network",
        summary="Asynchronous survival evolution and replay API",
        version="0.2.0",
    )
    application.state.runtime = runtime

    @application.get("/api/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "storage": str(runtime.root)}

    @application.post("/api/experiment")
    def experiment(payload: ExperimentPayload) -> dict[str, object]:
        request = ExperimentRequest(**payload.model_dump())
        result = run_experiment(request)
        return dashboard_document(result.graph, result.runs, result.config)

    @application.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict[str, object]:
        try:
            return runtime.job_snapshot(job_id)
        except KeyError as error:
            raise HTTPException(404, "unknown job") from error

    @application.post("/api/async/diagnostic")
    def start_async(
        payload: AsyncDiagnosticPayload, background_tasks: BackgroundTasks
    ) -> dict[str, str]:
        runtime.ensure_root()
        seed = _seed(payload.seed)
        job_id = runtime.new_job("async_diagnostic", seed, 80)
        run_directory = runtime.root / "async_runs" / job_id

        def worker() -> None:
            try:
                report = run_diagnostic_experiment(
                    run_directory,
                    seed,
                    progress=lambda event: runtime.update_job(
                        job_id, {"phase": "asynchronous", **event}
                    ),
                )
                result = runtime.async_run_summary(run_directory)
                result["report"] = report
                runtime.finish_job(job_id, result)
            except Exception as error:
                runtime.fail_job(job_id, error)

        background_tasks.add_task(worker)
        return {"job_id": job_id}

    @application.post("/api/async/train")
    def start_async_training(
        payload: AsyncTrainingPayload, background_tasks: BackgroundTasks
    ) -> dict[str, str]:
        runtime.ensure_root()
        seed = _seed(payload.seed)
        job_id = runtime.new_job("async_training", seed, payload.max_ticks)
        run_directory = runtime.root / "async_runs" / job_id
        config = build_async_training_config(payload, seed)

        def worker() -> None:
            try:
                report = run_async_experiment(
                    run_directory,
                    config,
                    progress=lambda event: runtime.update_job(
                        job_id, {"phase": "asynchronous", **event}
                    ),
                )
                result = runtime.async_run_summary(run_directory)
                result["report"] = report
                runtime.finish_job(job_id, result)
            except Exception as error:
                runtime.fail_job(job_id, error)

        background_tasks.add_task(worker)
        return {"job_id": job_id}

    @application.post("/api/embodied/food-web/train")
    def start_embodied_food_web_training(
        payload: EmbodiedFoodWebTrainingPayload, background_tasks: BackgroundTasks
    ) -> dict[str, str]:
        """Co-evolve prey and predator rules in matched random food-web episodes."""
        runtime.ensure_root()
        seed = _seed(payload.seed)
        initial_genome: tuple[float, ...] | None = None
        initial_prey_genome: tuple[float, ...] | None = None
        initial_predator_genome: tuple[float, ...] | None = None
        initialization: dict[str, str] = {"kind": "fresh"}
        architecture = RuleArchitecture(state_width=payload.state_width)
        edge_architecture = EdgeArchitecture(node_state_width=payload.state_width)
        if payload.model_id:
            document = runtime.load_trained_rule(payload.model_id)
            if document.get("target") != "joint" or not document.get("edge_architecture"):
                raise HTTPException(400, "selected basic model must contain both node and edge rules")
            architecture = RuleArchitecture(**document["architecture"])
            edge_architecture = EdgeArchitecture(**document["edge_architecture"])
            initial_genome = tuple(float(value) for value in document["genome"])
            initialization = {"kind": "basic_model", "model_id": payload.model_id}
        elif payload.continue_run_id:
            try:
                previous = runtime.load_embodied_report(payload.continue_run_id)
                architecture = RuleArchitecture(**previous["architecture"])
                edge_architecture = EdgeArchitecture(**previous["edge_architecture"])
                initial_prey_genome = tuple(float(value) for value in previous["prey_best_genome"])
                initial_predator_genome = tuple(float(value) for value in previous["predator_best_genome"])
            except (KeyError, TypeError, ValueError) as error:
                raise HTTPException(400, "selected embodied run cannot be continued") from error
            initialization = {"kind": "embodied_run", "run_id": payload.continue_run_id}
        if architecture.state_width < 2:
            raise HTTPException(400, "the embodied ray/body interface requires a rule with at least 2 node-state channels")
        if architecture.state_width != payload.state_width:
            raise HTTPException(
                400,
                f"selected rule uses {architecture.state_width} node-state channels; set Node state channels to {architecture.state_width}",
            )
        adapter = FoodWebAgentAdapter(vision_pixels=9, body_inputs=payload.body_inputs)
        boundary_nodes = adapter.input_count + adapter.action_count
        total_nodes = payload.hidden_nodes + boundary_nodes
        network = EmbodiedNetworkConfig(
            nodes=total_nodes, mean_degree=payload.mean_degree,
            state_width=architecture.state_width, initial_state_scale=payload.initial_state_scale,
            dt=payload.network_dt, max_delta=payload.max_delta,
            edge_step_scale=payload.edge_step_scale,
            vision_pixels=9,
            body_inputs=payload.body_inputs,
            execution_backend=payload.execution_backend, device=payload.device,
        )
        task = EmbodiedFoodWebTaskConfig(
            network=network,
            environment=FoodWebConfig(
                prey_initial_energy=9.0 * payload.initial_energy_scale,
                predator_initial_energy=14.0 * payload.initial_energy_scale,
                initial_plants=min(24, payload.max_food),
                max_plants=payload.max_food,
                plant_regrowth=payload.food_growth_rate,
                max_speed=payload.max_speed,
                max_turn=payload.max_turn,
                plant_cluster_count=payload.plant_cluster_count,
                plant_cluster_radius=payload.plant_cluster_radius,
                respawn_on_death=payload.training_mode != "batch",
            ),
            prey_count=payload.prey_count, predator_count=payload.predator_count,
            max_steps=payload.batch_episode_steps if payload.training_mode == "batch" else 1,
            trials=1, seed=seed,
        )
        evaluator = FoodWebCoevolutionEvaluator(architecture, edge_architecture, task)
        evolution = EmbodiedRuleEvolutionConfig(
            generations=payload.batch_generations if payload.training_mode == "batch" else 1,
            population_size=payload.population_size, seed=seed,
            initial_genome=initial_genome, algorithm=payload.algorithm,
        )
        if payload.training_mode == "batch":
            runner = BatchFoodWebCoevolutionRunner(
                evaluator, evolution,
                BatchFoodWebConfig(
                    generations=payload.batch_generations, episode_steps=payload.batch_episode_steps,
                    trials=payload.batch_trials, validation_trials=payload.batch_validation_trials,
                    test_trials=payload.batch_test_trials,
                    opponent_pool_size=payload.batch_opponents,
                    seed=seed, initial_genome=initial_genome,
                    initial_prey_genome=initial_prey_genome, initial_predator_genome=initial_predator_genome,
                    workers=payload.workers,
                ),
            )
            job_total = payload.batch_generations
        else:
            runner = ContinuousFoodWebCoevolutionRunner(
                evaluator, evolution,
                ContinuousFoodWebConfig(
                    ticks=payload.ticks, seed=seed, initial_genome=initial_genome,
                    initial_prey_genome=initial_prey_genome, initial_predator_genome=initial_predator_genome,
                ),
            )
            job_total = payload.ticks
        job_id = runtime.new_job("embodied_food_web", seed, job_total)
        task_config = {
            "training_mode": payload.training_mode, "algorithm": payload.algorithm,
            "objective": "restricted_mean_lifetime" if payload.training_mode == "batch" else "completed_lifetime",
            "objective_units": "ticks", "reward_shaping": False,
            "seed": seed, "population_size": payload.population_size,
            "initial_sigma": evolution.initial_sigma,
            "execution_backend": payload.execution_backend, "device": payload.device,
            "workers": payload.workers,
            "body_inputs": list(payload.body_inputs),
            "embodied_interface": "ray_image_v3_sparse_multichannel_v1",
            "network": asdict(network), "environment": asdict(task.environment),
            "prey_count": task.prey_count, "predator_count": task.predator_count,
            "batch_generations": payload.batch_generations, "batch_episode_steps": payload.batch_episode_steps,
            "batch_trials": payload.batch_trials, "batch_validation_trials": payload.batch_validation_trials,
            "batch_test_trials": payload.batch_test_trials, "batch_opponents": payload.batch_opponents,
            "enforce_survival_pressure": payload.enforce_survival_pressure,
            "diagnostics": {
                "boundary_nodes": boundary_nodes,
                "body_inputs": list(payload.body_inputs),
                "hidden_nodes": payload.hidden_nodes,
                "total_nodes": total_nodes,
                "no_food_lifetime_steps": 20.0 * payload.initial_energy_scale,
                "survival_horizon_multiple": (
                    payload.batch_episode_steps / max(20.0 * payload.initial_energy_scale, 1e-12)
                    if payload.training_mode == "batch" else None
                ),
                "survival_pressure_active": (
                    payload.training_mode != "batch"
                    or payload.batch_episode_steps >= 60.0 * payload.initial_energy_scale
                ),
                "population_sustainable_from_regrowth": (
                    (payload.food_growth_rate * task.environment.plant_energy if payload.max_food > 0 else 0.0)
                    >= payload.prey_count * task.environment.prey_metabolism
                ),
                "selection_objective": "first_life_restricted_mean_lifetime",
                "common_validation_bank_for_model_selection": True,
                "final_test_touched_only_after_selection": payload.training_mode == "batch",
                "causal_baselines": ["zero_rule", "vision_masked"] if payload.training_mode == "batch" else [],
            },
        }

        def checkpoint(event: dict[str, object]) -> None:
            prey, predator = dict(event["prey"]), dict(event["predator"])
            if payload.training_mode == "batch":
                progress_fields = {"generations": payload.batch_generations, "checkpoint_generation": event["generation"]}
                task_name = "batch_food_web_coevolution_checkpoint"
            else:
                progress_fields = {"ticks": payload.ticks, "checkpoint_tick": event["tick"]}
                task_name = "continuous_food_web_coevolution_checkpoint"
            snapshot = {
                "task": task_name, "training_mode": payload.training_mode, "algorithm": payload.algorithm,
                **progress_fields, "prey": prey, "predator": predator,
                "prey_best_genome": prey["best_genome"], "predator_best_genome": predator["best_genome"],
                "architecture": asdict(architecture), "edge_architecture": asdict(edge_architecture),
                "task_config": task_config, "initialization": initialization,
            }
            checkpoint_url = runtime.write_embodied_checkpoint(job_id, snapshot)
            runtime.update_job(job_id, {"phase": "embodied_food_web", **event, "checkpoint_url": checkpoint_url})

        def worker() -> None:
            try:
                report = runner.run(
                    progress=checkpoint,
                    should_stop=lambda: runtime.job_termination_requested(job_id),
                )
                report["architecture"] = asdict(architecture)
                report["edge_architecture"] = asdict(edge_architecture)
                report["initialization"] = initialization
                report["task_config"] = task_config
                output = runtime.root / "embodied_runs" / job_id
                output.mkdir(parents=True, exist_ok=True)
                (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
                report["output_url"] = runtime.artifact_url(output / "report.json")
                runtime.finish_job(job_id, report)
            except EvolutionTerminated:
                runtime.terminate_job(job_id)
            except Exception as error:
                runtime.fail_job(job_id, error)

        background_tasks.add_task(worker)
        return {"job_id": job_id}

    @application.post("/api/embodied/jobs/{job_id}/terminate")
    def terminate_embodied_food_web_training(job_id: str) -> dict[str, object]:
        try:
            requested = runtime.request_job_termination(job_id, kind="embodied_food_web")
        except KeyError as error:
            raise HTTPException(404, "unknown embodied evolution job") from error
        if not requested:
            raise HTTPException(409, "embodied evolution job is no longer running")
        return {"job_id": job_id, "termination_requested": True}

    @application.get("/api/embodied/runs")
    def embodied_runs() -> dict[str, object]:
        return {"runs": runtime.available_embodied_runs()}

    @application.post("/api/embodied/sessions")
    def create_embodied_session(payload: EmbodiedDemoPayload) -> dict[str, object]:
        try:
            return runtime.create_embodied_session(payload)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    @application.post("/api/embodied/sessions/{session_id}/step")
    def step_embodied_session(session_id: str, payload: EmbodiedDemoStepPayload) -> dict[str, object]:
        try:
            return runtime.advance_embodied_session(session_id, payload.ticks)
        except KeyError as error:
            raise HTTPException(404, "embodied demonstration session is unavailable") from error

    @application.get("/api/embodied/sessions/{session_id}/individuals/{individual_id}")
    def embodied_individual(session_id: str, individual_id: str) -> dict[str, object]:
        try:
            return runtime.embodied_individual_snapshot(session_id, individual_id)
        except KeyError as error:
            raise HTTPException(404, "embodied demonstration individual is unavailable") from error

    @application.get("/api/async/latest")
    def latest_async() -> dict[str, object]:
        return runtime.latest_async_summary()

    @application.get("/api/async/replays/{run_id}/{candidate_id}/{replica}")
    def async_replay(run_id: str, candidate_id: int, replica: int) -> dict[str, object]:
        try:
            return runtime.async_replay_document(run_id, candidate_id, replica)
        except KeyError as error:
            raise HTTPException(404, "asynchronous run is unavailable") from error
        except (IndexError, ValueError) as error:
            raise HTTPException(404, str(error)) from error

    def start_evolution_job(
        kind: Literal["random_search", "search"],
        payload: EvolutionPayload,
        background_tasks: BackgroundTasks,
    ) -> dict[str, str]:
        runtime.ensure_root()
        seed = _seed(payload.seed)
        total = payload.population if kind == "search" else payload.samples
        job_id = runtime.new_job(kind, seed, total)

        def worker() -> None:
            try:
                architecture = RuleArchitecture()
                edge_architecture = EdgeArchitecture(node_state_width=architecture.state_width)
                if kind == "random_search":
                    report = random_search_smoke_test(
                        CandidateEvaluator(
                            architecture,
                            edge_architecture=edge_architecture,
                            target="joint",
                        ),
                        samples=payload.samples,
                        seed=seed,
                        on_sample=lambda event: runtime.update_job(
                            job_id, {"phase": "smoke", **event}
                        ),
                    )
                    runtime.finish_job(job_id, {"smoke_report": report.to_dict()})
                    return
                run_directory = runtime.root / "evolution_runs" / job_id
                runner = EvolutionRunner(
                    EvolutionConfig(
                        seed=seed,
                        generations=payload.generations,
                        population_size=payload.population,
                        smoke_samples=payload.samples,
                        architecture=architecture,
                        edge_architecture=edge_architecture,
                        target="joint",
                    )
                )
                report = runner.run(
                    run_directory,
                    progress=lambda event: runtime.update_job(job_id, event),
                )
                runtime.finish_job(
                    job_id,
                    {
                        "best_fitness": report["best"]["fitness"],
                        "validation_fitness": report["validation"]["fitness"],
                        "test_fitness": report["test"]["fitness"],
                        "output_url": runtime.artifact_url(
                            run_directory / "evolution_report.json"
                        ),
                        "best_genome_url": runtime.artifact_url(
                            run_directory / "best_genome.json"
                        ),
                        "analysis_url": runtime.artifact_url(
                            run_directory / "analysis" / "analysis.json"
                        ),
                        "trajectory_svg_url": runtime.artifact_url(
                            run_directory / "analysis" / "trajectory.svg"
                        ),
                        "recovery_svg_url": runtime.artifact_url(
                            run_directory / "analysis" / "recovery.svg"
                        ),
                        "replay_index_url": runtime.artifact_url(
                            run_directory / "replays" / "index.json"
                        ),
                    },
                )
            except Exception as error:
                runtime.fail_job(job_id, error)

        background_tasks.add_task(worker)
        return {"job_id": job_id}

    @application.post("/api/evolution/random-search")
    def random_search(
        payload: EvolutionPayload, background_tasks: BackgroundTasks
    ) -> dict[str, str]:
        return start_evolution_job("random_search", payload, background_tasks)

    @application.post("/api/evolution/search")
    def evolution_search(
        payload: EvolutionPayload, background_tasks: BackgroundTasks
    ) -> dict[str, str]:
        return start_evolution_job("search", payload, background_tasks)

    @application.get("/api/live/models")
    def live_models() -> dict[str, object]:
        return {
            "models": runtime.available_live_models(),
            "latest_survival": runtime.latest_async_summary(),
        }

    @application.post("/api/live/sessions")
    def create_live(payload: LiveSessionPayload) -> dict[str, object]:
        try:
            return runtime.create_live_session(payload)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    @application.post("/api/live/sessions/{session_id}/step")
    def step_live(session_id: str, payload: LiveStepPayload) -> dict[str, object]:
        try:
            return runtime.advance_live_session(session_id, payload.steps)
        except KeyError as error:
            raise HTTPException(404, "live session is unavailable") from error

    @application.get("/dashboard", include_in_schema=False)
    @application.get("/dashboard/", include_in_schema=False)
    def old_dashboard_url() -> RedirectResponse:
        return RedirectResponse("/", status_code=307)

    application.mount(
        "/artifacts",
        StaticFiles(directory=str(runtime.root), check_dir=False),
        name="artifacts",
    )
    web = Path(__file__).with_name("web")
    application.mount("/", StaticFiles(directory=str(web), html=True), name="frontend")
    return application


app = create_app()
