"""FastAPI route composition for the state-network research workspace."""

from __future__ import annotations

import json
import secrets
from dataclasses import asdict
from math import sqrt
from pathlib import Path
from random import Random
from statistics import fmean, median, pstdev
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from .application.configuration import build_async_training_config
from .application.models import (
    AsyncDiagnosticPayload,
    AsyncTrainingPayload,
    EmbodiedFoodWebTrainingPayload,
    EmbodiedRandomGraphDiagnosticPayload,
    EmbodiedRunComparisonPayload,
    EmbodiedDemoPayload,
    EmbodiedDemoStepPayload,
    EvolutionPayload,
    LiveSessionPayload,
    LiveStepPayload,
)
from .application.runtime import ApplicationRuntime
from .checkpoint_diagnostics import evaluate_prey_checkpoints
from .evolution.asynchronous import run_async_experiment, run_diagnostic_experiment
from .evolution.candidate import EdgeArchitecture, RuleArchitecture, _forward
from .evolution.genome import GenomeCodec
from .dashboard import dashboard_document
from .evolution.evaluation import CandidateEvaluator
from .evolution import EvolutionConfig, EvolutionRunner, random_search_smoke_test
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


def _distribution_summary(values: tuple[float, ...]) -> dict[str, float]:
    """Return stable JSON-friendly descriptive statistics for diagnostic trials."""
    ordered = sorted(values)
    if not ordered:
        return {}

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    standard_deviation = pstdev(ordered)
    return {
        "mean": fmean(ordered), "standard_deviation": standard_deviation,
        "variance": standard_deviation ** 2, "minimum": ordered[0], "maximum": ordered[-1],
        "median": median(ordered), "p10": percentile(.10), "p25": percentile(.25),
        "p75": percentile(.75), "p90": percentile(.90),
    }


def _vector_comparison(left: tuple[float, ...], right: tuple[float, ...]) -> dict[str, float | int | bool]:
    """Compare vectors without silently aligning incompatible architectures."""
    if len(left) != len(right):
        return {"compatible": False, "left_dimension": len(left), "right_dimension": len(right)}
    l2 = sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))
    left_norm = sqrt(sum(value ** 2 for value in left))
    right_norm = sqrt(sum(value ** 2 for value in right))
    return {
        "compatible": True, "left_dimension": len(left), "right_dimension": len(right),
        "l2_distance": l2, "rms_distance": l2 / sqrt(max(1, len(left))),
        "cosine_similarity": sum(a * b for a, b in zip(left, right, strict=True)) / max(left_norm * right_norm, 1e-12),
        "left_rms": left_norm / sqrt(max(1, len(left))), "right_rms": right_norm / sqrt(max(1, len(right))),
    }


def _raw_output_summary(values: tuple[float, ...]) -> dict[str, float]:
    if not values:
        return {}
    absolute = tuple(abs(value) for value in values)
    ordered = sorted(absolute)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "mean": fmean(values), "standard_deviation": pstdev(values),
        "rms": sqrt(fmean(value ** 2 for value in values)), "minimum": min(values), "maximum": max(values),
        "abs_p50": percentile(.50), "abs_p90": percentile(.90), "abs_p99": percentile(.99),
        "abs_gt_1_fraction": sum(value > 1 for value in absolute) / len(absolute),
        "abs_gt_2_fraction": sum(value > 2 for value in absolute) / len(absolute),
        "abs_gt_3_fraction": sum(value > 3 for value in absolute) / len(absolute),
    }


def _synthetic_rule_outputs(
    genome: tuple[float, ...], architecture: RuleArchitecture, edge_architecture: EdgeArchitecture,
    *, state_limit: float, probe_count: int, seed: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Evaluate raw MLP outputs on a common bounded synthetic input distribution.

    These probes intentionally bypass the downstream ``tanh`` and update
    scaling.  They are a fast saturation signal, not a replacement for an
    episode trace drawn from the learned policy's real state distribution.
    """
    node_rule, edge_rule = GenomeCodec(architecture, edge_architecture, "joint").decode_groups(genome)
    assert node_rule is not None and edge_rule is not None
    random = Random(seed)
    node_values: list[float] = []
    edge_values: list[float] = []
    for _ in range(probe_count):
        node_features = tuple(random.uniform(-state_limit, state_limit) for _ in range(2 * architecture.state_width))
        node_values.extend(_forward(node_features, node_rule._layers, architecture.activation))
        edge_features = (
            tuple(random.uniform(-1.0, 1.0) for _ in range(edge_architecture.latent_width))
            + tuple(random.uniform(-state_limit, state_limit) for _ in range(3 * architecture.state_width))
        )
        edge_values.extend(_forward(edge_features, edge_rule._layers, edge_architecture.activation))
    return tuple(node_values), tuple(edge_values)


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
        architecture = RuleArchitecture(
            state_width=payload.state_width, hidden_layers=payload.node_hidden_layers,
            activation=payload.node_activation,
        )
        edge_architecture = EdgeArchitecture(
            node_state_width=payload.state_width, latent_width=payload.edge_latent_width,
            hidden_layers=payload.edge_hidden_layers, activation=payload.edge_activation,
        )
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
            rule_output_scale=payload.rule_output_scale,
            vision_pixels=9,
            body_inputs=payload.body_inputs,
            allow_input_output_connections=payload.allow_input_output_connections,
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
            mutation_sigma=payload.mutation_sigma, elite_fraction=payload.elite_fraction,
            immigrant_fraction=payload.immigrant_fraction, immigrant_sigma=payload.immigrant_sigma,
            immigrant_mode=payload.immigrant_mode, max_genome_norm=payload.max_genome_norm,
            max_parameter_magnitude=payload.max_parameter_magnitude,
            local_mutation_sigma=payload.local_mutation_sigma,
            local_offspring_fraction=payload.local_offspring_fraction,
            regional_fraction=payload.regional_fraction, regional_scale=payload.regional_scale,
            regional_min_std=payload.regional_min_std, global_fraction=payload.global_fraction,
            global_parameter_range=payload.global_parameter_range,
            global_viability_filter=payload.global_viability_filter,
            global_max_sampling_attempts=payload.global_max_sampling_attempts,
        )
        if payload.training_mode == "batch":
            runner = BatchFoodWebCoevolutionRunner(
                evaluator, evolution,
                BatchFoodWebConfig(
                    population_mode=payload.batch_population_mode,
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
            "training_mode": payload.training_mode, "batch_population_mode": payload.batch_population_mode,
            "algorithm": payload.algorithm,
            "objective": "restricted_mean_lifetime" if payload.training_mode == "batch" else "completed_lifetime",
            "objective_units": "ticks", "reward_shaping": False,
            "seed": seed, "population_size": payload.population_size,
            "initial_sigma": evolution.initial_sigma,
            "mutation_sigma": evolution.mutation_sigma or evolution.initial_sigma,
            "elite_fraction": evolution.elite_fraction, "immigrant_fraction": evolution.immigrant_fraction,
            "immigrant_sigma": evolution.immigrant_sigma or max(.05, evolution.initial_sigma * 3.0),
            "immigrant_mode": evolution.immigrant_mode, "max_genome_norm": evolution.max_genome_norm,
            "max_parameter_magnitude": evolution.max_parameter_magnitude,
            "local_mutation_sigma": evolution.local_mutation_sigma or evolution.mutation_sigma or evolution.initial_sigma,
            "local_offspring_fraction": evolution.local_offspring_fraction,
            "regional_fraction": evolution.regional_fraction, "regional_scale": evolution.regional_scale,
            "regional_min_std": evolution.regional_min_std, "global_fraction": evolution.global_fraction,
            "global_parameter_range": evolution.global_parameter_range,
            "global_viability_filter": evolution.global_viability_filter,
            "global_max_sampling_attempts": evolution.global_max_sampling_attempts,
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

    @application.post("/api/embodied/runs/{run_id}/diagnostics/random-graphs")
    def diagnose_embodied_random_graphs(
        run_id: str, payload: EmbodiedRandomGraphDiagnosticPayload, background_tasks: BackgroundTasks,
    ) -> dict[str, str]:
        """Evaluate saved prey/predator rules on matched fresh graph/state samples.

        This is strictly post-training: it does not mutate a checkpoint,
        consume evolution RNG, or affect the optimizer's stored results.
        """
        try:
            report = runtime.load_embodied_report(run_id)
            architecture = RuleArchitecture(**report["architecture"])
            edge_architecture = EdgeArchitecture(**report["edge_architecture"])
            task_data = report["task_config"]
            network = EmbodiedNetworkConfig(**task_data["network"])
            environment = FoodWebConfig(**task_data["environment"])
            prey_genome = tuple(float(value) for value in report["prey_best_genome"])
            predator_genome = tuple(float(value) for value in report["predator_best_genome"])
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(404, "selected embodied run is unavailable or incomplete") from error
        seed = _seed(payload.seed)
        job_id = runtime.new_job("embodied_random_graph_diagnostic", seed, payload.sample_count)

        def worker() -> None:
            try:
                task = EmbodiedFoodWebTaskConfig(
                    network=network, environment=environment,
                    prey_count=int(task_data["prey_count"]), predator_count=int(task_data["predator_count"]),
                    max_steps=int(report.get("episode_steps", task_data.get("max_steps", 200))),
                    trials=payload.sample_count, seed=seed,
                )
                evaluation = FoodWebCoevolutionEvaluator(architecture, edge_architecture, task).evaluate(prey_genome, predator_genome)
                result = {
                    "mode": "random_graph_random_state", "run_id": run_id, "seed": seed,
                    "sample_count": payload.sample_count,
                    "episode_seeds": [seed + 10_007 * index for index in range(payload.sample_count)],
                    "prey": {"fitness_values": list(evaluation.prey_trial_lifetimes), "fitness": _distribution_summary(evaluation.prey_trial_lifetimes), "behavior": dict(evaluation.prey_behavior)},
                    "predator": {"fitness_values": list(evaluation.predator_trial_lifetimes), "fitness": _distribution_summary(evaluation.predator_trial_lifetimes), "behavior": dict(evaluation.predator_behavior)},
                }
                runtime.finish_job(job_id, result)
            except Exception as error:
                runtime.fail_job(job_id, error)

        background_tasks.add_task(worker)
        return {"job_id": job_id}

    @application.post("/api/embodied/diagnostics/compare")
    def compare_embodied_runs(payload: EmbodiedRunComparisonPayload) -> dict[str, object]:
        """Compare two saved rule genomes and common raw-output probes."""
        try:
            left = runtime.load_embodied_report(payload.left_run_id)
            right = runtime.load_embodied_report(payload.right_run_id)
            left_architecture, right_architecture = RuleArchitecture(**left["architecture"]), RuleArchitecture(**right["architecture"])
            left_edge, right_edge = EdgeArchitecture(**left["edge_architecture"]), EdgeArchitecture(**right["edge_architecture"])
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(404, "one or both embodied runs are unavailable or incomplete") from error
        seed = _seed(payload.seed)
        left_limit = float(left.get("task_config", {}).get("network", {}).get("max_abs_state", 4.0))
        right_limit = float(right.get("task_config", {}).get("network", {}).get("max_abs_state", 4.0))
        probe_limit = max(left_limit, right_limit)

        def compare_species(species: Literal["prey", "predator"]) -> dict[str, object]:
            try:
                left_genome = tuple(float(value) for value in left[f"{species}_best_genome"])
                right_genome = tuple(float(value) for value in right[f"{species}_best_genome"])
                left_node, left_edge_group = GenomeCodec(left_architecture, left_edge, "joint").split(left_genome)
                right_node, right_edge_group = GenomeCodec(right_architecture, right_edge, "joint").split(right_genome)
                left_node_raw, left_edge_raw = _synthetic_rule_outputs(left_genome, left_architecture, left_edge, state_limit=probe_limit, probe_count=payload.probe_count, seed=seed)
                right_node_raw, right_edge_raw = _synthetic_rule_outputs(right_genome, right_architecture, right_edge, state_limit=probe_limit, probe_count=payload.probe_count, seed=seed)
            except (KeyError, TypeError, ValueError) as error:
                raise HTTPException(422, f"{species} genome is incompatible with its saved architecture") from error
            return {
                "joint_genome": _vector_comparison(left_genome, right_genome),
                "node_genome": _vector_comparison(left_node, right_node),
                "edge_genome": _vector_comparison(left_edge_group, right_edge_group),
                "node_rule_raw_output": {
                    "left": _raw_output_summary(left_node_raw), "right": _raw_output_summary(right_node_raw),
                    "common_probe_response": _vector_comparison(left_node_raw, right_node_raw),
                },
                "edge_rule_raw_output": {
                    "left": _raw_output_summary(left_edge_raw), "right": _raw_output_summary(right_edge_raw),
                    "common_probe_response": _vector_comparison(left_edge_raw, right_edge_raw),
                },
            }

        return {
            "left_run_id": payload.left_run_id, "right_run_id": payload.right_run_id,
            "seed": seed, "probe_count": payload.probe_count,
            "probe_protocol": {
                "kind": "synthetic_common_input", "node_state_range": [-probe_limit, probe_limit],
                "edge_state_range": [-1.0, 1.0], "saturation_warning": "more than 10% of raw outputs with abs(x) > 3",
                "note": "These are common bounded synthetic inputs before downstream tanh; they are not recorded episode activations.",
            },
            "prey": compare_species("prey"), "predator": compare_species("predator"),
        }

    @application.post("/api/embodied/diagnostics/checkpoints/evaluate")
    def evaluate_embodied_checkpoints(
        payload: EmbodiedRunComparisonPayload, background_tasks: BackgroundTasks,
    ) -> dict[str, str]:
        """Evaluate two prey checkpoints on matched random networks and worlds."""
        try:
            left = runtime.load_embodied_report(payload.left_run_id)
            right = runtime.load_embodied_report(payload.right_run_id)
            architecture = RuleArchitecture(**left["architecture"])
            edge_architecture = EdgeArchitecture(**left["edge_architecture"])
            right_architecture = RuleArchitecture(**right["architecture"])
            right_edge = EdgeArchitecture(**right["edge_architecture"])
            if (architecture.parameter_count != right_architecture.parameter_count or edge_architecture.parameter_count != right_edge.parameter_count or architecture.state_width != right_architecture.state_width):
                raise ValueError("saved architectures differ")
            task_data = left["task_config"]
            network = EmbodiedNetworkConfig(**task_data["network"])
            environment = FoodWebConfig(**task_data["environment"])
            left_genome = tuple(float(value) for value in left["prey_best_genome"])
            right_genome = tuple(float(value) for value in right["prey_best_genome"])
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(422, "checkpoints are unavailable or not architecture-compatible") from error
        seed = _seed(payload.seed)
        evaluation_seeds = tuple(seed + 10_007 * index for index in range(payload.evaluation_samples))
        job_id = runtime.new_job("embodied_checkpoint_diagnostic", seed, payload.evaluation_samples)

        def worker() -> None:
            try:
                report = evaluate_prey_checkpoints(
                    left_genome, right_genome, architecture, edge_architecture, network, environment,
                    prey_count=int(task_data["prey_count"]), predator_count=int(task_data["predator_count"]),
                    steps=int(left.get("episode_steps", task_data.get("max_steps", 200))),
                    seeds=evaluation_seeds, scales=payload.parameter_scales,
                )
                report.update({
                    "schema_version": 1, "checkpoint_a": {"run_id": payload.left_run_id, **report["checkpoint_a"]},
                    "checkpoint_b": {"run_id": payload.right_run_id, **report["checkpoint_b"]},
                    "seed": seed, "evaluation_seeds": list(evaluation_seeds),
                    "evaluation_config": {"network": asdict(network), "environment": asdict(environment), "steps": int(left.get("episode_steps", task_data.get("max_steps", 200))), "parameter_scales": list(payload.parameter_scales)},
                    "instrumentation": {"raw_outputs": "recorded before tanh on one deterministic representative prey network per matched episode", "effective_updates": "recorded after normal scaling/clipping on that same representative", "action_trajectories": "one deterministic representative prey trajectory per matched episode"},
                })
                output = runtime.root / "embodied_diagnostics"
                output.mkdir(parents=True, exist_ok=True)
                path = output / f"{job_id}.json"
                path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
                report["output_url"] = runtime.artifact_url(path)
                runtime.finish_job(job_id, report)
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
