"""Headless YAML-configured runner for embodied food-web evolution."""

from __future__ import annotations

import argparse
import json
import secrets
import signal
from dataclasses import asdict
from pathlib import Path
from threading import Event
from typing import Any, Sequence
from uuid import uuid4

import yaml
from pydantic import ValidationError

from .application.models import EmbodiedFoodWebTrainingPayload
from .application.runtime import ApplicationRuntime
from .embodied import EmbodiedNetworkConfig, FoodWebAgentAdapter
from .environments import FoodWebConfig
from .evolution.candidate import EdgeArchitecture, RuleArchitecture
from .tasks import BatchFoodWebCoevolutionRunner, BatchFoodWebConfig, ContinuousFoodWebConfig, ContinuousFoodWebCoevolutionRunner, EmbodiedFoodWebTaskConfig, EmbodiedRuleEvolutionConfig, EvolutionTerminated, FoodWebCoevolutionEvaluator


DEFAULT_CONFIG = """# Settings have the same names and defaults as the Embodied UI form.
seed: 41
training_mode: batch
algorithm: cma_es
execution_backend: torch
device: cpu
workers: 0                 # 0 selects up to eight CPU workers automatically
body_inputs: [hunger]      # choose from hunger, energy_change, ate, time_since_meal
population_size: 24
prey_count: 5
predator_count: 2
hidden_nodes: 31
state_width: 2
mean_degree: 6.0
initial_state_scale: 0.12
network_dt: 0.05
max_delta: 0.12
edge_step_scale: 0.06
initial_energy_scale: 1.0
max_food: 80
food_growth_rate: 24.0
max_speed: 20.0
max_turn: 6.283185307179586
plant_cluster_count: 4
plant_cluster_radius: 5.0
# Used only by continuous mode.
ticks: 600
# Used only by batch mode.
batch_generations: 48
batch_episode_steps: 256
batch_trials: 4
batch_validation_trials: 8
batch_test_trials: 16
batch_opponents: 2
enforce_survival_pressure: true
# Optional continuation sources, as in the UI. Choose at most one.
# model_id: survival:<run-id>:<elite-rank>
# continue_run_id: <embodied-run-id>
"""


def _load_config(path: Path) -> dict[str, Any]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML in {path}: {error}") from error
    if document is None:
        return {}
    if not isinstance(document, dict) or not all(isinstance(key, str) for key in document):
        raise ValueError("the YAML document must be a mapping with string keys")
    return document


def _build_runner(payload: EmbodiedFoodWebTrainingPayload, runtime: ApplicationRuntime, seed: int):
    initial_genome: tuple[float, ...] | None = None
    initial_prey_genome: tuple[float, ...] | None = None
    initial_predator_genome: tuple[float, ...] | None = None
    initialization: dict[str, str] = {"kind": "fresh"}
    architecture = RuleArchitecture(state_width=payload.state_width)
    edge_architecture = EdgeArchitecture(node_state_width=payload.state_width)
    if payload.model_id:
        document = runtime.load_trained_rule(payload.model_id)
        if document.get("target") != "joint" or not document.get("edge_architecture"):
            raise ValueError("selected basic model must contain both node and edge rules")
        architecture = RuleArchitecture(**document["architecture"])
        edge_architecture = EdgeArchitecture(**document["edge_architecture"])
        initial_genome = tuple(float(value) for value in document["genome"])
        initialization = {"kind": "basic_model", "model_id": payload.model_id}
    elif payload.continue_run_id:
        previous = runtime.load_embodied_report(payload.continue_run_id)
        architecture = RuleArchitecture(**previous["architecture"])
        edge_architecture = EdgeArchitecture(**previous["edge_architecture"])
        initial_prey_genome = tuple(float(value) for value in previous["prey_best_genome"])
        initial_predator_genome = tuple(float(value) for value in previous["predator_best_genome"])
        initialization = {"kind": "embodied_run", "run_id": payload.continue_run_id}
    if architecture.state_width != payload.state_width:
        raise ValueError(f"selected rule uses {architecture.state_width} node-state channels; set state_width to {architecture.state_width}")

    adapter = FoodWebAgentAdapter(vision_pixels=9, body_inputs=payload.body_inputs)
    boundary_nodes = adapter.input_count + adapter.action_count
    network = EmbodiedNetworkConfig(nodes=payload.hidden_nodes + boundary_nodes, mean_degree=payload.mean_degree, state_width=architecture.state_width, initial_state_scale=payload.initial_state_scale, dt=payload.network_dt, max_delta=payload.max_delta, edge_step_scale=payload.edge_step_scale, vision_pixels=9, body_inputs=payload.body_inputs, execution_backend=payload.execution_backend, device=payload.device)
    task = EmbodiedFoodWebTaskConfig(network=network, environment=FoodWebConfig(prey_initial_energy=9.0 * payload.initial_energy_scale, predator_initial_energy=14.0 * payload.initial_energy_scale, initial_plants=min(24, payload.max_food), max_plants=payload.max_food, plant_regrowth=payload.food_growth_rate, max_speed=payload.max_speed, max_turn=payload.max_turn, plant_cluster_count=payload.plant_cluster_count, plant_cluster_radius=payload.plant_cluster_radius, respawn_on_death=payload.training_mode != "batch"), prey_count=payload.prey_count, predator_count=payload.predator_count, max_steps=payload.batch_episode_steps if payload.training_mode == "batch" else 1, trials=1, seed=seed)
    evaluator = FoodWebCoevolutionEvaluator(architecture, edge_architecture, task)
    evolution = EmbodiedRuleEvolutionConfig(generations=payload.batch_generations if payload.training_mode == "batch" else 1, population_size=payload.population_size, seed=seed, initial_genome=initial_genome, algorithm=payload.algorithm)
    if payload.training_mode == "batch":
        runner = BatchFoodWebCoevolutionRunner(evaluator, evolution, BatchFoodWebConfig(generations=payload.batch_generations, episode_steps=payload.batch_episode_steps, trials=payload.batch_trials, validation_trials=payload.batch_validation_trials, test_trials=payload.batch_test_trials, opponent_pool_size=payload.batch_opponents, seed=seed, initial_genome=initial_genome, initial_prey_genome=initial_prey_genome, initial_predator_genome=initial_predator_genome, workers=payload.workers))
    else:
        runner = ContinuousFoodWebCoevolutionRunner(evaluator, evolution, ContinuousFoodWebConfig(ticks=payload.ticks, seed=seed, initial_genome=initial_genome, initial_prey_genome=initial_prey_genome, initial_predator_genome=initial_predator_genome))
    return runner, architecture, edge_architecture, network, task, evolution, initialization, boundary_nodes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run embodied food-web evolution from a YAML configuration.")
    parser.add_argument("--config", type=Path, help="YAML file containing UI-equivalent training settings")
    parser.add_argument("--data-dir", type=Path, default=Path(".outputs"), help="root directory for results")
    parser.add_argument("--run-id", help="optional unique run identifier (default: generated UUID)")
    parser.add_argument("--write-template", type=Path, metavar="PATH", help="write a commented YAML template and exit")
    args = parser.parse_args(argv)
    if args.write_template:
        args.write_template.parent.mkdir(parents=True, exist_ok=True)
        args.write_template.write_text(DEFAULT_CONFIG, encoding="utf-8")
        print(f"Wrote {args.write_template}")
        return 0
    if args.config is None:
        parser.error("--config is required unless --write-template is used")
    if args.run_id is not None and not args.run_id.isalnum():
        parser.error("--run-id must contain only letters and digits")
    try:
        payload = EmbodiedFoodWebTrainingPayload(**_load_config(args.config))
    except (ValueError, ValidationError) as error:
        parser.error(str(error))
    seed = secrets.randbelow(2**32) if payload.seed is None else payload.seed
    root = args.data_dir.expanduser().resolve()
    runtime = ApplicationRuntime(root)
    runtime.ensure_root()
    run_id = args.run_id or uuid4().hex
    stop_requested = Event()

    def request_stop(signum: int, _frame: object) -> None:
        if not stop_requested.is_set():
            print(f"Received signal {signum}; saving the next safe checkpoint before stopping.", flush=True)
        stop_requested.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        runner, architecture, edge_architecture, network, task, evolution, initialization, boundary_nodes = _build_runner(payload, runtime, seed)
    except (KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    task_config = {"training_mode": payload.training_mode, "algorithm": payload.algorithm, "objective": "restricted_mean_lifetime" if payload.training_mode == "batch" else "completed_lifetime", "objective_units": "ticks", "reward_shaping": False, "seed": seed, "population_size": payload.population_size, "initial_sigma": evolution.initial_sigma, "execution_backend": payload.execution_backend, "device": payload.device, "workers": payload.workers, "body_inputs": list(payload.body_inputs), "embodied_interface": "ray_image_v3_sparse_multichannel_v1", "network": asdict(network), "environment": asdict(task.environment), "prey_count": task.prey_count, "predator_count": task.predator_count, "batch_generations": payload.batch_generations, "batch_episode_steps": payload.batch_episode_steps, "batch_trials": payload.batch_trials, "batch_validation_trials": payload.batch_validation_trials, "batch_test_trials": payload.batch_test_trials, "batch_opponents": payload.batch_opponents, "enforce_survival_pressure": payload.enforce_survival_pressure, "diagnostics": {"boundary_nodes": boundary_nodes, "body_inputs": list(payload.body_inputs), "hidden_nodes": payload.hidden_nodes, "total_nodes": network.nodes, "selection_objective": "first_life_restricted_mean_lifetime"}}

    def checkpoint(event: dict[str, object]) -> None:
        prey, predator = dict(event["prey"]), dict(event["predator"])
        progress = ({"generations": payload.batch_generations, "checkpoint_generation": event["generation"]} if payload.training_mode == "batch" else {"ticks": payload.ticks, "checkpoint_tick": event["tick"]})
        snapshot = {"task": "batch_food_web_coevolution_checkpoint" if payload.training_mode == "batch" else "continuous_food_web_coevolution_checkpoint", "training_mode": payload.training_mode, "algorithm": payload.algorithm, **progress, "prey": prey, "predator": predator, "prey_best_genome": prey["best_genome"], "predator_best_genome": predator["best_genome"], "architecture": asdict(architecture), "edge_architecture": asdict(edge_architecture), "task_config": task_config, "initialization": initialization}
        runtime.write_embodied_checkpoint(run_id, snapshot)
        print(f"Checkpoint saved: {root / 'embodied_runs' / run_id / 'checkpoint.json'}", flush=True)

    output = root / "embodied_runs" / run_id
    print(f"Starting embodied run {run_id} (seed {seed}); results: {output}", flush=True)
    try:
        report = runner.run(progress=checkpoint, should_stop=stop_requested.is_set)
    except EvolutionTerminated:
        print(f"Run stopped safely; latest checkpoint: {output / 'checkpoint.json'}", flush=True)
        return 143
    report.update({"architecture": asdict(architecture), "edge_architecture": asdict(edge_architecture), "initialization": initialization, "task_config": task_config})
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Completed. Wrote {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
