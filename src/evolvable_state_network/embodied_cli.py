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
from .plot_data import embodied_plot_rows, write_plot_table
from .tasks import BatchFoodWebCoevolutionRunner, BatchFoodWebConfig, ContinuousFoodWebConfig, ContinuousFoodWebCoevolutionRunner, EmbodiedFoodWebTaskConfig, EmbodiedRuleEvolutionConfig, EvolutionTerminated, FoodWebCoevolutionEvaluator


DEFAULT_CONFIG = """# Settings have the same names and defaults as the Embodied UI form.
seed: 41
training_mode: batch
# shared_rule_cohort: one genome controls all same-species agents in a world.
# mixed_individual_population: each agent has its own genome; population_size
# then means worlds per generation, so total prey genomes are population_size * prey_count.
batch_population_mode: shared_rule_cohort
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
node_hidden_layers: [8]
node_activation: tanh
edge_hidden_layers: [12]
edge_activation: tanh
edge_latent_width: 3
mean_degree: 6.0
# Default topology: input -> hidden -> output; inputs cannot connect to each other.
allow_input_output_connections: false
# Direct sensor-to-action ablation example (replace the preceding value):
# allow_input_output_connections: true
initial_state_scale: 0.12
network_dt: 0.05
max_delta: 0.12
edge_step_scale: 0.06
rule_output_scale: 1.0
# Genetic-algorithm exploration controls. Local is the remainder after elite,
# regional, and global fractions; the values below match the Embodied UI.
elite_fraction: 0.10
local_mutation_sigma: 0.05
regional_fraction: 0.15
regional_scale: 1.0
regional_min_std: 0.02
global_fraction: 0.15
global_parameter_range: 1.0
global_viability_filter: true
global_max_sampling_attempts: 20
# Legacy comparison only (set regional_fraction and global_fraction to 0):
# immigrant_fraction: 0.25
# immigrant_mode: zero  # zero or population Gaussian
# mutation_sigma: 0.1
# max_genome_norm: 20.0
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


def _load_network_config(path: Path) -> dict[str, Any]:
    """Translate a focused node/edge architecture YAML into payload fields."""
    document = _load_config(path)
    allowed = {"node", "edge"}
    unknown = set(document) - allowed
    if unknown:
        raise ValueError(f"unknown network-config section(s): {', '.join(sorted(unknown))}")
    node = document.get("node", {})
    edge = document.get("edge", {})
    if not isinstance(node, dict) or not isinstance(edge, dict):
        raise ValueError("network-config node and edge entries must be mappings")
    allowed_node = {"hidden_layers", "activation"}
    allowed_edge = {"hidden_layers", "activation", "latent_width"}
    if set(node) - allowed_node or set(edge) - allowed_edge:
        raise ValueError("network-config supports node/edge hidden_layers, activation, and edge latent_width")
    translated: dict[str, Any] = {}
    if "hidden_layers" in node:
        translated["node_hidden_layers"] = node["hidden_layers"]
    if "activation" in node:
        translated["node_activation"] = node["activation"]
    if "hidden_layers" in edge:
        translated["edge_hidden_layers"] = edge["hidden_layers"]
    if "activation" in edge:
        translated["edge_activation"] = edge["activation"]
    if "latent_width" in edge:
        translated["edge_latent_width"] = edge["latent_width"]
    return translated


def _build_runner(payload: EmbodiedFoodWebTrainingPayload, runtime: ApplicationRuntime, seed: int):
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
    network = EmbodiedNetworkConfig(nodes=payload.hidden_nodes + boundary_nodes, mean_degree=payload.mean_degree, state_width=architecture.state_width, initial_state_scale=payload.initial_state_scale, dt=payload.network_dt, max_delta=payload.max_delta, edge_step_scale=payload.edge_step_scale, rule_output_scale=payload.rule_output_scale, vision_pixels=9, body_inputs=payload.body_inputs, allow_input_output_connections=payload.allow_input_output_connections, execution_backend=payload.execution_backend, device=payload.device)
    task = EmbodiedFoodWebTaskConfig(network=network, environment=FoodWebConfig(prey_initial_energy=9.0 * payload.initial_energy_scale, predator_initial_energy=14.0 * payload.initial_energy_scale, initial_plants=min(24, payload.max_food), max_plants=payload.max_food, plant_regrowth=payload.food_growth_rate, max_speed=payload.max_speed, max_turn=payload.max_turn, plant_cluster_count=payload.plant_cluster_count, plant_cluster_radius=payload.plant_cluster_radius, respawn_on_death=payload.training_mode != "batch"), prey_count=payload.prey_count, predator_count=payload.predator_count, max_steps=payload.batch_episode_steps if payload.training_mode == "batch" else 1, trials=1, seed=seed)
    evaluator = FoodWebCoevolutionEvaluator(architecture, edge_architecture, task)
    evolution = EmbodiedRuleEvolutionConfig(generations=payload.batch_generations if payload.training_mode == "batch" else 1, population_size=payload.population_size, seed=seed, initial_genome=initial_genome, algorithm=payload.algorithm, mutation_sigma=payload.mutation_sigma, elite_fraction=payload.elite_fraction, immigrant_fraction=payload.immigrant_fraction, immigrant_sigma=payload.immigrant_sigma, immigrant_mode=payload.immigrant_mode, local_mutation_sigma=payload.local_mutation_sigma, local_offspring_fraction=payload.local_offspring_fraction, regional_fraction=payload.regional_fraction, regional_scale=payload.regional_scale, regional_min_std=payload.regional_min_std, global_fraction=payload.global_fraction, global_parameter_range=payload.global_parameter_range, global_viability_filter=payload.global_viability_filter, global_max_sampling_attempts=payload.global_max_sampling_attempts, max_genome_norm=payload.max_genome_norm, max_parameter_magnitude=payload.max_parameter_magnitude)
    if payload.training_mode == "batch":
        runner = BatchFoodWebCoevolutionRunner(evaluator, evolution, BatchFoodWebConfig(population_mode=payload.batch_population_mode, generations=payload.batch_generations, episode_steps=payload.batch_episode_steps, trials=payload.batch_trials, validation_trials=payload.batch_validation_trials, test_trials=payload.batch_test_trials, opponent_pool_size=payload.batch_opponents, seed=seed, initial_genome=initial_genome, initial_prey_genome=initial_prey_genome, initial_predator_genome=initial_predator_genome, workers=payload.workers))
    else:
        runner = ContinuousFoodWebCoevolutionRunner(evaluator, evolution, ContinuousFoodWebConfig(ticks=payload.ticks, seed=seed, initial_genome=initial_genome, initial_prey_genome=initial_prey_genome, initial_predator_genome=initial_predator_genome))
    return runner, architecture, edge_architecture, network, task, evolution, initialization, boundary_nodes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run embodied food-web evolution from a YAML configuration.")
    parser.add_argument("--config", type=Path, help="YAML file containing UI-equivalent training settings")
    parser.add_argument("--network-config", type=Path, help="optional YAML containing only node/edge MLP architecture")
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
        config = _load_config(args.config)
        if args.network_config is not None:
            config.update(_load_network_config(args.network_config))
        payload = EmbodiedFoodWebTrainingPayload(**config)
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
    task_config = {"training_mode": payload.training_mode, "batch_population_mode": payload.batch_population_mode, "algorithm": payload.algorithm, "objective": "restricted_mean_lifetime" if payload.training_mode == "batch" else "completed_lifetime", "objective_units": "ticks", "reward_shaping": False, "seed": seed, "population_size": payload.population_size, "initial_sigma": evolution.initial_sigma, "mutation_sigma": evolution.mutation_sigma or evolution.initial_sigma, "elite_fraction": evolution.elite_fraction, "immigrant_fraction": evolution.immigrant_fraction, "immigrant_sigma": evolution.immigrant_sigma or max(.05, evolution.initial_sigma * 3.0), "immigrant_mode": evolution.immigrant_mode, "local_mutation_sigma": evolution.local_mutation_sigma or evolution.mutation_sigma or evolution.initial_sigma, "local_offspring_fraction": evolution.local_offspring_fraction, "regional_fraction": evolution.regional_fraction, "regional_scale": evolution.regional_scale, "regional_min_std": evolution.regional_min_std, "global_fraction": evolution.global_fraction, "global_parameter_range": evolution.global_parameter_range, "global_viability_filter": evolution.global_viability_filter, "global_max_sampling_attempts": evolution.global_max_sampling_attempts, "max_genome_norm": evolution.max_genome_norm, "max_parameter_magnitude": evolution.max_parameter_magnitude, "execution_backend": payload.execution_backend, "device": payload.device, "workers": payload.workers, "body_inputs": list(payload.body_inputs), "embodied_interface": "ray_image_v3_sparse_multichannel_v1", "network": asdict(network), "environment": asdict(task.environment), "prey_count": task.prey_count, "predator_count": task.predator_count, "batch_generations": payload.batch_generations, "batch_episode_steps": payload.batch_episode_steps, "batch_trials": payload.batch_trials, "batch_validation_trials": payload.batch_validation_trials, "batch_test_trials": payload.batch_test_trials, "batch_opponents": payload.batch_opponents, "enforce_survival_pressure": payload.enforce_survival_pressure, "diagnostics": {"boundary_nodes": boundary_nodes, "body_inputs": list(payload.body_inputs), "hidden_nodes": payload.hidden_nodes, "total_nodes": network.nodes, "selection_objective": "first_life_restricted_mean_lifetime"}}

    def checkpoint(event: dict[str, object]) -> None:
        prey, predator = dict(event["prey"]), dict(event["predator"])
        progress = ({"generations": payload.batch_generations, "checkpoint_generation": event["generation"]} if payload.training_mode == "batch" else {"ticks": payload.ticks, "checkpoint_tick": event["tick"]})
        snapshot = {"task": "batch_food_web_coevolution_checkpoint" if payload.training_mode == "batch" else "continuous_food_web_coevolution_checkpoint", "training_mode": payload.training_mode, "algorithm": payload.algorithm, **progress, "prey": prey, "predator": predator, "prey_best_genome": prey["best_genome"], "predator_best_genome": predator["best_genome"], "architecture": asdict(architecture), "edge_architecture": asdict(edge_architecture), "task_config": task_config, "initialization": initialization}
        runtime.write_embodied_checkpoint(run_id, snapshot)
        print(f"Checkpoint saved: {root / 'embodied_runs' / run_id / 'checkpoint.json'}", flush=True)

    output = root / "embodied_runs" / run_id
    progress_events: list[dict[str, object]] = []
    print(f"Starting embodied run {run_id} (seed {seed}); results: {output}", flush=True)
    try:
        def record_progress(event: dict[str, object]) -> None:
            progress_events.append(event)
            checkpoint(event)
            rows = embodied_plot_rows({"training_mode": payload.training_mode}, progress_events)
            write_plot_table(
                output / "training_curves.txt", rows,
                description="Embodied training curves (one row per generation or simulation tick).",
            )

        report = runner.run(progress=record_progress, should_stop=stop_requested.is_set)
    except EvolutionTerminated:
        print(f"Run stopped safely; latest checkpoint: {output / 'checkpoint.json'}", flush=True)
        return 143
    report.update({"architecture": asdict(architecture), "edge_architecture": asdict(edge_architecture), "initialization": initialization, "task_config": task_config})
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    curves_path = output / "training_curves.txt"
    write_plot_table(
        curves_path, embodied_plot_rows(report, progress_events),
        description="Embodied training curves (one row per generation or simulation tick).",
    )
    print(f"Completed. Wrote {report_path}", flush=True)
    print(f"Completed. Wrote {curves_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
