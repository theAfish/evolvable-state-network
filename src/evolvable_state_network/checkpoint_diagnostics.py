"""Evaluation-only diagnostics for comparing saved embodied prey checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import sqrt, tanh
from random import Random
from statistics import fmean, pstdev
from typing import Mapping, Sequence

from .embodied import EmbodiedNetwork, EmbodiedNetworkConfig, FoodWebAgentAdapter
from .environments import (
    Action, Controller, ControllerBlueprint, EpisodeRunner, FoodWebConfig,
    FoodWebEnvironment, Observation, RandomControllerBlueprint, Species, make_reference_population,
)
from .evolution.candidate import EdgeArchitecture, MLPEdgeRule, MLPUpdateRule, RuleArchitecture, _forward
from .evolution.genome import GenomeCodec


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = fraction * (len(ordered) - 1)
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _raw_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    absolute = [abs(value) for value in values]
    return {
        "mean": fmean(values), "standard_deviation": pstdev(values),
        "rms": sqrt(fmean(value * value for value in values)),
        "p50": _percentile(absolute, .50), "p90": _percentile(absolute, .90),
        "p95": _percentile(absolute, .95), "p99": _percentile(absolute, .99),
        "abs_gt_1_fraction": sum(value > 1 for value in absolute) / len(absolute),
        "abs_gt_3_fraction": sum(value > 3 for value in absolute) / len(absolute),
        "abs_gt_5_fraction": sum(value > 5 for value in absolute) / len(absolute),
        "abs_gt_10_fraction": sum(value > 10 for value in absolute) / len(absolute),
    }


def _delta_summary(values: Sequence[float], *, limit: float | None) -> dict[str, float]:
    if not values:
        return {}
    absolute = [abs(value) for value in values]
    result = {
        "mean_absolute_delta": fmean(absolute), "rms_delta": sqrt(fmean(value * value for value in values)),
        "maximum_absolute_delta": max(absolute), "near_zero_fraction": sum(value <= 1e-8 for value in absolute) / len(absolute),
    }
    if limit is not None:
        result["near_limit_fraction"] = sum(value >= .99 * limit for value in absolute) / len(absolute)
        result["configured_limit"] = limit
    return result


def _fitness_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    return {"mean": fmean(values), "standard_deviation": pstdev(values), "minimum": min(values), "maximum": max(values)}


@dataclass(slots=True)
class _Collector:
    node_raw: list[float] = field(default_factory=list)
    edge_raw: list[float] = field(default_factory=list)
    node_delta: list[float] = field(default_factory=list)
    edge_delta: list[float] = field(default_factory=list)
    action_trace: list[tuple[float, float]] = field(default_factory=list)
    _next_controller: int = 0

    def controller_index(self) -> int:
        index = self._next_controller
        self._next_controller += 1
        return index


class _DiagnosticController(Controller):
    def __init__(
        self, node_rule: MLPUpdateRule, edge_rule: MLPEdgeRule, config: EmbodiedNetworkConfig,
        seed: int, collector: _Collector, index: int,
    ) -> None:
        self.node_rule, self.edge_rule, self.config, self.seed = node_rule, edge_rule, config, seed
        self.collector, self.index = collector, index
        self.network = EmbodiedNetwork(node_rule, edge_rule, FoodWebAgentAdapter(vision_pixels=config.vision_pixels, body_inputs=config.body_inputs), config, seed=seed)

    def begin_episode(self, *, seed: int | None = None) -> None:
        episode_seed = self.seed if seed is None else self.seed + seed
        self.network = EmbodiedNetwork(self.node_rule, self.edge_rule, FoodWebAgentAdapter(vision_pixels=self.config.vision_pixels, body_inputs=self.config.body_inputs), self.config, seed=episode_seed)

    def _capture_pre_step(self, observation: Observation) -> tuple[list[tuple[float, ...]], list[tuple[float, ...]]]:
        """Mirror the Python runtime's synchronous raw rule calculations."""
        values = self.network.adapter.encode_observation(observation)
        for node, value, channel in zip(self.network.interface.input_nodes, values, self.network.adapter.input_signal_channels, strict=True):
            vector = [0.0] * self.config.state_width
            vector[channel] = max(-1.0, min(1.0, float(value)))
            self.network.state.node[0][node] = tuple(vector)
        node = self.network.state.node[0]
        edge = self.network.state.edge[0]
        edge_raw: list[tuple[float, ...]] = []
        next_edge: list[tuple[float, ...]] = []
        for item, state in zip(self.network.graph.edges, edge, strict=True):
            message = self.edge_rule.message(state, node[item.source])
            raw = _forward(state + node[item.source] + node[item.target] + message, self.edge_rule._layers, self.edge_rule.architecture.activation)
            edge_raw.append(raw)
            next_edge.append(tuple(value + self.config.edge_step_scale * tanh(update * self.config.rule_output_scale) for value, update in zip(state, raw, strict=True)))
        aggregate = [[0.0] * self.config.state_width for _ in range(self.config.nodes)]
        counts = [0] * self.config.nodes
        for item, state in zip(self.network.graph.edges, next_edge, strict=True):
            message = self.edge_rule.message(state, node[item.source])
            aggregate[item.target] = [a + b for a, b in zip(aggregate[item.target], message, strict=True)]
            counts[item.target] += 1
        node_raw = [
            _forward(node[index] + tuple(value / max(1, counts[index]) for value in aggregate[index]), self.node_rule._layers, self.node_rule.architecture.activation)
            for index in range(self.config.nodes)
        ]
        return node_raw, edge_raw

    def act(self, observation: Observation, *, available_actions: Sequence[Action]) -> Action:
        if self.index != 0:
            return self.network.act(observation)
        node_raw, edge_raw = self._capture_pre_step(observation)
        before_node = [tuple(value) for value in self.network.state.node[0]]
        before_edge = [tuple(value) for value in self.network.state.edge[0]]
        action = self.network.act(observation)
        self.collector.node_raw.extend(value for row in node_raw for value in row)
        self.collector.edge_raw.extend(value for row in edge_raw for value in row)
        self.collector.node_delta.extend(
            after - before
            for index, (before_row, after_row) in enumerate(zip(before_node, self.network.state.node[0], strict=True))
            if index not in self.network.interface.input_nodes
            for before, after in zip(before_row, after_row, strict=True)
        )
        self.collector.edge_delta.extend(
            after - before for before_row, after_row in zip(before_edge, self.network.state.edge[0], strict=True)
            for before, after in zip(before_row, after_row, strict=True)
        )
        if self.index == 0:
            self.collector.action_trace.append((float(action.get("turn", 0.0)), float(action.get("speed", 0.0))))
        return action


class _DiagnosticBlueprint(ControllerBlueprint):
    def __init__(self, node_rule: MLPUpdateRule, edge_rule: MLPEdgeRule, config: EmbodiedNetworkConfig, seed: int, collector: _Collector) -> None:
        self.node_rule, self.edge_rule, self.config, self.seed, self.collector = node_rule, edge_rule, config, seed, collector

    def build(self, *, seed: int | None = None) -> Controller:
        index = self.collector.controller_index()
        # Only the first focal prey is instrumented through the reference
        # Python backend. The remaining same-genome prey retain the saved
        # backend, preserving the full fitness ecology without multiplying
        # capture overhead by the population size.
        config = replace(self.config, execution_backend="python", device="cpu") if index == 0 else self.config
        return _DiagnosticController(self.node_rule, self.edge_rule, config, self.seed + (seed or 0), self.collector, index)


def _one_checkpoint_episode(
    genome: Sequence[float], architecture: RuleArchitecture, edge_architecture: EdgeArchitecture,
    network: EmbodiedNetworkConfig, environment: FoodWebConfig, *, prey_count: int, predator_count: int,
    steps: int, seed: int,
) -> tuple[dict[str, float], _Collector]:
    node_rule, edge_rule = GenomeCodec(architecture, edge_architecture, "joint").decode_groups(genome)
    assert node_rule is not None and edge_rule is not None
    collector = _Collector()
    blueprint = _DiagnosticBlueprint(node_rule, edge_rule, network, seed * 1_000_003 + 0x51A7, collector)
    agents = make_reference_population(prey_count=prey_count, predator_count=predator_count, width=environment.width, height=environment.height, prey_initial_energy=environment.prey_initial_energy, predator_initial_energy=environment.predator_initial_energy, controller=RandomControllerBlueprint(), seed=seed)
    prey_ids = []
    for agent in agents:
        if agent.species is Species.PREY:
            agent.controller = blueprint
            prey_ids.append(agent.id)
    result = EpisodeRunner(FoodWebEnvironment(environment, seed=seed)).run(agents, max_steps=steps, seed=seed)
    rows = [result.behavior[agent_id] for agent_id in prey_ids]
    return {
        "fitness": fmean(float(row["restricted_lifetime"]) for row in rows),
        "mean_speed": fmean(float(row["mean_speed"]) for row in rows),
        "mean_turn": fmean(float(row["mean_turn"]) for row in rows),
        "mean_action_change": fmean(float(row["mean_action_change"]) for row in rows),
    }, collector


def evaluate_prey_checkpoints(
    left_genome: Sequence[float], right_genome: Sequence[float], architecture: RuleArchitecture,
    edge_architecture: EdgeArchitecture, network: EmbodiedNetworkConfig, environment: FoodWebConfig,
    *, prey_count: int, predator_count: int, steps: int, seeds: Sequence[int], scales: Sequence[float] = (1.0, .5, .1),
) -> dict[str, object]:
    """Run matching real-world checkpoint evaluations and aggregate diagnostics."""
    def evaluate(genome: Sequence[float]) -> dict[str, object]:
        rows, collectors = zip(*(_one_checkpoint_episode(genome, architecture, edge_architecture, network, environment, prey_count=prey_count, predator_count=predator_count, steps=steps, seed=seed) for seed in seeds), strict=True)
        return {
            "fitness_values": [row["fitness"] for row in rows], "fitness": _fitness_summary([row["fitness"] for row in rows]),
            "behavior": {key: fmean(row[key] for row in rows) for key in ("mean_speed", "mean_turn", "mean_action_change")},
            "node_rule_raw_output": _raw_summary([value for item in collectors for value in item.node_raw]),
            "edge_rule_raw_output": _raw_summary([value for item in collectors for value in item.edge_raw]),
            "node_update": _delta_summary([value for item in collectors for value in item.node_delta], limit=network.max_delta * architecture.increment_fraction * (network.dt / .05)),
            "edge_update": _delta_summary([value for item in collectors for value in item.edge_delta], limit=None),
            "action_trajectories": [[list(action) for action in item.action_trace] for item in collectors],
        }

    left, right = evaluate(left_genome), evaluate(right_genome)
    differences = []
    for first, second in zip(left["action_trajectories"], right["action_trajectories"], strict=True):
        differences.extend(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(first, second))
    scaled = {
        str(scale): evaluate([float(value) * float(scale) for value in right_genome])
        for scale in scales
    }
    return {
        "evaluation_seeds": list(seeds), "checkpoint_a": left, "checkpoint_b": right,
        "differences": {
            "fitness": right["fitness"]["mean"] - left["fitness"]["mean"],
            "mean_speed": right["behavior"]["mean_speed"] - left["behavior"]["mean_speed"],
            "mean_turn": right["behavior"]["mean_turn"] - left["behavior"]["mean_turn"],
            "mean_action_change": right["behavior"]["mean_action_change"] - left["behavior"]["mean_action_change"],
            "mean_absolute_action_difference": fmean(differences) if differences else 0.0,
            "maximum_action_difference": max(differences, default=0.0), "cumulative_action_difference": sum(differences),
        },
        "parameter_scaling": {"target": "checkpoint_b", "kind": "parameter_scale", "results": scaled},
    }
