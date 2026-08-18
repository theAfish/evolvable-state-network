"""Post-training, observation-only diagnostics for embodied prey rules.

This module deliberately does not participate in optimisation.  It loads an
already saved prey genome, probes its local rule, and runs matched evaluation
episodes with optional *evaluation-only* ablations.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from math import sqrt, tanh
from pathlib import Path
from statistics import fmean, pstdev
from typing import Callable, Mapping, Sequence

import numpy as np

from .embodied import EmbodiedNetwork, EmbodiedNetworkConfig, FoodWebAgentAdapter
from .environments import AgentId, EpisodeRunner, FoodWebConfig, FoodWebEnvironment, RandomControllerBlueprint, Species, make_reference_population
from .evolution.candidate import EdgeArchitecture, MLPEdgeRule, MLPUpdateRule, RuleArchitecture
from .evolution.genome import GenomeCodec
from .tasks.embodied_food_web import EmbodiedFoodWebControllerBlueprint, EmbodiedFoodWebTaskConfig, _network_seed, _species_behavior


def _rms(values: Sequence[float]) -> float:
    return sqrt(fmean(value * value for value in values)) if values else 0.0


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    return (fmean(values), pstdev(values)) if values else (0.0, 0.0)


@dataclass(frozen=True, slots=True)
class PreyDiagnosticConfig:
    sweep_points: int = 81
    episodes: int = 3
    episode_steps: int | None = None
    fixed_point_tolerance: float = 1e-4
    plateau_tolerance: float = .03
    sustained_window: int = 8
    large_bias_ratio: float = 1.5
    dominance_ratio: float = .75
    rapid_convergence_steps: int = 20
    vector_grid_points: int = 25
    vector_field_stride: int = 1
    jacobian_epsilon: float = 1e-3
    synchronization_variance_threshold: float = .01
    synchronization_distance_threshold: float = .15
    cross_coupling_derivative_threshold: float = .02
    fixed_point_max_iterations: int = 20


class _Trace:
    def __init__(self, network: EmbodiedNetwork, event: dict[str, object]) -> None:
        self.network, self.event = network, event
        self.pending: dict[int, list[tuple[int, tuple[float, ...], tuple[float, ...]]]] = {}
        self.samples: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
        self.actions: list[Mapping[str, object]] = []

    def observe(self, step: int, node: int, state: tuple[float, ...], aggregate: tuple[float, ...]) -> None:
        self.pending.setdefault(step, []).append((node, state, aggregate))
        if node not in self.network.interface.input_nodes and node not in self.network.interface.action_nodes:
            # Rule probes intentionally exclude externally overwritten sensory
            # ports and action readouts; they are not autonomous tissue.
            self.samples.append((state, aggregate))
            cast = self.event.setdefault("samples", [])
            assert isinstance(cast, list)
            cast.append((state, aggregate))

    def acted(self, action: Mapping[str, object]) -> None:
        self.actions.append(action)
        cast_actions = self.event.setdefault("actions", [])
        assert isinstance(cast_actions, list)
        cast_actions.append(dict(action))
        step = max(self.pending, default=-1)
        rows = self.pending.get(step, [])
        groups = _group_rows(rows, self.network)
        cast = self.event.setdefault("steps", [])
        assert isinstance(cast, list)
        cast.append(groups)


def _group_rows(rows: Sequence[tuple[int, tuple[float, ...], tuple[float, ...]]], network: EmbodiedNetwork) -> dict[str, dict[str, list[float]]]:
    input_nodes, action_nodes = set(network.interface.input_nodes), set(network.interface.action_nodes)
    groups: dict[str, list[tuple[tuple[float, ...], tuple[float, ...]]]] = {"sensory_input": [], "action_output": [], "anonymous_hidden": []}
    for node, state, aggregate in rows:
        groups["sensory_input" if node in input_nodes else "action_output" if node in action_nodes else "anonymous_hidden"].append((state, aggregate))
    result: dict[str, dict[str, list[float]]] = {}
    for name, paired in groups.items():
        vectors = [state for state, _ in paired]
        aggregates = [aggregate for _, aggregate in paired]
        width = network.config.state_width
        distances = [sqrt(sum((left[index] - right[index]) ** 2 for index in range(width))) for left_index, left in enumerate(vectors) for right in vectors[left_index + 1:]]
        aggregate_distances = [sqrt(sum((left[index] - right[index]) ** 2 for index in range(width))) for left_index, left in enumerate(aggregates) for right in aggregates[left_index + 1:]]
        centroid_distances = [sqrt(sum((value - (fmean(row[index] for row in vectors) if vectors else 0.0)) ** 2 for index, value in enumerate(row))) for row in vectors]
        result[name] = {
            "mean": [fmean(row[index] for row in vectors) if vectors else 0.0 for index in range(width)],
            "variance": [pstdev(row[index] for row in vectors) ** 2 if len(vectors) > 1 else 0.0 for index in range(width)],
            "mean_pairwise_distance": fmean(distances) if distances else 0.0,
            "median_pairwise_distance": float(np.median(distances)) if distances else 0.0,
            "maximum_pairwise_distance": max(distances, default=0.0),
            "mean_distance_from_centroid": fmean(centroid_distances) if centroid_distances else 0.0,
            "aggregate_mean": [fmean(row[index] for row in aggregates) if aggregates else 0.0 for index in range(width)],
            "aggregate_variance": [pstdev(row[index] for row in aggregates) ** 2 if len(aggregates) > 1 else 0.0 for index in range(width)],
            "aggregate_min": [min((row[index] for row in aggregates), default=0.0) for index in range(width)],
            "aggregate_max": [max((row[index] for row in aggregates), default=0.0) for index in range(width)],
            "mean_pairwise_aggregate_distance": fmean(aggregate_distances) if aggregate_distances else 0.0,
        }
    return result


class _DiagnosticController:
    """A normal embodied controller with trace capture and optional ablation."""

    def __init__(self, node_rule: MLPUpdateRule, edge_rule: MLPEdgeRule, config: EmbodiedNetworkConfig, seed: int, events: list[dict[str, object]], zero_messages: bool) -> None:
        self.node_rule, self.edge_rule, self.config, self.seed = node_rule, edge_rule, config, seed
        self.events, self.zero_messages = events, zero_messages
        self.network: EmbodiedNetwork | None = None
        self.trace: _Trace | None = None

    def begin_episode(self, *, seed: int | None = None) -> None:
        episode_seed = self.seed if seed is None else self.seed + seed
        event: dict[str, object] = {"steps": []}
        def observer(step: int, node: int, state: tuple[float, ...], aggregate: tuple[float, ...]) -> None:
            assert self.trace is not None
            self.trace.observe(step, node, state, aggregate)
        self.network = EmbodiedNetwork(self.node_rule, self.edge_rule, FoodWebAgentAdapter(vision_pixels=self.config.vision_pixels, body_inputs=self.config.body_inputs), self.config, seed=episode_seed, node_observer=observer, force_zero_messages=self.zero_messages)
        self.trace = _Trace(self.network, event)
        # Initial state statistics are an explicit init/respawn anchor.
        event["initial"] = _group_rows([(node, state, tuple(0.0 for _ in state)) for node, state in enumerate(self.network.state.node[0])], self.network)
        event["graph"] = {
            "hidden_nodes": [node for node in range(self.network.config.nodes) if node not in self.network.interface.input_nodes and node not in self.network.interface.action_nodes],
            "edges": [[edge.source, edge.target] for edge in self.network.graph.edges],
        }
        self.events.append(event)

    def act(self, observation: Mapping[str, object], *, available_actions: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
        assert self.network is not None and self.trace is not None
        action = self.network.act(observation)
        self.trace.acted(action)
        return action

    def learn(self, transition: object) -> None: pass
    def end_episode(self) -> None: pass


class _DiagnosticBlueprint:
    def __init__(self, node: MLPUpdateRule, edge: MLPEdgeRule, network: EmbodiedNetworkConfig, seed: int, events: list[dict[str, object]], zero_messages: bool) -> None:
        self.node, self.edge, self.network, self.seed, self.events, self.zero_messages = node, edge, network, seed, events, zero_messages
    def build(self, *, seed: int | None = None) -> _DiagnosticController:
        return _DiagnosticController(self.node, self.edge, self.network, self.seed + (seed or 0), self.events, self.zero_messages)


def _run_episode(
    prey: MLPUpdateRule, predator: tuple[MLPUpdateRule, MLPEdgeRule] | None,
    edge: MLPEdgeRule, task: EmbodiedFoodWebTaskConfig, seed: int, *, zero_messages: bool,
) -> tuple[dict[str, float], list[dict[str, object]], list[tuple[tuple[float, ...], tuple[float, ...]]], list[Mapping[str, object]]]:
    # Observation hooks are implemented by the reference Python backend so
    # that diagnostics never perturb the accelerated training path.
    network = replace(task.network, execution_backend="python", device="cpu")
    events: list[dict[str, object]] = []
    blueprint = _DiagnosticBlueprint(prey, edge, network, _network_seed(seed, Species.PREY), events, zero_messages)
    agents = make_reference_population(prey_count=task.prey_count, predator_count=task.predator_count, width=task.environment.width, height=task.environment.height, prey_initial_energy=task.environment.prey_initial_energy, predator_initial_energy=task.environment.predator_initial_energy, controller=RandomControllerBlueprint(), seed=seed)
    prey_ids: list[AgentId] = []
    for agent in agents:
        if agent.species is Species.PREY:
            agent.controller = blueprint
            prey_ids.append(agent.id)
        elif predator is not None:
            node, edge_rule = predator
            agent.controller = EmbodiedFoodWebControllerBlueprint(node, edge_rule, network, _network_seed(seed, Species.PREDATOR))
    result = EpisodeRunner(FoodWebEnvironment(task.environment, seed=seed)).run(agents, max_steps=task.max_steps, seed=seed)
    samples: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
    actions: list[Mapping[str, object]] = []
    # Event lifetime is limited to each fresh network, exactly matching respawn.
    # The controller objects are intentionally not retained by EpisodeRunner;
    # their traces are already stored in the shared event list.
    for event in events:
        # samples/actions are attached by a small closure below if present.
        samples.extend(event.get("samples", []))  # type: ignore[arg-type]
        actions.extend(event.get("actions", []))  # type: ignore[arg-type]
    return _species_behavior(result, prey_ids), events, samples, actions


def _final_bias_zeroed(rule: MLPUpdateRule) -> MLPUpdateRule:
    values = list(rule.parameters)
    # Every layer stores weights then its bias.  The final layer is last.
    values[-rule.state_width:] = [0.0] * rule.state_width
    return MLPUpdateRule(rule.architecture, values, output_scale=rule.output_scale)


def _bias_report(rule: MLPUpdateRule, samples: Sequence[tuple[tuple[float, ...], tuple[float, ...]]], config: PreyDiagnosticConfig) -> tuple[dict[str, object], bool]:
    layers = rule._layers  # Decoded, immutable rule representation.
    hidden = [list(bias) for _, bias in layers[:-1]]
    final = list(layers[-1][1])
    weights = [weight for matrix, _ in layers for row in matrix for weight in row]
    biases = [bias for _, row in layers for bias in row]
    outputs = [rule.raw_output(state, aggregate)[0] for state, aggregate in samples]
    variation = pstdev(outputs) if len(outputs) > 1 else 0.0
    ratio = abs(final[0]) / max(variation, 1e-12)
    return ({"hidden_layer_biases": hidden, "final_layer_biases": final, "final_layer_channel_0_bias": final[0], "weight_rms": _rms(weights), "bias_rms": _rms(biases), "channel_0_input_dependent_std": variation, "final_bias_to_variation_ratio": ratio}, ratio >= config.large_bias_ratio)


def _probe_samples(rule: MLPUpdateRule, samples: Sequence[tuple[tuple[float, ...], tuple[float, ...]]], network: EmbodiedNetworkConfig) -> dict[str, dict[str, float]]:
    zero = (0.0,) * rule.state_width
    cases = {"normal_state_normal_aggregate": [(state, aggregate) for state, aggregate in samples], "normal_state_zero_aggregate": [(state, zero) for state, _ in samples], "zero_state_normal_aggregate": [(zero, aggregate) for _, aggregate in samples], "zero_state_zero_aggregate": [(zero, zero) for _ in samples]}
    result: dict[str, dict[str, float]] = {}
    limit = network.max_delta * rule.architecture.increment_fraction * (network.dt / .05)
    for name, rows in cases.items():
        raw = [rule.raw_output(state, aggregate) for state, aggregate in rows]
        updates = [tuple(limit * tanh(value * rule.output_scale) for value in vector) for vector in raw]
        channel0 = [row[0] for row in updates]
        result[name] = {"raw_output_rms": _rms([value for row in raw for value in row]), "effective_update_rms": _rms([value for row in updates for value in row]), "channel_0_mean_update": fmean(channel0) if channel0 else 0.0, "channel_0_update_std": pstdev(channel0) if len(channel0) > 1 else 0.0}
    normal = result["normal_state_normal_aggregate"]
    for name, label in (("normal_state_zero_aggregate", "zero_aggregate_update_rms_over_normal"), ("zero_state_normal_aggregate", "zero_state_update_rms_over_normal")):
        result[name][label] = result[name]["effective_update_rms"] / max(normal["effective_update_rms"], 1e-12)
    return result


def _drift_curves(rule: MLPUpdateRule, samples: Sequence[tuple[tuple[float, ...], tuple[float, ...]]], network: EmbodiedNetworkConfig, config: PreyDiagnosticConfig) -> dict[str, object]:
    width, zero = rule.state_width, (0.0,) * rule.state_width
    representative = samples[len(samples) // 2][1] if samples else zero
    aggregates = {"zero": zero, "representative_negative": tuple(-.5 for _ in range(width)), "representative_positive": tuple(.5 for _ in range(width)), "episode_sample": representative}
    limit = network.max_delta * rule.architecture.increment_fraction * (network.dt / .05)
    xs = [-network.max_abs_state + 2 * network.max_abs_state * index / max(1, config.sweep_points - 1) for index in range(config.sweep_points)]
    report: dict[str, object] = {}
    for name, aggregate in aggregates.items():
        rows = []
        deltas: list[float] = []
        for value in xs:
            state = (value,) + (0.0,) * (width - 1)
            raw = rule.raw_output(state, aggregate)
            update = limit * tanh(raw[0] * rule.output_scale)
            # This is the same final state bound used by Simulation, so the
            # plotted delta represents the actual state transition, including
            # a possible endpoint clip at +/- max_abs_state.
            resulting_delta = max(-network.max_abs_state, min(network.max_abs_state, value + update)) - value
            deltas.append(resulting_delta)
            rows.append({"state_channel_0": value, "raw_output_channel_0": raw[0], "bounded_effective_update_channel_0": update, "delta_state_channel_0": resulting_delta})
        fixed = []
        for index in range(len(xs) - 1):
            left, right = deltas[index], deltas[index + 1]
            if left == 0.0 or left * right < 0.0:
                fraction = 0.0 if left == right else -left / (right - left)
                location = xs[index] + fraction * (xs[index + 1] - xs[index])
                stable = left > 0.0 and right < 0.0
                unstable = left < 0.0 and right > 0.0
                fixed.append({"location": location, "classification": "stable" if stable else "unstable" if unstable else "indeterminate", "left_delta": left, "right_delta": right})
        report[name] = {"aggregate": list(aggregate), "curve": rows, "fixed_points": fixed}
    return report


def _effective_delta(rule: MLPUpdateRule, state: Sequence[float], aggregate: Sequence[float], network: EmbodiedNetworkConfig) -> np.ndarray:
    """Exact local effective update, including the node state bound."""
    state_vector, aggregate_vector = tuple(float(value) for value in state), tuple(float(value) for value in aggregate)
    limit = network.max_delta * rule.architecture.increment_fraction * (network.dt / .05)
    raw = rule.raw_output(state_vector, aggregate_vector)
    proposed = np.asarray(state_vector) + np.asarray([limit * tanh(value * rule.output_scale) for value in raw])
    bounded = np.clip(proposed, -network.max_abs_state, network.max_abs_state)
    return bounded - np.asarray(state_vector)


def _local_jacobian(rule: MLPUpdateRule, state: Sequence[float], aggregate: Sequence[float], network: EmbodiedNetworkConfig, config: PreyDiagnosticConfig) -> np.ndarray:
    epsilon = config.jacobian_epsilon
    matrix = np.zeros((2, 2), dtype=float)
    for column in range(2):
        plus = np.asarray(state, dtype=float).copy()
        minus = np.asarray(state, dtype=float).copy()
        plus[column] += epsilon; minus[column] -= epsilon
        matrix[:, column] = (_effective_delta(rule, plus, aggregate, network) - _effective_delta(rule, minus, aggregate, network)) / (2 * epsilon)
    return matrix


def _stability(rule: MLPUpdateRule, state: Sequence[float], aggregate: Sequence[float], network: EmbodiedNetworkConfig, config: PreyDiagnosticConfig) -> dict[str, object]:
    jacobian = _local_jacobian(rule, state, aggregate, network, config)
    eigenvalues = np.linalg.eigvals(np.eye(2) + jacobian)
    radii = np.abs(eigenvalues)
    tolerance = .01
    if np.all(radii < 1.0 - tolerance):
        classification = "stable"
    elif np.any(radii < 1.0 - tolerance) and np.any(radii > 1.0 + tolerance):
        classification = "saddle"
    elif np.any(radii > 1.0 + tolerance):
        classification = "unstable"
    else:
        classification = "marginal / indeterminate"
    return {"jacobian_effective_delta": jacobian.tolist(), "transition_eigenvalues": [[float(value.real), float(value.imag)] for value in eigenvalues], "spectral_radius": float(max(radii)), "classification": classification}


def _refine_fixed_point(rule: MLPUpdateRule, initial: Sequence[float], aggregate: Sequence[float], network: EmbodiedNetworkConfig, config: PreyDiagnosticConfig) -> tuple[np.ndarray, float] | None:
    point = np.asarray(initial, dtype=float)
    for _ in range(config.fixed_point_max_iterations):
        delta = _effective_delta(rule, point, aggregate, network)
        residual = float(np.linalg.norm(delta))
        if residual <= config.fixed_point_tolerance:
            return point, residual
        jacobian = _local_jacobian(rule, point, aggregate, network, config)
        try:
            step = np.linalg.solve(jacobian, delta)
        except np.linalg.LinAlgError:
            return None
        candidate = np.clip(point - step, -network.max_abs_state + config.jacobian_epsilon, network.max_abs_state - config.jacobian_epsilon)
        if float(np.linalg.norm(candidate - point)) < config.jacobian_epsilon * .1:
            break
        point = candidate
    residual = float(np.linalg.norm(_effective_delta(rule, point, aggregate, network)))
    return (point, residual) if residual <= config.fixed_point_tolerance else None


def _vector_field(rule: MLPUpdateRule, samples: Sequence[tuple[tuple[float, ...], tuple[float, ...]]], network: EmbodiedNetworkConfig, config: PreyDiagnosticConfig) -> dict[str, object]:
    if rule.state_width != 2:
        return {"available": False, "reason": "2D vector fields require state_width=2"}
    zero = (0.0, 0.0)
    mean = tuple(float(fmean(aggregate[index] for _, aggregate in samples)) if samples else 0.0 for index in range(2))
    ordered = sorted((aggregate for _, aggregate in samples), key=lambda row: sum(row))
    low = ordered[max(0, len(ordered) // 10)] if ordered else zero
    high = ordered[min(len(ordered) - 1, 9 * len(ordered) // 10)] if ordered else zero
    conditions = {"zero": zero, "episode_mean": mean, "representative_low": low, "representative_high": high}
    coordinates = np.linspace(-network.max_abs_state, network.max_abs_state, config.vector_grid_points)
    result: dict[str, object] = {}
    for name, aggregate in conditions.items():
        rows = []
        candidates: list[tuple[float, tuple[float, float]]] = []
        for h0 in coordinates:
            for h1 in coordinates:
                state = (float(h0), float(h1))
                raw = rule.raw_output(state, aggregate)
                delta = _effective_delta(rule, state, aggregate, network)
                residual = float(np.linalg.norm(delta))
                rows.append({"h0": state[0], "h1": state[1], "raw_output": list(raw), "effective_delta": delta.tolist(), "residual": residual})
                candidates.append((residual, state))
        fixed: list[dict[str, object]] = []
        for _, seed in sorted(candidates)[:16]:
            refined = _refine_fixed_point(rule, seed, aggregate, network, config)
            if refined is None:
                continue
            point, residual = refined
            if any(float(np.linalg.norm(point - np.asarray(existing["state"]))) < .02 for existing in fixed):
                continue
            fixed.append({"state": point.tolist(), "residual_update_magnitude": residual, "aggregate_condition": name, **_stability(rule, point, aggregate, network, config)})
        result[name] = {"aggregate": list(aggregate), "samples": rows, "fixed_points": fixed}
    return {"available": True, "grid_points": config.vector_grid_points, "conditions": result}


def _cross_channel_coupling(rule: MLPUpdateRule, state: Sequence[float], aggregate: Sequence[float], network: EmbodiedNetworkConfig, config: PreyDiagnosticConfig) -> dict[str, object]:
    extent = min(1.5, network.max_abs_state)
    h0, h1 = float(state[0]), float(state[1])
    h1_sweep = [{"h1": float(value), "delta_h0": float(_effective_delta(rule, (h0, value), aggregate, network)[0])} for value in np.linspace(max(-network.max_abs_state, h1 - extent), min(network.max_abs_state, h1 + extent), 41)]
    h0_sweep = [{"h0": float(value), "delta_h1": float(_effective_delta(rule, (value, h1), aggregate, network)[1])} for value in np.linspace(max(-network.max_abs_state, h0 - extent), min(network.max_abs_state, h0 + extent), 41)]
    jacobian = _local_jacobian(rule, state, aggregate, network, config)
    return {"empirical_state": [h0, h1], "aggregate": list(aggregate), "hold_h0_sweep_h1_to_delta_h0": h1_sweep, "hold_h1_sweep_h0_to_delta_h1": h0_sweep, "d_delta_h0_d_h1": float(jacobian[0, 1]), "d_delta_h1_d_h0": float(jacobian[1, 0])}


def _synchronization(events: Sequence[dict[str, object]], config: PreyDiagnosticConfig) -> dict[str, object]:
    event_rows: list[dict[str, object]] = []
    for event in events:
        steps = [row["anonymous_hidden"] for row in event.get("steps", []) if isinstance(row, dict) and "anonymous_hidden" in row]
        if not steps:
            continue
        state_variance = [float(fmean(row["variance"])) for row in steps]
        aggregate_variance = [float(fmean(row["aggregate_variance"])) for row in steps]
        distances = [float(row["mean_pairwise_distance"]) for row in steps]
        aggregate_distances = [float(row["mean_pairwise_aggregate_distance"]) for row in steps]
        def first_below(values: Sequence[float], threshold: float) -> int | None:
            return next((index for index, value in enumerate(values) if value <= threshold), None)
        plateau = next(
            (index for index in range(len(distances)) if all(
                abs(later - distances[-1]) <= config.synchronization_distance_threshold
                for later in distances[index:]
            )),
            len(distances) - 1,
        )
        correlation = float(np.corrcoef(state_variance, aggregate_variance)[0, 1]) if len(state_variance) > 1 and np.std(state_variance) > 0 and np.std(aggregate_variance) > 0 else 0.0
        lagged = {str(lag): float(np.corrcoef(state_variance[lag:], aggregate_variance[:-lag])[0, 1]) if len(state_variance) > lag + 1 and np.std(state_variance[lag:]) > 0 and np.std(aggregate_variance[:-lag]) > 0 else 0.0 for lag in (1, 2, 3) if len(state_variance) > lag + 1}
        event_rows.append({"state_variance": state_variance, "message_variance": aggregate_variance, "mean_pairwise_distance": distances, "mean_pairwise_message_distance": aggregate_distances, "synchronization_time_state_variance": first_below(state_variance, config.synchronization_variance_threshold), "synchronization_time_channel_0": first_below([float(row["variance"][0]) for row in steps], config.synchronization_variance_threshold), "synchronization_time_channel_1": first_below([float(row["variance"][1]) for row in steps], config.synchronization_variance_threshold), "pairwise_distance_plateau": plateau, "state_message_variance_correlation": correlation, "lagged_state_to_message_correlations": lagged})
    if not event_rows:
        return {"events": [], "aggregate": {}}
    numeric = ("synchronization_time_state_variance", "synchronization_time_channel_0", "synchronization_time_channel_1", "pairwise_distance_plateau", "state_message_variance_correlation")
    aggregate = {name: fmean(float(row[name]) for row in event_rows if row[name] is not None) if any(row[name] is not None for row in event_rows) else None for name in numeric}
    return {"events": event_rows, "aggregate": aggregate}


def _graph_roles(events: Sequence[dict[str, object]]) -> dict[str, object]:
    # Graphs are held on trace events as compact, JSON-safe snapshots.
    rows = []
    for event in events:
        graph = event.get("graph")
        if not isinstance(graph, dict):
            continue
        hidden = set(graph["hidden_nodes"])
        edges = [tuple(edge) for edge in graph["edges"]]
        in_degree = {node: sum(target == node for _, target in edges) for node in hidden}
        out_degree = {node: sum(source == node for source, _ in edges) for node in hidden}
        signatures: dict[tuple[int, int], list[int]] = {}
        for node in hidden: signatures.setdefault((in_degree[node], out_degree[node]), []).append(node)
        rows.append({"hidden_in_degree": list(in_degree.values()), "hidden_out_degree": list(out_degree.values()), "same_degree_role_groups": [members for members in signatures.values() if len(members) > 1]})
    return {"networks": rows}


def _convergence(events: Sequence[dict[str, object]], width: int, config: PreyDiagnosticConfig) -> dict[str, object]:
    summaries = []
    for event in events:
        initial = event.get("initial", {})
        steps = event.get("steps", [])
        if not isinstance(initial, dict) or not isinstance(steps, list):
            continue
        series = [initial.get("anonymous_hidden", {})] + [row.get("anonymous_hidden", {}) for row in steps if isinstance(row, dict)]
        means = [[float(value) for value in row.get("mean", [0.0] * width)] for row in series if isinstance(row, dict)]
        variances = [[float(value) for value in row.get("variance", [0.0] * width)] for row in series if isinstance(row, dict)]
        if not means:
            continue
        final = means[-1]
        plateau = next(
            (index for index in range(len(means)) if all(
                abs(means[later][0] - final[0]) <= config.plateau_tolerance
                for later in range(index, len(means))
            )),
            len(means) - 1,
        )
        variance_final = variances[-1][0]
        variance_stable = next(
            (index for index in range(len(variances)) if all(
                abs(variances[later][0] - variance_final) <= config.plateau_tolerance
                for later in range(index, len(variances))
            )),
            len(variances) - 1,
        )
        summaries.append({"initial_mean": means[0], "final_mean": final, "initial_std": [sqrt(max(0.0, value)) for value in variances[0]], "final_std": [sqrt(max(0.0, value)) for value in variances[-1]], "minimum_mean": [min(row[channel] for row in means) for channel in range(width)], "time_to_final_plateau": plateau, "time_to_sustained_tolerance": plateau, "time_to_variance_stabilization": variance_stable})
    if not summaries:
        return {"events": [], "aggregate": {}}
    keys = ("time_to_final_plateau", "time_to_sustained_tolerance", "time_to_variance_stabilization")
    return {"events": summaries, "aggregate": {key: fmean(float(row[key]) for row in summaries) for key in keys}}


def _node_group_state_summaries(events: Sequence[dict[str, object]], width: int) -> dict[str, object]:
    """Keep boundary ports separate from anonymous recurrent tissue."""
    result: dict[str, object] = {}
    for group in ("sensory_input", "action_output", "anonymous_hidden"):
        initial: list[list[float]] = []
        final: list[list[float]] = []
        final_variance: list[list[float]] = []
        for event in events:
            start = event.get("initial", {})
            steps = event.get("steps", [])
            if not isinstance(start, dict) or not isinstance(steps, list):
                continue
            start_group = start.get(group, {})
            end_group = steps[-1].get(group, {}) if steps and isinstance(steps[-1], dict) else start_group
            if isinstance(start_group, dict) and isinstance(end_group, dict):
                initial.append([float(value) for value in start_group.get("mean", [0.0] * width)])
                final.append([float(value) for value in end_group.get("mean", [0.0] * width)])
                final_variance.append([float(value) for value in end_group.get("variance", [0.0] * width)])
        result[group] = {
            "initial_mean": [fmean(row[channel] for row in initial) if initial else 0.0 for channel in range(width)],
            "final_mean": [fmean(row[channel] for row in final) if final else 0.0 for channel in range(width)],
            "final_node_to_node_variance": [fmean(row[channel] for row in final_variance) if final_variance else 0.0 for channel in range(width)],
        }
    return result


def _ablation_summary(behavior_rows: Sequence[Mapping[str, float]], events: Sequence[dict[str, object]], width: int, config: PreyDiagnosticConfig) -> dict[str, object]:
    actions: list[Mapping[str, object]] = []
    for event in events:
        actions.extend(event.get("actions", []))  # type: ignore[arg-type]
    convergence = _convergence(events, width, config)
    channels = [row["final_mean"] for row in convergence.get("events", [])]  # type: ignore[union-attr]
    variances = [row["final_std"] for row in convergence.get("events", [])]  # type: ignore[union-attr]
    turns = [float(action.get("turn", 0.0)) for action in actions]
    speeds = [float(action.get("speed", 0.0)) for action in actions]
    changes = [abs(turns[i] - turns[i - 1]) + abs(speeds[i] - speeds[i - 1]) for i in range(1, len(actions))]
    return {"fitness_lifetime": fmean(float(row.get("restricted_mean_lifetime", 0.0)) for row in behavior_rows) if behavior_rows else 0.0, "hidden_channel_0_mean": fmean(row[0] for row in channels) if channels else 0.0, "hidden_channel_0_variance": fmean(row[0] ** 2 for row in variances) if variances else 0.0, "convergence_time": float(convergence.get("aggregate", {}).get("time_to_final_plateau", 0.0)), "mean_speed": fmean(speeds) if speeds else 0.0, "mean_turn": fmean(turns) if turns else 0.0, "mean_action_change": fmean(changes) if changes else 0.0, "convergence": convergence}


def diagnose_prey_genome(prey_genome: Sequence[float], architecture: RuleArchitecture, edge_architecture: EdgeArchitecture, task: EmbodiedFoodWebTaskConfig, *, predator_genome: Sequence[float] | None = None, config: PreyDiagnosticConfig = PreyDiagnosticConfig()) -> dict[str, object]:
    """Return a JSON-ready diagnostic report; no training state is modified."""
    if config.sweep_points < 3 or config.episodes < 1:
        raise ValueError("sweep_points must be at least 3 and episodes must be positive")
    codec = GenomeCodec(architecture, edge_architecture, "joint")
    prey, edge = codec.decode_groups(prey_genome, output_scale=task.network.rule_output_scale)
    assert prey is not None and edge is not None
    predator = None
    if predator_genome is not None:
        predator = codec.decode_groups(predator_genome, output_scale=task.network.rule_output_scale)
        assert predator[0] is not None and predator[1] is not None
    seeds = [task.seed + 10_007 * index for index in range(config.episodes)]
    normal_behavior: list[dict[str, float]] = []
    normal_events: list[dict[str, object]] = []
    samples: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
    for seed in seeds:
        behavior, events, _, _ = _run_episode(prey, predator, edge, task, seed, zero_messages=False)
        normal_behavior.append(behavior); normal_events.extend(events)
    # The trace stores a compact per-step series.  Recover probe samples by
    # re-running a short direct network is unnecessary: observer records them.
    # Store samples directly on the event at capture time (see below).
    for event in normal_events:
        samples.extend(event.get("samples", []))  # type: ignore[arg-type]
    drift = _drift_curves(prey, samples, task.network, config)
    vector_field = _vector_field(prey, samples, task.network, config)
    probes = _probe_samples(prey, samples, task.network)
    bias, strong_bias = _bias_report(prey, samples, config)
    ablations: dict[str, object] = {"normal": _ablation_summary(normal_behavior, normal_events, architecture.state_width, config)}
    # The aggregate node-group summary is across fresh networks; use it as a
    # replay attractor estimate rather than mixing boundary ports into it.
    groups = _node_group_state_summaries(normal_events, architecture.state_width)
    empirical_state = list(groups["anonymous_hidden"]["final_mean"])
    empirical_aggregate = tuple(float(fmean(aggregate[index] for _, aggregate in samples)) if samples else 0.0 for index in range(2))
    empirical_stability = {"residual_update_magnitude": float(np.linalg.norm(_effective_delta(prey, empirical_state, empirical_aggregate, task.network))), **_stability(prey, empirical_state, empirical_aggregate, task.network, config)}
    coupling = _cross_channel_coupling(prey, empirical_state, empirical_aggregate, task.network, config)
    zero_message_events: list[dict[str, object]] = []
    for name, rule, zero_messages in (("zero_final_output_bias", _final_bias_zeroed(prey), False), ("zero_messages", prey, True)):
        rows: list[dict[str, float]] = []; events: list[dict[str, object]] = []
        for seed in seeds:
            behavior, trace, _, _ = _run_episode(rule, predator, edge, task, seed, zero_messages=zero_messages)
            rows.append(behavior); events.extend(trace)
        ablations[name] = _ablation_summary(rows, events, architecture.state_width, config)
        if name == "zero_messages":
            zero_message_events = events
    fixed = [point for condition in drift.values() for point in condition["fixed_points"]]  # type: ignore[index]
    normal_summary = ablations["normal"]  # type: ignore[assignment]
    zero_message_summary = ablations["zero_messages"]  # type: ignore[assignment]
    zero_bias_summary = ablations["zero_final_output_bias"]  # type: ignore[assignment]
    normal_sync = _synchronization(normal_events, config)
    zero_message_sync = _synchronization(zero_message_events, config)
    ablations["normal"]["synchronization"] = normal_sync  # type: ignore[index]
    ablations["zero_messages"]["synchronization"] = zero_message_sync  # type: ignore[index]
    normal_rms = probes["normal_state_normal_aggregate"]["effective_update_rms"]
    self_ratio = probes["normal_state_zero_aggregate"]["effective_update_rms"] / max(normal_rms, 1e-12)
    message_ratio = probes["zero_state_normal_aggregate"]["effective_update_rms"] / max(normal_rms, 1e-12)
    flags = []
    if any(point["classification"] == "stable" and point["location"] < 0 for point in fixed): flags.append("NEGATIVE_FIXED_POINT_DETECTED")
    if strong_bias: flags.append("STRONG_NEGATIVE_OUTPUT_BIAS")
    if self_ratio >= config.dominance_ratio and message_ratio < config.dominance_ratio: flags.append("SELF_DYNAMICS_DOMINATE_MESSAGES")
    if message_ratio >= config.dominance_ratio and self_ratio < config.dominance_ratio: flags.append("MESSAGES_DOMINATE_SELF_DYNAMICS")
    if float(normal_summary["hidden_channel_0_mean"]) < -config.plateau_tolerance and abs(float(zero_message_summary["hidden_channel_0_mean"]) - float(normal_summary["hidden_channel_0_mean"])) <= config.plateau_tolerance: flags.append("ATTRACTOR_PERSISTS_WITH_ZERO_MESSAGES")
    if abs(float(zero_bias_summary["hidden_channel_0_mean"])) < abs(float(normal_summary["hidden_channel_0_mean"])) * .5: flags.append("BIAS_ABLATION_REMOVES_ATTRACTOR")
    if float(normal_summary["convergence_time"]) <= config.rapid_convergence_steps: flags.append("RAPID_POST_INIT_CONVERGENCE")
    if vector_field.get("available"):
        all_fixed = [point for condition in vector_field["conditions"].values() for point in condition["fixed_points"]]
        if any(point["classification"] == "stable" for point in all_fixed): flags.append("STABLE_2D_FIXED_POINT")
        near_empirical = [point for point in all_fixed if float(np.linalg.norm(np.asarray(point["state"]) - np.asarray(empirical_state))) < .5]
        if near_empirical: flags.append("COUPLED_STATE_ATTRACTOR_DETECTED")
    if coupling["d_delta_h0_d_h1"] <= -config.cross_coupling_derivative_threshold: flags.append("STRONG_CROSS_CHANNEL_COUPLING")
    if normal_sync.get("aggregate", {}).get("synchronization_time_state_variance") is not None: flags.append("HIDDEN_NODE_SYNCHRONIZATION")
    if normal_sync.get("aggregate", {}).get("synchronization_time_state_variance") is not None and zero_message_sync.get("aggregate", {}).get("synchronization_time_state_variance") is not None: flags.append("SELF_DYNAMICS_SUFFICIENT_FOR_COLLAPSE")
    if float(normal_sync.get("aggregate", {}).get("state_message_variance_correlation", 0.0)) > .5: flags.append("MESSAGE_SYNCHRONIZATION")
    if normal_sync.get("aggregate", {}).get("state_message_variance_correlation", 0.0) > .5: flags.append("STATE_MESSAGE_POSITIVE_FEEDBACK")
    shifts: list[dict[str, object]] = []
    if vector_field.get("available"):
        conditions = vector_field["conditions"]
        zero_fixed = conditions["zero"]["fixed_points"]
        for name, condition in conditions.items():
            if name == "zero" or not zero_fixed or not condition["fixed_points"]:
                continue
            start = min(zero_fixed, key=lambda row: float(np.linalg.norm(np.asarray(row["state"]) - np.asarray(empirical_state))))
            end = min(condition["fixed_points"], key=lambda row: float(np.linalg.norm(np.asarray(row["state"]) - np.asarray(empirical_state))))
            shifts.append({"condition": name, "from_zero_state": start["state"], "to_state": end["state"], "displacement": (np.asarray(end["state"]) - np.asarray(start["state"])).tolist(), "from_stability": start["classification"], "to_stability": end["classification"]})
    return {"kind": "prey_negative_state_attractor_diagnostic", "config": asdict(config), "seeds": seeds, "drift_curves_channel_0": drift, "vector_field_2d": vector_field, "empirical_replay_attractor": {"state": empirical_state, "aggregate": list(empirical_aggregate), **empirical_stability}, "cross_channel_coupling": coupling, "attractor_message_shifts": shifts, "self_state_vs_message": probes, "zero_self_state_probe": {"real_state": probes["normal_state_normal_aggregate"], "zero_state": probes["zero_state_normal_aggregate"]}, "mlp_bias": bias, "ablations": ablations, "post_initialization_convergence": normal_summary["convergence"], "per_channel_hidden_summary": normal_summary["convergence"], "node_groups": groups, "hidden_synchronization": normal_sync, "graph_induced_symmetry": _graph_roles(normal_events), "flags": flags}


def _load_report(path: Path) -> tuple[Sequence[float], Sequence[float] | None, RuleArchitecture, EdgeArchitecture, EmbodiedFoodWebTaskConfig]:
    document = json.loads(path.read_text(encoding="utf-8"))
    architecture = RuleArchitecture(**document["architecture"])
    edge_architecture = EdgeArchitecture(**document["edge_architecture"])
    settings = document["task_config"]
    network = EmbodiedNetworkConfig(**settings["network"])
    environment = FoodWebConfig(**settings["environment"])
    task = EmbodiedFoodWebTaskConfig(network=network, environment=environment, prey_count=int(settings["prey_count"]), predator_count=int(settings["predator_count"]), max_steps=int(settings.get("batch_episode_steps", 256)), trials=1, seed=int(settings["seed"]))
    return document["prey_best_genome"], document.get("predator_best_genome"), architecture, edge_architecture, task


def write_diagnostic_plots(report: Mapping[str, object], output_dir: Path) -> list[Path]:
    """Render compact, publication-friendly views from a diagnostic JSON report."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    vector = report.get("vector_field_2d", {})
    if isinstance(vector, Mapping) and vector.get("available"):
        conditions = vector.get("conditions", {})
        if isinstance(conditions, Mapping):
            names = [name for name in ("zero", "episode_mean", "representative_low", "representative_high") if name in conditions]
            figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
            empirical = report.get("empirical_replay_attractor", {})
            for axis, name in zip(axes.flat, names, strict=False):
                condition = conditions[name]
                rows = condition.get("samples", [])
                h0 = np.asarray([row["h0"] for row in rows]); h1 = np.asarray([row["h1"] for row in rows])
                d0 = np.asarray([row["effective_delta"][0] for row in rows]); d1 = np.asarray([row["effective_delta"][1] for row in rows])
                stride = max(1, int(np.sqrt(len(rows)) // 15))
                axis.quiver(h0[::stride], h1[::stride], d0[::stride], d1[::stride], np.hypot(d0[::stride], d1[::stride]), cmap="viridis", scale=2.5)
                for point in condition.get("fixed_points", []):
                    state = point["state"]
                    axis.plot(state[0], state[1], "o", color="crimson" if point["classification"] == "stable" else "black", ms=6)
                if isinstance(empirical, Mapping):
                    state = empirical.get("state", [])
                    if len(state) == 2: axis.plot(state[0], state[1], "*", color="orange", ms=11, label="replay mean")
                axis.set(title=name.replace("_", " "), xlabel="hidden channel 0", ylabel="hidden channel 1", xlim=(-4, 4), ylim=(-4, 4))
            for axis in axes.flat[len(names):]: axis.set_visible(False)
            path = output_dir / "vector_fields_2d.png"; figure.savefig(path, dpi=160); plt.close(figure); paths.append(path)

    coupling = report.get("cross_channel_coupling", {})
    if isinstance(coupling, Mapping):
        left, right = coupling.get("hold_h0_sweep_h1_to_delta_h0", []), coupling.get("hold_h1_sweep_h0_to_delta_h1", [])
        if left and right:
            figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
            axes[0].plot([row["h1"] for row in left], [row["delta_h0"] for row in left]); axes[0].axhline(0, color="black", lw=.7); axes[0].set(xlabel="channel 1 (channel 0 held)", ylabel="effective Δ channel 0")
            axes[1].plot([row["h0"] for row in right], [row["delta_h1"] for row in right]); axes[1].axhline(0, color="black", lw=.7); axes[1].set(xlabel="channel 0 (channel 1 held)", ylabel="effective Δ channel 1")
            path = output_dir / "cross_channel_coupling.png"; figure.savefig(path, dpi=160); plt.close(figure); paths.append(path)

    synchronization = report.get("hidden_synchronization", {})
    if isinstance(synchronization, Mapping) and synchronization.get("events"):
        events = synchronization["events"]
        def average_series(name: str) -> list[float]:
            width = max(len(event.get(name, [])) for event in events)
            return [fmean(float(event[name][index]) for event in events if index < len(event.get(name, []))) for index in range(width)]
        state_variance, message_variance = average_series("state_variance"), average_series("message_variance")
        distance, message_distance = average_series("mean_pairwise_distance"), average_series("mean_pairwise_message_distance")
        zero_events = report.get("ablations", {}).get("zero_messages", {}).get("synchronization", {}).get("events", [])  # type: ignore[union-attr]
        def average_zero(name: str) -> list[float]:
            if not zero_events:
                return []
            width = max(len(event.get(name, [])) for event in zero_events)
            return [fmean(float(event[name][index]) for event in zero_events if index < len(event.get(name, []))) for index in range(width)]
        zero_variance, zero_distance = average_zero("state_variance"), average_zero("mean_pairwise_distance")
        figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        axes[0].plot(state_variance, label="normal state variance"); axes[0].plot(message_variance, label="normal aggregate variance");
        if zero_variance: axes[0].plot(zero_variance, "--", label="zero-message state variance")
        axes[0].set(xlabel="network tick", ylabel="across-hidden-node variance"); axes[0].legend()
        axes[1].plot(distance, label="normal state pairwise distance"); axes[1].plot(message_distance, label="normal aggregate pairwise distance")
        if zero_distance: axes[1].plot(zero_distance, "--", label="zero-message state distance")
        axes[1].set(xlabel="network tick", ylabel="mean pairwise distance"); axes[1].legend()
        path = output_dir / "hidden_synchronization.png"; figure.savefig(path, dpi=160); plt.close(figure); paths.append(path)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose a saved embodied prey genome without changing training.")
    parser.add_argument("--report", required=True, type=Path, help="saved embodied report.json or checkpoint.json")
    parser.add_argument("--output", type=Path, help="report path (default: prey_diagnostic.json beside input)")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--plot-dir", type=Path, help="optional directory for PNG diagnostic plots")
    args = parser.parse_args(argv)
    prey, predator, architecture, edge_architecture, task = _load_report(args.report)
    if args.steps is not None:
        task = replace(task, max_steps=args.steps)
    report = diagnose_prey_genome(prey, architecture, edge_architecture, task, predator_genome=predator, config=PreyDiagnosticConfig(episodes=args.episodes))
    output = args.output or args.report.with_name("prey_diagnostic.json")
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.plot_dir is not None:
        for path in write_diagnostic_plots(report, args.plot_dir):
            print(path)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
