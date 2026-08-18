"""Asynchronous, death-driven evolution of generic local graph dynamics.

The module deliberately separates the inherited rule parameters from local
runtime state.  Candidates occupy independent slots and are replaced as soon
as they die or reach the active survival milestone.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from math import isfinite
from pathlib import Path
from queue import Empty, SimpleQueue
from random import Random
from statistics import fmean
from typing import Callable, Literal, Mapping, Sequence

from .candidate import EdgeArchitecture, RuleArchitecture
from .cmaes import CMAES, CMAESConfig
from .genome import EvolutionTarget, GenomeCodec
from ..graph import Graph, generate_random_graph
from ..perturbations import ImpulseInjection
from ..simulation import (
    EventWindow,
    NetworkState,
    Simulation,
    SimulationConfig,
    Trajectory,
    TransitionDiagnostics,
)


@dataclass(frozen=True, slots=True)
class PathologyConfig:
    fatal_threshold: float = 8.0
    increase: float = 1.0
    recovery: float = 0.65
    communication_floor: float = 0.04
    boundary_fraction: float = 0.80
    homogenization_variance: float = 1e-4
    # A brief probe is allowed to make a resting state non-uniform, but it
    # cannot repeatedly reset this counter.  A network that always collapses
    # back to one shared vector is non-functional, not a viable equilibrium.
    rest_homogenization_steps: int = 6
    response_floor: float = 2e-3
    propagation_floor: float = 5e-4
    distinguishability_floor: float = 2e-3
    recovery_limit: float = 0.35
    one_direction_steps: int = 12
    # A direction is only pathological once the coordinate is materially near
    # the +/- 4 safety boundary. Ordinary transient adjustment is allowed.
    node_growth_alert: float = 3.2
    node_growth_delta: float = 1e-4
    absolute_node_limit: float = 4.0
    absolute_edge_limit: float = 12.0
    # These are survival tests, not constraints on the edge rule.  Edge
    # dynamics remain entirely encoded by the genome.
    edge_activity_delta: float = 1e-5
    edge_saturation_strength: float = .98
    edge_saturation_fraction: float = .80
    edge_growth_alert: float = 1.0
    edge_growth_delta: float = 1e-4
    edge_growth_steps: int = 12
    edge_activity_grace_steps: int = 8


@dataclass(frozen=True, slots=True)
class ProbeConfig:
    interval: int = 8
    duration: int = 2
    amplitude: float = 0.08


@dataclass(frozen=True, slots=True)
class CurriculumLevel:
    lifetime: int
    disturbance_frequency: int = 0
    disturbance_strength: float = 0.0
    compound_disturbances: bool = False
    # Retained only so old persisted scenario records remain readable.  New
    # survival dynamics have no external-input channel and ignore this field.
    input_scale: float = 0.0
    graph_nodes: int = 8
    mean_degree: float = 3.0


@dataclass(frozen=True, slots=True)
class AsyncEvolutionConfig:
    slots: int = 8
    replicas: int = 3
    result_batch_size: int = 8
    # Normal survival training has no tick cap: completion is exclusively the
    # final-stage stable-population gate.  A finite cap is retained only for
    # diagnostics, tests, or an explicit caller-requested interruption.
    max_ticks: int | None = None
    # A reporting checkpoint, not a completion criterion.  Training remains in
    # its current curriculum stage until a stable survivor population exists.
    candidate_budget: int | None = None
    seed: int = 41
    target: EvolutionTarget = "joint"
    architecture: RuleArchitecture = RuleArchitecture(state_width=1, hidden_width=8)
    edge_architecture: EdgeArchitecture | None = EdgeArchitecture(
        node_state_width=1, latent_width=2, hidden_width=12
    )
    levels: tuple[CurriculumLevel, ...] = (
        CurriculumLevel(20),
        CurriculumLevel(40, 10, .12, graph_nodes=10),
        CurriculumLevel(70, 7, .20, graph_nodes=12, mean_degree=4.0),
    )
    pathology: PathologyConfig = PathologyConfig()
    probes: ProbeConfig = ProbeConfig()
    curriculum_window: int = 20
    curriculum_pass_fraction: float = .60
    stable_population_size: int = 4
    censor_interval: int = 5
    elite_size: int = 4
    initial_state_scale: float = .12
    # Final-stage candidates must also remain functional on unseen graphs of
    # the default Live size.  This is a viability gate, not another objective.
    deployment_validation_replicas: int = 3
    deployment_validation_nodes: int = 24
    deployment_validation_mean_degree: float = 5.0
    deployment_autonomous_steps: int = 200
    # Held-out replicas are independent.  None uses one worker per replica;
    # set this to 1 to retain serial validation or cap CPU use explicitly.
    deployment_validation_workers: int | None = None
    # CMA-ES is the sole proposal mechanism.  The archive records survival
    # evidence and deployment candidates; it never breeds a hand-mutated
    # offspring population.
    initial_sigma: float = .35
    initial_genomes: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self) -> None:
        if (
            self.slots < 1
            or self.replicas < 1
            or self.result_batch_size < 2
            or self.stable_population_size < 1
            or self.deployment_validation_replicas < 1
            or self.deployment_validation_nodes < 2
            or self.initial_state_scale <= 0
            or self.deployment_autonomous_steps < 1
        ):
            raise ValueError("slot, replica, optimizer, and population counts must be positive")
        if self.candidate_budget is not None and self.candidate_budget < 1:
            raise ValueError("candidate_budget must be positive when provided")
        if self.max_ticks is not None and self.max_ticks < 1:
            raise ValueError("max_ticks must be positive when provided")
        if not self.levels or any(level.lifetime < 1 for level in self.levels):
            raise ValueError("at least one positive curriculum lifetime is required")
        if self.target in {"edge", "joint"} and self.edge_architecture is None:
            raise ValueError("edge architecture is required for edge or joint evolution")
        if self.deployment_validation_mean_degree > self.deployment_validation_nodes - 1:
            raise ValueError("deployment validation mean degree cannot exceed nodes - 1")
        if (
            self.deployment_validation_workers is not None
            and self.deployment_validation_workers < 1
        ):
            raise ValueError("deployment_validation_workers must be positive when provided")


@dataclass(frozen=True, slots=True)
class Scenario:
    replica: int
    graph_seed: int
    initial_state_seed: int
    input_seed: int
    nodes: int
    mean_degree: float
    input_scale: float


@dataclass(frozen=True, slots=True)
class ScenarioBank:
    bank_id: str
    level: int
    scenarios: tuple[Scenario, ...]

    @classmethod
    def create(cls, seed: int, level_index: int, replicas: int, level: CurriculumLevel) -> "ScenarioBank":
        # No candidate or genome identifier participates in these seeds.
        base = seed + 100_003 * level_index
        scenarios = tuple(
            Scenario(
                replica=index,
                graph_seed=base + 101 * index,
                initial_state_seed=base + 211 * index + 1,
                input_seed=base + 307 * index + 2,
                nodes=level.graph_nodes + (index % 2),
                mean_degree=level.mean_degree,
                input_scale=level.input_scale,
            )
            for index in range(replicas)
        )
        return cls(f"level-{level_index}-seed-{seed}", level_index, scenarios)


@dataclass(slots=True)
class ProbeSummary:
    step: int
    response: float
    propagation: float
    distinguishability: float
    recovered: bool
    coordinate_response: tuple[float, ...] = ()
    coordinate_propagation: tuple[float, ...] = ()
    coordinate_distinguishability: tuple[float, ...] = ()
    coordinate_recovered: tuple[bool, ...] = ()


@dataclass(slots=True)
class HealthMonitor:
    config: PathologyConfig
    burdens: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    peak_burdens: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    violation_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    probe_count: int = 0
    minimum_response: float | None = None
    minimum_propagation: float | None = None
    minimum_distinguishability: float | None = None
    all_probes_recovered: bool = True
    one_direction_signs: list[int] = field(default_factory=list)
    one_direction_runs: list[int] = field(default_factory=list)
    node_growth_runs: list[int] = field(default_factory=list)
    rest_homogenization_runs: list[int] = field(default_factory=list)
    edge_ever_active: bool = False
    edge_growth_runs: list[int] = field(default_factory=list)
    death_cause: str | None = None
    death_time: int | None = None

    def _burden(self, name: str, violation: bool, amount: float = 1.0) -> None:
        current = self.burdens[name]
        if violation:
            self.violation_counts[name] += 1
        self.burdens[name] = (
            current + self.config.increase * amount
            if violation
            else max(0.0, current - self.config.recovery)
        )
        self.peak_burdens[name] = max(self.peak_burdens[name], self.burdens[name])

    def observe(
        self,
        step: int,
        previous: NetworkState,
        current: NetworkState,
        strengths: Sequence[float],
        diagnostics: TransitionDiagnostics,
        probe: ProbeSummary | None = None,
        simulator_failed: bool = False,
        edge_gates: Sequence[Sequence[float]] | None = None,
        resting: bool = True,
    ) -> str | None:
        values = [value for row in current.node for vector in row for value in vector]
        edges = [value for row in current.edge for vector in row for value in vector]
        if simulator_failed:
            return self._die("simulator_failure", step)
        if diagnostics.nonfinite_proposals or any(not isfinite(value) for value in values + edges):
            return self._die("nonfinite", step)
        if (
            diagnostics.state_clipped
            or any(abs(value) > self.config.absolute_node_limit for value in values)
            or any(abs(value) > self.config.absolute_edge_limit for value in edges)
        ):
            return self._die("absolute_safety_limit", step)

        prior_values = [value for row in previous.node for vector in row for value in vector]
        width = len(current.node[0][0]) if current.node and current.node[0] else 0
        coordinate_values = [values[coordinate::width] for coordinate in range(width)] if width else []
        coordinate_prior_values = [prior_values[coordinate::width] for coordinate in range(width)] if width else []
        coordinate_means = [fmean(items) if items else 0.0 for items in coordinate_values]
        prior_coordinate_means = [fmean(items) if items else 0.0 for items in coordinate_prior_values]
        coordinate_variances = [
            fmean((value - mean) ** 2 for value in items) if items else 0.0
            for items, mean in zip(coordinate_values, coordinate_means, strict=True)
        ]
        if len(self.one_direction_signs) != width:
            self.one_direction_signs = [0] * width
            self.one_direction_runs = [0] * width
            self.node_growth_runs = [0] * width
            self.rest_homogenization_runs = [0] * width
        for coordinate, (mean, prior_mean) in enumerate(
            zip(coordinate_means, prior_coordinate_means, strict=True)
        ):
            delta = mean - prior_mean
            sign = 1 if delta > 1e-7 else -1 if delta < -1e-7 else 0
            if sign and sign == self.one_direction_signs[coordinate]:
                self.one_direction_runs[coordinate] += 1
            else:
                self.one_direction_signs[coordinate] = sign
                self.one_direction_runs[coordinate] = int(bool(sign))
            magnitude = fmean(abs(value) for value in coordinate_values[coordinate])
            prior_magnitude = fmean(
                abs(value) for value in coordinate_prior_values[coordinate]
            )
            if magnitude - prior_magnitude > self.config.node_growth_delta:
                self.node_growth_runs[coordinate] += 1
            else:
                self.node_growth_runs[coordinate] = 0
            if resting:
                self.rest_homogenization_runs[coordinate] = (
                    self.rest_homogenization_runs[coordinate] + 1
                    if coordinate_variances[coordinate] < self.config.homogenization_variance
                    else 0
                )

        communication_collapsed = (
            not strengths or fmean(strengths) < self.config.communication_floor
        )
        self._burden("communication_collapse", communication_collapsed)
        self._burden(
            "boundary_saturation",
            any(
                fmean(abs(value) > .95 * self.config.absolute_node_limit for value in items)
                >= self.config.boundary_fraction
                for items in coordinate_values
            ),
        )
        self._burden(
            "state_homogenization",
            any(variance < self.config.homogenization_variance for variance in coordinate_variances),
        )
        if any(run >= self.config.rest_homogenization_steps for run in self.rest_homogenization_runs):
            return self._die("rest_state_homogenization", step)
        self._burden(
            "one_direction_degeneration",
            any(
                run >= self.config.one_direction_steps
                and growth_run >= self.config.one_direction_steps
                and magnitude >= self.config.node_growth_alert
                for run, growth_run, magnitude in zip(
                    self.one_direction_runs,
                    self.node_growth_runs,
                    (fmean(abs(value) for value in items) for items in coordinate_values),
                    strict=True,
                )
            ),
        )

        prior_edges = [value for row in previous.edge for vector in row for value in vector]
        edge_updates = [
            abs(current_value - previous_value)
            for previous_value, current_value in zip(prior_edges, edges, strict=True)
        ]
        edge_activity = fmean(edge_updates) if edge_updates else 0.0
        if edge_activity > self.config.edge_activity_delta:
            self.edge_ever_active = True
        # An edge rule that has never moved its latent state is a fixed
        # communication ablation, even if the node rule itself is healthy.
        self._burden(
            "edge_dynamics_inactive",
            bool(edges)
            and step > self.config.edge_activity_grace_steps
            and not self.edge_ever_active,
        )

        edge_width = len(current.edge[0][0]) if current.edge and current.edge[0] else 0
        coordinate_edges = [edges[coordinate::edge_width] for coordinate in range(edge_width)]
        prior_coordinate_edges = [
            prior_edges[coordinate::edge_width] for coordinate in range(edge_width)
        ]
        if len(self.edge_growth_runs) != edge_width:
            self.edge_growth_runs = [0] * edge_width
        edge_magnitudes = [
            fmean(abs(value) for value in coordinate_values) if coordinate_values else 0.0
            for coordinate_values in coordinate_edges
        ]
        prior_edge_magnitudes = [
            fmean(abs(value) for value in coordinate_values) if coordinate_values else 0.0
            for coordinate_values in prior_coordinate_edges
        ]
        for coordinate, (magnitude, prior_magnitude) in enumerate(
            zip(edge_magnitudes, prior_edge_magnitudes, strict=True)
        ):
            self.edge_growth_runs[coordinate] = (
                self.edge_growth_runs[coordinate] + 1
                if magnitude - prior_magnitude > self.config.edge_growth_delta
                else 0
            )
        # ``strengths`` is a scalar visual summary.  Safety must instead
        # inspect every communication coordinate, so one saturated gate cannot
        # be hidden by another coordinate's healthy average.
        gate_vectors = edge_gates if edge_gates is not None else tuple((value,) for value in strengths)
        gate_width = len(gate_vectors[0]) if gate_vectors else 0
        saturated_edges = any(
            sum(
                value <= 1.0 - self.config.edge_saturation_strength
                or value >= self.config.edge_saturation_strength
                for value in (vector[coordinate] for vector in gate_vectors)
            )
            / len(gate_vectors)
            >= self.config.edge_saturation_fraction
            for coordinate in range(gate_width)
        )
        # Saturation is a dead communication state, and continued latent
        # growth while saturated is an earlier, directional runaway signal.
        self._burden(
            "edge_gate_saturation",
            saturated_edges
            and not communication_collapsed
            and not any(run >= self.config.edge_growth_steps for run in self.edge_growth_runs),
        )
        self._burden(
            "edge_runaway_growth",
            any(
                magnitude >= self.config.edge_growth_alert
                and run >= self.config.edge_growth_steps
                for magnitude, run in zip(edge_magnitudes, self.edge_growth_runs, strict=True)
            ),
        )
        if probe is not None:
            self.probe_count += 1
            self.minimum_response = (
                probe.response
                if self.minimum_response is None
                else min(self.minimum_response, probe.response)
            )
            self.minimum_propagation = (
                probe.propagation
                if self.minimum_propagation is None
                else min(self.minimum_propagation, probe.propagation)
            )
            self.minimum_distinguishability = (
                probe.distinguishability
                if self.minimum_distinguishability is None
                else min(self.minimum_distinguishability, probe.distinguishability)
            )
            self.all_probes_recovered = self.all_probes_recovered and probe.recovered
            responses = probe.coordinate_response or (probe.response,)
            propagations = probe.coordinate_propagation or (probe.propagation,)
            distinguishabilities = (
                probe.coordinate_distinguishability or (probe.distinguishability,)
            )
            recovered = probe.coordinate_recovered or (probe.recovered,)
            self._burden(
                "input_unresponsive",
                any(value < self.config.response_floor for value in responses),
            )
            self._burden(
                "communication_unresponsive",
                any(value < self.config.propagation_floor for value in propagations),
            )
            self._burden(
                "trajectory_indistinguishable",
                any(value < self.config.distinguishability_floor for value in distinguishabilities),
            )
            self._burden("disturbance_unrecovered", not all(recovered))
        # Probe-derived health evidence is updated only by another probe.
        # Absence of a probe is not evidence of functional recovery.  The
        # monitor deliberately keeps aggregates only; no training trajectory is
        # retained unless a user later requests a debug reconstruction.
        fatal = [(value, name) for name, value in self.burdens.items() if value >= self.config.fatal_threshold]
        return self._die(max(fatal)[1], step) if fatal else None

    def _die(self, cause: str, step: int) -> str:
        self.death_cause, self.death_time = cause, step
        return cause

    @property
    def normalized_burden(self) -> float:
        return max(self.burdens.values(), default=0.0) / self.config.fatal_threshold


@dataclass(slots=True)
class ReplicaRuntime:
    scenario: Scenario
    simulation: Simulation
    config: SimulationConfig
    state: NetworkState
    monitor: HealthMonitor
    age: int = 0
    active_probe_start: int | None = None
    probe_node: int | None = None
    probe_packet: tuple[float, ...] | None = None
    probe_baseline: NetworkState | None = None
    shadow_state: NetworkState | None = None
    probe_peak_response: list[float] = field(default_factory=list)
    probe_peak_propagation: list[float] = field(default_factory=list)
    probe_peak_distinguishability: list[float] = field(default_factory=list)
    diagnostics: TransitionDiagnostics = field(default_factory=TransitionDiagnostics)
    fatal_cause: str | None = None

    def _gaussian_state_packet(
        self, step: int, standard_deviation: float, salt: int
    ) -> tuple[int, tuple[float, ...]]:
        """Return one reproducible, local, coordinate-wise state mutation."""
        rng = Random(self.scenario.initial_state_seed + 1_000_003 * step + salt)
        node = rng.randrange(self.simulation.graph.n_nodes)
        return (
            node,
            tuple(rng.gauss(0.0, standard_deviation) for _ in range(self.simulation.node_rule.state_width)),
        )

    def advance(self, level: CurriculumLevel, probe_config: ProbeConfig) -> None:
        if self.fatal_cause:
            return
        previous = deepcopy(self.state)
        width = self.simulation.node_rule.state_width
        external = [[(0.0,) * width for _ in range(self.simulation.graph.n_nodes)]]
        disturbances: list[ImpulseInjection] = []
        probe_active = self.active_probe_start is not None
        if level.disturbance_frequency and self.age and self.age % level.disturbance_frequency == 0:
            # A rare local perturbation: random node and independent Gaussian
            # displacement in every state coordinate.  Its seed is scenario
            # derived, so all genomes still receive an identical test.
            node, packet = self._gaussian_state_packet(
                self.age, level.disturbance_strength, salt=17
            )
            disturbances.append(ImpulseInjection(self.age, (node,), packet))
        if self.active_probe_start is None and self.age and self.age % probe_config.interval == 0:
            self.active_probe_start = self.age
            probe_active = True
            self.probe_node, self.probe_packet = self._gaussian_state_packet(
                self.age, probe_config.amplitude, salt=31
            )
            self.probe_baseline = deepcopy(self.state)
            self.shadow_state = deepcopy(self.state)
            self.probe_peak_response = [0.0] * width
            self.probe_peak_propagation = [0.0] * width
            self.probe_peak_distinguishability = [0.0] * width
        shadow_disturbances = list(disturbances)
        if self.active_probe_start == self.age:
            assert self.probe_node is not None and self.probe_packet is not None
            disturbances.append(ImpulseInjection(self.age, (self.probe_node,), self.probe_packet))
            shadow_disturbances.append(
                ImpulseInjection(self.age, (self.probe_node,), tuple(-value for value in self.probe_packet))
            )
        prior_nonfinite = self.diagnostics.nonfinite_proposals
        prior_clipped = self.diagnostics.state_clipped
        try:
            self.state = self.simulation._step(
                self.state, external, self.age, self.config, tuple(disturbances), self.diagnostics, None
            )
            if self.shadow_state is not None:
                shadow_diagnostics = TransitionDiagnostics()
                self.shadow_state = self.simulation._step(
                    self.shadow_state, external, self.age, self.config, tuple(shadow_disturbances), shadow_diagnostics, None
                )
                self.diagnostics.nonfinite_proposals += shadow_diagnostics.nonfinite_proposals
                self.diagnostics.state_clipped += shadow_diagnostics.state_clipped
                self._update_probe_peaks()
        except Exception:
            self.fatal_cause = self.monitor.observe(
                self.age + 1, previous, previous, (), self.diagnostics, simulator_failed=True
            )
            return
        self.age += 1
        step_diagnostics = TransitionDiagnostics(
            nonfinite_proposals=self.diagnostics.nonfinite_proposals - prior_nonfinite,
            state_clipped=self.diagnostics.state_clipped - prior_clipped,
        )
        probe = None
        if (
            self.active_probe_start is not None
            and self.age - self.active_probe_start >= 2 * probe_config.duration
            and self.probe_baseline is not None
            and self.shadow_state is not None
        ):
            probe = self._probe_summary()
            self.active_probe_start = None
            self.probe_node = None
            self.probe_packet = None
            self.probe_baseline = None
            self.shadow_state = None
        strengths = [
            self.simulation.edge_rule.communication_strength(vector)
            for row in self.state.edge
            for vector in row
        ]
        edge_gates = (
            [
                self.simulation.edge_rule.communication_gates(vector)
                for row in self.state.edge
                for vector in row
            ]
            if self.simulation.edge_rule.state_width
            else ()
        )
        self.fatal_cause = self.monitor.observe(
            self.age, previous, self.state, strengths, step_diagnostics, probe,
            edge_gates=edge_gates, resting=not probe_active and not disturbances,
        )

    def _probe_summary(self) -> ProbeSummary:
        assert self.probe_baseline is not None and self.shadow_state is not None
        base = self.probe_baseline.node[0]
        live = self.state.node[0]
        width = self.simulation.node_rule.state_width
        baseline_scale = [
            fmean(abs(vector[coordinate]) for vector in base) if base else 0.0
            for coordinate in range(width)
        ]
        current_scale = [
            fmean(abs(vector[coordinate]) for vector in live) if live else 0.0
            for coordinate in range(width)
        ]
        recovered = tuple(
            abs(current - baseline) <= self.monitor.config.recovery_limit
            for baseline, current in zip(baseline_scale, current_scale, strict=True)
        )
        return ProbeSummary(
            self.age,
            min(self.probe_peak_response, default=0.0),
            min(self.probe_peak_propagation, default=0.0),
            min(self.probe_peak_distinguishability, default=0.0),
            all(recovered),
            tuple(self.probe_peak_response),
            tuple(self.probe_peak_propagation),
            tuple(self.probe_peak_distinguishability),
            recovered,
        )

    def _update_probe_peaks(self) -> None:
        assert self.probe_baseline is not None and self.shadow_state is not None
        base = self.probe_baseline.node[0]
        live = self.state.node[0]
        shadow = self.shadow_state.node[0]
        assert self.probe_node is not None
        for coordinate in range(self.simulation.node_rule.state_width):
            response = abs(
                live[self.probe_node][coordinate] - base[self.probe_node][coordinate]
            )
            propagation_values = [
                abs(live_vector[coordinate] - base_vector[coordinate])
                for node, (base_vector, live_vector) in enumerate(zip(base, live, strict=True))
                if node != self.probe_node
            ]
            distinguishability = fmean(
                abs(live_vector[coordinate] - shadow_vector[coordinate])
                for live_vector, shadow_vector in zip(live, shadow, strict=True)
            )
            self.probe_peak_response[coordinate] = max(
                self.probe_peak_response[coordinate], response
            )
            self.probe_peak_propagation[coordinate] = max(
                self.probe_peak_propagation[coordinate],
                fmean(propagation_values) if propagation_values else 0.0,
            )
            self.probe_peak_distinguishability[coordinate] = max(
                self.probe_peak_distinguishability[coordinate], distinguishability
            )

    def summary(self) -> dict[str, object]:
        values = [value for vector in self.state.node[0] for value in vector]
        edges = [value for vector in self.state.edge[0] for value in vector]
        return {
            "scenario": asdict(self.scenario),
            "age": self.age,
            "death_cause": self.fatal_cause,
            "normalized_pathology_burden": self.monitor.normalized_burden,
            "current_pathology_burdens": dict(self.monitor.burdens),
            "peak_pathology_burdens": dict(self.monitor.peak_burdens),
            "pathology_violation_counts": dict(self.monitor.violation_counts),
            "probe_count": self.monitor.probe_count,
            "responsiveness": self.monitor.minimum_response or 0.0,
            "propagation": self.monitor.minimum_propagation or 0.0,
            "distinguishability": self.monitor.minimum_distinguishability or 0.0,
            "recovered": self.monitor.probe_count > 0 and self.monitor.all_probes_recovered,
            "final_node_statistics": _stats(values),
            "final_edge_statistics": _stats(edges),
            "update_cost": self.diagnostics.components / max(1, self.age),
        }


@dataclass(frozen=True, slots=True)
class CandidateProposal:
    genome: tuple[float, ...]
    source: Literal["cma", "initial"]
    sample_id: int | None
    optimizer_update: int
    lineage_id: int | None = None


@dataclass(slots=True)
class CandidateSlot:
    slot: int
    candidate_id: int
    proposal: CandidateProposal
    bank: ScenarioBank
    level: int
    replicas: list[ReplicaRuntime]
    milestone_history: list[dict[str, object]] = field(default_factory=list)
    published_ages: set[int] = field(default_factory=set)

    @property
    def age(self) -> int:
        return min(replica.age for replica in self.replicas)

    @property
    def dead(self) -> bool:
        return any(replica.fatal_cause is not None for replica in self.replicas)

    def rank_key(self, graduated: bool = False) -> tuple[float, ...]:
        summaries = [replica.summary() for replica in self.replicas]
        responsive = min(float(item["responsiveness"]) for item in summaries)
        propagated = min(float(item["propagation"]) for item in summaries)
        distinguishable = min(float(item["distinguishability"]) for item in summaries)
        recovered = min(float(bool(item["recovered"])) for item in summaries)
        burden = max(float(item["normalized_pathology_burden"]) for item in summaries)
        update_cost = max(float(item["update_cost"]) for item in summaries)
        # Feasibility and function precede lifetime. A frozen survivor therefore
        # stays below a responsive survivor in the same curriculum level.
        health = self.replicas[0].monitor.config
        functional = float(
            responsive >= health.response_floor
            and propagated >= health.propagation_floor
            and distinguishable >= health.distinguishability_floor
            and recovered > 0
        )
        return (
            float(self.level + int(graduated)),
            functional,
            float(self.age),
            -burden,
            responsive,
            propagated,
            recovered,
            distinguishable,
            -update_cost,
        )


class SteadyStateCMA:
    """One comparable CMA-ES cohort at a time.

    Candidates can finish at different times, but ``tell`` is deliberately
    delayed until the entire population was evaluated on one scenario bank.
    A curriculum transition cancels an incomplete cohort and starts a fresh,
    comparable one for the new stage.
    """

    def __init__(self, dimension: int, batch_size: int, sigma: float, seed: int) -> None:
        self.optimizer = CMAES(CMAESConfig(dimension, batch_size, sigma, seed))
        self.batch_size = batch_size
        self.pending: deque[tuple[int, tuple[float, ...]]] = deque()
        self.results: list[tuple[int, tuple[float, ...], float]] = []
        self.inflight: set[int] = set()
        self.bank_id: str | None = None
        self.next_sample_id = 0
        self.update_count = 0
        self.observed_samples: set[int] = set()
        self.cancelled_samples = 0
        self._refill()

    def _refill(self) -> None:
        for genome in self.optimizer.ask():
            self.pending.append((self.next_sample_id, genome))
            self.next_sample_id += 1

    def ask(self) -> tuple[int, tuple[float, ...]] | None:
        if not self.pending:
            return None
        sample_id, genome = self.pending.popleft()
        self.inflight.add(sample_id)
        return sample_id, genome

    def begin_stage(self, bank_id: str) -> None:
        """Discard incomparable unfinished evidence and issue a fresh cohort."""
        if self.bank_id == bank_id:
            return
        self.cancelled_samples += len(self.inflight) + len(self.results)
        self.pending.clear()
        self.inflight.clear()
        self.results.clear()
        self.bank_id = bank_id
        self._refill()

    @property
    def waiting_for_results(self) -> bool:
        return bool(self.inflight or self.results)

    def progress(self) -> dict[str, int | str | None]:
        return {
            "bank_id": self.bank_id,
            "batch_size": self.batch_size,
            "completed": len(self.results),
            "inflight": len(self.inflight),
            "unissued": len(self.pending),
            "cancelled": self.cancelled_samples,
        }

    def observe(
        self, bank_id: str, sample_id: int | None, genome: tuple[float, ...], lifetime: float
    ) -> bool:
        if sample_id is None or sample_id in self.observed_samples:
            return False
        if self.bank_id is None:
            self.bank_id = bank_id
        if self.bank_id != bank_id or sample_id not in self.inflight:
            return False
        self.observed_samples.add(sample_id)
        self.inflight.remove(sample_id)
        self.results.append((sample_id, genome, float(lifetime)))
        if len(self.results) < self.batch_size:
            return False
        records = self.results
        self.results = []
        # CMA-ES sees survival time only.  Death causes, probe measures,
        # pathology burden, and archive ranking never enter this comparison.
        # The sample id is only a deterministic tie-breaker.
        ordered = sorted(records, key=lambda item: (item[2], item[0]))
        scalar_by_id = {sample: float(index) for index, (sample, _, _) in enumerate(ordered)}
        population = [genome_value for sample, genome_value, _ in records]
        fitness = [scalar_by_id[sample] for sample, _, _ in records]
        self.optimizer.tell(population, fitness)
        self.update_count += 1
        self._refill()
        return True


class EliteArchive:
    def __init__(self, size: int) -> None:
        self.size = size
        self.records: list[dict[str, object]] = []
        self.changes = 0

    def consider(self, record: dict[str, object]) -> bool:
        before = [(item["candidate_id"], item["rank_key"]) for item in self.records]
        combined = self.records + [record]
        combined.sort(key=lambda item: tuple(item["rank_key"]), reverse=True)
        unique: list[dict[str, object]] = []
        seen: set[tuple[float, ...]] = set()
        for item in combined:
            genome = tuple(float(value) for value in item["genome"])
            if genome not in seen:
                seen.add(genome)
                unique.append(item)
        self.records = unique[: self.size]
        after = [(item["candidate_id"], item["rank_key"]) for item in self.records]
        changed = before != after
        self.changes += int(changed)
        return changed


class SurvivorArchive(EliteArchive):
    """One shared functional-survivor archive across curriculum levels.

    A functional graduate at a harder level has a strictly higher primary
    rank.  Its arrival removes lower-rank earlier-stage records immediately,
    rather than retaining them as permanent evidence leaders.
    """

    def consider(self, record: dict[str, object]) -> bool:
        primary_rank = float(record["rank_key"][0])
        self.records = [
            item for item in self.records if float(item["rank_key"][0]) >= primary_rank
        ]
        return super().consider(record)


class AsyncEvolutionRunner:
    def __init__(self, config: AsyncEvolutionConfig) -> None:
        self.config = config
        self.codec = GenomeCodec(config.architecture, config.edge_architecture, config.target)
        self.rng = Random(config.seed)
        self.optimizer = SteadyStateCMA(
            self.codec.dimension, config.result_batch_size, config.initial_sigma, config.seed
        )
        self.elites = EliteArchive(config.elite_size)
        self.survivors = SurvivorArchive(config.elite_size)
        self.level = 0
        self.banks = [
            ScenarioBank.create(config.seed, index, config.replicas, level)
            for index, level in enumerate(config.levels)
        ]
        self.slots: list[CandidateSlot] = []
        self.archive: list[dict[str, object]] = []
        self.censored: list[dict[str, object]] = []
        self.stage_survivors: dict[int, list[dict[str, object]]] = defaultdict(list)
        self.next_candidate_id = 0
        self.initial = deque(config.initial_genomes)
        self.utilization_samples: list[float] = []
        self.ticks_elapsed = 0
        self.stop_reason = "running"
        self.optimizer.begin_stage(self.banks[self.level].bank_id)
        self.stage_optimizer_updates_at_entry = self.optimizer.update_count

    def run(
        self,
        output: Path | None = None,
        progress: Callable[[dict[str, object]], None] | None = None,
    ) -> dict[str, object]:
        self._fill_slots()
        tick = 0
        while self.config.max_ticks is None or tick < self.config.max_ticks:
            self.utilization_samples.append(len(self.slots) / self.config.slots)
            finished: list[CandidateSlot] = []
            for slot in list(self.slots):
                level = self.config.levels[slot.level]
                for replica in slot.replicas:
                    replica.advance(level, self.config.probes)
                if slot.dead:
                    self._finish(slot, "death", progress=progress, tick=tick + 1)
                    finished.append(slot)
                elif slot.age >= level.lifetime:
                    slot.milestone_history.append(
                        {"level": slot.level, "time": slot.age, "kind": "graduation"}
                    )
                    self._finish(slot, "graduation", progress=progress, tick=tick + 1)
                    finished.append(slot)
            self.slots = [slot for slot in self.slots if slot not in finished]
            level_before_transition = self.level
            completed = self._maybe_advance_curriculum()
            if completed:
                self.ticks_elapsed = tick + 1
                break
            if self.level != level_before_transition:
                # The previous stage has supplied one full CMA cohort.  Any
                # remaining non-CMA references are now censored rather than
                # silently carried into an incomparable L2 evaluation.
                for slot in self.slots:
                    self._publish_censored(slot, "stage_transition")
                self.slots = []
                # Stage-one graduates are not parents.  Once the curriculum
                # advances, retaining them cannot affect future proposals.
                self.stage_survivors.pop(level_before_transition, None)
                self.optimizer.begin_stage(self.banks[self.level].bank_id)
                self.stage_optimizer_updates_at_entry = self.optimizer.update_count
            self._fill_slots()
            self.ticks_elapsed = tick + 1
            if progress is not None:
                progress(self._progress_snapshot(tick + 1))
            tick += 1
        if self.stop_reason == "running":
            self.stop_reason = "stage_not_passed_tick_limit"
        for slot in self.slots:
            self._publish_censored(slot, "run_stop")
        report = self.report()
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            (output / "candidate_archive.json").write_text(
                json.dumps(self.archive, indent=2, sort_keys=True), encoding="utf-8"
            )
            (output / "elite_archive.json").write_text(
                json.dumps(self.elites.records, indent=2, sort_keys=True), encoding="utf-8"
            )
            (output / "living_censored.json").write_text(
                json.dumps(self.censored, indent=2, sort_keys=True), encoding="utf-8"
            )
            (output / "diagnostic_report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
            )
        return report

    def _proposal(self) -> CandidateProposal | None:
        # Deterministic diagnostic references are evidence only.  Normal
        # training has no such queue, so every candidate is CMA-generated.
        if self.initial:
            return CandidateProposal(tuple(self.initial.popleft()), "initial", None, self.optimizer.update_count)
        asked = self.optimizer.ask()
        if asked is None:
            return None
        sample, genome = asked
        return CandidateProposal(genome, "cma", sample, self.optimizer.update_count)

    def _fill_slots(self) -> None:
        occupied = {slot.slot for slot in self.slots}
        for slot_index in range(self.config.slots):
            if slot_index in occupied:
                continue
            proposal = self._proposal()
            if proposal is None:
                return
            self.slots.append(self._new_slot(slot_index, proposal))

    def _progress_snapshot(
        self, tick: int, validation: dict[str, object] | None = None
    ) -> dict[str, object]:
        """Return a single UI-safe view of evolution, including final validation.

        Deployment validation is intentionally synchronous: its result is part
        of a candidate's comparable lifetime.  It must nevertheless remain
        observable so a long held-out test never looks like a stalled worker.
        """
        return {
            "tick": tick,
            "max_ticks": self.config.max_ticks,
            "report": self.report(),
            "slots": [
                {
                    "slot": slot.slot,
                    "candidate_id": slot.candidate_id,
                    "age": slot.age,
                    "level": slot.level,
                    "milestone": self.config.levels[slot.level].lifetime,
                    "source": slot.proposal.source,
                    "worst_burden": max(
                        replica.monitor.normalized_burden for replica in slot.replicas
                    ),
                }
                for slot in self.slots
            ],
            "validation": validation,
        }

    def _new_slot(self, slot_index: int, proposal: CandidateProposal | None = None) -> CandidateSlot:
        proposal = proposal or self._proposal()
        if proposal is None:
            raise RuntimeError("CMA cohort is awaiting comparable results")
        if proposal.lineage_id is None:
            proposal = replace(proposal, lineage_id=self.next_candidate_id)
        node_rule, edge_rule = self.codec.decode_groups(proposal.genome)
        if node_rule is None:
            from .baselines import HomeostaticRule
            node_rule = HomeostaticRule()
        replicas = []
        bank = self.banks[self.level]
        for scenario in bank.scenarios:
            graph = generate_random_graph(
                scenario.nodes, scenario.mean_degree, scenario.graph_seed
            )
            simulation = Simulation(graph, node_rule, edge_rule)
            simulation_config = SimulationConfig(
                steps=1, batch_size=1, max_abs_state=4.0, record_every=1
            )
            state = simulation.initial_state(1)
            init_rng = Random(scenario.initial_state_seed)
            state.node = [[
                tuple(
                    init_rng.gauss(0.0, self.config.initial_state_scale)
                    for _ in range(node_rule.state_width)
                )
                for _ in range(graph.n_nodes)
            ]]
            replicas.append(
                ReplicaRuntime(
                    scenario,
                    simulation,
                    simulation_config,
                    state,
                    HealthMonitor(self.config.pathology),
                )
            )
        candidate = CandidateSlot(
            slot_index, self.next_candidate_id, proposal, bank, self.level, replicas
        )
        self.next_candidate_id += 1
        return candidate

    def _publish_censored(
        self, slot: CandidateSlot, kind: Literal["run_stop", "stage_transition"]
    ) -> None:
        record = {
            "candidate_id": slot.candidate_id,
            "lineage_id": slot.proposal.lineage_id,
            "slot": slot.slot,
            "kind": kind,
            "right_censored": True,
            "age": slot.age,
            "level": slot.level,
            "bank_id": slot.bank.bank_id,
            "source": slot.proposal.source,
            "milestone": self.config.levels[slot.level].lifetime,
            "rank_key": list(slot.rank_key()),
        }
        self.censored.append(record)
        # Only lives still alive when the run stops are right-censored.  Periodic
        # health checks and stage graduation are observations, not censoring.

    def _validation_replicas(
        self,
        genome: tuple[float, ...],
        level: CurriculumLevel,
        probes: ProbeConfig,
        steps: int,
        *,
        candidate_id: int,
        phase: Literal["autonomous", "perturbed"],
        progress: Callable[[dict[str, object]], None] | None,
        tick: int,
    ) -> list[dict[str, object]]:
        """Run independent held-out replicas concurrently, preserving their order."""
        progress_events: SimpleQueue[tuple[int, int]] = SimpleQueue()

        def emit_progress(replica_index: int, step: int) -> None:
            if progress is not None:
                progress_events.put((replica_index, step))

        def flush_progress() -> None:
            if progress is None:
                return
            while True:
                try:
                    replica_index, step = progress_events.get_nowait()
                except Empty:
                    return
                progress(
                    self._progress_snapshot(
                        tick,
                        {
                            "candidate_id": candidate_id,
                            "phase": phase,
                            "replica": replica_index + 1,
                            "replicas": self.config.deployment_validation_replicas,
                            "step": step,
                            "steps": steps,
                        },
                    )
                )

        worker_count = min(
            self.config.deployment_validation_replicas,
            self.config.deployment_validation_workers
            or self.config.deployment_validation_replicas,
        )
        summaries: list[dict[str, object] | None] = [
            None
        ] * self.config.deployment_validation_replicas
        with ThreadPoolExecutor(
            max_workers=worker_count, thread_name_prefix="deployment-validation"
        ) as executor:
            pending = {
                executor.submit(
                    self._validation_replica,
                    genome,
                    level,
                    probes,
                    steps,
                    replica_index=replica_index,
                    progress=emit_progress,
                ): replica_index
                for replica_index in range(self.config.deployment_validation_replicas)
            }
            while pending:
                done, _ = wait(
                    pending, timeout=0.05, return_when=FIRST_COMPLETED
                )
                flush_progress()
                for future in done:
                    replica_index = pending.pop(future)
                    summaries[replica_index] = future.result()
        flush_progress()
        return [summary for summary in summaries if summary is not None]

    def _validation_replica(
        self,
        genome: tuple[float, ...],
        level: CurriculumLevel,
        probes: ProbeConfig,
        steps: int,
        *,
        replica_index: int,
        progress: Callable[[int, int], None],
    ) -> dict[str, object]:
        """Run one isolated held-out replica and report compact progress events."""
        node_rule, edge_rule = self.codec.decode_groups(genome)
        if node_rule is None:
            from .baselines import HomeostaticRule
            node_rule = HomeostaticRule()
        base_seed = self.config.seed + 700_001 + 10_007 * replica_index
        nodes = self.config.deployment_validation_nodes + (replica_index % 2)
        graph = generate_random_graph(
            nodes,
            min(self.config.deployment_validation_mean_degree, nodes - 1),
            base_seed,
        )
        simulation = Simulation(graph, node_rule, edge_rule)
        state = simulation.initial_state(1)
        initial_rng = Random(base_seed + 1)
        state.node = [[
            tuple(
                initial_rng.gauss(0.0, self.config.initial_state_scale)
                for _ in range(node_rule.state_width)
            )
            for _ in range(graph.n_nodes)
        ]]
        scenario = Scenario(
            replica=replica_index,
            graph_seed=base_seed,
            initial_state_seed=base_seed + 1,
            input_seed=base_seed + 2,
            nodes=nodes,
            mean_degree=min(self.config.deployment_validation_mean_degree, nodes - 1),
            input_scale=0.0,
        )
        replica = ReplicaRuntime(
            scenario,
            simulation,
            SimulationConfig(steps=1, batch_size=1, max_abs_state=4.0, record_every=1),
            state,
            HealthMonitor(self.config.pathology),
        )
        update_interval = max(1, steps // 20)
        while replica.age < steps and replica.fatal_cause is None:
            replica.advance(level, probes)
            if (
                replica.age == 1
                or replica.age % update_interval == 0
                or replica.fatal_cause is not None
            ):
                progress(replica_index, replica.age)
        return replica.summary()

    def _deployment_validation(
        self,
        genome: tuple[float, ...],
        *,
        candidate_id: int,
        progress: Callable[[dict[str, object]], None] | None,
        tick: int,
    ) -> dict[str, object]:
        """Require autonomous stability before testing recovery on held-out graphs."""
        level = self.config.levels[-1]
        autonomous_level = replace(
            level, disturbance_frequency=0, disturbance_strength=0.0
        )
        autonomous_probes = ProbeConfig(
            interval=self.config.deployment_autonomous_steps + 1,
            duration=self.config.probes.duration,
            amplitude=self.config.probes.amplitude,
        )
        autonomous = self._validation_replicas(
            genome,
            autonomous_level,
            autonomous_probes,
            self.config.deployment_autonomous_steps,
            candidate_id=candidate_id,
            phase="autonomous",
            progress=progress,
            tick=tick,
        )
        health = self.config.pathology
        autonomous_passed = all(
            summary["death_cause"] is None
            and int(summary["age"]) >= self.config.deployment_autonomous_steps
            and float(summary["normalized_pathology_burden"]) <= 1e-12
            for summary in autonomous
        )
        perturbed = (
            self._validation_replicas(
                genome,
                level,
                self.config.probes,
                level.lifetime,
                candidate_id=candidate_id,
                phase="perturbed",
                progress=progress,
                tick=tick,
            )
            if autonomous_passed else []
        )
        perturbed_passed = autonomous_passed and all(
            summary["death_cause"] is None
            and int(summary["age"]) >= level.lifetime
            and float(summary["normalized_pathology_burden"]) <= 1e-12
            and float(summary["responsiveness"]) >= health.response_floor
            and float(summary["propagation"]) >= health.propagation_floor
            and float(summary["distinguishability"]) >= health.distinguishability_floor
            and bool(summary["recovered"])
            for summary in perturbed
        )
        return {
            "passed": autonomous_passed and perturbed_passed,
            "autonomous": autonomous,
            "perturbed": perturbed,
            "autonomous_steps": self.config.deployment_autonomous_steps,
            "nodes": self.config.deployment_validation_nodes,
            "mean_degree": self.config.deployment_validation_mean_degree,
        }

    def _finish(
        self,
        slot: CandidateSlot,
        status: Literal["death", "graduation"],
        *,
        progress: Callable[[dict[str, object]], None] | None = None,
        tick: int = 0,
    ) -> bool:
        graduated = status == "graduation"
        deployment_validation: dict[str, object] | None = None
        if graduated and slot.level == len(self.config.levels) - 1:
            deployment_validation = self._deployment_validation(
                slot.proposal.genome,
                candidate_id=slot.candidate_id,
                progress=progress,
                tick=tick,
            )
        causes = [
            replica.fatal_cause for replica in slot.replicas if replica.fatal_cause is not None
        ]
        rank_key = slot.rank_key(graduated)
        if deployment_validation is not None and not bool(deployment_validation["passed"]):
            rank_key = (rank_key[0], 0.0, *rank_key[2:])
        cma_lifetime = float(slot.age)
        if deployment_validation is not None:
            # These are consecutive required survival exposures.  CMA-ES sees
            # only their total duration, never the failure label or metric.
            cma_lifetime += min(
                (float(summary["age"]) for summary in deployment_validation["autonomous"]),
                default=0.0,
            )
            cma_lifetime += min(
                (float(summary["age"]) for summary in deployment_validation["perturbed"]),
                default=0.0,
            )
        replica_results = [replica.summary() for replica in slot.replicas]
        functional_at_last_probe = bool(rank_key[1]) if len(rank_key) > 1 else False
        functional = graduated and functional_at_last_probe
        unresolved_burden = max(
            (float(replica["normalized_pathology_burden"]) for replica in replica_results),
            default=0.0,
        )
        record = {
            "candidate_id": slot.candidate_id,
            "slot": slot.slot,
            "status": status,
            "lineage_id": slot.proposal.lineage_id,
            "genome": list(slot.proposal.genome),
            "parameter_groups": self.codec.export_groups(slot.proposal.genome),
            "sampling": asdict(slot.proposal),
            "bank": asdict(slot.bank),
            "level": slot.level,
            "age": slot.age,
            "cma_lifetime": cma_lifetime,
            "death_time": slot.age if causes else None,
            "death_cause": causes[0] if causes else None,
            "milestone_history": slot.milestone_history,
            "per_replica_results": replica_results,
            "rank_key": list(rank_key),
            "functional": functional,
            "functional_at_last_probe": functional_at_last_probe,
            "deployment_validation": deployment_validation,
            # A model is deployable only after the final curriculum test. Earlier
            # graduations remain useful training evidence, but are not Live models.
            "live_eligible": (
                graduated
                and slot.level == len(self.config.levels) - 1
                and functional
                and unresolved_burden <= 1e-12
                and deployment_validation is not None
                and bool(deployment_validation["passed"])
            ),
        }
        self.archive.append(record)
        updated = self.optimizer.observe(
            slot.bank.bank_id,
            slot.proposal.sample_id,
            slot.proposal.genome,
            cma_lifetime,
        )
        if graduated and functional:
            self.stage_survivors[slot.level].append(record)
            self.stage_survivors[slot.level].sort(
                key=lambda item: tuple(item["rank_key"]), reverse=True
            )
            del self.stage_survivors[slot.level][
                max(self.config.elite_size, self.config.stable_population_size) :
            ]
            self.survivors.consider(record)
            self.elites.consider(record)
        return updated

    def _stable_records(self, level: int) -> list[dict[str, object]]:
        records = list(self.stage_survivors[level])
        records.sort(key=lambda item: tuple(item["rank_key"]), reverse=True)
        return records[: self.config.stable_population_size]

    def _maybe_advance_curriculum(self) -> bool:
        """Advance only after stable survivors and one complete CMA cohort."""
        if len(self._stable_records(self.level)) < self.config.stable_population_size:
            return False
        if self.optimizer.update_count <= self.stage_optimizer_updates_at_entry:
            return False
        if self.level + 1 >= len(self.config.levels):
            self.stop_reason = "final_stage_population_established"
            return True
        self.level += 1
        return False

    def report(self) -> dict[str, object]:
        deaths = [item for item in self.archive if item["status"] == "death"]
        graduations = [item for item in self.archive if item["status"] == "graduation"]
        lifetimes = [int(item["age"]) for item in self.archive]
        disagreements = []
        burden_ranges = []
        outcome_disagreements = []
        for item in self.archive:
            ages = [int(replica["age"]) for replica in item["per_replica_results"]]
            disagreements.append(max(ages) - min(ages))
            burdens = [
                float(replica["normalized_pathology_burden"])
                for replica in item["per_replica_results"]
            ]
            burden_ranges.append(max(burdens) - min(burdens))
            outcomes = [
                (replica["death_cause"] is None, replica["death_cause"])
                for replica in item["per_replica_results"]
            ]
            outcome_disagreements.append(len(set(outcomes)) > 1)
        responsive = sum(
            item["status"] == "graduation"
            and min(float(replica["responsiveness"]) for replica in item["per_replica_results"]) > 0
            for item in self.archive
        )
        nonresponsive = len(graduations) - responsive
        return {
            "schema_version": 1,
            "mode": "asynchronous_death_driven_joint_evolution",
            "active_slot_utilization": fmean(self.utilization_samples) if self.utilization_samples else 0.0,
            "active_slots": len(self.slots),
            "ticks_elapsed": self.ticks_elapsed,
            "tick_limit": self.config.max_ticks,
            "stop_reason": self.stop_reason,
            "candidate_budget": self.config.candidate_budget,
            "candidate_evidence_checkpoint_reached": (
                self.config.candidate_budget is not None
                and len(self.archive) >= self.config.candidate_budget
            ),
            "candidates_started": self.next_candidate_id,
            "completed_candidates": len(self.archive),
            "completed_replica_lives": len(self.archive) * self.config.replicas,
            "active_replica_lives": len(self.slots) * self.config.replicas,
            "deaths": len(deaths),
            "graduations": len(graduations),
            "proposals_by_source": dict(Counter(item["sampling"]["source"] for item in self.archive)),
            "deaths_per_cause": dict(Counter(item["death_cause"] for item in deaths)),
            "lifetime_distribution": {
                "count": len(lifetimes),
                "minimum": min(lifetimes, default=None),
                "mean": fmean(lifetimes) if lifetimes else None,
                "maximum": max(lifetimes, default=None),
            },
            "milestone_passage_rates": {
                str(level): (
                    fmean(item["status"] == "graduation" for item in self.archive if item["level"] == level)
                    if any(item["level"] == level for item in self.archive)
                    else None
                )
                for level in range(len(self.config.levels))
            },
            "right_censored_candidate_count": len(self.censored),
            "currently_living_right_censored": len(self.slots),
            "optimizer_updates": self.optimizer.update_count,
            "optimizer_update_frequency": self.optimizer.update_count / max(1, self.ticks_elapsed),
            "optimizer_batch_progress": self.optimizer.progress(),
            "replica_disagreement": {
                "mean_age_range": fmean(disagreements) if disagreements else 0.0,
                "maximum_age_range": max(disagreements, default=0),
                "mean_pathology_burden_range": (
                    fmean(burden_ranges) if burden_ranges else 0.0
                ),
                "outcome_disagreement_fraction": (
                    fmean(outcome_disagreements) if outcome_disagreements else 0.0
                ),
            },
            "viable_survivors": responsive,
            "nonresponsive_survivors": nonresponsive,
            "elite_archive_changes": self.elites.changes,
            "elite_archive_size": len(self.elites.records),
            "curriculum_level": self.level,
            "stable_population_size": self.config.stable_population_size,
            "survivor_archive_size": len(self.survivors.records),
            "stable_survivor_count": len(self._stable_records(self.level)),
            "stage_survivor_counts": {
                str(level): len(self._stable_records(level))
                for level in range(len(self.config.levels))
            },
        }

    @classmethod
    def replay_record(cls, record: dict[str, object], config: AsyncEvolutionConfig) -> dict[str, object]:
        replay = cls(
            replace(
                config,
                slots=1,
                max_ticks=int(record["age"]),
                initial_genomes=(tuple(float(value) for value in record["genome"]),),
            )
        )
        replay.level = int(record["level"])
        bank_data = record["bank"]
        replay.banks[replay.level] = ScenarioBank(
            str(bank_data["bank_id"]),
            int(bank_data["level"]),
            tuple(Scenario(**scenario) for scenario in bank_data["scenarios"]),
        )
        replay.slots = [replay._new_slot(0)]
        slot = replay.slots[0]
        for _ in range(int(record["age"])):
            for replica in slot.replicas:
                replica.advance(config.levels[slot.level], config.probes)
            if slot.dead:
                break
        return {
            "age": slot.age,
            "death_cause": next((replica.fatal_cause for replica in slot.replicas if replica.fatal_cause), None),
            "per_replica_results": [replica.summary() for replica in slot.replicas],
        }


def diagnostic_reference_genomes(codec: GenomeCodec) -> tuple[tuple[float, ...], ...]:
    """Return deterministic frozen, responsive, and drifting reference genomes."""
    frozen = [0.0] * codec.dimension
    responsive = [0.0] * codec.dimension
    if codec.node_dimension:
        architecture = codec.architecture
        # Hidden units form an approximate antisymmetric (input - state) pair.
        input_width = architecture.input_width
        if architecture.hidden_width >= 2:
            responsive[0] = -2.0
            responsive[input_width] = 2.0
            responsive[input_width + 2 * architecture.state_width] = -2.0
            output_start = architecture.hidden_width * input_width + architecture.hidden_width
            responsive[output_start] = 1.5
            responsive[output_start + 1] = -1.5
    drifting = responsive.copy()
    if codec.node_dimension:
        drifting[codec.node_dimension - codec.architecture.state_width] = .8
    return tuple(frozen), tuple(responsive), tuple(drifting)


def run_diagnostic_experiment(
    output: Path,
    seed: int = 41,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    architecture = RuleArchitecture(state_width=1, hidden_width=8)
    edge_architecture = EdgeArchitecture(node_state_width=1, latent_width=2, hidden_width=12)
    codec = GenomeCodec(architecture, edge_architecture, "joint")
    config = AsyncEvolutionConfig(
        slots=6,
        replicas=3,
        result_batch_size=4,
        max_ticks=80,
        seed=seed,
        architecture=architecture,
        edge_architecture=edge_architecture,
        target="joint",
        levels=(
            CurriculumLevel(16, graph_nodes=7),
            CurriculumLevel(28, 8, .10, graph_nodes=9),
        ),
        pathology=PathologyConfig(fatal_threshold=3.0),
        probes=ProbeConfig(interval=4, duration=2, amplitude=.10),
        curriculum_window=8,
        censor_interval=4,
        initial_genomes=diagnostic_reference_genomes(codec),
    )
    return run_async_experiment(output, config, progress)


def run_async_experiment(
    output: Path,
    config: AsyncEvolutionConfig,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Run and persist one configured asynchronous survival experiment."""
    report = AsyncEvolutionRunner(config).run(output, progress=progress)
    (output / "diagnostic_config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def async_config_from_dict(data: Mapping[str, object]) -> AsyncEvolutionConfig:
    """Restore a persisted asynchronous configuration for exact replay."""
    architecture = RuleArchitecture(**dict(data["architecture"]))
    raw_edge = data.get("edge_architecture")
    edge_architecture = EdgeArchitecture(**dict(raw_edge)) if raw_edge is not None else None
    return AsyncEvolutionConfig(
        slots=int(data["slots"]),
        replicas=int(data["replicas"]),
        result_batch_size=int(data["result_batch_size"]),
        max_ticks=(int(data["max_ticks"]) if data.get("max_ticks") is not None else None),
        candidate_budget=(
            int(data["candidate_budget"]) if data.get("candidate_budget") is not None else None
        ),
        seed=int(data["seed"]),
        target=str(data["target"]),  # type: ignore[arg-type]
        architecture=architecture,
        edge_architecture=edge_architecture,
        levels=tuple(CurriculumLevel(**dict(item)) for item in data["levels"]),
        pathology=PathologyConfig(**dict(data["pathology"])),
        probes=ProbeConfig(**dict(data["probes"])),
        curriculum_window=int(data["curriculum_window"]),
        curriculum_pass_fraction=float(data["curriculum_pass_fraction"]),
        stable_population_size=int(data.get("stable_population_size", 4)),
        censor_interval=int(data["censor_interval"]),
        elite_size=int(data["elite_size"]),
        deployment_validation_replicas=int(data.get("deployment_validation_replicas", 3)),
        deployment_validation_nodes=int(data.get("deployment_validation_nodes", 24)),
        deployment_validation_mean_degree=float(data.get("deployment_validation_mean_degree", 5.0)),
        deployment_autonomous_steps=int(data.get("deployment_autonomous_steps", 200)),
        deployment_validation_workers=(
            int(data["deployment_validation_workers"])
            if data.get("deployment_validation_workers") is not None
            else None
        ),
        initial_state_scale=float(data.get("initial_state_scale", .12)),
        initial_sigma=float(data["initial_sigma"]),
        initial_genomes=tuple(tuple(float(value) for value in genome) for genome in data["initial_genomes"]),
    )


def replay_archived_candidate(
    record: Mapping[str, object], config: AsyncEvolutionConfig, replica_index: int
) -> tuple[Graph, Trajectory, SimulationConfig, dict[str, object]]:
    """Re-run one archived candidate on its saved scenario until its exact stop age.

    This intentionally does not reuse a broad evaluation horizon.  The graph,
    initial state, input stream, probes, and curriculum disturbances all come
    from the completed candidate record and its diagnostic configuration.
    """
    replica_results = list(record["per_replica_results"])
    if replica_index < 0 or replica_index >= len(replica_results):
        raise IndexError("replica is outside the archived candidate")
    result = dict(replica_results[replica_index])
    scenario = Scenario(**dict(result["scenario"]))
    level_index = int(record["level"])
    if level_index < 0 or level_index >= len(config.levels):
        raise ValueError("archived curriculum level is unavailable")
    level = config.levels[level_index]
    codec = GenomeCodec(config.architecture, config.edge_architecture, config.target)
    genome = tuple(float(value) for value in record["genome"])
    node_rule, edge_rule = codec.decode_groups(genome)
    if node_rule is None:
        from .baselines import HomeostaticRule

        node_rule = HomeostaticRule()
    graph = generate_random_graph(scenario.nodes, scenario.mean_degree, scenario.graph_seed)
    simulation = Simulation(graph, node_rule, edge_rule)
    simulation_config = SimulationConfig(
        steps=max(1, int(result["age"])),
        batch_size=1,
        max_abs_state=4.0,
        record_every=1,
    )
    state = simulation.initial_state(1)
    initial_rng = Random(scenario.initial_state_seed)
    state.node = [[
        tuple(
            initial_rng.gauss(0.0, config.initial_state_scale)
            for _ in range(node_rule.state_width)
        )
        for _ in range(graph.n_nodes)
    ]]
    runtime = ReplicaRuntime(
        scenario,
        simulation,
        simulation_config,
        state,
        HealthMonitor(config.pathology),
    )
    trajectory = Trajectory()
    initial_external = [[(0.0,) * node_rule.state_width for _ in range(graph.n_nodes)]]
    trajectory.append(
        0,
        0.0,
        runtime.state.node,
        initial_external,
        runtime.state.edge,
        simulation._effective_strengths(runtime.state.edge),
    )
    stop_age = int(result["age"])
    expected_cause = result.get("death_cause")
    while runtime.age < stop_age or (expected_cause is not None and runtime.fatal_cause is None):
        prior_age = runtime.age
        if prior_age and level.disturbance_frequency and prior_age % level.disturbance_frequency == 0:
            trajectory.events.append(EventWindow("curriculum disturbance", prior_age, prior_age))
        if prior_age and prior_age % config.probes.interval == 0:
            trajectory.events.append(
                EventWindow("paired state probe", prior_age, prior_age)
            )
        runtime.advance(level, config.probes)
        if runtime.age == prior_age and runtime.fatal_cause is not None:
            break
        if runtime.age <= prior_age:
            raise RuntimeError("archived candidate could not advance during replay")
        trajectory.append(
            runtime.age,
            runtime.age * simulation_config.dt,
            runtime.state.node,
            [[(0.0,) * node_rule.state_width for _ in range(graph.n_nodes)]],
            runtime.state.edge,
            simulation._effective_strengths(runtime.state.edge),
        )
    replay_summary = runtime.summary()
    if replay_summary["age"] != stop_age:
        raise RuntimeError("replay age differs from archived result")
    if expected_cause != replay_summary["death_cause"]:
        raise RuntimeError("replay death cause differs from archived result")
    metrics = {
        "candidate": int(record["candidate_id"]),
        "outcome": record["status"],
        "curriculum_level": level_index,
        "replica": replica_index,
        "stop_age": stop_age,
        "death_cause": replay_summary["death_cause"] or "graduated",
        "normalized_pathology_burden": replay_summary["normalized_pathology_burden"],
        "responsiveness": replay_summary["responsiveness"],
        "propagation": replay_summary["propagation"],
        "distinguishability": replay_summary["distinguishability"],
        "recovered": replay_summary["recovered"],
    }
    return graph, trajectory, simulation_config, metrics


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "minimum": None, "mean": None, "maximum": None}
    return {
        "count": len(values),
        "minimum": min(values),
        "mean": fmean(values),
        "maximum": max(values),
    }
