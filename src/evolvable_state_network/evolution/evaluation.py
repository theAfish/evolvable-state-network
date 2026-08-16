"""Deterministic batched viability evaluation for Phase 1A genomes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
from math import isfinite, sqrt
from random import Random
from statistics import fmean
from typing import Iterable, Literal, Sequence

from .candidate import (
    EdgeArchitecture,
    FixedEdgeRule,
    MLPEdgeRule,
    MLPUpdateRule,
    RuleArchitecture,
)
from .genome import EvolutionTarget, GenomeCodec
from ..graph import generate_random_graph
from ..inputs import GaussianInput
from ..interventions import StateIntervention
from ..metrics import MetricReport, evaluate_metrics
from ..perturbations import (
    EdgeStateImpulse,
    ImpulseInjection,
    InputDistributionShift,
    NodeLesion,
    Perturbation,
    WeightNoise,
)
from ..simulation import Simulation, SimulationConfig, Trajectory, TransitionDiagnostics

Split = Literal["train", "validation", "test"]


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    name: str
    graph_seed: int
    initial_state_seed: int
    input_seed: int
    nodes: int = 24
    mean_degree: float = 5.0
    steps: int = 300
    batch_size: int = 4
    perturbation_strength: float = 0.35
    topology: str = "erdos_renyi"
    initial_state_scale: float = 0.05
    include_perturbations: bool = True
    initial_edge_state_seed: int | None = None


@dataclass(frozen=True, slots=True)
class ScenarioSuite:
    train: tuple[ScenarioConfig, ...]
    validation: tuple[ScenarioConfig, ...]
    test: tuple[ScenarioConfig, ...]

    def for_split(self, split: Split) -> tuple[ScenarioConfig, ...]:
        return getattr(self, split)


def default_scenario_suite() -> ScenarioSuite:
    """Disjoint seeds, scales, strengths, and horizons for Phase 1A."""
    train = tuple(
        ScenarioConfig(f"train-{index}", 100 + index, 200 + index, 300 + index, nodes=12, mean_degree=4, steps=80, batch_size=2, perturbation_strength=0.28 + .05 * index)
        for index in range(3)
    )
    validation = tuple(
        ScenarioConfig(f"validation-{index}", 500 + index, 600 + index, 700 + index, nodes=24 + 8 * index, steps=160, batch_size=2, perturbation_strength=.5 + .08 * index)
        for index in range(2)
    )
    test = tuple(
        ScenarioConfig(f"test-{index}", 900 + index, 1000 + index, 1100 + index, nodes=40 + 12 * index, steps=260, batch_size=2, perturbation_strength=.7 + .1 * index)
        for index in range(2)
    )
    return ScenarioSuite(train, validation, test)


@dataclass(frozen=True, slots=True)
class FailureReport:
    nonfinite: bool
    numerical_explosion: bool
    persistent_silence: bool
    persistent_saturation: bool
    persistent_state_bias: bool
    excessive_synchronization: bool
    persistent_update_clipping: bool
    input_unresponsive: bool
    failed_recovery: bool
    edge_collapse: bool
    edge_saturation: bool
    persistent_maximum_edge_updates: bool
    uncontrolled_edge_growth: bool
    identical_edge_dynamics: bool
    communication_elimination_stability: bool
    costly_edge_oscillation: bool

    @property
    def failed(self) -> bool:
        return any(asdict(self).values())

    def labels(self) -> tuple[str, ...]:
        return tuple(name for name, value in asdict(self).items() if value)


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    scenario: ScenarioConfig
    metrics: MetricReport
    diagnostics: TransitionDiagnostics
    failures: FailureReport
    score: float
    mean_abs_correlation: float
    trajectory: Trajectory | None = field(default=None, compare=False)

    def to_dict(self, include_trajectory: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "scenario": asdict(self.scenario), "metrics": self.metrics.to_dict(), "diagnostics": self.diagnostics.to_dict(),
            "failures": asdict(self.failures), "score": self.score, "mean_abs_correlation": self.mean_abs_correlation,
        }
        if include_trajectory and self.trajectory is not None:
            result["trajectory_frames"] = len(self.trajectory.steps)
        return result


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    genome: tuple[float, ...]
    split: str
    scenario_results: tuple[ScenarioResult, ...]
    mean_score: float
    score_standard_deviation: float
    fitness: float
    penalties: dict[str, float] = field(default_factory=dict)

    @property
    def viable_fraction(self) -> float:
        return fmean(not result.failures.failed for result in self.scenario_results)

    def to_dict(self) -> dict[str, object]:
        return {
            "genome": list(self.genome), "split": self.split, "mean_score": self.mean_score,
            "score_standard_deviation": self.score_standard_deviation, "fitness": self.fitness,
            "viable_fraction": self.viable_fraction, "penalties": self.penalties,
            "scenario_results": [item.to_dict() for item in self.scenario_results],
        }


class CandidateEvaluator:
    """Evaluate one or many fixed-length genomes over deterministic scenario batches."""

    def __init__(
        self, architecture: RuleArchitecture | None = None, suite: ScenarioSuite | None = None,
        *, edge_architecture: EdgeArchitecture | None = None, target: EvolutionTarget = "node",
        rule_output_scale: float = 1.0, saturation_penalty_weight: float = 0.0,
        clipping_penalty_weight: float = 0.0, saturation_threshold: float = 3.0,
    ) -> None:
        self.architecture = architecture or RuleArchitecture()
        self.edge_architecture = edge_architecture
        self.target = target
        self.codec = GenomeCodec(self.architecture, edge_architecture, target)
        self.suite = suite or default_scenario_suite()
        if rule_output_scale <= 0 or saturation_penalty_weight < 0 or clipping_penalty_weight < 0 or saturation_threshold <= 0:
            raise ValueError("invalid rule-dynamics evaluation configuration")
        self.rule_output_scale = rule_output_scale
        self.saturation_penalty_weight = saturation_penalty_weight
        self.clipping_penalty_weight = clipping_penalty_weight
        self.saturation_threshold = saturation_threshold

    def evaluate(
        self, genome: Sequence[float], split: Split = "train", *, retain_trajectories: bool = False,
        intervention: StateIntervention | None = None, scenarios: Sequence[ScenarioConfig] | None = None,
        independently_seed_candidate: bool = True,
    ) -> EvaluationResult:
        encoded = tuple(float(value) for value in genome)
        node_rule, edge_rule = self.codec.decode_groups(encoded, output_scale=self.rule_output_scale)
        node_rule = node_rule or MLPUpdateRule(
            self.architecture, (0.0,) * self.architecture.parameter_count, output_scale=self.rule_output_scale,
        )
        if edge_rule is None:
            edge_rule = FixedEdgeRule(self.edge_architecture) if self.edge_architecture else None
        active_scenarios = tuple(scenarios) if scenarios is not None else self.suite.for_split(split)
        seeded_scenarios = tuple(_candidate_scenario(scenario, encoded) for scenario in active_scenarios) if independently_seed_candidate else active_scenarios
        results = tuple(self._evaluate_scenario(node_rule, edge_rule, scenario, retain_trajectories, intervention) for scenario in seeded_scenarios)
        scores = [item.score for item in results]
        mean_score = fmean(scores)
        deviation = sqrt(fmean((score - mean_score) ** 2 for score in scores))
        # A candidate may not buy one high-scoring trajectory by failing the
        # remaining seeded scenarios. Keep a small diagnostic gradient among
        # wholly nonviable candidates, but make viability the primary gate.
        viable_fraction = fmean(not item.failures.failed for item in results)
        robust_score = mean_score - 0.30 * deviation
        base_fitness = viable_fraction * robust_score - 0.20 * (1.0 - viable_fraction) + 0.02 * mean_score
        saturation = fmean(_saturation_fraction(item.diagnostics, self.saturation_threshold) for item in results)
        clipping = fmean(_clipping_fraction(item.diagnostics) for item in results)
        penalties = {
            "rule_output_saturation": self.saturation_penalty_weight * saturation,
            "update_clipping": self.clipping_penalty_weight * clipping,
        }
        fitness = base_fitness - sum(penalties.values())
        return EvaluationResult(encoded, split, results, mean_score, deviation, fitness, penalties)

    def evaluate_batch(
        self, genomes: Iterable[Sequence[float]], split: Split = "train", *, scenarios: Sequence[ScenarioConfig] | None = None,
    ) -> tuple[EvaluationResult, ...]:
        """Evaluate the population with the tensorized MLP execution path."""
        items = tuple(tuple(float(value) for value in genome) for genome in genomes)
        return tuple(self.evaluate(genome, split, scenarios=scenarios) for genome in items)

    def training_scenarios(self, generation: int, seed: int) -> tuple[ScenarioConfig, ...]:
        """Deterministically resample train-only graphs and disturbance regimes."""
        rng = Random(seed + 10_000_019 * generation)
        scenarios = []
        for index in range(len(self.suite.train)):
            nodes = rng.choice((10, 12, 16, 20))
            topology = rng.choice(("erdos_renyi", "ring"))
            scenarios.append(ScenarioConfig(
                name=f"generation-{generation}-train-{index}", graph_seed=rng.randrange(2**32),
                initial_state_seed=rng.randrange(2**32), input_seed=rng.randrange(2**32), nodes=nodes,
                mean_degree=min(4.0, nodes - 1), steps=80, batch_size=2, topology=topology,
                perturbation_strength=0.0 if index == 0 else rng.uniform(.2, .7),
                include_perturbations=index != 0,
            ))
        return tuple(scenarios)

    def _evaluate_scenario(
        self, rule: MLPUpdateRule, edge_rule: MLPEdgeRule | FixedEdgeRule | None, scenario: ScenarioConfig,
        retain: bool, intervention: StateIntervention | None,
    ) -> ScenarioResult:
        graph = generate_random_graph(scenario.nodes, scenario.mean_degree, scenario.graph_seed, scenario.topology)
        config = SimulationConfig(steps=scenario.steps, batch_size=scenario.batch_size)
        disturbances = _scenario_perturbations(scenario)
        if edge_rule is not None and edge_rule.state_width:
            disturbances += (EdgeStateImpulse(scenario.steps // 2, (0,), scenario.perturbation_strength),)
        initial_node = _initial_state(scenario, self.architecture.state_width)
        initial_edge = _initial_edge_state(scenario, len(graph.edges), edge_rule.state_width if edge_rule else 0)
        if intervention is None:
            from ..simulation.torch_backend import TorchMLPSimulator, resolve_device
            result = TorchMLPSimulator(graph, rule, edge_rule, resolve_device("auto")).run(
                config, GaussianInput(scenario.input_seed, standard_deviation=.28), disturbances, initial_node, initial_edge,
            )
        else:
            result = Simulation(graph, rule, edge_rule).run(
                config, GaussianInput(scenario.input_seed, standard_deviation=.28), disturbances,
                initial_node, initial_edge, intervention,
            )
        metrics = evaluate_metrics(result.trajectory, safety_bound=config.max_abs_state)
        correlation = _mean_abs_node_correlation(result.trajectory)
        failures = _failure_report(
            metrics, result.diagnostics, correlation, config, result.trajectory, edge_rule
        )
        score = _scenario_score(metrics, failures)
        return ScenarioResult(scenario, metrics, result.diagnostics, failures, score, correlation, result.trajectory if retain else None)


    def evaluate_ablations(self, node_parameters: Sequence[float], edge_parameters: Sequence[float], split: Split = "validation") -> dict[str, EvaluationResult]:
        """Run the four fixed/adaptive rule combinations on matched scenarios."""
        if self.edge_architecture is None:
            raise ValueError("ablations require an edge architecture")
        node = tuple(float(value) for value in node_parameters)
        edge = tuple(float(value) for value in edge_parameters)
        if len(node) != self.architecture.parameter_count or len(edge) != self.edge_architecture.parameter_count:
            raise ValueError("ablation parameter groups have incorrect dimensions")
        fixed_node = (0.0,) * self.architecture.parameter_count
        return {
            "fixed_node_fixed_edge": CandidateEvaluator(self.architecture, self.suite, edge_architecture=self.edge_architecture, target="joint").evaluate(fixed_node + (0.0,) * len(edge), split, independently_seed_candidate=False),
            "adaptive_node_fixed_edge": CandidateEvaluator(self.architecture, self.suite, edge_architecture=self.edge_architecture, target="joint").evaluate(node + (0.0,) * len(edge), split, independently_seed_candidate=False),
            "fixed_node_adaptive_edge": CandidateEvaluator(self.architecture, self.suite, edge_architecture=self.edge_architecture, target="joint").evaluate(fixed_node + edge, split, independently_seed_candidate=False),
            "adaptive_node_adaptive_edge": CandidateEvaluator(self.architecture, self.suite, edge_architecture=self.edge_architecture, target="joint").evaluate(node + edge, split, independently_seed_candidate=False),
        }


def _initial_state(scenario: ScenarioConfig, width: int) -> list[list[tuple[float, ...]]]:
    return [
        [tuple(Random(scenario.initial_state_seed + 100_003 * batch + 101 * node + component).gauss(0, scenario.initial_state_scale) for component in range(width)) for node in range(scenario.nodes)]
        for batch in range(scenario.batch_size)
    ]


def _initial_edge_state(scenario: ScenarioConfig, edges: int, width: int) -> list[list[tuple[float, ...]]]:
    seed = scenario.initial_edge_state_seed if scenario.initial_edge_state_seed is not None else scenario.initial_state_seed + 50_000_017
    return [
        [tuple(Random(seed + 100_003 * batch + 101 * edge + component).gauss(0, scenario.initial_state_scale) for component in range(width)) for edge in range(edges)]
        for batch in range(scenario.batch_size)
    ]


def _candidate_scenario(scenario: ScenarioConfig, genome: Sequence[float]) -> ScenarioConfig:
    """Give each candidate reproducible, independent graph/state/input streams.

    The digest makes this invariant to evaluation order, so permuting a batch
    cannot alter any candidate's environment.
    """
    digest = sha256(repr(tuple(float(value) for value in genome)).encode("utf-8")).digest()
    salt = int.from_bytes(digest[:4], "big")
    return replace(
        scenario,
        graph_seed=(scenario.graph_seed + salt) % 2**32,
        initial_state_seed=(scenario.initial_state_seed + 3 * salt) % 2**32,
        initial_edge_state_seed=((scenario.initial_edge_state_seed if scenario.initial_edge_state_seed is not None else scenario.initial_state_seed + 50_000_017) + 5 * salt) % 2**32,
        input_seed=(scenario.input_seed + 7 * salt) % 2**32,
    )


def _saturation_fraction(diagnostics: TransitionDiagnostics, threshold: float) -> float:
    values = diagnostics.node_rule_outputs + diagnostics.edge_rule_outputs
    return sum(abs(value) > threshold for value in values) / max(1, len(values))


def _clipping_fraction(diagnostics: TransitionDiagnostics) -> float:
    values = diagnostics.node_applied_deltas + diagnostics.edge_applied_deltas
    limits = [diagnostics.node_update_limit] * len(diagnostics.node_applied_deltas) + [diagnostics.edge_update_limit] * len(diagnostics.edge_applied_deltas)
    valid = [(abs(value), limit) for value, limit in zip(values, limits, strict=True) if limit is not None]
    return sum(value >= .99 * float(limit) for value, limit in valid) / max(1, len(valid))


def _scenario_perturbations(scenario: ScenarioConfig) -> tuple[Perturbation, ...]:
    if not scenario.include_perturbations:
        return ()
    quarter, half = scenario.steps // 4, scenario.steps // 2
    strength = scenario.perturbation_strength
    return (
        InputDistributionShift(quarter, quarter + max(2, scenario.steps // 12), offset=strength, scale=1 + strength),
        ImpulseInjection(half, (0,), 2 * strength),
        NodeLesion(half + max(2, scenario.steps // 12), (1,), half + max(4, scenario.steps // 6)),
        WeightNoise(half, min(scenario.steps - 1, half + max(2, scenario.steps // 10)), strength * .35, scenario.graph_seed + 19),
    )


def _failure_report(
    metrics: MetricReport, diagnostics: TransitionDiagnostics, correlation: float, config: SimulationConfig,
    trajectory: Trajectory | None = None,
    edge_rule: MLPEdgeRule | FixedEdgeRule | None = None,
) -> FailureReport:
    clipping = (diagnostics.delta_clipped + diagnostics.state_clipped) / max(1, diagnostics.components)
    tail_silent = not metrics.non_silence.non_silent
    tail_saturated = metrics.saturation.saturated
    tail_biased = False
    tail_correlation = correlation
    if trajectory is not None:
        start = max(0, len(trajectory.node_states) * 3 // 4)
        tail_values = [vector[0] for frame in trajectory.node_states[start:] for batch in frame for vector in batch]
        tail_rms = sqrt(fmean(value * value for value in tail_values)) if tail_values else 0.0
        tail_silent = tail_rms <= 1e-3
        tail_saturated = sum(abs(value) >= .95 for value in tail_values) / max(1, len(tail_values)) >= .5
        tail_rms = sqrt(fmean(value * value for value in tail_values)) if tail_values else 0.0
        dominant_sign = max(
            sum(value >= 0 for value in tail_values), sum(value <= 0 for value in tail_values)
        ) / max(1, len(tail_values))
        # A large, persistent one-sided offset is a degenerate trajectory even
        # when it has not yet crossed the hard saturation threshold.
        tail_biased = tail_rms >= .5 and abs(fmean(tail_values)) >= .5 and dominant_sign >= .70
        tail_correlation = _mean_abs_node_correlation(trajectory, start)
        tail_clips = diagnostics.clipped_components_per_step[start:]
        tail_components = diagnostics.components_per_step[start:]
        clipping = sum(tail_clips) / max(1, sum(tail_components))
    edge = _edge_pathologies(trajectory, config, metrics, edge_rule)
    return FailureReport(
        nonfinite=diagnostics.nonfinite_proposals > 0 or not metrics.boundedness.finite,
        numerical_explosion=diagnostics.raw_maximum_absolute_value > config.max_abs_state * 4 or diagnostics.raw_maximum_delta > config.max_delta * 20,
        persistent_silence=tail_silent,
        persistent_saturation=tail_saturated,
        persistent_state_bias=tail_biased,
        excessive_synchronization=not metrics.activity_diversity.diverse or tail_correlation > .985,
        persistent_update_clipping=clipping > .20,
        input_unresponsive=not metrics.perturbation_response.responsive,
        failed_recovery=not metrics.recovery.recovered,
        edge_collapse=edge["collapse"],
        edge_saturation=edge["saturation"],
        persistent_maximum_edge_updates=edge["maximum_updates"],
        uncontrolled_edge_growth=edge["growth"],
        identical_edge_dynamics=edge["identical"],
        communication_elimination_stability=edge["elimination_stability"],
        costly_edge_oscillation=edge["costly_oscillation"],
    )


def _edge_pathologies(
    trajectory: Trajectory | None,
    config: SimulationConfig,
    metrics: MetricReport,
    edge_rule: MLPEdgeRule | FixedEdgeRule | None = None,
) -> dict[str, bool]:
    absent = {name: False for name in ("collapse", "saturation", "maximum_updates", "growth", "identical", "elimination_stability", "costly_oscillation")}
    if trajectory is None or not trajectory.edge_states or not trajectory.edge_states[0] or not trajectory.edge_states[0][0] or not trajectory.edge_states[0][0][0]:
        return absent
    tail = max(0, len(trajectory.edge_states) * 3 // 4)
    strengths = [value for frame in trajectory.effective_edge_strengths[tail:] for batch in frame for value in batch]
    latent = [abs(value) for frame in trajectory.edge_states[tail:] for batch in frame for vector in batch for value in vector]
    updates = [abs(current - previous) for before, after in zip(trajectory.edge_states[tail:], trajectory.edge_states[tail + 1:]) for row_before, row_after in zip(before, after, strict=True) for vector_before, vector_after in zip(row_before, row_after, strict=True) for previous, current in zip(vector_before, vector_after, strict=True)]
    if edge_rule is None:
        gate_coordinates = [strengths]
    else:
        gate_coordinates = [
            [
                edge_rule.communication_gates(vector)[coordinate]
                for frame in trajectory.edge_states[tail:]
                for batch in frame
                for vector in batch
            ]
            for coordinate in range(edge_rule.architecture.node_state_width)
        ]
    # A scalar mean gate is sufficient for graph styling but not viability:
    # every communication coordinate must independently remain usable.
    collapse = any(values and max(values) <= .02 for values in gate_coordinates)
    saturation = any(
        values and sum(value <= .02 or value >= .98 for value in values) / len(values) >= .98
        for values in gate_coordinates
    )
    maximum_updates = bool(updates) and sum(value >= .98 * config.edge_step_scale for value in updates) / len(updates) >= .9
    # Mirror first-passage survival: a non-zero latent is acceptable when it
    # has settled.  It is pathological only when it remains above the alert
    # regime and keeps moving away from it for a sustained interval.
    growth = False
    for coordinate in range(len(trajectory.edge_states[0][0][0])):
        latent_magnitudes = [
            fmean(abs(vector[coordinate]) for batch in frame for vector in batch)
            for frame in trajectory.edge_states[tail:]
        ]
        growth_run = 0
        maximum_growth_run = 0
        for previous, current in zip(latent_magnitudes, latent_magnitudes[1:]):
            if current - previous > config.edge_growth_delta:
                growth_run += 1
                maximum_growth_run = max(maximum_growth_run, growth_run)
            else:
                growth_run = 0
        growth = growth or (
            bool(latent_magnitudes)
            and max(latent_magnitudes) >= config.edge_latent_alert
            and maximum_growth_run >= config.edge_growth_steps
        )
    # Compare channels at every retained frame; one channel cannot be diverse.
    frame_spreads = [
        max(max(vector[coordinate] for vector in batch) - min(vector[coordinate] for vector in batch) for coordinate in range(len(batch[0])))
        for frame in trajectory.edge_states[tail:] for batch in frame if len(batch) > 1
    ]
    identical = bool(frame_spreads) and max(frame_spreads) <= 1e-8
    node_updates = [abs(current - previous) for before, after in zip(trajectory.node_states[tail:], trajectory.node_states[tail + 1:]) for row_before, row_after in zip(before, after, strict=True) for vector_before, vector_after in zip(row_before, row_after, strict=True) for previous, current in zip(vector_before, vector_after, strict=True)]
    elimination_stability = collapse and (not node_updates or fmean(node_updates) < 1e-3)
    signed = [current - previous for before, after in zip(trajectory.edge_states[tail:], trajectory.edge_states[tail + 1:]) for row_before, row_after in zip(before, after, strict=True) for vector_before, vector_after in zip(row_before, row_after, strict=True) for previous, current in zip(vector_before, vector_after, strict=True)]
    reversals = sum(left * right < 0 for left, right in zip(signed, signed[1:])) / max(1, len(signed) - 1)
    costly_oscillation = bool(updates) and fmean(updates) > .45 * config.edge_step_scale and reversals > .35 and (not metrics.perturbation_response.responsive or not metrics.recovery.recovered)
    return {"collapse": collapse, "saturation": saturation, "maximum_updates": maximum_updates, "growth": growth, "identical": identical, "elimination_stability": elimination_stability, "costly_oscillation": costly_oscillation}


def _scenario_score(metrics: MetricReport, failures: FailureReport) -> float:
    if failures.failed:
        # Preserve a small diagnostic gradient among nonviable candidates.
        penalties = sum(asdict(failures).values())
        return max(0.0, .12 - .02 * penalties)
    bounded = max(0.0, 1 - metrics.boundedness.maximum_absolute_value / 4.0)
    activity = min(1.0, metrics.non_silence.rms / .15)
    diversity = min(1.0, metrics.activity_diversity.node_time_mean_std / .12)
    response = min(1.0, metrics.perturbation_response.magnitude / .12)
    recovery = max(0.0, 1 - metrics.recovery.error / .25)
    saturation = max(0.0, 1 - metrics.saturation.fraction)
    return .15 * bounded + .18 * activity + .18 * diversity + .20 * response + .18 * recovery + .11 * saturation


def _mean_abs_node_correlation(trajectory: Trajectory, start_frame: int = 0) -> float:
    # A fixed deterministic subsample keeps the evaluator scalable; the full
    # node-by-node matrix remains available from analysis.node_correlation_matrix.
    node_count = min(32, len(trajectory.node_states[0][0]))
    series = [
        [snapshot[0][node][0] for snapshot in trajectory.node_states[start_frame:]]
        for node in range(node_count)
    ]
    correlations: list[float] = []
    for left in range(len(series)):
        for right in range(left + 1, len(series)):
            a, b = series[left], series[right]
            am, bm = fmean(a), fmean(b)
            numerator = sum((x - am) * (y - bm) for x, y in zip(a, b, strict=True))
            denominator = sqrt(sum((x - am) ** 2 for x in a) * sum((y - bm) ** 2 for y in b))
            correlations.append(abs(numerator / denominator) if denominator > 1e-12 else 1.0)
    return fmean(correlations) if correlations else 1.0
