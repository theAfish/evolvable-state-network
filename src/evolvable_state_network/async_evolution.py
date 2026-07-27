"""Asynchronous, death-driven evolution of generic local graph dynamics.

The module deliberately separates the inherited rule parameters from local
runtime state.  Candidates occupy independent slots and are replaced as soon
as they die or reach the active survival milestone.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from math import isfinite
from pathlib import Path
from random import Random
from statistics import fmean
from typing import Callable, Literal, Sequence

from .candidate import EdgeArchitecture, RuleArchitecture
from .cmaes import CMAES, CMAESConfig
from .genome import EvolutionTarget, GenomeCodec
from .graph import generate_random_graph
from .inputs import GaussianInput
from .simulation import NetworkState, Simulation, SimulationConfig, TransitionDiagnostics


@dataclass(frozen=True, slots=True)
class PathologyConfig:
    fatal_threshold: float = 8.0
    increase: float = 1.0
    recovery: float = 0.65
    communication_floor: float = 0.04
    boundary_fraction: float = 0.80
    homogenization_variance: float = 1e-6
    response_floor: float = 2e-3
    propagation_floor: float = 5e-4
    distinguishability_floor: float = 2e-3
    recovery_limit: float = 0.35
    one_direction_steps: int = 12
    absolute_node_limit: float = 4.0
    absolute_edge_limit: float = 12.0


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
    input_scale: float = 0.12
    graph_nodes: int = 8
    mean_degree: float = 3.0


@dataclass(frozen=True, slots=True)
class AsyncEvolutionConfig:
    slots: int = 8
    replicas: int = 3
    result_batch_size: int = 8
    max_ticks: int = 200
    seed: int = 41
    target: EvolutionTarget = "joint"
    architecture: RuleArchitecture = RuleArchitecture(state_width=1, hidden_width=3)
    edge_architecture: EdgeArchitecture | None = EdgeArchitecture(
        node_state_width=1, latent_width=2, hidden_width=3
    )
    levels: tuple[CurriculumLevel, ...] = (
        CurriculumLevel(20),
        CurriculumLevel(40, 10, .12, input_scale=.16, graph_nodes=10),
        CurriculumLevel(70, 7, .20, True, .20, 12, 4.0),
    )
    pathology: PathologyConfig = PathologyConfig()
    probes: ProbeConfig = ProbeConfig()
    curriculum_window: int = 20
    curriculum_pass_fraction: float = .60
    censor_interval: int = 5
    elite_size: int = 4
    cma_fraction: float = .70
    elite_fraction: float = .20
    exploration_sigma: float = 1.0
    initial_sigma: float = .35
    initial_genomes: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self) -> None:
        if self.slots < 1 or self.replicas < 1 or self.result_batch_size < 2:
            raise ValueError("slots, replicas, and result_batch_size must be positive")
        if not self.levels or any(level.lifetime < 1 for level in self.levels):
            raise ValueError("at least one positive curriculum lifetime is required")
        if self.target in {"edge", "joint"} and self.edge_architecture is None:
            raise ValueError("edge architecture is required for edge or joint evolution")
        if (
            not 0 <= self.cma_fraction <= 1
            or not 0 <= self.elite_fraction <= 1
            or self.cma_fraction + self.elite_fraction > 1.0
        ):
            raise ValueError("replacement mixture fractions cannot exceed one")


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


@dataclass(slots=True)
class HealthMonitor:
    config: PathologyConfig
    burdens: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    burden_history: list[dict[str, object]] = field(default_factory=list)
    probes: list[ProbeSummary] = field(default_factory=list)
    previous_mean: float | None = None
    one_direction_sign: int = 0
    one_direction_run: int = 0
    death_cause: str | None = None
    death_time: int | None = None

    def _burden(self, name: str, violation: bool, amount: float = 1.0) -> None:
        current = self.burdens[name]
        self.burdens[name] = (
            current + self.config.increase * amount
            if violation
            else max(0.0, current - self.config.recovery)
        )

    def observe(
        self,
        step: int,
        previous: NetworkState,
        current: NetworkState,
        strengths: Sequence[float],
        diagnostics: TransitionDiagnostics,
        probe: ProbeSummary | None = None,
        simulator_failed: bool = False,
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

        mean = fmean(values) if values else 0.0
        variance = fmean((value - mean) ** 2 for value in values) if values else 0.0
        prior_values = [value for row in previous.node for vector in row for value in vector]
        prior_mean = fmean(prior_values) if prior_values else 0.0
        delta = mean - prior_mean
        sign = 1 if delta > 1e-7 else -1 if delta < -1e-7 else 0
        if sign and sign == self.one_direction_sign:
            self.one_direction_run += 1
        else:
            self.one_direction_sign, self.one_direction_run = sign, int(bool(sign))

        self._burden(
            "communication_collapse",
            not strengths or fmean(strengths) < self.config.communication_floor,
        )
        self._burden(
            "boundary_saturation",
            bool(values)
            and fmean(abs(value) > .95 * self.config.absolute_node_limit for value in values)
            >= self.config.boundary_fraction,
        )
        self._burden("state_homogenization", variance < self.config.homogenization_variance)
        self._burden("one_direction_degeneration", self.one_direction_run >= self.config.one_direction_steps)
        if probe is not None:
            self.probes.append(probe)
            self._burden("input_unresponsive", probe.response < self.config.response_floor)
            self._burden("communication_unresponsive", probe.propagation < self.config.propagation_floor)
            self._burden(
                "trajectory_indistinguishable",
                probe.distinguishability < self.config.distinguishability_floor,
            )
            self._burden("disturbance_unrecovered", not probe.recovered)
        # Probe-derived health evidence is updated only by another probe.
        # Absence of a probe is not evidence of functional recovery.
        self.burden_history.append({"step": step, "burdens": dict(self.burdens)})
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
    input_provider: GaussianInput
    state: NetworkState
    monitor: HealthMonitor
    age: int = 0
    active_probe_start: int | None = None
    probe_baseline: NetworkState | None = None
    shadow_state: NetworkState | None = None
    probe_peak_response: float = 0.0
    probe_peak_propagation: float = 0.0
    probe_peak_distinguishability: float = 0.0
    diagnostics: TransitionDiagnostics = field(default_factory=TransitionDiagnostics)
    fatal_cause: str | None = None

    def advance(self, level: CurriculumLevel, probe_config: ProbeConfig) -> None:
        if self.fatal_cause:
            return
        previous = deepcopy(self.state)
        external = self.input_provider.sample(
            self.age, 1, self.simulation.graph.n_nodes, self.simulation.node_rule.state_width
        )
        if level.disturbance_frequency and self.age and self.age % level.disturbance_frequency == 0:
            for node in range(len(external[0])):
                if level.compound_disturbances or node == 0:
                    external[0][node] = tuple(value + level.disturbance_strength for value in external[0][node])
        if self.active_probe_start is None and self.age and self.age % probe_config.interval == 0:
            self.active_probe_start = self.age
            self.probe_baseline = deepcopy(self.state)
            self.shadow_state = deepcopy(self.state)
            self.probe_peak_response = 0.0
            self.probe_peak_propagation = 0.0
            self.probe_peak_distinguishability = 0.0
        shadow_external = deepcopy(external)
        if (
            self.active_probe_start is not None
            and self.age - self.active_probe_start < probe_config.duration
        ):
            external[0][0] = tuple(value + probe_config.amplitude for value in external[0][0])
            shadow_external[0][0] = tuple(value - probe_config.amplitude for value in shadow_external[0][0])
        prior_nonfinite = self.diagnostics.nonfinite_proposals
        prior_clipped = self.diagnostics.state_clipped
        try:
            self.state = self.simulation._step(
                self.state, external, self.age, self.config, (), self.diagnostics, None
            )
            if self.shadow_state is not None:
                shadow_diagnostics = TransitionDiagnostics()
                self.shadow_state = self.simulation._step(
                    self.shadow_state, shadow_external, self.age, self.config, (), shadow_diagnostics, None
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
            self.probe_baseline = None
            self.shadow_state = None
        strengths = [
            self.simulation.edge_rule.communication_strength(vector)
            for row in self.state.edge
            for vector in row
        ]
        self.fatal_cause = self.monitor.observe(
            self.age, previous, self.state, strengths, step_diagnostics, probe
        )

    def _probe_summary(self) -> ProbeSummary:
        assert self.probe_baseline is not None and self.shadow_state is not None
        base = self.probe_baseline.node[0]
        live = self.state.node[0]
        baseline_scale = fmean(abs(value) for vector in base for value in vector) if base else 0.0
        current_scale = fmean(abs(value) for vector in live for value in vector) if live else 0.0
        return ProbeSummary(
            self.age,
            self.probe_peak_response,
            self.probe_peak_propagation,
            self.probe_peak_distinguishability,
            abs(current_scale - baseline_scale) <= self.monitor.config.recovery_limit,
        )

    def _update_probe_peaks(self) -> None:
        assert self.probe_baseline is not None and self.shadow_state is not None
        base = self.probe_baseline.node[0]
        live = self.state.node[0]
        shadow = self.shadow_state.node[0]
        response = fmean(
            abs(after - before) for before, after in zip(base[0], live[0], strict=True)
        )
        propagation_values = [
            abs(after - before)
            for base_vector, live_vector in zip(base[1:], live[1:], strict=True)
            for before, after in zip(base_vector, live_vector, strict=True)
        ]
        distinguishability = fmean(
            abs(left - right)
            for live_vector, shadow_vector in zip(live, shadow, strict=True)
            for left, right in zip(live_vector, shadow_vector, strict=True)
        )
        self.probe_peak_response = max(self.probe_peak_response, response)
        self.probe_peak_propagation = max(
            self.probe_peak_propagation,
            fmean(propagation_values) if propagation_values else 0.0,
        )
        self.probe_peak_distinguishability = max(
            self.probe_peak_distinguishability, distinguishability
        )

    def summary(self) -> dict[str, object]:
        values = [value for vector in self.state.node[0] for value in vector]
        edges = [value for vector in self.state.edge[0] for value in vector]
        latest = self.monitor.probes[-1] if self.monitor.probes else None
        return {
            "scenario": asdict(self.scenario),
            "age": self.age,
            "death_cause": self.fatal_cause,
            "normalized_pathology_burden": self.monitor.normalized_burden,
            "pathology_trajectories": self.monitor.burden_history,
            "response_probes": [asdict(item) for item in self.monitor.probes],
            "responsiveness": latest.response if latest else 0.0,
            "propagation": latest.propagation if latest else 0.0,
            "distinguishability": latest.distinguishability if latest else 0.0,
            "recovered": latest.recovered if latest else False,
            "final_node_statistics": _stats(values),
            "final_edge_statistics": _stats(edges),
            "update_cost": self.diagnostics.components / max(1, self.age),
        }


@dataclass(frozen=True, slots=True)
class CandidateProposal:
    genome: tuple[float, ...]
    source: Literal["cma", "elite", "exploration", "initial"]
    sample_id: int | None
    optimizer_update: int
    parent_candidate_id: int | None = None


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
    """Buffered adapter that never blocks slot replacement on a generation."""

    def __init__(self, dimension: int, batch_size: int, sigma: float, seed: int) -> None:
        self.optimizer = CMAES(CMAESConfig(dimension, batch_size, sigma, seed))
        self.batch_size = batch_size
        self.pending: deque[tuple[int, tuple[float, ...]]] = deque()
        self.results: dict[str, list[tuple[int, tuple[float, ...], tuple[float, ...]]]] = defaultdict(list)
        self.next_sample_id = 0
        self.update_count = 0
        self.observed_samples: set[int] = set()
        self._refill()

    def _refill(self) -> None:
        for genome in self.optimizer.ask():
            self.pending.append((self.next_sample_id, genome))
            self.next_sample_id += 1

    def ask(self) -> tuple[int, tuple[float, ...]] | None:
        return self.pending.popleft() if self.pending else None

    def observe(
        self, bank_id: str, sample_id: int | None, genome: tuple[float, ...], rank: tuple[float, ...]
    ) -> bool:
        if sample_id is None:
            return False
        if sample_id in self.observed_samples:
            return False
        self.observed_samples.add(sample_id)
        bucket = self.results[bank_id]
        bucket.append((sample_id, genome, rank))
        if len(bucket) < self.batch_size:
            return False
        records = bucket[: self.batch_size]
        del bucket[: self.batch_size]
        ordered = sorted(records, key=lambda item: item[2])
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


class AsyncEvolutionRunner:
    def __init__(self, config: AsyncEvolutionConfig) -> None:
        self.config = config
        self.codec = GenomeCodec(config.architecture, config.edge_architecture, config.target)
        self.rng = Random(config.seed)
        self.optimizer = SteadyStateCMA(
            self.codec.dimension, config.result_batch_size, config.initial_sigma, config.seed
        )
        self.elites = EliteArchive(config.elite_size)
        self.level = 0
        self.banks = [
            ScenarioBank.create(config.seed, index, config.replicas, level)
            for index, level in enumerate(config.levels)
        ]
        self.slots: list[CandidateSlot] = []
        self.archive: list[dict[str, object]] = []
        self.censored: list[dict[str, object]] = []
        self.recent_outcomes: deque[bool] = deque(maxlen=config.curriculum_window)
        self.next_candidate_id = 0
        self.initial = deque(config.initial_genomes)
        self.utilization_samples: list[float] = []

    def run(
        self,
        output: Path | None = None,
        progress: Callable[[dict[str, object]], None] | None = None,
    ) -> dict[str, object]:
        while len(self.slots) < self.config.slots:
            self.slots.append(self._new_slot(len(self.slots)))
        for tick in range(self.config.max_ticks):
            self.utilization_samples.append(len(self.slots) / self.config.slots)
            for slot in list(self.slots):
                level = self.config.levels[slot.level]
                for replica in slot.replicas:
                    replica.advance(level, self.config.probes)
                if slot.dead:
                    self._finish(slot, "death")
                    self._replace(slot)
                elif slot.age >= level.lifetime:
                    slot.milestone_history.append(
                        {"level": slot.level, "time": slot.age, "kind": "graduation"}
                    )
                    self._publish_censored(slot, "milestone")
                    self._finish(slot, "graduation")
                    self._replace(slot)
                elif slot.age % self.config.censor_interval == 0 and slot.age not in slot.published_ages:
                    self._publish_censored(slot, "living")
            self._maybe_advance_curriculum()
            if progress is not None:
                progress(
                    {
                        "tick": tick + 1,
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
                                    replica.monitor.normalized_burden
                                    for replica in slot.replicas
                                ),
                            }
                            for slot in self.slots
                        ],
                    }
                )
        for slot in self.slots:
            self._publish_censored(slot, "living_at_stop")
        report = self.report()
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            (output / "candidate_archive.json").write_text(
                json.dumps(self.archive, indent=2, sort_keys=True), encoding="utf-8"
            )
            (output / "living_censored.json").write_text(
                json.dumps(self.censored, indent=2, sort_keys=True), encoding="utf-8"
            )
            (output / "diagnostic_report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
            )
        return report

    def _proposal(self) -> CandidateProposal:
        if self.initial:
            return CandidateProposal(tuple(self.initial.popleft()), "initial", None, self.optimizer.update_count)
        draw = self.rng.random()
        if draw < self.config.cma_fraction:
            asked = self.optimizer.ask()
            if asked is not None:
                sample, genome = asked
                return CandidateProposal(genome, "cma", sample, self.optimizer.update_count)
        if draw < self.config.cma_fraction + self.config.elite_fraction and self.elites.records:
            parent = self.rng.choice(self.elites.records)
            genome = tuple(float(value) + self.rng.gauss(0, .08) for value in parent["genome"])
            return CandidateProposal(
                genome,
                "elite",
                None,
                self.optimizer.update_count,
                int(parent["candidate_id"]),
            )
        genome = tuple(self.rng.gauss(0, self.config.exploration_sigma) for _ in range(self.codec.dimension))
        return CandidateProposal(genome, "exploration", None, self.optimizer.update_count)

    def _new_slot(self, slot_index: int) -> CandidateSlot:
        proposal = self._proposal()
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
                tuple(init_rng.gauss(0, .03) for _ in range(node_rule.state_width))
                for _ in range(graph.n_nodes)
            ]]
            replicas.append(
                ReplicaRuntime(
                    scenario,
                    simulation,
                    simulation_config,
                    GaussianInput(scenario.input_seed, 0.0, scenario.input_scale),
                    state,
                    HealthMonitor(self.config.pathology),
                )
            )
        candidate = CandidateSlot(
            slot_index, self.next_candidate_id, proposal, bank, self.level, replicas
        )
        self.next_candidate_id += 1
        return candidate

    def _replace(self, old: CandidateSlot) -> None:
        self.slots[old.slot] = self._new_slot(old.slot)

    def _publish_censored(self, slot: CandidateSlot, kind: str) -> None:
        slot.published_ages.add(slot.age)
        record = {
            "candidate_id": slot.candidate_id,
            "slot": slot.slot,
            "kind": kind,
            "right_censored": True,
            "age": slot.age,
            "level": slot.level,
            "bank_id": slot.bank.bank_id,
            "source": slot.proposal.source,
            "milestone": self.config.levels[slot.level].lifetime,
            "rank_key": list(slot.rank_key(kind == "milestone")),
        }
        self.censored.append(record)
        self.optimizer.observe(
            slot.bank.bank_id,
            slot.proposal.sample_id,
            slot.proposal.genome,
            slot.rank_key(kind == "milestone"),
        )

    def _finish(self, slot: CandidateSlot, status: Literal["death", "graduation"]) -> None:
        graduated = status == "graduation"
        causes = [
            replica.fatal_cause for replica in slot.replicas if replica.fatal_cause is not None
        ]
        record = {
            "candidate_id": slot.candidate_id,
            "slot": slot.slot,
            "status": status,
            "genome": list(slot.proposal.genome),
            "parameter_groups": self.codec.export_groups(slot.proposal.genome),
            "sampling": asdict(slot.proposal),
            "bank": asdict(slot.bank),
            "level": slot.level,
            "age": slot.age,
            "death_time": slot.age if causes else None,
            "death_cause": causes[0] if causes else None,
            "milestone_history": slot.milestone_history,
            "per_replica_results": [replica.summary() for replica in slot.replicas],
            "rank_key": list(slot.rank_key(graduated)),
        }
        self.archive.append(record)
        self.recent_outcomes.append(graduated)
        self.optimizer.observe(
            slot.bank.bank_id,
            slot.proposal.sample_id,
            slot.proposal.genome,
            slot.rank_key(graduated),
        )
        if graduated:
            self.elites.consider(record)

    def _maybe_advance_curriculum(self) -> None:
        if self.level + 1 >= len(self.config.levels):
            return
        if (
            len(self.recent_outcomes) == self.recent_outcomes.maxlen
            and fmean(self.recent_outcomes) >= self.config.curriculum_pass_fraction
        ):
            self.level += 1
            self.recent_outcomes.clear()

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
            "completed_candidates": len(self.archive),
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
            "optimizer_update_frequency": self.optimizer.update_count / max(1, self.config.max_ticks),
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
            responsive[2 * architecture.state_width] = 2.0
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
    architecture = RuleArchitecture(state_width=1, hidden_width=3)
    edge_architecture = EdgeArchitecture(node_state_width=1, latent_width=2, hidden_width=3)
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
            CurriculumLevel(16, input_scale=.10, graph_nodes=7),
            CurriculumLevel(28, 8, .10, input_scale=.15, graph_nodes=9),
        ),
        pathology=PathologyConfig(fatal_threshold=3.0),
        probes=ProbeConfig(interval=4, duration=2, amplitude=.10),
        curriculum_window=8,
        censor_interval=4,
        initial_genomes=diagnostic_reference_genomes(codec),
    )
    report = AsyncEvolutionRunner(config).run(output, progress=progress)
    (output / "diagnostic_config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "minimum": None, "mean": None, "maximum": None}
    return {
        "count": len(values),
        "minimum": min(values),
        "mean": fmean(values),
        "maximum": max(values),
    }
