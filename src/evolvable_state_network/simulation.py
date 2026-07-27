"""Batched, bounded, synchronous integration of local graph dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Iterable

from .graph import Graph
from .inputs import GaussianInput, InputProvider
from .interventions import StateIntervention
from .perturbations import (
    ImpulseInjection,
    InputDistributionShift,
    NodeLesion,
    Perturbation,
    WeightNoise,
    EdgeStateImpulse,
)
from .rules import EdgeRule, NodeRule, StatelessEdgeRule
from .types import EdgeTensor, NodeTensor, StateVector, clip, zeros


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    steps: int = 300
    dt: float = 0.05
    batch_size: int = 4
    max_delta: float = 0.12
    max_abs_state: float = 4.0
    edge_step_scale: float = 0.06
    edge_latent_alert: float = 12.0
    record_every: int = 1

    def __post_init__(self) -> None:
        if self.steps < 1 or self.batch_size < 1 or self.record_every < 1:
            raise ValueError("steps, batch_size, and record_every must be positive")
        if self.dt <= 0 or self.max_delta <= 0 or self.max_abs_state <= 0 or self.edge_step_scale <= 0 or self.edge_latent_alert <= 0:
            raise ValueError("dt and bounds must be positive")


@dataclass(slots=True)
class NetworkState:
    node: NodeTensor
    edge: EdgeTensor


@dataclass(frozen=True, slots=True)
class EventWindow:
    kind: str
    start: int
    end: int


@dataclass(slots=True)
class Trajectory:
    """Recorded bounded state snapshots; dimensions remain intentionally unnamed."""

    times: list[float] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    node_states: list[NodeTensor] = field(default_factory=list)
    edge_states: list[EdgeTensor] = field(default_factory=list)
    effective_edge_strengths: list[list[list[float]]] = field(default_factory=list)
    inputs: list[NodeTensor] = field(default_factory=list)
    events: list[EventWindow] = field(default_factory=list)

    def append(
        self, step: int, time: float, state: NodeTensor, external: NodeTensor, edge: EdgeTensor | None = None,
        effective_edge_strengths: list[list[float]] | None = None,
    ) -> None:
        self.steps.append(step)
        self.times.append(time)
        self.node_states.append([[tuple(vector) for vector in row] for row in state])
        self.edge_states.append([[tuple(vector) for vector in row] for row in edge] if edge is not None else [])
        self.effective_edge_strengths.append([list(row) for row in effective_edge_strengths] if effective_edge_strengths is not None else [])
        self.inputs.append([[tuple(vector) for vector in row] for row in external])


@dataclass(slots=True)
class SimulationResult:
    final_state: NetworkState
    trajectory: Trajectory
    diagnostics: "TransitionDiagnostics"


@dataclass(slots=True)
class TransitionDiagnostics:
    """Raw transition observations retained before numerical safety clipping."""

    nonfinite_proposals: int = 0
    delta_clipped: int = 0
    state_clipped: int = 0
    components: int = 0
    raw_maximum_absolute_value: float = 0.0
    raw_maximum_delta: float = 0.0
    last_state_clip: dict[str, int | float] | None = None
    clipped_components_per_step: list[int] = field(default_factory=list)
    components_per_step: list[int] = field(default_factory=list)


class Simulation:
    """Runs a shared local rule across graph sizes and batch members.

    The simulator only exposes each node's own vector, a permutation-invariant
    sum of incoming messages, and its external vector. It never passes a node
    index, global trajectory, future input, or experimental score into rules.
    """

    def __init__(self, graph: Graph, node_rule: NodeRule, edge_rule: EdgeRule | None = None) -> None:
        self.graph = graph
        self._incoming = graph.incoming
        self.node_rule = node_rule
        self.edge_rule = edge_rule or StatelessEdgeRule()
        if len(node_rule.initial_state()) != node_rule.state_width:
            raise ValueError("node initial state has incorrect width")
        if len(self.edge_rule.initial_state()) != self.edge_rule.state_width:
            raise ValueError("edge initial state has incorrect width")

    def initial_state(self, batch_size: int) -> NetworkState:
        return NetworkState(
            node=[[self.node_rule.initial_state() for _ in range(self.graph.n_nodes)] for _ in range(batch_size)],
            edge=[[self.edge_rule.initial_state() for _ in self.graph.edges] for _ in range(batch_size)],
        )

    def run(
        self,
        config: SimulationConfig,
        input_provider: InputProvider | None = None,
        perturbations: Iterable[Perturbation] = (),
        initial_node_state: NodeTensor | None = None,
        initial_edge_state: EdgeTensor | None = None,
        intervention: StateIntervention | None = None,
    ) -> SimulationResult:
        provider = input_provider or GaussianInput(seed=0)
        disturbances = tuple(perturbations)
        state = self.initial_state(config.batch_size)
        if initial_node_state is not None:
            self._validate_initial_node_state(initial_node_state, config)
            state.node = [[tuple(vector) for vector in row] for row in initial_node_state]
        if initial_edge_state is not None:
            self._validate_initial_edge_state(initial_edge_state, config)
            state.edge = [[tuple(vector) for vector in row] for row in initial_edge_state]
        if intervention is not None:
            intervention.validate(self.node_rule.state_width)
            state.node = [[intervention.initial(vector) for vector in row] for row in state.node]
        diagnostics = TransitionDiagnostics()
        trajectory = Trajectory(events=self._event_windows(disturbances, config.steps))
        initial_external = provider.sample(0, config.batch_size, self.graph.n_nodes, self.node_rule.state_width)
        trajectory.append(0, 0.0, state.node, self._apply_input_shifts(initial_external, 0, disturbances), state.edge, self._effective_strengths(state.edge))
        for step in range(config.steps):
            prior_components = diagnostics.components
            prior_clips = diagnostics.delta_clipped + diagnostics.state_clipped
            external = provider.sample(step, config.batch_size, self.graph.n_nodes, self.node_rule.state_width)
            external = self._apply_input_shifts(external, step, disturbances)
            state = self._step(state, external, step, config, disturbances, diagnostics, intervention)
            diagnostics.components_per_step.append(diagnostics.components - prior_components)
            diagnostics.clipped_components_per_step.append(diagnostics.delta_clipped + diagnostics.state_clipped - prior_clips)
            if (step + 1) % config.record_every == 0 or step + 1 == config.steps:
                trajectory.append(step + 1, (step + 1) * config.dt, state.node, external, state.edge, self._effective_strengths(state.edge))
        return SimulationResult(final_state=state, trajectory=trajectory, diagnostics=diagnostics)

    def _step(
        self,
        state: NetworkState,
        external: NodeTensor,
        step: int,
        config: SimulationConfig,
        disturbances: tuple[Perturbation, ...],
        diagnostics: TransitionDiagnostics,
        intervention: StateIntervention | None,
    ) -> NetworkState:
        width = self.node_rule.state_width
        incoming = self._incoming
        lesions = {
            node
            for disturbance in disturbances
            if isinstance(disturbance, NodeLesion) and disturbance.active(step)
            for node in disturbance.nodes
        }
        if any(node < 0 or node >= self.graph.n_nodes for node in lesions):
            raise ValueError("lesion node is outside graph")
        next_edges: EdgeTensor = []
        next_nodes: NodeTensor = []
        for batch in range(config.batch_size):
            edge_row: list[StateVector] = []
            messages: list[StateVector] = []
            for edge_index, edge in enumerate(self.graph.edges):
                if edge.source in lesions or edge.target in lesions:
                    edge_row.append(state.edge[batch][edge_index])
                    messages.append(zeros(width))
                    continue
                current_message = self.edge_rule.message(state.edge[batch][edge_index], state.node[batch][edge.source])
                edge_state = self.edge_rule.update(
                    state.edge[batch][edge_index],
                    state.node[batch][edge.source],
                    state.node[batch][edge.target],
                    current_message,
                    external[batch][edge.source],
                    external[batch][edge.target],
                    config.edge_step_scale,
                )
                edge_row.append(self._edge_transition(state.edge[batch][edge_index], edge_state, diagnostics))
                messages.append(self.edge_rule.message(edge_row[-1], state.node[batch][edge.source]))
            next_edges.append(edge_row)
            node_row: list[StateVector] = []
            for node in range(self.graph.n_nodes):
                if node in lesions:
                    node_row.append(zeros(width))
                    continue
                aggregate = [0.0] * width
                count = 0
                for edge_index in incoming[node]:
                    edge = self.graph.edges[edge_index]
                    if edge.source in lesions:
                        continue
                    weight = edge.weight + sum(
                        disturbance.sample(step, edge_index, batch)
                        for disturbance in disturbances
                        if isinstance(disturbance, WeightNoise) and disturbance.active(step)
                    )
                    message = messages[edge_index]
                    if len(message) != width:
                        raise ValueError("edge message width must match node state width")
                    aggregate = [value + weight * component for value, component in zip(aggregate, message, strict=True)]
                    count += 1
                aggregate_vector = tuple(value / count for value in aggregate) if count else zeros(width)
                if intervention is not None:
                    aggregate_vector, local_external = intervention.local_inputs(aggregate_vector, external[batch][node])
                else:
                    local_external = external[batch][node]
                proposed = self.node_rule.update(
                    state.node[batch][node], aggregate_vector, local_external, config.dt, config.max_delta
                )
                if intervention is not None:
                    proposed = intervention.transition(state.node[batch][node], proposed)
                node_row.append(
                    self._bounded_transition(
                        state.node[batch][node], proposed, config, diagnostics, batch, node
                    )
                )
            next_nodes.append(node_row)
        next_nodes = self._apply_impulses(next_nodes, step, config, disturbances)
        next_edges = self._apply_edge_impulses(next_edges, step, disturbances)
        return NetworkState(node=next_nodes, edge=next_edges)

    def _edge_transition(
        self, previous: StateVector, proposed: StateVector, diagnostics: TransitionDiagnostics
    ) -> StateVector:
        """Keep latent channels unconstrained while safely containing non-finite proposals.

        The effective communication value is bounded smoothly by the edge rule;
        retaining the latent values makes uncontrolled growth observable rather
        than hiding it behind a hard clip.
        """
        if len(previous) != len(proposed):
            raise ValueError("rule changed edge-state width")
        values: list[float] = []
        for before, after in zip(previous, proposed, strict=True):
            diagnostics.components += 1
            if not isfinite(after):
                diagnostics.nonfinite_proposals += 1
                after = before
            diagnostics.raw_maximum_absolute_value = max(diagnostics.raw_maximum_absolute_value, abs(after))
            diagnostics.raw_maximum_delta = max(diagnostics.raw_maximum_delta, abs(after - before))
            values.append(after)
        return tuple(values)

    def _bounded_transition(
        self,
        previous: StateVector,
        proposed: StateVector,
        config: SimulationConfig,
        diagnostics: TransitionDiagnostics,
        batch: int,
        node: int,
    ) -> StateVector:
        if len(previous) != len(proposed):
            raise ValueError("rule changed state-vector width")
        values: list[float] = []
        for coordinate, (before, after) in enumerate(zip(previous, proposed, strict=True)):
            diagnostics.components += 1
            if not isfinite(after):
                # A failed rule cannot propagate non-finite values through a batch.
                diagnostics.nonfinite_proposals += 1
                after = 0.0
            raw_delta = after - before
            diagnostics.raw_maximum_absolute_value = max(diagnostics.raw_maximum_absolute_value, abs(after))
            diagnostics.raw_maximum_delta = max(diagnostics.raw_maximum_delta, abs(raw_delta))
            if abs(raw_delta) > config.max_delta:
                diagnostics.delta_clipped += 1
            bounded_before_state = before + clip(raw_delta, config.max_delta)
            if abs(bounded_before_state) > config.max_abs_state:
                diagnostics.state_clipped += 1
                diagnostics.last_state_clip = {
                    "batch": batch,
                    "node": node,
                    "coordinate": coordinate,
                    "previous": before,
                    "proposed": after,
                    "after_delta_limit": bounded_before_state,
                    "bound": config.max_abs_state,
                }
            values.append(clip(before + clip(after - before, config.max_delta), config.max_abs_state))
        return tuple(values)

    def _validate_initial_node_state(self, node: NodeTensor, config: SimulationConfig) -> None:
        if len(node) != config.batch_size or any(len(row) != self.graph.n_nodes for row in node):
            raise ValueError("initial node state must match simulation batch and graph shape")
        if any(len(vector) != self.node_rule.state_width for row in node for vector in row):
            raise ValueError("initial node state vector width does not match node rule")

    def _validate_initial_edge_state(self, edge: EdgeTensor, config: SimulationConfig) -> None:
        if len(edge) != config.batch_size or any(len(row) != len(self.graph.edges) for row in edge):
            raise ValueError("initial edge state must match simulation batch and graph shape")
        if any(len(vector) != self.edge_rule.state_width for row in edge for vector in row):
            raise ValueError("initial edge state vector width does not match edge rule")

    def _effective_strengths(self, edges: EdgeTensor) -> list[list[float]]:
        return [[self.edge_rule.communication_strength(vector) for vector in row] for row in edges]

    def _apply_input_shifts(
        self, external: NodeTensor, step: int, disturbances: tuple[Perturbation, ...]
    ) -> NodeTensor:
        shifts = [d for d in disturbances if isinstance(d, InputDistributionShift) and d.active(step)]
        if not shifts:
            return external
        return [
            [
                tuple(
                    component * self._scale(shifts) + sum(shift.offset for shift in shifts) for component in vector
                )
                for vector in row
            ]
            for row in external
        ]

    @staticmethod
    def _scale(shifts: list[InputDistributionShift]) -> float:
        value = 1.0
        for shift in shifts:
            value *= shift.scale
        return value

    def _apply_impulses(
        self, nodes: NodeTensor, step: int, config: SimulationConfig, disturbances: tuple[Perturbation, ...]
    ) -> NodeTensor:
        for impulse in (d for d in disturbances if isinstance(d, ImpulseInjection) and d.active(step)):
            for node in impulse.nodes:
                if node < 0 or node >= self.graph.n_nodes:
                    raise ValueError("impulse node is outside graph")
                amount = (
                    (float(impulse.amount),) * self.node_rule.state_width
                    if isinstance(impulse.amount, (int, float))
                    else impulse.amount
                )
                if len(amount) != self.node_rule.state_width:
                    raise ValueError("impulse width does not match node state")
                for batch in range(len(nodes)):
                    nodes[batch][node] = tuple(
                        clip(value + change, config.max_abs_state)
                        for value, change in zip(nodes[batch][node], amount, strict=True)
                    )
        return nodes

    def _apply_edge_impulses(self, edges: EdgeTensor, step: int, disturbances: tuple[Perturbation, ...]) -> EdgeTensor:
        for impulse in (item for item in disturbances if isinstance(item, EdgeStateImpulse) and item.active(step)):
            for edge_index in impulse.edges:
                if edge_index < 0 or edge_index >= len(self.graph.edges):
                    raise ValueError("edge impulse index is outside graph")
                amount = ((float(impulse.amount),) * self.edge_rule.state_width if isinstance(impulse.amount, (int, float)) else impulse.amount)
                if len(amount) != self.edge_rule.state_width:
                    raise ValueError("edge impulse width does not match edge state")
                for batch in range(len(edges)):
                    edges[batch][edge_index] = tuple(value + change for value, change in zip(edges[batch][edge_index], amount, strict=True))
        return edges

    @staticmethod
    def _event_windows(disturbances: tuple[Perturbation, ...], steps: int) -> list[EventWindow]:
        windows: list[EventWindow] = []
        for disturbance in disturbances:
            if isinstance(disturbance, ImpulseInjection):
                windows.append(EventWindow("impulse", disturbance.step, disturbance.step))
            elif isinstance(disturbance, NodeLesion):
                windows.append(EventWindow("lesion", disturbance.start, disturbance.end if disturbance.end is not None else steps - 1))
            elif isinstance(disturbance, EdgeStateImpulse):
                windows.append(EventWindow("edge_impulse", disturbance.step, disturbance.step))
            elif isinstance(disturbance, (InputDistributionShift, WeightNoise)):
                windows.append(EventWindow(type(disturbance).__name__, disturbance.start, disturbance.end))
        return windows
