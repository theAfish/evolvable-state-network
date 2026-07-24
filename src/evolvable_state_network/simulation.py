"""Batched, bounded, synchronous integration of local graph dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Iterable

from .graph import Graph
from .inputs import GaussianInput, InputProvider
from .perturbations import (
    ImpulseInjection,
    InputDistributionShift,
    NodeLesion,
    Perturbation,
    WeightNoise,
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
    record_every: int = 1

    def __post_init__(self) -> None:
        if self.steps < 1 or self.batch_size < 1 or self.record_every < 1:
            raise ValueError("steps, batch_size, and record_every must be positive")
        if self.dt <= 0 or self.max_delta <= 0 or self.max_abs_state <= 0:
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
    inputs: list[NodeTensor] = field(default_factory=list)
    events: list[EventWindow] = field(default_factory=list)

    def append(
        self, step: int, time: float, state: NodeTensor, external: NodeTensor, edge: EdgeTensor | None = None
    ) -> None:
        self.steps.append(step)
        self.times.append(time)
        self.node_states.append([[tuple(vector) for vector in row] for row in state])
        self.edge_states.append([[tuple(vector) for vector in row] for row in edge] if edge is not None else [])
        self.inputs.append([[tuple(vector) for vector in row] for row in external])


@dataclass(slots=True)
class SimulationResult:
    final_state: NetworkState
    trajectory: Trajectory


class Simulation:
    """Runs a shared local rule across graph sizes and batch members.

    The simulator only exposes each node's own vector, a permutation-invariant
    sum of incoming messages, and its external vector. It never passes a node
    index, global trajectory, future input, or experimental score into rules.
    """

    def __init__(self, graph: Graph, node_rule: NodeRule, edge_rule: EdgeRule | None = None) -> None:
        self.graph = graph
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
    ) -> SimulationResult:
        provider = input_provider or GaussianInput(seed=0)
        disturbances = tuple(perturbations)
        state = self.initial_state(config.batch_size)
        trajectory = Trajectory(events=self._event_windows(disturbances, config.steps))
        initial_external = provider.sample(0, config.batch_size, self.graph.n_nodes, self.node_rule.state_width)
        trajectory.append(0, 0.0, state.node, self._apply_input_shifts(initial_external, 0, disturbances), state.edge)
        for step in range(config.steps):
            external = provider.sample(step, config.batch_size, self.graph.n_nodes, self.node_rule.state_width)
            external = self._apply_input_shifts(external, step, disturbances)
            state = self._step(state, external, step, config, disturbances)
            if (step + 1) % config.record_every == 0 or step + 1 == config.steps:
                trajectory.append(step + 1, (step + 1) * config.dt, state.node, external, state.edge)
        return SimulationResult(final_state=state, trajectory=trajectory)

    def _step(
        self,
        state: NetworkState,
        external: NodeTensor,
        step: int,
        config: SimulationConfig,
        disturbances: tuple[Perturbation, ...],
    ) -> NetworkState:
        width = self.node_rule.state_width
        incoming = self.graph.incoming
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
                edge_state = self.edge_rule.update(
                    state.edge[batch][edge_index],
                    state.node[batch][edge.source],
                    state.node[batch][edge.target],
                    config.dt,
                    config.max_delta,
                )
                edge_row.append(self._bounded_transition(state.edge[batch][edge_index], edge_state, config))
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
                proposed = self.node_rule.update(
                    state.node[batch][node], aggregate_vector, external[batch][node], config.dt, config.max_delta
                )
                node_row.append(self._bounded_transition(state.node[batch][node], proposed, config))
            next_nodes.append(node_row)
        next_nodes = self._apply_impulses(next_nodes, step, config, disturbances)
        return NetworkState(node=next_nodes, edge=next_edges)

    def _bounded_transition(self, previous: StateVector, proposed: StateVector, config: SimulationConfig) -> StateVector:
        if len(previous) != len(proposed):
            raise ValueError("rule changed state-vector width")
        values: list[float] = []
        for before, after in zip(previous, proposed, strict=True):
            if not isfinite(after):
                # A failed rule cannot propagate non-finite values through a batch.
                after = 0.0
            values.append(clip(before + clip(after - before, config.max_delta), config.max_abs_state))
        return tuple(values)

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

    @staticmethod
    def _event_windows(disturbances: tuple[Perturbation, ...], steps: int) -> list[EventWindow]:
        windows: list[EventWindow] = []
        for disturbance in disturbances:
            if isinstance(disturbance, ImpulseInjection):
                windows.append(EventWindow("impulse", disturbance.step, disturbance.step))
            elif isinstance(disturbance, NodeLesion):
                windows.append(EventWindow("lesion", disturbance.start, disturbance.end if disturbance.end is not None else steps - 1))
            elif isinstance(disturbance, (InputDistributionShift, WeightNoise)):
                windows.append(EventWindow(type(disturbance).__name__, disturbance.start, disturbance.end))
        return windows
