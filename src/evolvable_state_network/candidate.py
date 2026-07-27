"""Fixed-architecture, shared MLP node rules for Phase 1A."""

from __future__ import annotations

from dataclasses import dataclass
from math import tanh
from typing import Sequence

from .rules import EdgeRule, NodeRule
from .types import StateVector


@dataclass(frozen=True, slots=True)
class RuleArchitecture:
    """Configurable but non-evolvable architecture for one experiment."""

    state_width: int = 4
    hidden_width: int = 8
    increment_fraction: float = 0.8

    def __post_init__(self) -> None:
        if self.state_width < 1 or self.hidden_width < 1 or not 0 < self.increment_fraction <= 1:
            raise ValueError("invalid fixed rule architecture")

    @property
    def input_width(self) -> int:
        # current local state, mean incoming message, external vector, bias
        return 3 * self.state_width + 1

    @property
    def parameter_count(self) -> int:
        return self.hidden_width * self.input_width + self.hidden_width + self.state_width * self.hidden_width + self.state_width


class MLPUpdateRule(NodeRule):
    """One shared local rule with a bounded, unnamed-vector state increment."""

    def __init__(self, architecture: RuleArchitecture, parameters: Sequence[float]) -> None:
        self.architecture = architecture
        self.state_width = architecture.state_width
        if len(parameters) != architecture.parameter_count:
            raise ValueError(f"expected {architecture.parameter_count} rule parameters, received {len(parameters)}")
        self.parameters = tuple(float(value) for value in parameters)
        cursor = 0
        count = architecture.hidden_width * architecture.input_width
        self._input_weights = self.parameters[cursor : cursor + count]
        cursor += count
        self._hidden_bias = self.parameters[cursor : cursor + architecture.hidden_width]
        cursor += architecture.hidden_width
        count = architecture.state_width * architecture.hidden_width
        self._output_weights = self.parameters[cursor : cursor + count]
        cursor += count
        self._output_bias = self.parameters[cursor : cursor + architecture.state_width]

    def initial_state(self) -> StateVector:
        return (0.0,) * self.state_width

    def update(
        self, state: StateVector, aggregate: StateVector, external: StateVector, dt: float, max_delta: float
    ) -> StateVector:
        if len(state) != self.state_width or len(aggregate) != self.state_width or len(external) != self.state_width:
            raise ValueError("MLP update inputs must match configured state width")
        features = state + aggregate + external + (1.0,)
        hidden = []
        for row in range(self.architecture.hidden_width):
            offset = row * self.architecture.input_width
            total = self._hidden_bias[row] + sum(
                self._input_weights[offset + column] * value for column, value in enumerate(features)
            )
            hidden.append(tanh(total))
        increment_limit = max_delta * self.architecture.increment_fraction
        result = []
        for row in range(self.state_width):
            offset = row * self.architecture.hidden_width
            total = self._output_bias[row] + sum(
                self._output_weights[offset + column] * value for column, value in enumerate(hidden)
            )
            result.append(state[row] + increment_limit * tanh(total))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class EdgeArchitecture:
    """Fixed experiment configuration for a shared local channel rule.

    Coordinates in ``latent_width`` are intentionally unnamed.  The optional
    projection matrix is configuration, never a genome field.
    """

    node_state_width: int = 4
    latent_width: int = 3
    hidden_width: int = 8
    gate_index: int = 0
    message_projection: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        if self.node_state_width < 1 or self.latent_width < 1 or self.hidden_width < 1:
            raise ValueError("edge architecture widths must be positive")
        if not 0 <= self.gate_index < self.latent_width:
            raise ValueError("gate_index is outside the edge-state vector")
        if self.message_projection is not None and (
            len(self.message_projection) != self.node_state_width
            or any(len(row) != self.node_state_width for row in self.message_projection)
        ):
            raise ValueError("message projection must be square in node-state width")

    @property
    def input_width(self) -> int:
        # edge vector, source/target vectors, current message, endpoint input, bias
        return self.latent_width + 5 * self.node_state_width + 1

    @property
    def parameter_count(self) -> int:
        return self.hidden_width * self.input_width + self.hidden_width + self.latent_width * self.hidden_width + self.latent_width

    @property
    def projection(self) -> tuple[tuple[float, ...], ...]:
        if self.message_projection is not None:
            return self.message_projection
        return tuple(tuple(1.0 if row == column else 0.0 for column in range(self.node_state_width)) for row in range(self.node_state_width))


class MLPEdgeRule(EdgeRule):
    """Shared edge update with bounded increments and learned coordinate-wise gates."""

    def __init__(self, architecture: EdgeArchitecture, parameters: Sequence[float]) -> None:
        self.architecture = architecture
        self.state_width = architecture.latent_width
        if len(parameters) != architecture.parameter_count:
            raise ValueError(f"expected {architecture.parameter_count} edge-rule parameters, received {len(parameters)}")
        self.parameters = tuple(float(value) for value in parameters)
        cursor = 0
        count = architecture.hidden_width * architecture.input_width
        self._input_weights = self.parameters[cursor : cursor + count]
        cursor += count
        self._hidden_bias = self.parameters[cursor : cursor + architecture.hidden_width]
        cursor += architecture.hidden_width
        count = architecture.latent_width * architecture.hidden_width
        self._output_weights = self.parameters[cursor : cursor + count]
        cursor += count
        self._output_bias = self.parameters[cursor : cursor + architecture.latent_width]

    def initial_state(self) -> StateVector:
        return (0.0,) * self.state_width

    def update(
        self, state: StateVector, source: StateVector, target: StateVector, message: StateVector,
        source_external: StateVector, target_external: StateVector, edge_step_scale: float,
    ) -> StateVector:
        width = self.architecture.node_state_width
        if (len(state) != self.state_width or any(len(vector) != width for vector in (source, target, message, source_external, target_external))):
            raise ValueError("edge update inputs must match the configured architecture")
        features = state + source + target + message + source_external + target_external + (1.0,)
        hidden = []
        for row in range(self.architecture.hidden_width):
            offset = row * self.architecture.input_width
            hidden.append(tanh(self._hidden_bias[row] + sum(self._input_weights[offset + column] * value for column, value in enumerate(features))))
        increments = []
        for row in range(self.state_width):
            offset = row * self.architecture.hidden_width
            raw = self._output_bias[row] + sum(self._output_weights[offset + column] * value for column, value in enumerate(hidden))
            increments.append(edge_step_scale * tanh(raw))
        return tuple(value + increment for value, increment in zip(state, increments, strict=True))

    def communication_gates(self, state: StateVector) -> StateVector:
        """Map latent coordinates to one smooth communication gate per node coordinate.

        New survival runs use matching node and edge widths.  Cycling remains
        backward-compatible with older exports whose edge state was narrower
        than their node vector.
        """
        if len(state) != self.state_width:
            raise ValueError("edge state width does not match the configured architecture")
        return tuple(
            0.5 * (1.0 + tanh(state[(self.architecture.gate_index + coordinate) % self.state_width]))
            for coordinate in range(self.architecture.node_state_width)
        )

    def communication_strength(self, state: StateVector) -> float:
        # A scalar mean is retained for graph styling and health summaries.
        gates = self.communication_gates(state)
        return sum(gates) / len(gates)

    def message(self, state: StateVector, source: StateVector) -> StateVector:
        projected = tuple(sum(weight * value for weight, value in zip(row, source, strict=True)) for row in self.architecture.projection)
        return tuple(
            gate * value for gate, value in zip(self.communication_gates(state), projected, strict=True)
        )


class FixedEdgeRule(MLPEdgeRule):
    """Fixed-channel ablation with the same state shape and message map."""

    def __init__(self, architecture: EdgeArchitecture) -> None:
        super().__init__(architecture, (0.0,) * architecture.parameter_count)

    def update(
        self, state: StateVector, source: StateVector, target: StateVector, message: StateVector,
        source_external: StateVector, target_external: StateVector, edge_step_scale: float,
    ) -> StateVector:
        return state
