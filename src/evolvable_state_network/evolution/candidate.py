"""Fixed-architecture, shared MLP node rules for Phase 1A."""

from __future__ import annotations

from dataclasses import dataclass
from math import tanh
from typing import Literal, Sequence

from ..rules import EdgeRule, NodeRule
from ..types import StateVector


def _activation(value: float, name: str) -> float:
    if name == "tanh":
        return tanh(value)
    if name == "relu":
        return max(0.0, value)
    if name == "silu":
        return value / (1.0 + __import__("math").exp(-value))
    # Numerically stable enough for the small evolved rule ranges used here.
    import math
    return .5 * value * (1.0 + math.tanh(.7978845608 * (value + .044715 * value ** 3)))


def _decode_layers(values: tuple[float, ...], input_width: int, widths: tuple[int, ...]) -> tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]:
    cursor = 0
    previous = input_width
    layers = []
    for width in widths:
        count = width * previous
        weights = tuple(tuple(values[cursor + row * previous + column] for column in range(previous)) for row in range(width))
        cursor += count
        bias = values[cursor : cursor + width]
        cursor += width
        layers.append((weights, bias))
        previous = width
    return tuple(layers)


def _forward(features: StateVector, layers: tuple[tuple[tuple[float, ...], tuple[float, ...]], ...], activation: str) -> StateVector:
    values = features
    for index, (weights, bias) in enumerate(layers):
        values = tuple(sum(weight * value for weight, value in zip(row, values, strict=True)) + offset for row, offset in zip(weights, bias, strict=True))
        if index < len(layers) - 1:
            values = tuple(_activation(value, activation) for value in values)
    return values


@dataclass(frozen=True, slots=True)
class RuleArchitecture:
    """Configurable but non-evolvable architecture for one experiment."""

    state_width: int = 4
    hidden_width: int = 8
    hidden_layers: tuple[int, ...] | None = None
    activation: Literal["tanh", "relu", "gelu", "silu"] = "tanh"
    increment_fraction: float = 0.8

    def __post_init__(self) -> None:
        if self.state_width < 1 or self.hidden_width < 1 or not 0 < self.increment_fraction <= 1:
            raise ValueError("invalid fixed rule architecture")
        if self.hidden_layers is not None and (not self.hidden_layers or any(width < 1 for width in self.hidden_layers)):
            raise ValueError("hidden_layers must contain positive widths")

    @property
    def input_width(self) -> int:
        # current local state and mean incoming message; each MLP layer owns
        # an explicit bias vector.
        return 2 * self.state_width

    @property
    def parameter_count(self) -> int:
        widths = self.layers + (self.state_width,)
        return sum(output * (input_ + 1) for input_, output in zip((self.input_width,) + widths[:-1], widths, strict=True))

    @property
    def layers(self) -> tuple[int, ...]:
        """Hidden widths, with ``hidden_width`` retained for old exports."""
        # JSON checkpoints decode tuples as lists; normalize on read so both
        # persisted reports and programmatic construction are accepted.
        return tuple(self.hidden_layers) if self.hidden_layers is not None else (self.hidden_width,)


class MLPUpdateRule(NodeRule):
    """One shared local rule with a bounded, unnamed-vector state increment."""

    def __init__(self, architecture: RuleArchitecture, parameters: Sequence[float], *, output_scale: float = 1.0) -> None:
        self.architecture = architecture
        self.state_width = architecture.state_width
        if len(parameters) != architecture.parameter_count:
            raise ValueError(f"expected {architecture.parameter_count} rule parameters, received {len(parameters)}")
        self.parameters = tuple(float(value) for value in parameters)
        if output_scale <= 0:
            raise ValueError("rule output scale must be positive")
        self.output_scale = float(output_scale)
        self._layers = _decode_layers(self.parameters, architecture.input_width, architecture.layers + (architecture.state_width,))

    def initial_state(self) -> StateVector:
        return (0.0,) * self.state_width

    def update(
        self, state: StateVector, aggregate: StateVector, dt: float, max_delta: float
    ) -> StateVector:
        if len(state) != self.state_width or len(aggregate) != self.state_width:
            raise ValueError("MLP update inputs must match configured state width")
        output = self.raw_output(state, aggregate)
        # Keep the established update magnitude at dt=.05, while making the
        # integration step meaningful for all other caller-selected values.
        increment_limit = max_delta * self.architecture.increment_fraction * (dt / .05)
        result = []
        for row in range(self.state_width):
            result.append(state[row] + increment_limit * tanh(output[row] * self.output_scale))
        return tuple(result)

    def raw_output(self, state: StateVector, aggregate: StateVector) -> StateVector:
        """Return the final MLP layer before output scaling and bounding."""
        if len(state) != self.state_width or len(aggregate) != self.state_width:
            raise ValueError("MLP update inputs must match configured state width")
        return _forward(state + aggregate, self._layers, self.architecture.activation)


@dataclass(frozen=True, slots=True)
class EdgeArchitecture:
    """Fixed experiment configuration for a shared local channel rule.

    Coordinates in ``latent_width`` are intentionally unnamed.  The optional
    projection matrix is configuration, never a genome field.
    """

    node_state_width: int = 4
    latent_width: int = 3
    hidden_width: int = 12
    hidden_layers: tuple[int, ...] | None = None
    activation: Literal["tanh", "relu", "gelu", "silu"] = "tanh"
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
        if self.hidden_layers is not None and (not self.hidden_layers or any(width < 1 for width in self.hidden_layers)):
            raise ValueError("hidden_layers must contain positive widths")

    @property
    def input_width(self) -> int:
        # edge vector, source/target vectors, and current message. Each MLP
        # layer owns an explicit bias vector.
        return self.latent_width + 3 * self.node_state_width

    @property
    def parameter_count(self) -> int:
        widths = self.layers + (self.latent_width,)
        return sum(output * (input_ + 1) for input_, output in zip((self.input_width,) + widths[:-1], widths, strict=True))

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(self.hidden_layers) if self.hidden_layers is not None else (self.hidden_width,)

    @property
    def projection(self) -> tuple[tuple[float, ...], ...]:
        if self.message_projection is not None:
            return self.message_projection
        return tuple(tuple(1.0 if row == column else 0.0 for column in range(self.node_state_width)) for row in range(self.node_state_width))


class MLPEdgeRule(EdgeRule):
    """Shared edge update with bounded increments and learned coordinate-wise gates."""

    def __init__(self, architecture: EdgeArchitecture, parameters: Sequence[float], *, output_scale: float = 1.0) -> None:
        self.architecture = architecture
        self.state_width = architecture.latent_width
        if len(parameters) != architecture.parameter_count:
            raise ValueError(f"expected {architecture.parameter_count} edge-rule parameters, received {len(parameters)}")
        self.parameters = tuple(float(value) for value in parameters)
        if output_scale <= 0:
            raise ValueError("rule output scale must be positive")
        self.output_scale = float(output_scale)
        self._layers = _decode_layers(self.parameters, architecture.input_width, architecture.layers + (architecture.latent_width,))

    def initial_state(self) -> StateVector:
        return (0.0,) * self.state_width

    def update(
        self, state: StateVector, source: StateVector, target: StateVector, message: StateVector,
        edge_step_scale: float,
    ) -> StateVector:
        width = self.architecture.node_state_width
        if (len(state) != self.state_width or any(len(vector) != width for vector in (source, target, message))):
            raise ValueError("edge update inputs must match the configured architecture")
        output = self.raw_output(state, source, target, message)
        increments = []
        for row in range(self.state_width):
            increments.append(edge_step_scale * tanh(output[row] * self.output_scale))
        return tuple(value + increment for value, increment in zip(state, increments, strict=True))

    def raw_output(
        self, state: StateVector, source: StateVector, target: StateVector, message: StateVector,
    ) -> StateVector:
        width = self.architecture.node_state_width
        if len(state) != self.state_width or any(len(vector) != width for vector in (source, target, message)):
            raise ValueError("edge update inputs must match the configured architecture")
        return _forward(state + source + target + message, self._layers, self.architecture.activation)

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
        edge_step_scale: float,
    ) -> StateVector:
        return state
