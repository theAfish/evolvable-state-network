"""Small fixed reference rules for comparison, not claims about biology."""

from __future__ import annotations

from math import tanh

from .types import StateVector, clip


class FixedRNNRule:
    """A conventional fixed one-coordinate recurrent rule reference."""

    state_width = 1

    def __init__(self, recurrent_scale: float = 1.25, input_scale: float = 1.0, leak: float = 1.0) -> None:
        self.recurrent_scale = recurrent_scale
        self.input_scale = input_scale
        self.leak = leak

    def initial_state(self) -> StateVector:
        return (0.0,)

    def update(
        self, state: StateVector, aggregate: StateVector, dt: float, max_delta: float
    ) -> StateVector:
        candidate = tanh(self.recurrent_scale * aggregate[0])
        return (state[0] + clip(dt * self.leak * (candidate - state[0]) / 0.2, max_delta),)


class HomeostaticRule:
    """A hand-designed four-coordinate stabilizing reference rule.

    The coordinates are deliberately not public semantic categories. This rule
    merely provides a fixed, engineered contrast to the one-coordinate RNN;
    later evolved rules may use any finite vector dimension instead.
    """

    state_width = 4

    def __init__(self, target_scale: float = 0.35) -> None:
        self.target_scale = target_scale

    def initial_state(self) -> StateVector:
        return (0.0, 0.0, 0.0, 0.0)

    def update(
        self, state: StateVector, aggregate: StateVector, dt: float, max_delta: float
    ) -> StateVector:
        x, average, spread, control = state
        magnitude = abs(x)
        next_average = average + clip(dt * (magnitude - average) / 0.7, max_delta)
        next_spread = spread + clip(dt * ((magnitude - average) ** 2 - spread) / 1.1, max_delta)
        next_control = control + clip(
            dt * (0.6 * (self.target_scale - next_average) - 0.2 * control - 0.1 * next_spread), max_delta
        )
        candidate = tanh((1.0 + next_control) * aggregate[0] - 0.18 * x)
        next_x = x + clip(dt * (candidate - x) / 0.2, max_delta)
        return (next_x, next_average, next_spread, next_control)
