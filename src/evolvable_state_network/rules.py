"""Replaceable local rules. They receive no global state or node identifiers."""

from __future__ import annotations

from typing import Protocol

from .types import StateVector


class NodeRule(Protocol):
    """Shared node transition interface for one local state vector."""

    state_width: int

    def initial_state(self) -> StateVector:
        """Return the common initial vector for every node."""

    def update(
        self,
        state: StateVector,
        aggregate: StateVector,
        external: StateVector,
        dt: float,
        max_delta: float,
    ) -> StateVector:
        """Return the next state; implementations must honour ``max_delta``."""


class EdgeRule(Protocol):
    """Optional shared connection-state transition and local message interface."""

    state_width: int

    def initial_state(self) -> StateVector:
        """Return the common initial vector for every edge."""

    def update(
        self,
        state: StateVector,
        source: StateVector,
        target: StateVector,
        dt: float,
        max_delta: float,
    ) -> StateVector:
        """Update only from the endpoint states and its own state."""

    def message(self, state: StateVector, source: StateVector) -> StateVector:
        """Emit a vector whose width equals the node-state width."""


class StatelessEdgeRule:
    """A default identity connection rule with zero-dimensional edge state."""

    state_width = 0

    def initial_state(self) -> StateVector:
        return ()

    def update(
        self, state: StateVector, source: StateVector, target: StateVector, dt: float, max_delta: float
    ) -> StateVector:
        return state

    def message(self, state: StateVector, source: StateVector) -> StateVector:
        return source
