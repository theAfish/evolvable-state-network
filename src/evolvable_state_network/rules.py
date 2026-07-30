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
        dt: float,
        max_delta: float,
    ) -> StateVector:
        """Return the next state; implementations must honour ``max_delta``."""


class EdgeRule(Protocol):
    """Shared local transition for a generic stateful communication channel."""

    state_width: int

    def initial_state(self) -> StateVector:
        """Return the common initial vector for every edge."""

    def update(
        self,
        state: StateVector,
        source: StateVector,
        target: StateVector,
        message: StateVector,
        edge_step_scale: float,
    ) -> StateVector:
        """Update from strictly local, current quantities only."""

    def message(self, state: StateVector, source: StateVector) -> StateVector:
        """Emit a vector whose width equals the node-state width."""

    def communication_strength(self, state: StateVector) -> float:
        """Return the bounded scalar used to inspect effective communication."""


class StatelessEdgeRule:
    """A fixed identity channel used where runtime adaptation is disabled."""

    state_width = 0

    def initial_state(self) -> StateVector:
        return ()

    def update(
        self, state: StateVector, source: StateVector, target: StateVector, message: StateVector,
        edge_step_scale: float,
    ) -> StateVector:
        return state

    def message(self, state: StateVector, source: StateVector) -> StateVector:
        return source

    def communication_strength(self, state: StateVector) -> float:
        return 1.0
