"""Deterministic external signal providers for batched simulations."""

from __future__ import annotations

from random import Random
from typing import Protocol

from .types import NodeTensor, StateVector


class InputProvider(Protocol):
    def sample(self, step: int, batch_size: int, n_nodes: int, width: int) -> NodeTensor:
        """Return an external vector for every batch item and node."""


class GaussianInput:
    """Index-addressed pseudo-random input, invariant to simulator traversal."""

    def __init__(self, seed: int, mean: float = 0.0, standard_deviation: float = 0.25) -> None:
        if standard_deviation < 0:
            raise ValueError("standard_deviation cannot be negative")
        self.seed = seed
        self.mean = mean
        self.standard_deviation = standard_deviation

    def sample(self, step: int, batch_size: int, n_nodes: int, width: int) -> NodeTensor:
        return [
            [
                tuple(
                    Random(self.seed + 1_000_003 * step + 10_007 * batch + 101 * node + component).gauss(
                        self.mean, self.standard_deviation
                    )
                    for component in range(width)
                )
                for node in range(n_nodes)
            ]
            for batch in range(batch_size)
        ]


class ConstantInput:
    """A simple deterministic signal useful for controlled tests."""

    def __init__(self, value: float | StateVector = 0.0) -> None:
        self.value = value

    def sample(self, step: int, batch_size: int, n_nodes: int, width: int) -> NodeTensor:
        vector = (float(self.value),) * width if isinstance(self.value, (int, float)) else self.value
        if len(vector) != width:
            raise ValueError("constant input width does not match node rule")
        return [[tuple(vector) for _ in range(n_nodes)] for _ in range(batch_size)]
