"""Local disturbances applied deterministically by the simulator."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Protocol

from .types import StateVector


class Perturbation(Protocol):
    start: int

    def active(self, step: int) -> bool:
        """Whether this perturbation affects the specified integration step."""


@dataclass(frozen=True, slots=True)
class InputDistributionShift:
    """Scale and offset all external signals during a finite step interval."""

    start: int
    end: int
    offset: float = 0.0
    scale: float = 1.0

    def active(self, step: int) -> bool:
        return self.start <= step <= self.end


@dataclass(frozen=True, slots=True)
class ImpulseInjection:
    """Add a finite vector to selected node states at one integration step."""

    step: int
    nodes: tuple[int, ...]
    amount: float | StateVector

    @property
    def start(self) -> int:
        return self.step

    def active(self, step: int) -> bool:
        return step == self.step


@dataclass(frozen=True, slots=True)
class NodeLesion:
    """Clamp selected nodes to zero and remove their messages for an interval."""

    start: int
    nodes: tuple[int, ...]
    end: int | None = None

    def active(self, step: int) -> bool:
        return step >= self.start and (self.end is None or step <= self.end)


@dataclass(frozen=True, slots=True)
class WeightNoise:
    """Add reproducible, independent connection-weight noise for an interval."""

    start: int
    end: int
    standard_deviation: float
    seed: int

    def active(self, step: int) -> bool:
        return self.start <= step <= self.end

    def sample(self, step: int, edge_index: int, batch_index: int) -> float:
        return Random(self.seed + 1_000_003 * step + 10_007 * edge_index + batch_index).gauss(
            0.0, self.standard_deviation
        )
