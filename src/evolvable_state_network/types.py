"""Shared type aliases and deliberately small numerical helpers."""

from __future__ import annotations

from math import isfinite
from typing import TypeAlias

StateVector: TypeAlias = tuple[float, ...]
NodeTensor: TypeAlias = list[list[StateVector]]  # batch, node, component
EdgeTensor: TypeAlias = list[list[StateVector]]  # batch, edge, component


def clip(value: float, limit: float) -> float:
    """Clamp a scalar symmetrically, requiring a positive finite limit."""
    if limit <= 0 or not isfinite(limit):
        raise ValueError("limit must be positive and finite")
    return max(-limit, min(limit, value))


def add(left: StateVector, right: StateVector) -> StateVector:
    if len(left) != len(right):
        raise ValueError("state-vector dimensions must match")
    return tuple(a + b for a, b in zip(left, right, strict=True))


def zeros(width: int) -> StateVector:
    if width < 1:
        raise ValueError("node state width must be at least one")
    return (0.0,) * width
