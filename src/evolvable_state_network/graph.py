"""Reproducible directed graphs used by local message-passing simulations."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random


@dataclass(frozen=True, slots=True)
class Edge:
    """One directed connection; its mutable state lives in a simulation state."""

    source: int
    target: int
    weight: float


@dataclass(frozen=True, slots=True)
class Graph:
    """An immutable directed graph with deterministic incoming-edge lookup."""

    n_nodes: int
    edges: tuple[Edge, ...]

    def __post_init__(self) -> None:
        if self.n_nodes < 1:
            raise ValueError("n_nodes must be positive")
        for edge in self.edges:
            if not (0 <= edge.source < self.n_nodes and 0 <= edge.target < self.n_nodes):
                raise ValueError("edge endpoint is outside graph")

    @property
    def incoming(self) -> tuple[tuple[int, ...], ...]:
        buckets: list[list[int]] = [[] for _ in range(self.n_nodes)]
        for index, edge in enumerate(self.edges):
            buckets[edge.target].append(index)
        return tuple(tuple(bucket) for bucket in buckets)


def generate_random_graph(
    n_nodes: int,
    mean_degree: float,
    seed: int,
    topology: str = "erdos_renyi",
    weight_scale: float = 0.8,
) -> Graph:
    """Create a deterministic graph without node IDs exposed to update rules.

    ``erdos_renyi`` samples each permitted directed edge independently.
    ``ring`` creates a deterministic bidirectional local ring, useful for a
    topology contrast. Edge weights are independently sampled in both cases.
    """
    if n_nodes < 1 or mean_degree < 0 or weight_scale <= 0:
        raise ValueError("invalid graph generation parameters")
    rng = Random(seed)
    pairs: list[tuple[int, int]] = []
    if topology == "erdos_renyi":
        probability = min(1.0, mean_degree / max(1, n_nodes - 1))
        pairs = [
            (source, target)
            for source in range(n_nodes)
            for target in range(n_nodes)
            if source != target and rng.random() < probability
        ]
    elif topology == "ring":
        radius = min((n_nodes - 1) // 2, max(0, round(mean_degree / 2)))
        pairs = [
            (source, (source + offset) % n_nodes)
            for source in range(n_nodes)
            for offset in range(1, radius + 1)
        ] + [
            (source, (source - offset) % n_nodes)
            for source in range(n_nodes)
            for offset in range(1, radius + 1)
        ]
    else:
        raise ValueError(f"unknown topology: {topology}")
    edges = tuple(Edge(source, target, rng.uniform(-weight_scale, weight_scale)) for source, target in pairs)
    return Graph(n_nodes=n_nodes, edges=edges)
