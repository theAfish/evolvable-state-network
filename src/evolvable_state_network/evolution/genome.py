"""Deterministic flat-vector representation for evolution parameter groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from .candidate import EdgeArchitecture, MLPEdgeRule, MLPUpdateRule, RuleArchitecture

EvolutionTarget = Literal["node", "edge", "joint"]


@dataclass(frozen=True, slots=True)
class GenomeCodec:
    architecture: RuleArchitecture
    edge_architecture: EdgeArchitecture | None = None
    target: EvolutionTarget = "node"

    def __post_init__(self) -> None:
        if self.target in ("edge", "joint") and self.edge_architecture is None:
            raise ValueError("an edge architecture is required when edge parameters evolve")
        if self.edge_architecture is not None and self.edge_architecture.node_state_width != self.architecture.state_width:
            raise ValueError("node and edge architectures must agree on node-state width")

    @property
    def dimension(self) -> int:
        return self.node_dimension + self.edge_dimension

    @property
    def node_dimension(self) -> int:
        return self.architecture.parameter_count if self.target in ("node", "joint") else 0

    @property
    def edge_dimension(self) -> int:
        return self.edge_architecture.parameter_count if self.target in ("edge", "joint") and self.edge_architecture else 0

    def decode(self, genome: Sequence[float]) -> MLPUpdateRule:
        """Backwards-compatible node-only decode; use ``decode_groups`` otherwise."""
        if self.target != "node":
            raise ValueError("decode_groups is required for edge or joint genomes")
        return self.decode_node(genome)

    def encode(self, rule: MLPUpdateRule) -> tuple[float, ...]:
        if self.target != "node":
            raise ValueError("export_groups is required for edge or joint genomes")
        if rule.architecture != self.architecture:
            raise ValueError("rule architecture does not match this codec")
        return rule.parameters

    def decode_node(self, parameters: Sequence[float]) -> MLPUpdateRule:
        if len(parameters) != self.node_dimension:
            raise ValueError(f"node parameter dimension must be {self.node_dimension}")
        return MLPUpdateRule(self.architecture, tuple(float(value) for value in parameters))

    def decode_edge(self, parameters: Sequence[float]) -> MLPEdgeRule:
        if self.edge_architecture is None or len(parameters) != self.edge_dimension:
            raise ValueError(f"edge parameter dimension must be {self.edge_dimension}")
        return MLPEdgeRule(self.edge_architecture, tuple(float(value) for value in parameters))

    def split(self, genome: Sequence[float]) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if len(genome) != self.dimension:
            raise ValueError(f"genome dimension must be {self.dimension}")
        values = tuple(float(value) for value in genome)
        return values[: self.node_dimension], values[self.node_dimension :]

    def decode_groups(self, genome: Sequence[float]) -> tuple[MLPUpdateRule | None, MLPEdgeRule | None]:
        node, edge = self.split(genome)
        return (self.decode_node(node) if node else None, self.decode_edge(edge) if edge else None)

    def export_groups(self, genome: Sequence[float]) -> dict[str, list[float]]:
        node, edge = self.split(genome)
        return {"node": list(node), "edge": list(edge)}

    def restore_groups(self, groups: dict[str, Sequence[float]]) -> tuple[float, ...]:
        node = tuple(float(value) for value in groups.get("node", ()))
        edge = tuple(float(value) for value in groups.get("edge", ()))
        if len(node) != self.node_dimension or len(edge) != self.edge_dimension:
            raise ValueError("parameter groups do not match this codec")
        return node + edge
