"""Evaluation-time ablations for generic, unnamed state dimensions."""

from __future__ import annotations

from dataclasses import dataclass

from .types import StateVector


@dataclass(frozen=True, slots=True)
class StateIntervention:
    """Local ablations applied uniformly to every node during one evaluation.

    These controls are deliberately expressed only in terms of vector indices;
    they do not assign a meaning to any state component.
    """

    freeze_dimensions: tuple[int, ...] = ()
    zero_dimensions: tuple[int, ...] = ()
    mask_dimensions: tuple[int, ...] = ()
    permutation: tuple[int, ...] | None = None
    disable_neighbor_messages: bool = False
    disable_external_inputs: bool = False

    def validate(self, width: int) -> None:
        indices = self.freeze_dimensions + self.zero_dimensions + self.mask_dimensions
        if any(index < 0 or index >= width for index in indices):
            raise ValueError("intervention dimension is outside node state width")
        if self.permutation is not None and sorted(self.permutation) != list(range(width)):
            raise ValueError("permutation must contain every state dimension exactly once")

    def initial(self, vector: StateVector) -> StateVector:
        self.validate(len(vector))
        return tuple(0.0 if index in self.zero_dimensions or index in self.mask_dimensions else value for index, value in enumerate(vector))

    def local_inputs(self, aggregate: StateVector, external: StateVector) -> tuple[StateVector, StateVector]:
        self.validate(len(aggregate))
        if self.disable_neighbor_messages:
            aggregate = (0.0,) * len(aggregate)
        if self.disable_external_inputs:
            external = (0.0,) * len(external)
        return aggregate, external

    def transition(self, previous: StateVector, proposed: StateVector) -> StateVector:
        self.validate(len(previous))
        values = list(proposed)
        if self.permutation is not None:
            values = [values[index] for index in self.permutation]
        for index in self.freeze_dimensions:
            values[index] = previous[index]
        for index in self.zero_dimensions + self.mask_dimensions:
            values[index] = 0.0
        return tuple(values)
