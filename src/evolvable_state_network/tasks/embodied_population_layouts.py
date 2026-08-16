"""Population layouts for batch embodied-food-web evolution.

Layouts decide what an optimizer population means.  They deliberately do not
know how an episode is simulated: the food-web task owns that concern.  This
keeps the legacy shared-rule evaluation and the mixed-individual evaluation
independently removable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence


BatchPopulationMode = Literal["shared_rule_cohort", "mixed_individual_population"]
Genome = tuple[float, ...]


class BatchPopulationLayout:
    """Map optimizer genomes to one or more world-level evaluation groups."""

    mode: BatchPopulationMode

    def genome_population_size(self, world_count: int, agents_per_world: int) -> int:
        raise NotImplementedError

    def groups(
        self, population: Sequence[Genome], agents_per_world: int,
    ) -> tuple[tuple[Genome, ...], ...]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class SharedRuleCohortLayout(BatchPopulationLayout):
    """Evaluate one inherited rule through a same-rule cohort in each world."""

    mode: BatchPopulationMode = "shared_rule_cohort"

    def genome_population_size(self, world_count: int, agents_per_world: int) -> int:
        _validate_counts(world_count, agents_per_world)
        return world_count

    def groups(
        self, population: Sequence[Genome], agents_per_world: int,
    ) -> tuple[tuple[Genome, ...], ...]:
        if agents_per_world < 1:
            raise ValueError("agents_per_world must be positive")
        return tuple((tuple(genome),) for genome in population)


@dataclass(frozen=True, slots=True)
class MixedIndividualPopulationLayout(BatchPopulationLayout):
    """Place distinct inherited rules together and select every individual."""

    mode: BatchPopulationMode = "mixed_individual_population"

    def genome_population_size(self, world_count: int, agents_per_world: int) -> int:
        _validate_counts(world_count, agents_per_world)
        return world_count * agents_per_world

    def groups(
        self, population: Sequence[Genome], agents_per_world: int,
    ) -> tuple[tuple[Genome, ...], ...]:
        if agents_per_world < 1 or len(population) % agents_per_world:
            raise ValueError("mixed populations must divide evenly into complete worlds")
        return tuple(
            tuple(tuple(genome) for genome in population[index:index + agents_per_world])
            for index in range(0, len(population), agents_per_world)
        )


def get_batch_population_layout(mode: BatchPopulationMode) -> BatchPopulationLayout:
    if mode == "shared_rule_cohort":
        return SharedRuleCohortLayout()
    if mode == "mixed_individual_population":
        return MixedIndividualPopulationLayout()
    raise ValueError(f"unknown batch population mode: {mode}")


def _validate_counts(world_count: int, agents_per_world: int) -> None:
    if world_count < 1 or agents_per_world < 1:
        raise ValueError("world_count and agents_per_world must be positive")
