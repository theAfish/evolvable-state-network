"""A deterministic, diversity-preserving real-valued genetic algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from math import sqrt
from statistics import fmean, pstdev
from typing import Literal, Sequence


@dataclass(frozen=True, slots=True)
class GeneticAlgorithmConfig:
    dimension: int
    population_size: int = 16
    mutation_sigma: float = .25
    seed: int = 1
    elite_fraction: float = .25
    immigrant_fraction: float = .25
    immigrant_sigma: float = 1.0
    tournament_size: int = 3
    immigrant_mode: Literal["zero", "population"] = "zero"
    max_genome_norm: float | None = None
    max_parameter_magnitude: float | None = None

    def __post_init__(self) -> None:
        if self.dimension < 1 or self.population_size < 2 or self.mutation_sigma <= 0 or self.immigrant_sigma <= 0:
            raise ValueError("invalid genetic-algorithm configuration")
        if not 0 < self.elite_fraction < 1 or not 0 <= self.immigrant_fraction < 1:
            raise ValueError("genetic-algorithm fractions are invalid")
        if self.tournament_size < 2:
            raise ValueError("genetic-algorithm tournament size must be at least two")
        if self.immigrant_mode not in {"zero", "population"}:
            raise ValueError("unknown immigrant mode")
        if self.max_genome_norm is not None and self.max_genome_norm <= 0:
            raise ValueError("max_genome_norm must be positive when enabled")
        if self.max_parameter_magnitude is not None and self.max_parameter_magnitude <= 0:
            raise ValueError("max_parameter_magnitude must be positive when enabled")


class GeneticAlgorithm:
    """Maximising generational GA with elitism, crossover, mutation, and immigrants."""

    def __init__(self, config: GeneticAlgorithmConfig, center: Sequence[float] | None = None) -> None:
        self.config = config
        self._random = Random(config.seed)
        self._center = tuple(float(value) for value in center) if center is not None else (0.0,) * config.dimension
        if len(self._center) != config.dimension:
            raise ValueError("genetic-algorithm center has the wrong dimension")
        self._parents: tuple[tuple[float, ...], ...] = ()
        self._fitnesses: tuple[float, ...] = ()
        self._pending: tuple[tuple[float, ...], ...] | None = None
        self._generation = 0
        self._normalizations = 0
        self._last_ask_normalizations = 0

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def sigma(self) -> float:
        """Mutation scale, exposed alongside CMA-ES sigma for shared telemetry."""
        return self.config.mutation_sigma

    @property
    def normalization_count(self) -> int:
        return self._normalizations

    @property
    def last_ask_normalizations(self) -> int:
        return self._last_ask_normalizations

    def ask(self) -> tuple[tuple[float, ...], ...]:
        if self._pending is not None:
            raise RuntimeError("tell must consume the current genetic-algorithm population before ask")
        normalizations_before = self._normalizations
        if not self._parents:
            population = [self._center]
            population.extend(self._mutate(self._center, self.config.immigrant_sigma) for _ in range(self.config.population_size - 1))
        else:
            elite_count = max(1, min(self.config.population_size - 1, round(self.config.population_size * self.config.elite_fraction)))
            immigrant_count = min(self.config.population_size - elite_count, round(self.config.population_size * self.config.immigrant_fraction))
            ranked = sorted(zip(self._fitnesses, self._parents, strict=True), key=lambda item: item[0], reverse=True)
            population = [genome for _, genome in ranked[:elite_count]]
            offspring_count = self.config.population_size - elite_count - immigrant_count
            for _ in range(offspring_count):
                first, second = self._select(), self._select()
                child = tuple(a if self._random.random() < .5 else b for a, b in zip(first, second, strict=True))
                population.append(self._mutate(child, self.config.mutation_sigma))
            population.extend(self._immigrant() for _ in range(immigrant_count))
        self._pending = tuple(population)
        self._last_ask_normalizations = self._normalizations - normalizations_before
        return self._pending

    def tell(self, population: Sequence[Sequence[float]], fitnesses: Sequence[float]) -> None:
        encoded = tuple(tuple(float(value) for value in genome) for genome in population)
        if self._pending is None or encoded != self._pending:
            raise ValueError("tell must receive the exact population returned by ask")
        if len(fitnesses) != self.config.population_size:
            raise ValueError("tell requires exactly one full genetic-algorithm population")
        self._parents, self._fitnesses = encoded, tuple(float(value) for value in fitnesses)
        self._pending = None
        self._generation += 1

    def _select(self) -> tuple[float, ...]:
        choices = [self._random.randrange(len(self._parents)) for _ in range(self.config.tournament_size)]
        return self._parents[max(choices, key=lambda index: self._fitnesses[index])]

    def _mutate(self, genome: Sequence[float], sigma: float) -> tuple[float, ...]:
        return self._protect(tuple(float(value) + self._random.gauss(0.0, sigma) for value in genome))

    def _immigrant(self) -> tuple[float, ...]:
        if self.config.immigrant_mode == "zero":
            return self._mutate(self._center, self.config.immigrant_sigma)
        means = tuple(fmean(parent[index] for parent in self._parents) for index in range(self.config.dimension))
        deviations = tuple(pstdev(parent[index] for parent in self._parents) for index in range(self.config.dimension))
        return self._protect(tuple(self._random.gauss(mean, deviation) for mean, deviation in zip(means, deviations, strict=True)))

    def _protect(self, genome: tuple[float, ...]) -> tuple[float, ...]:
        values = genome
        changed = False
        if self.config.max_parameter_magnitude is not None:
            limit = self.config.max_parameter_magnitude
            bounded = tuple(max(-limit, min(limit, value)) for value in values)
            changed = changed or bounded != values
            values = bounded
        if self.config.max_genome_norm is not None:
            norm = sqrt(sum(value * value for value in values))
            if norm > self.config.max_genome_norm:
                factor = self.config.max_genome_norm / norm
                values = tuple(value * factor for value in values)
                changed = True
        if changed:
            self._normalizations += 1
        return values


def population_statistics(population: Sequence[Sequence[float]], *, node_dimension: int | None = None) -> dict[str, float]:
    """Compact, comparable genotype telemetry for every GA generation."""
    if not population:
        return {"genome_rms": 0.0, "genome_l2_norm": 0.0, "population_parameter_diversity": 0.0}
    rows = [tuple(float(value) for value in genome) for genome in population]
    flat = [value for row in rows for value in row]
    rms = sqrt(fmean(value * value for value in flat))
    norms = [sqrt(sum(value * value for value in row)) for row in rows]
    diversity = fmean(pstdev(row[index] for row in rows) for index in range(len(rows[0])))
    result = {
        "genome_rms": rms, "genome_l2_norm": fmean(norms),
        "population_parameter_diversity": diversity,
    }
    if node_dimension is not None:
        result["node_rule_parameter_norm"] = fmean(
            sqrt(sum(value * value for value in row[:node_dimension])) for row in rows
        )
        result["edge_rule_parameter_norm"] = fmean(
            sqrt(sum(value * value for value in row[node_dimension:])) for row in rows
        )
    return result
