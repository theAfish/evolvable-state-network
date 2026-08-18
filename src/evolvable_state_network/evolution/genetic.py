"""A deterministic, explicitly multi-scale real-valued genetic algorithm."""

from __future__ import annotations

from base64 import b64decode, b64encode
from dataclasses import asdict, dataclass
from math import isfinite, sqrt, tanh
import pickle
from random import Random
from statistics import fmean, pstdev
from typing import Callable, Literal, Sequence

from .genome import GenomeCodec


SamplingSource = Literal["elite", "local_offspring", "regional_immigrant", "global_immigrant"]


@dataclass(frozen=True, slots=True)
class ViabilityResult:
    """Result of a task-independent rule-dynamics probe."""

    viable: bool
    diagnostics: dict[str, float]
    reasons: tuple[str, ...] = ()

    @property
    def degeneracy(self) -> float:
        return sum(max(0.0, value) for value in self.diagnostics.values() if isfinite(value)) + 1000.0 * len(self.reasons)


@dataclass(frozen=True, slots=True)
class RuleDynamicsViabilityProbe:
    """Synthetic rule probe that never evaluates task fitness."""

    codec: GenomeCodec
    node_update_limit: float = .12
    edge_update_limit: float = .06
    state_limit: float = 1.0
    probe_count: int = 12
    raw_saturation_threshold: float = 3.0
    max_raw_saturation_fraction: float = .85
    max_update_limit_fraction: float = .85
    max_near_zero_update_fraction: float = .95
    near_zero_update_fraction: float = .01
    seed: int = 0

    def __post_init__(self) -> None:
        if self.probe_count < 1 or self.node_update_limit <= 0 or self.edge_update_limit <= 0:
            raise ValueError("rule-dynamics probe controls are invalid")
        if any(not 0 <= value <= 1 for value in (self.max_raw_saturation_fraction, self.max_update_limit_fraction, self.max_near_zero_update_fraction, self.near_zero_update_fraction)):
            raise ValueError("rule-dynamics probe fractions must be in [0, 1]")

    def __call__(self, genome: Sequence[float]) -> ViabilityResult:
        node, edge = self.codec.decode_groups(genome)
        random = Random(self.seed)
        node_raw: list[float] = []
        edge_raw: list[float] = []
        for _ in range(self.probe_count):
            state = tuple(random.uniform(-self.state_limit, self.state_limit) for _ in range(self.codec.architecture.state_width))
            aggregate = tuple(random.uniform(-self.state_limit, self.state_limit) for _ in range(self.codec.architecture.state_width))
            if node is not None:
                node_raw.extend(node.raw_output(state, aggregate))
            if edge is not None:
                edge_state = tuple(random.uniform(-1.0, 1.0) for _ in range(edge.architecture.latent_width))
                source = tuple(random.uniform(-self.state_limit, self.state_limit) for _ in range(edge.architecture.node_state_width))
                target = tuple(random.uniform(-self.state_limit, self.state_limit) for _ in range(edge.architecture.node_state_width))
                message = tuple(random.uniform(-self.state_limit, self.state_limit) for _ in range(edge.architecture.node_state_width))
                edge_raw.extend(edge.raw_output(edge_state, source, target, message))
        diagnostics = _dynamics_summary(node_raw, self.node_update_limit, "node", self.raw_saturation_threshold, self.near_zero_update_fraction)
        diagnostics.update(_dynamics_summary(edge_raw, self.edge_update_limit, "edge", self.raw_saturation_threshold, self.near_zero_update_fraction))
        reasons: list[str] = []
        for kind in ("node", "edge"):
            if not diagnostics.get(f"{kind}_output_count", 0):
                continue
            if diagnostics[f"{kind}_nonfinite_fraction"] > 0: reasons.append(f"{kind}_nonfinite")
            if diagnostics[f"{kind}_raw_saturation_fraction"] > self.max_raw_saturation_fraction: reasons.append(f"{kind}_raw_saturation")
            if diagnostics[f"{kind}_update_limit_fraction"] > self.max_update_limit_fraction: reasons.append(f"{kind}_update_limit")
            if diagnostics[f"{kind}_near_zero_update_fraction"] > self.max_near_zero_update_fraction: reasons.append(f"{kind}_near_zero")
        return ViabilityResult(not reasons, diagnostics, tuple(reasons))


def _dynamics_summary(raw: Sequence[float], limit: float, prefix: str, saturation: float, near_zero: float) -> dict[str, float]:
    if not raw:
        return {f"{prefix}_output_count": 0.0}
    finite = [value for value in raw if isfinite(value)]
    count = len(raw)
    updates = [limit * tanh(value) for value in finite]
    return {
        f"{prefix}_output_count": float(count),
        f"{prefix}_nonfinite_fraction": 1.0 - len(finite) / count,
        f"{prefix}_raw_saturation_fraction": sum(abs(value) > saturation for value in finite) / count,
        f"{prefix}_near_zero_update_fraction": sum(abs(value) <= near_zero * limit for value in updates) / count,
        f"{prefix}_update_limit_fraction": sum(abs(value) >= .98 * limit for value in updates) / count,
    }


@dataclass(frozen=True, slots=True)
class GeneticAlgorithmConfig:
    dimension: int
    population_size: int = 16
    # Compatibility alias. New callers should use local_mutation_sigma.
    mutation_sigma: float = .25
    seed: int = 1
    elite_fraction: float = .25
    immigrant_fraction: float = .25
    immigrant_sigma: float = 1.0
    tournament_size: int = 3
    immigrant_mode: Literal["zero", "population"] = "zero"
    local_mutation_sigma: float | None = None
    local_offspring_fraction: float | None = None
    regional_fraction: float = 0.0
    regional_scale: float = 1.0
    regional_min_std: float = .02
    global_fraction: float = 0.0
    global_distribution: Literal["uniform"] = "uniform"
    global_parameter_range: float = 1.0
    global_viability_filter: bool = False
    global_max_sampling_attempts: int = 20
    max_genome_norm: float | None = None
    max_parameter_magnitude: float | None = None

    def __post_init__(self) -> None:
        if self.dimension < 1 or self.population_size < 2 or self.local_sigma <= 0 or self.immigrant_sigma <= 0:
            raise ValueError("invalid genetic-algorithm configuration")
        if not 0 < self.elite_fraction < 1 or any(not 0 <= value < 1 for value in (self.regional_fraction, self.global_fraction)):
            raise ValueError("genetic-algorithm fractions are invalid")
        if self.local_offspring_fraction is not None and (
            not 0 <= self.local_offspring_fraction < 1
            or abs(self.elite_fraction + self.local_offspring_fraction + self.regional_fraction + self.global_fraction - 1.0) > 1e-12
        ):
            raise ValueError("explicit sampling fractions are invalid")
        if self.local_offspring_fraction is None and self.elite_fraction + self.regional_fraction + self.global_fraction > 1 + 1e-12:
            raise ValueError("elite, regional, and global fractions cannot exceed one")
        if not 0 <= self.immigrant_fraction < 1 or self.tournament_size < 2:
            raise ValueError("genetic-algorithm controls are invalid")
        if self.immigrant_mode not in {"zero", "population"} or self.global_distribution != "uniform":
            raise ValueError("unknown immigrant distribution")
        if self.regional_scale <= 0 or self.regional_min_std <= 0 or self.global_parameter_range <= 0 or self.global_max_sampling_attempts < 1:
            raise ValueError("exploration controls are invalid")
        if self.max_genome_norm is not None and self.max_genome_norm <= 0: raise ValueError("max_genome_norm must be positive when enabled")
        if self.max_parameter_magnitude is not None and self.max_parameter_magnitude <= 0: raise ValueError("max_parameter_magnitude must be positive when enabled")

    @property
    def local_sigma(self) -> float:
        return self.mutation_sigma if self.local_mutation_sigma is None else self.local_mutation_sigma

    @property
    def uses_explicit_multiscale_sampling(self) -> bool:
        return self.local_offspring_fraction is not None or self.regional_fraction > 0 or self.global_fraction > 0


class GeneticAlgorithm:
    """Maximising GA with explicit local, regional, and global sampling."""

    def __init__(self, config: GeneticAlgorithmConfig, center: Sequence[float] | None = None, *, global_viability_probe: Callable[[Sequence[float]], ViabilityResult | bool] | None = None) -> None:
        self.config = config
        self._random = Random(config.seed)
        self._center = tuple(float(value) for value in center) if center is not None else (0.0,) * config.dimension
        if len(self._center) != config.dimension: raise ValueError("genetic-algorithm center has the wrong dimension")
        if config.global_viability_filter and global_viability_probe is None: raise ValueError("global_viability_filter requires a task-independent global_viability_probe")
        self._global_viability_probe = global_viability_probe
        self._parents: tuple[tuple[float, ...], ...] = ()
        self._parent_sources: tuple[SamplingSource, ...] = ()
        self._fitnesses: tuple[float, ...] = ()
        self._pending: tuple[tuple[float, ...], ...] | None = None
        self._pending_sources: tuple[SamplingSource, ...] = ()
        self._generation = self._normalizations = self._last_ask_normalizations = 0
        self._last_sampling: dict[str, object] = {}

    @property
    def generation(self) -> int: return self._generation
    @property
    def sigma(self) -> float: return self.config.local_sigma
    @property
    def normalization_count(self) -> int: return self._normalizations
    @property
    def last_ask_normalizations(self) -> int: return self._last_ask_normalizations
    @property
    def pending_sources(self) -> tuple[SamplingSource, ...]: return self._pending_sources
    @property
    def last_sampling_telemetry(self) -> dict[str, object]: return dict(self._last_sampling)

    def ask(self) -> tuple[tuple[float, ...], ...]:
        if self._pending is not None: raise RuntimeError("tell must consume the current genetic-algorithm population before ask")
        normalizations_before = self._normalizations
        # Rejection counters are generation-local telemetry, not run totals.
        self._last_sampling = {}
        counts = self._source_counts(); population: list[tuple[float, ...]] = []; sources: list[SamplingSource] = []
        if not self._parents:
            population.append(self._protect(self._center)); sources.append("elite")
            for source, count in counts.items():
                for _ in range(count - (1 if source == "elite" else 0)):
                    population.append(self._initial_sample(source)); sources.append(source)
        else:
            selected_indices = sorted(range(len(self._parents)), key=lambda index: self._fitnesses[index], reverse=True)[:counts["elite"]]
            self._last_sampling["source_survival"] = {
                source: {
                    "parent_count": self._parent_sources.count(source),
                    "elite_survivors": sum(self._parent_sources[index] == source for index in selected_indices),
                    "elite_survival_fraction": sum(self._parent_sources[index] == source for index in selected_indices) / max(1, self._parent_sources.count(source)),
                }
                for source in ("elite", "local_offspring", "regional_immigrant", "global_immigrant")
            }
            ranked = sorted(zip(self._fitnesses, self._parents, strict=True), key=lambda item: item[0], reverse=True)
            population.extend(genome for _, genome in ranked[:counts["elite"]]); sources.extend(["elite"] * counts["elite"])
            for _ in range(counts["local_offspring"]):
                first, second = self._select(), self._select()
                population.append(self._mutate(tuple(a if self._random.random() < .5 else b for a, b in zip(first, second, strict=True)), self.config.local_sigma)); sources.append("local_offspring")
            for _ in range(counts["regional_immigrant"]): population.append(self._regional_immigrant()); sources.append("regional_immigrant")
            for _ in range(counts["global_immigrant"]): population.append(self._global_immigrant()); sources.append("global_immigrant")
        assert len(population) == self.config.population_size
        self._pending, self._pending_sources = tuple(population), tuple(sources)
        self._last_ask_normalizations = self._normalizations - normalizations_before
        self._last_sampling = self._sampling_summary(self._pending, self._pending_sources)
        return self._pending

    def tell(self, population: Sequence[Sequence[float]], fitnesses: Sequence[float]) -> None:
        encoded = tuple(tuple(float(value) for value in genome) for genome in population)
        if self._pending is None or encoded != self._pending: raise ValueError("tell must receive the exact population returned by ask")
        if len(fitnesses) != self.config.population_size: raise ValueError("tell requires exactly one full genetic-algorithm population")
        self._parents, self._fitnesses, self._parent_sources = encoded, tuple(float(value) for value in fitnesses), self._pending_sources
        self._last_sampling = self._sampling_summary(encoded, self._parent_sources, self._fitnesses)
        self._pending, self._pending_sources = None, (); self._generation += 1

    def state_dict(self) -> dict[str, object]:
        return {"config": asdict(self.config), "center": list(self._center), "generation": self._generation, "parents": [list(row) for row in self._parents], "parent_sources": list(self._parent_sources), "fitnesses": list(self._fitnesses), "pending": [list(row) for row in self._pending] if self._pending else None, "pending_sources": list(self._pending_sources), "normalizations": self._normalizations, "last_ask_normalizations": self._last_ask_normalizations, "last_sampling": self._last_sampling, "rng_pickle": b64encode(pickle.dumps(self._random.getstate())).decode("ascii")}

    @classmethod
    def from_state_dict(cls, state: dict[str, object], *, global_viability_probe: Callable[[Sequence[float]], ViabilityResult | bool] | None = None) -> "GeneticAlgorithm":
        instance = cls(GeneticAlgorithmConfig(**dict(state["config"])), state["center"], global_viability_probe=global_viability_probe)
        instance._generation = int(state["generation"]); instance._parents = tuple(tuple(float(value) for value in row) for row in state["parents"]); instance._parent_sources = tuple(state.get("parent_sources", ()))  # type: ignore[assignment]
        instance._fitnesses = tuple(float(value) for value in state["fitnesses"]); pending = state.get("pending"); instance._pending = tuple(tuple(float(value) for value in row) for row in pending) if pending else None; instance._pending_sources = tuple(state.get("pending_sources", ()))  # type: ignore[assignment]
        instance._normalizations = int(state.get("normalizations", 0)); instance._last_ask_normalizations = int(state.get("last_ask_normalizations", 0)); instance._last_sampling = dict(state.get("last_sampling", {})); instance._random.setstate(pickle.loads(b64decode(str(state["rng_pickle"])))); return instance

    def _source_counts(self) -> dict[SamplingSource, int]:
        size = self.config.population_size; elite = max(1, min(size - 1, round(size * self.config.elite_fraction)))
        if self.config.uses_explicit_multiscale_sampling:
            regional = min(size - elite, round(size * self.config.regional_fraction)); global_ = min(size - elite - regional, round(size * self.config.global_fraction))
        else:
            regional = min(size - elite, round(size * self.config.immigrant_fraction)) if self.config.immigrant_mode == "population" else 0
            global_ = min(size - elite - regional, round(size * self.config.immigrant_fraction)) if self.config.immigrant_mode == "zero" else 0
        return {"elite": elite, "local_offspring": size - elite - regional - global_, "regional_immigrant": regional, "global_immigrant": global_}

    def _initial_sample(self, source: SamplingSource) -> tuple[float, ...]:
        if source == "global_immigrant" and self.config.uses_explicit_multiscale_sampling: return self._global_immigrant()
        if source == "regional_immigrant": return self._mutate(self._center, self.config.regional_scale * self.config.regional_min_std)
        return self._mutate(self._center, self.config.immigrant_sigma)
    def _select(self) -> tuple[float, ...]:
        choices = [self._random.randrange(len(self._parents)) for _ in range(self.config.tournament_size)]; return self._parents[max(choices, key=lambda index: self._fitnesses[index])]
    def _mutate(self, genome: Sequence[float], sigma: float) -> tuple[float, ...]: return self._protect(tuple(float(value) + self._random.gauss(0.0, sigma) for value in genome))
    def _regional_immigrant(self) -> tuple[float, ...]:
        reference = self._parents or (self._center,); means = tuple(fmean(row[index] for row in reference) for index in range(self.config.dimension)); deviations = tuple(max(self.config.regional_min_std, pstdev(row[index] for row in reference)) for index in range(self.config.dimension)); return self._protect(tuple(self._random.gauss(mean, self.config.regional_scale * deviation) for mean, deviation in zip(means, deviations, strict=True)))

    def _global_immigrant(self) -> tuple[float, ...]:
        attempts: list[tuple[tuple[float, ...], ViabilityResult]] = []
        for _ in range(self.config.global_max_sampling_attempts):
            candidate = self._protect(tuple(self._random.uniform(-self.config.global_parameter_range, self.config.global_parameter_range) for _ in range(self.config.dimension))) if self.config.uses_explicit_multiscale_sampling else self._mutate(self._center, self.config.immigrant_sigma)
            result = self._probe(candidate); attempts.append((candidate, result))
            if result.viable: break
        candidate, result = attempts[-1] if attempts[-1][1].viable else min(attempts, key=lambda item: item[1].degeneracy)
        telemetry = self._last_sampling.setdefault("global_viability", {"attempts": 0, "rejections": 0, "fallbacks": 0, "rejection_reasons": {}}); assert isinstance(telemetry, dict)
        telemetry["attempts"] = int(telemetry["attempts"]) + len(attempts); telemetry["rejections"] = int(telemetry["rejections"]) + sum(not item[1].viable for item in attempts)
        if not result.viable: telemetry["fallbacks"] = int(telemetry["fallbacks"]) + 1
        reasons = telemetry["rejection_reasons"]; assert isinstance(reasons, dict)
        for _, attempt in attempts:
            if not attempt.viable:
                for reason in attempt.reasons: reasons[reason] = int(reasons.get(reason, 0)) + 1
        return candidate
    def _probe(self, genome: Sequence[float]) -> ViabilityResult:
        if not self.config.global_viability_filter: return ViabilityResult(True, {})
        assert self._global_viability_probe is not None; result = self._global_viability_probe(genome); return result if isinstance(result, ViabilityResult) else ViabilityResult(bool(result), {}, () if result else ("rejected",))

    def _sampling_summary(self, population: Sequence[Sequence[float]], sources: Sequence[SamplingSource], fitnesses: Sequence[float] | None = None) -> dict[str, object]:
        reference = self._parents or (self._center,); mean = tuple(fmean(row[index] for row in reference) for index in range(self.config.dimension)); elites = sorted(zip(self._fitnesses, self._parents, strict=True), reverse=True)[:max(1, min(len(reference), round(self.config.population_size * self.config.elite_fraction)))] if self._parents else [(0.0, self._center)]
        result: dict[str, object] = {"generation": self._generation, "sources": {}, "global_viability": self._last_sampling.get("global_viability", {"attempts": 0, "rejections": 0, "fallbacks": 0, "rejection_reasons": {}}), "source_survival": self._last_sampling.get("source_survival", {})}
        records = result["sources"]; assert isinstance(records, dict)
        for source in ("elite", "local_offspring", "regional_immigrant", "global_immigrant"):
            indices = [index for index, value in enumerate(sources) if value == source]; rows = [population[index] for index in indices]
            if not rows: records[source] = {"count": 0}; continue
            distances = [sqrt(sum((value - center) ** 2 for value, center in zip(row, mean, strict=True))) for row in rows]; nearest = [min(sqrt(sum((value - elite_value) ** 2 for value, elite_value in zip(row, elite, strict=True))) for _, elite in elites) for row in rows]; values = [value for row in rows for value in row]
            record: dict[str, object] = {"count": len(rows), "mean_l2_distance_to_population_mean": fmean(distances), "mean_l2_distance_to_nearest_elite": fmean(nearest), "genome_rms": sqrt(fmean(value * value for value in values))}
            if source == "regional_immigrant": record["mean_displacement_from_population_mean"] = fmean(distances)
            if isinstance(self._global_viability_probe, RuleDynamicsViabilityProbe):
                probe_results = [self._global_viability_probe(row) for row in rows]
                keys = {key for probe_result in probe_results for key in probe_result.diagnostics}
                record["functional_dynamics"] = {
                    key: fmean(probe_result.diagnostics.get(key, 0.0) for probe_result in probe_results)
                    for key in keys
                }
            if fitnesses is not None: record["fitness"] = {"minimum": min(fitnesses[index] for index in indices), "maximum": max(fitnesses[index] for index in indices), "mean": fmean(fitnesses[index] for index in indices)}
            records[source] = record
        return result

    def _protect(self, genome: tuple[float, ...]) -> tuple[float, ...]:
        values = genome; changed = False
        if self.config.max_parameter_magnitude is not None:
            limit = self.config.max_parameter_magnitude; bounded = tuple(max(-limit, min(limit, value)) for value in values); changed = bounded != values; values = bounded
        if self.config.max_genome_norm is not None:
            norm = sqrt(sum(value * value for value in values))
            if norm > self.config.max_genome_norm: values = tuple(value * self.config.max_genome_norm / norm for value in values); changed = True
        if changed: self._normalizations += 1
        return values


def calibrate_global_prior(codec: GenomeCodec, *, ranges: Sequence[float] = (.25, .5, 1.0, 2.0), samples: int = 128, seed: int = 1, probe: RuleDynamicsViabilityProbe | None = None) -> dict[str, dict[str, float]]:
    """Sample candidate global priors without running embodied training."""
    if samples < 1: raise ValueError("samples must be positive")
    viability_probe = probe or RuleDynamicsViabilityProbe(codec); report: dict[str, dict[str, float]] = {}
    for offset, parameter_range in enumerate(ranges):
        if parameter_range <= 0: raise ValueError("global ranges must be positive")
        random = Random(seed + offset); results = [viability_probe(tuple(random.uniform(-parameter_range, parameter_range) for _ in range(codec.dimension))) for _ in range(samples)]; keys = {key for result in results for key in result.diagnostics}; record = {key: fmean(result.diagnostics.get(key, 0.0) for result in results) for key in keys}; record["viable_sample_rate"] = fmean(result.viable for result in results); report[f"[-{parameter_range:g}, {parameter_range:g}]"] = record
    return report


def population_statistics(population: Sequence[Sequence[float]], *, node_dimension: int | None = None) -> dict[str, float]:
    """Compact, comparable genotype telemetry for every GA generation."""
    if not population: return {"genome_rms": 0.0, "genome_l2_norm": 0.0, "population_parameter_diversity": 0.0}
    rows = [tuple(float(value) for value in genome) for genome in population]; flat = [value for row in rows for value in row]; result = {"genome_rms": sqrt(fmean(value * value for value in flat)), "genome_l2_norm": fmean(sqrt(sum(value * value for value in row)) for row in rows), "population_parameter_diversity": fmean(pstdev(row[index] for row in rows) for index in range(len(rows[0])))}
    if node_dimension is not None:
        result["node_rule_parameter_norm"] = fmean(sqrt(sum(value * value for value in row[:node_dimension])) for row in rows); result["edge_rule_parameter_norm"] = fmean(sqrt(sum(value * value for value in row[node_dimension:])) for row in rows)
    return result
