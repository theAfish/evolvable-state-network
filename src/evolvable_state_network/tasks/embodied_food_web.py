"""Evolution task: shared local rules learn through random embodied food-web agents."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, replace
from multiprocessing import get_context
import os
from random import Random
from statistics import fmean, pstdev
from typing import Callable, Literal, Mapping, Sequence

from ..embodied import EmbodiedNetwork, EmbodiedNetworkConfig, FoodWebAgentAdapter
from ..environments import (
    Action, AgentId, Controller, ControllerBlueprint, EpisodeRunner, FoodWebConfig,
    FoodWebEnvironment, Observation, RandomControllerBlueprint, Species,
    make_reference_population,
)
from ..evolution.candidate import EdgeArchitecture, MLPEdgeRule, MLPUpdateRule, RuleArchitecture
from ..evolution.cmaes import CMAES, CMAESConfig
from ..evolution.genetic import (
    GeneticAlgorithm, GeneticAlgorithmConfig, RuleDynamicsViabilityProbe,
    population_statistics,
)
from ..evolution.genome import GenomeCodec
from ..simulation.torch_backend import resolve_device
from .embodied_population_layouts import (
    BatchPopulationMode,
    get_batch_population_layout,
)


_WORKER_EVALUATOR: FoodWebCoevolutionEvaluator | None = None


def _initialize_evaluation_worker(evaluator: "FoodWebCoevolutionEvaluator") -> None:
    """Install immutable task state once in each spawned evaluation process."""
    global _WORKER_EVALUATOR
    _WORKER_EVALUATOR = evaluator
    # Tiny MLPs should not create a full BLAS thread team inside every process.
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:  # pragma: no cover - Torch is a declared dependency.
        pass


def _evaluate_focal_worker(
    job: tuple[
        tuple[float, ...], tuple[tuple[float, ...], ...], Species, tuple[int, ...],
        Literal["none", "vision"],
    ],
) -> tuple[float, tuple[float, ...], dict[str, float]]:
    if _WORKER_EVALUATOR is None:
        raise RuntimeError("embodied evaluation worker was not initialized")
    genome, opponents, species, seeds, observation_mask = job
    return _WORKER_EVALUATOR.evaluate_focal(
        genome, opponents, species, seeds, observation_mask=observation_mask,
    )


def _evaluate_population_group_worker(
    job: tuple[
        BatchPopulationMode, tuple[tuple[float, ...], ...],
        tuple[tuple[float, ...], ...], Species, tuple[int, ...],
    ],
) -> tuple[tuple[float, tuple[float, ...], dict[str, float]], ...]:
    """Evaluate one world-level group and return one score row per genome."""
    if _WORKER_EVALUATOR is None:
        raise RuntimeError("embodied evaluation worker was not initialized")
    return _evaluate_population_group_worker_with(_WORKER_EVALUATOR, job)


def _evaluate_population_group_worker_with(
    evaluator: "FoodWebCoevolutionEvaluator",
    job: tuple[
        BatchPopulationMode, tuple[tuple[float, ...], ...],
        tuple[tuple[float, ...], ...], Species, tuple[int, ...],
    ],
) -> tuple[tuple[float, tuple[float, ...], dict[str, float]], ...]:
    mode, genomes, opponents, species, seeds = job
    if mode == "shared_rule_cohort":
        score = evaluator.evaluate_focal(genomes[0], opponents, species, seeds)
        return (score,)
    if mode == "mixed_individual_population":
        return evaluator.evaluate_mixed_focal(genomes, opponents, species, seeds)
    raise ValueError(f"unknown batch population mode: {mode}")


BEHAVIOR_KEYS = (
    "meals", "meal_rate", "mean_hunger", "mean_abs_energy_change",
    "mean_action_change", "early_return_rate", "late_return_rate", "adaptation_delta",
    "mean_turn", "mean_abs_turn", "mean_speed", "early_mean_turn", "late_mean_turn",
    "turn_drift", "early_mean_abs_turn", "late_mean_abs_turn", "abs_turn_drift",
    "early_mean_speed", "late_mean_speed", "speed_drift", "turn_saturation_rate",
    "plant_visible_rate", "plant_steering_alignment", "deaths_per_1000_steps",
    "mean_completed_lifetime", "restricted_mean_lifetime", "horizon_survival_rate",
    "final_energy_fraction",
)

_LIFETIME_TOTAL_KEYS = (
    "_death_count", "_completed_lifetime_sum", "_exposure_steps",
    "_first_lifetime_sum", "_first_lifetime_count", "_horizon_survivors",
)
_DERIVED_LIFETIME_KEYS = {
    "deaths_per_1000_steps", "mean_completed_lifetime",
    "restricted_mean_lifetime", "horizon_survival_rate",
}


class EvolutionTerminated(Exception):
    """Raised when a caller requests cooperative termination of an evolution run."""


def _network_seed(seed: int, species: Species) -> int:
    """Return scenario randomness that is independent of candidate parameters."""
    species_salt = 0x51A7 if species is Species.PREY else 0xA93D
    return (int(seed) * 1_000_003 + species_salt) % (2**32)


def _mean_behavior(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    result = {
        key: fmean(float(row.get(key, 0.0)) for row in rows) if rows else 0.0
        for key in BEHAVIOR_KEYS if key not in _DERIVED_LIFETIME_KEYS
    }
    totals = {
        key: sum(float(row.get(key, 0.0)) for row in rows)
        for key in _LIFETIME_TOTAL_KEYS
    }
    deaths = totals["_death_count"]
    exposure = totals["_exposure_steps"]
    first_lives = totals["_first_lifetime_count"]
    result.update({
        "deaths_per_1000_steps": 1000.0 * deaths / max(exposure, 1.0),
        "mean_completed_lifetime": totals["_completed_lifetime_sum"] / max(deaths, 1.0),
        "restricted_mean_lifetime": totals["_first_lifetime_sum"] / max(first_lives, 1.0),
        "horizon_survival_rate": totals["_horizon_survivors"] / max(first_lives, 1.0),
        **totals,
    })
    return result


def _public_behavior(behavior: Mapping[str, float]) -> dict[str, float]:
    return {key: float(value) for key, value in behavior.items() if not key.startswith("_")}


def _phenotype_diversity(rows: Sequence[Mapping[str, float]]) -> float:
    """Mean normalized spread of the established behavioral descriptors."""
    if len(rows) < 2:
        return 0.0
    spreads = []
    for key in BEHAVIOR_KEYS:
        values = [float(row.get(key, 0.0)) for row in rows]
        scale = max(1.0, abs(fmean(values)))
        spreads.append(pstdev(values) / scale)
    return fmean(spreads)


def _species_behavior(result: object, ids: Sequence[AgentId]) -> dict[str, float]:
    behavior = getattr(result, "behavior")
    return _mean_behavior([behavior[agent_id] for agent_id in ids])


def _execution_device(config: EmbodiedNetworkConfig) -> str:
    return resolve_device(config.device).type if config.execution_backend == "torch" else "cpu"


class EmbodiedFoodWebController(Controller):
    """One episode-local random network driven by inherited update rules."""

    def __init__(
        self, node_rule: MLPUpdateRule, edge_rule: MLPEdgeRule,
        config: EmbodiedNetworkConfig, seed: int, observation_mask: Literal["none", "vision"] = "none",
    ) -> None:
        self._node_rule, self._edge_rule, self._config, self._seed = node_rule, edge_rule, config, seed
        self._observation_mask = observation_mask
        self._network = EmbodiedNetwork(node_rule, edge_rule, self._adapter(), config, seed=seed)

    def _adapter(self) -> FoodWebAgentAdapter:
        return FoodWebAgentAdapter(
            vision_pixels=self._config.vision_pixels,
            body_inputs=self._config.body_inputs,
        )

    def begin_episode(self, *, seed: int | None = None) -> None:
        # A new episode creates a fresh random graph and node-state sample;
        # rule parameters are the only values inherited across episodes.
        episode_seed = self._seed if seed is None else self._seed + seed
        self._network = EmbodiedNetwork(self._node_rule, self._edge_rule, self._adapter(), self._config, seed=episode_seed)

    def act(self, observation: Observation, *, available_actions: Sequence[Action]) -> Action:
        if not available_actions:
            raise ValueError("environment offered no actions")
        if self._observation_mask == "vision":
            observation = {**observation, "vision": ()}
        return self._network.act(observation)

    def inspection_snapshot(self) -> dict[str, object]:
        return self._network.inspection_snapshot()


@dataclass(frozen=True, slots=True)
class EmbodiedFoodWebControllerBlueprint(ControllerBlueprint):
    """Heritable rules plus configuration, never a graph or node state."""

    node_rule: MLPUpdateRule
    edge_rule: MLPEdgeRule
    network: EmbodiedNetworkConfig
    individual_seed: int
    observation_mask: Literal["none", "vision"] = "none"

    def build(self, *, seed: int | None = None) -> Controller:
        return EmbodiedFoodWebController(
            self.node_rule, self.edge_rule, self.network,
            self.individual_seed + (seed or 0), self.observation_mask,
        )


@dataclass(frozen=True, slots=True)
class EmbodiedFoodWebTaskConfig:
    """One food-web task whose shared rule is exercised in both roles."""

    network: EmbodiedNetworkConfig = EmbodiedNetworkConfig()
    environment: FoodWebConfig = FoodWebConfig()
    focal_species: Species = Species.PREY
    prey_count: int = 5
    predator_count: int = 2
    max_steps: int = 200
    trials: int = 3
    seed: int = 1

    def __post_init__(self) -> None:
        if self.prey_count < 1 or self.predator_count < 0 or self.max_steps < 1 or self.trials < 1:
            raise ValueError("population, step, and trial counts must be positive")
        if self.network.state_width < 2:
            raise ValueError("food-web embodiment needs channel 0 for vision and channel 1 for interoception")


@dataclass(frozen=True, slots=True)
class EmbodiedFoodWebEvaluation:
    genome: tuple[float, ...]
    trial_lifetimes: tuple[float, ...]
    mean_lifetime: float
    behavior: Mapping[str, float]


class EmbodiedFoodWebEvaluator:
    """Score joint node/edge update rules without inheriting a concrete network."""

    def __init__(
        self, architecture: RuleArchitecture | None = None, edge_architecture: EdgeArchitecture | None = None,
        config: EmbodiedFoodWebTaskConfig | None = None,
    ) -> None:
        self.architecture = architecture or RuleArchitecture(state_width=2)
        self.edge_architecture = edge_architecture or EdgeArchitecture(node_state_width=self.architecture.state_width)
        self.config = config or EmbodiedFoodWebTaskConfig(network=EmbodiedNetworkConfig(state_width=self.architecture.state_width))
        if self.config.network.state_width != self.architecture.state_width:
            raise ValueError("task network and node-rule state widths must agree")
        self.codec = GenomeCodec(self.architecture, self.edge_architecture, "joint")

    def evaluate(self, genome: Sequence[float]) -> EmbodiedFoodWebEvaluation:
        encoded = tuple(float(value) for value in genome)
        node_rule, edge_rule = self.codec.decode_groups(encoded, output_scale=self.config.network.rule_output_scale)
        assert node_rule is not None and edge_rule is not None
        trials = tuple(
            self._trial(node_rule, edge_rule, self.config.seed + 10_007 * index)
            for index in range(self.config.trials)
        )
        lifetimes = tuple(item[0] for item in trials)
        behavior = _mean_behavior([item[1] for item in trials])
        return EmbodiedFoodWebEvaluation(
            encoded, lifetimes, fmean(lifetimes), _public_behavior(behavior),
        )

    def evaluate_batch(self, genomes: Sequence[Sequence[float]]) -> tuple[EmbodiedFoodWebEvaluation, ...]:
        return tuple(self.evaluate(genome) for genome in genomes)

    def _trial(self, node_rule: MLPUpdateRule, edge_rule: MLPEdgeRule, seed: int) -> tuple[float, dict[str, float]]:
        focal = EmbodiedFoodWebControllerBlueprint(
            node_rule, edge_rule, self.config.network, _network_seed(seed, self.config.focal_species)
        )
        agents = make_reference_population(
            prey_count=self.config.prey_count, predator_count=self.config.predator_count,
            width=self.config.environment.width, height=self.config.environment.height,
            prey_initial_energy=self.config.environment.prey_initial_energy, predator_initial_energy=self.config.environment.predator_initial_energy, controller=RandomControllerBlueprint(), seed=seed,
        )
        ids = [agent.id for agent in agents if agent.species is self.config.focal_species]
        for agent in agents:
            if agent.species is self.config.focal_species:
                agent.controller = focal
        result = EpisodeRunner(FoodWebEnvironment(self.config.environment, seed=seed)).run(agents, max_steps=self.config.max_steps, seed=seed)
        behavior = _species_behavior(result, ids)
        return float(behavior["restricted_mean_lifetime"]), behavior


@dataclass(frozen=True, slots=True)
class FoodWebCoevolutionEvaluation:
    """Matched prey/predator rule evaluations from one shared set of episodes."""

    prey_genome: tuple[float, ...]
    predator_genome: tuple[float, ...]
    prey_trial_lifetimes: tuple[float, ...]
    predator_trial_lifetimes: tuple[float, ...]
    prey_behavior: Mapping[str, float]
    predator_behavior: Mapping[str, float]

    @property
    def prey_mean_lifetime(self) -> float:
        return fmean(self.prey_trial_lifetimes)

    @property
    def predator_mean_lifetime(self) -> float:
        return fmean(self.predator_trial_lifetimes)


class FoodWebCoevolutionEvaluator:
    """Evaluate food-web rules in matched episodes.

    ``evaluate`` remains available for reading legacy two-genome experiments.
    New evolution uses :meth:`evaluate_shared`, which installs one decoded
    node/edge rule in every organism regardless of species.
    """

    def __init__(
        self, architecture: RuleArchitecture | None = None, edge_architecture: EdgeArchitecture | None = None,
        config: EmbodiedFoodWebTaskConfig | None = None,
    ) -> None:
        self.architecture = architecture or RuleArchitecture(state_width=2)
        self.edge_architecture = edge_architecture or EdgeArchitecture(node_state_width=self.architecture.state_width)
        self.config = config or EmbodiedFoodWebTaskConfig(network=EmbodiedNetworkConfig(state_width=self.architecture.state_width))
        if self.config.network.state_width != self.architecture.state_width:
            raise ValueError("task network and node-rule state widths must agree")
        self.codec = GenomeCodec(self.architecture, self.edge_architecture, "joint")

    def evaluate(self, prey_genome: Sequence[float], predator_genome: Sequence[float]) -> FoodWebCoevolutionEvaluation:
        prey = tuple(float(value) for value in prey_genome)
        predator = tuple(float(value) for value in predator_genome)
        trial_lifetimes = tuple(
            self._trial(prey, predator, self.config.seed + 10_007 * index)
            for index in range(self.config.trials)
        )
        return FoodWebCoevolutionEvaluation(
            prey, predator,
            tuple(item[0] for item in trial_lifetimes), tuple(item[1] for item in trial_lifetimes),
            _public_behavior(_mean_behavior([item[2] for item in trial_lifetimes])),
            _public_behavior(_mean_behavior([item[3] for item in trial_lifetimes])),
        )

    def evaluate_shared(
        self, genome: Sequence[float], *, seeds: Sequence[int] | None = None,
        observation_mask: Literal["none", "vision"] = "none",
    ) -> FoodWebCoevolutionEvaluation:
        """Score one rule after it adapts separate random graphs to both roles.

        The rule parameters are deliberately identical for prey and predators.
        Their different sensory/body streams, graph samples, and recurrent
        state are the only sources of role-specific behaviour.
        """
        shared = tuple(float(value) for value in genome)
        trial_seeds = (
            tuple(int(seed) for seed in seeds)
            if seeds is not None
            else tuple(self.config.seed + 10_007 * index for index in range(self.config.trials))
        )
        if not trial_seeds:
            raise ValueError("shared evaluation needs at least one seed")
        trials = tuple(
            self._trial(
                shared, shared, seed,
                prey_observation_mask=observation_mask,
                predator_observation_mask=observation_mask,
            )
            for seed in trial_seeds
        )
        return FoodWebCoevolutionEvaluation(
            shared, shared,
            tuple(item[0] for item in trials), tuple(item[1] for item in trials),
            _public_behavior(_mean_behavior([item[2] for item in trials])),
            _public_behavior(_mean_behavior([item[3] for item in trials])),
        )

    def shared_score(self, evaluation: FoodWebCoevolutionEvaluation) -> float:
        """Balance the selection pressure across the roles that are present."""
        scores = [evaluation.prey_mean_lifetime]
        if self.config.predator_count:
            scores.append(evaluation.predator_mean_lifetime)
        return fmean(scores)

    def evaluate_focal(
        self, genome: Sequence[float], opponents: Sequence[Sequence[float]],
        focal_species: Species, seeds: Sequence[int], *, observation_mask: Literal["none", "vision"] = "none",
    ) -> tuple[float, tuple[float, ...], dict[str, float]]:
        """Evaluate one candidate against frozen opponents and matched random networks."""
        focal = tuple(float(value) for value in genome)
        if not opponents or not seeds:
            raise ValueError("batch evaluation needs at least one opponent and seed")
        lifetimes: list[float] = []
        behavior: list[Mapping[str, float]] = []
        for opponent_values in opponents:
            opponent = tuple(float(value) for value in opponent_values)
            prey_genome, predator_genome = (focal, opponent) if focal_species is Species.PREY else (opponent, focal)
            for seed in seeds:
                result = self._trial(
                    prey_genome, predator_genome, int(seed),
                    prey_observation_mask=observation_mask if focal_species is Species.PREY else "none",
                    predator_observation_mask=observation_mask if focal_species is Species.PREDATOR else "none",
                )
                lifetimes.append(result[0] if focal_species is Species.PREY else result[1])
                behavior.append(result[2] if focal_species is Species.PREY else result[3])
        aggregate = _mean_behavior(behavior)
        return fmean(lifetimes), tuple(lifetimes), _public_behavior(aggregate)

    def evaluate_mixed_focal(
        self, genomes: Sequence[Sequence[float]], opponents: Sequence[Sequence[float]],
        focal_species: Species, seeds: Sequence[int], *, observation_mask: Literal["none", "vision"] = "none",
    ) -> tuple[tuple[float, tuple[float, ...], dict[str, float]], ...]:
        """Score every distinct focal genome from shared mixed-population worlds.

        A group contains exactly one genome per focal organism.  Each genome's
        score is its own first-life outcome, averaged across the common
        opponent/seed bank.  Random world and graph seeds remain independent
        of genome values, so the group members are selected fairly despite
        interacting in the same ecology.
        """
        focal = tuple(tuple(float(value) for value in genome) for genome in genomes)
        expected = self.config.prey_count if focal_species is Species.PREY else self.config.predator_count
        if not focal or len(focal) != expected:
            raise ValueError("mixed focal group must contain one genome for every focal organism")
        if not opponents or not seeds:
            raise ValueError("batch evaluation needs at least one opponent and seed")
        lifetimes: list[list[float]] = [[] for _ in focal]
        behavior: list[list[Mapping[str, float]]] = [[] for _ in focal]
        for opponent_values in opponents:
            opponent = tuple(float(value) for value in opponent_values)
            for seed in seeds:
                trial_lifetimes, trial_behavior = self._mixed_trial(
                    focal, opponent, focal_species, int(seed), observation_mask=observation_mask,
                )
                for index, (lifetime, row) in enumerate(zip(trial_lifetimes, trial_behavior, strict=True)):
                    lifetimes[index].append(lifetime)
                    behavior[index].append(row)
        return tuple(
            (
                fmean(lifetimes[index]),
                tuple(lifetimes[index]),
                _public_behavior(_mean_behavior(behavior[index])),
            )
            for index in range(len(focal))
        )

    def _blueprint(
        self, genome: tuple[float, ...], seed: int, species: Species,
        observation_mask: Literal["none", "vision"] = "none",
    ) -> EmbodiedFoodWebControllerBlueprint:
        node_rule, edge_rule = self.codec.decode_groups(genome, output_scale=self.config.network.rule_output_scale)
        assert node_rule is not None and edge_rule is not None
        return EmbodiedFoodWebControllerBlueprint(
            node_rule, edge_rule, self.config.network, _network_seed(seed, species), observation_mask
        )

    def _trial(
        self, prey_genome: tuple[float, ...], predator_genome: tuple[float, ...], seed: int,
        *, prey_observation_mask: Literal["none", "vision"] = "none",
        predator_observation_mask: Literal["none", "vision"] = "none",
    ) -> tuple[float, float, dict[str, float], dict[str, float]]:
        prey = self._blueprint(prey_genome, seed, Species.PREY, prey_observation_mask)
        predator = self._blueprint(predator_genome, seed, Species.PREDATOR, predator_observation_mask)
        agents = make_reference_population(
            prey_count=self.config.prey_count, predator_count=self.config.predator_count,
            width=self.config.environment.width, height=self.config.environment.height,
            prey_initial_energy=self.config.environment.prey_initial_energy, predator_initial_energy=self.config.environment.predator_initial_energy, controller=RandomControllerBlueprint(), seed=seed,
        )
        prey_ids, predator_ids = [], []
        for agent in agents:
            if agent.species is Species.PREY:
                agent.controller = prey
                prey_ids.append(agent.id)
            else:
                agent.controller = predator
                predator_ids.append(agent.id)
        result = EpisodeRunner(FoodWebEnvironment(self.config.environment, seed=seed)).run(agents, max_steps=self.config.max_steps, seed=seed)
        prey_behavior = _species_behavior(result, prey_ids)
        prey_lifetime = float(prey_behavior["restricted_mean_lifetime"])
        # A prey-only ecology has no predator genome to score.  Keep the
        # evaluator total so shared callers can still construct it safely.
        predator_behavior = _species_behavior(result, predator_ids) if predator_ids else _mean_behavior([])
        predator_lifetime = float(predator_behavior["restricted_mean_lifetime"]) if predator_ids else 0.0
        return (
            prey_lifetime, predator_lifetime, prey_behavior, predator_behavior,
        )

    def _mixed_trial(
        self, focal_genomes: Sequence[tuple[float, ...]], opponent_genome: tuple[float, ...],
        focal_species: Species, seed: int, *, observation_mask: Literal["none", "vision"] = "none",
    ) -> tuple[tuple[float, ...], tuple[Mapping[str, float], ...]]:
        """Run one ecology where every focal organism carries its own genome."""
        agents = make_reference_population(
            prey_count=self.config.prey_count, predator_count=self.config.predator_count,
            width=self.config.environment.width, height=self.config.environment.height,
            prey_initial_energy=self.config.environment.prey_initial_energy,
            predator_initial_energy=self.config.environment.predator_initial_energy,
            controller=RandomControllerBlueprint(), seed=seed,
        )
        focal_agents = [agent for agent in agents if agent.species is focal_species]
        if len(focal_agents) != len(focal_genomes):
            raise ValueError("mixed focal genomes do not match the world population")
        opponent_species = Species.PREDATOR if focal_species is Species.PREY else Species.PREY
        opponent = self._blueprint(opponent_genome, seed, opponent_species)
        focal_ids: list[AgentId] = []
        for agent, genome in zip(focal_agents, focal_genomes, strict=True):
            agent.controller = self._blueprint(genome, seed, focal_species, observation_mask)
            focal_ids.append(agent.id)
        for agent in agents:
            if agent.species is not focal_species:
                agent.controller = opponent
        result = EpisodeRunner(FoodWebEnvironment(self.config.environment, seed=seed)).run(
            agents, max_steps=self.config.max_steps, seed=seed,
        )
        behavior = getattr(result, "behavior")
        rows = tuple(behavior[agent_id] for agent_id in focal_ids)
        return (
            tuple(float(row["restricted_lifetime"]) for row in rows),
            rows,
        )


@dataclass(frozen=True, slots=True)
class EmbodiedRuleEvolutionConfig:
    """Optimizer settings for one agent species' local update-rule genome."""

    generations: int = 8
    population_size: int = 8
    initial_sigma: float = .12
    seed: int = 1
    initial_genome: tuple[float, ...] | None = None
    algorithm: Literal["cma_es", "genetic"] = "cma_es"
    mutation_sigma: float | None = None
    elite_fraction: float = .25
    immigrant_fraction: float = .25
    immigrant_sigma: float | None = None
    immigrant_mode: Literal["zero", "population"] = "zero"
    local_mutation_sigma: float | None = None
    local_offspring_fraction: float | None = None
    regional_fraction: float = 0.0
    regional_scale: float = 1.0
    regional_min_std: float = .02
    global_fraction: float = 0.0
    global_parameter_range: float = 1.0
    global_viability_filter: bool = False
    global_max_sampling_attempts: int = 20
    max_genome_norm: float | None = None
    max_parameter_magnitude: float | None = None

    def __post_init__(self) -> None:
        if self.generations < 1 or self.population_size < 2 or self.initial_sigma <= 0:
            raise ValueError("evolution configuration is invalid")
        if self.mutation_sigma is not None and self.mutation_sigma <= 0:
            raise ValueError("mutation_sigma must be positive when provided")
        if self.immigrant_sigma is not None and self.immigrant_sigma <= 0:
            raise ValueError("immigrant_sigma must be positive when provided")
        if self.local_mutation_sigma is not None and self.local_mutation_sigma <= 0:
            raise ValueError("local_mutation_sigma must be positive when provided")
        if self.elite_fraction + self.regional_fraction + self.global_fraction > 1:
            raise ValueError("elite, regional, and global fractions cannot exceed one")


RuleOptimizer = CMAES | GeneticAlgorithm


def _multiscale_kwargs(config: EmbodiedRuleEvolutionConfig) -> dict[str, object]:
    """Keep copied species/library configs in lockstep with GA exploration."""
    return {
        "local_mutation_sigma": config.local_mutation_sigma,
        "local_offspring_fraction": config.local_offspring_fraction,
        "regional_fraction": config.regional_fraction,
        "regional_scale": config.regional_scale,
        "regional_min_std": config.regional_min_std,
        "global_fraction": config.global_fraction,
        "global_parameter_range": config.global_parameter_range,
        "global_viability_filter": config.global_viability_filter,
        "global_max_sampling_attempts": config.global_max_sampling_attempts,
    }


def _make_optimizer(
    dimension: int, config: EmbodiedRuleEvolutionConfig, *, seed: int | None = None,
    codec: GenomeCodec | None = None,
) -> RuleOptimizer:
    optimizer_seed = config.seed if seed is None else seed
    if config.algorithm == "genetic":
        probe = RuleDynamicsViabilityProbe(codec) if config.global_viability_filter and codec is not None else None
        if config.global_viability_filter and probe is None:
            raise ValueError("a codec is required for global viability filtering")
        return GeneticAlgorithm(
            GeneticAlgorithmConfig(
                dimension, config.population_size, config.mutation_sigma or config.initial_sigma, optimizer_seed,
                elite_fraction=config.elite_fraction, immigrant_fraction=config.immigrant_fraction,
                immigrant_sigma=config.immigrant_sigma or max(.05, config.initial_sigma * 3.0),
                immigrant_mode=config.immigrant_mode,
                max_genome_norm=config.max_genome_norm,
                max_parameter_magnitude=config.max_parameter_magnitude,
                **_multiscale_kwargs(config),
            ),
            config.initial_genome,
            global_viability_probe=probe,
        )
    return CMAES(
        CMAESConfig(dimension, config.population_size, config.initial_sigma, optimizer_seed),
        config.initial_genome,
    )


def _shared_initial_genome(
    dimension: int, *, initial: Sequence[float] | None,
    prey_initial: Sequence[float] | None = None,
    predator_initial: Sequence[float] | None = None,
) -> tuple[float, ...]:
    """Resolve a warm start without silently retaining role-specific rules."""
    candidates = [
        tuple(float(value) for value in genome)
        for genome in (initial, prey_initial, predator_initial) if genome is not None
    ]
    if any(len(genome) != dimension for genome in candidates):
        raise ValueError("initial genome does not match the joint rule architecture")
    if len(set(candidates)) > 1:
        raise ValueError(
            "shared-rule evolution cannot warm start from different prey and predator genomes; "
            "select one shared genome or a shared-rule run"
        )
    return candidates[0] if candidates else (0.0,) * dimension


class EmbodiedRuleEvolutionRunner:
    """Optimises only rule genomes; every evaluation rebuilds random agents."""

    def __init__(self, evaluator: EmbodiedFoodWebEvaluator, config: EmbodiedRuleEvolutionConfig) -> None:
        self.evaluator, self.config = evaluator, config
        if config.initial_genome is not None and len(config.initial_genome) != evaluator.codec.dimension:
            raise ValueError("initial genome does not match the joint rule architecture")

    def run(self, progress: Callable[[dict[str, object]], None] | None = None) -> dict[str, object]:
        optimizer = _make_optimizer(self.evaluator.codec.dimension, self.config, codec=self.evaluator.codec)
        history: list[dict[str, object]] = []
        best: EmbodiedFoodWebEvaluation | None = None
        for _ in range(self.config.generations):
            population = optimizer.ask()
            evaluations = self.evaluator.evaluate_batch(population)
            winner = max(evaluations, key=lambda item: item.mean_lifetime)
            if best is None or winner.mean_lifetime > best.mean_lifetime:
                best = winner
            optimizer.tell(population, [item.mean_lifetime for item in evaluations])
            row = {
                "generation": optimizer.generation, "best_lifetime": winner.mean_lifetime,
                "mean_lifetime": fmean(item.mean_lifetime for item in evaluations), "sigma": optimizer.sigma,
                "sampling": getattr(optimizer, "last_sampling_telemetry", {}),
            }
            history.append(row)
            if progress:
                progress(row)
        assert best is not None
        return {"task": "food_web", "algorithm": self.config.algorithm, "objective": "restricted_mean_lifetime", "objective_units": "ticks", "focal_species": str(self.evaluator.config.focal_species), "best_genome": list(best.genome), "best_lifetime": best.mean_lifetime, "best_trial_lifetimes": list(best.trial_lifetimes), "history": history}


class FoodWebCoevolutionRunner:
    """Legacy entry point for evolving one shared rule in matched episodes."""

    def __init__(self, evaluator: FoodWebCoevolutionEvaluator, config: EmbodiedRuleEvolutionConfig) -> None:
        self.evaluator, self.config = evaluator, config
        if config.initial_genome is not None and len(config.initial_genome) != evaluator.codec.dimension:
            raise ValueError("initial genome does not match the joint rule architecture")

    def run(self, progress: Callable[[dict[str, object]], None] | None = None) -> dict[str, object]:
        optimizer = _make_optimizer(self.evaluator.codec.dimension, self.config, codec=self.evaluator.codec)
        history: list[dict[str, object]] = []
        best: FoodWebCoevolutionEvaluation | None = None
        for _ in range(self.config.generations):
            population = optimizer.ask()
            evaluations = tuple(self.evaluator.evaluate_shared(genome) for genome in population)
            scores = [self.evaluator.shared_score(item) for item in evaluations]
            winner = max(evaluations, key=self.evaluator.shared_score)
            if best is None or self.evaluator.shared_score(winner) > self.evaluator.shared_score(best):
                best = winner
            optimizer.tell(population, scores)
            row = {
                "generation": optimizer.generation,
                "shared_best_lifetime": self.evaluator.shared_score(winner),
                "prey_best_lifetime": winner.prey_mean_lifetime,
                "prey_mean_lifetime": fmean(item.prey_mean_lifetime for item in evaluations),
                "predator_best_lifetime": winner.predator_mean_lifetime,
                "shared_sampling": getattr(optimizer, "last_sampling_telemetry", {}),
                "predator_mean_lifetime": fmean(item.predator_mean_lifetime for item in evaluations),
                "shared_sigma": optimizer.sigma,
            }
            history.append(row)
            if progress:
                progress(row)
        assert best is not None
        return {
            "task": "food_web_shared_rule_evolution",
            "algorithm": self.config.algorithm, "objective": "mean_role_lifetime", "objective_units": "ticks",
            "rule_sharing": "one_genome_for_prey_and_predator",
            "shared_best_genome": list(best.prey_genome),
            "prey_best_genome": list(best.prey_genome), "prey_best_lifetime": best.prey_mean_lifetime,
            "predator_best_genome": list(best.prey_genome), "predator_best_lifetime": best.predator_mean_lifetime,
            "history": history,
        }


@dataclass(frozen=True, slots=True)
class BatchFoodWebConfig:
    """Comparable episodic batches with frozen opponents and common seeds."""

    population_mode: BatchPopulationMode = "shared_rule_cohort"
    generations: int = 8
    episode_steps: int = 256
    trials: int = 4
    validation_trials: int = 2
    test_trials: int = 4
    opponent_pool_size: int = 2
    seed: int = 1
    initial_genome: tuple[float, ...] | None = None
    initial_prey_genome: tuple[float, ...] | None = None
    initial_predator_genome: tuple[float, ...] | None = None
    workers: int = 1

    def __post_init__(self) -> None:
        if (
            self.generations < 1 or self.episode_steps < 1 or self.trials < 1
            or self.validation_trials < 1 or self.test_trials < 1 or self.opponent_pool_size < 1
            or self.workers < 0
        ):
            raise ValueError("batch food-web configuration is invalid")


class BatchFoodWebCoevolutionRunner:
    """Evolve one local update rule in complete, directly comparable batches."""

    def __init__(
        self, evaluator: FoodWebCoevolutionEvaluator, evolution: EmbodiedRuleEvolutionConfig,
        config: BatchFoodWebConfig,
    ) -> None:
        batch_environment = replace(evaluator.config.environment, respawn_on_death=False)
        batch_task = replace(evaluator.config, environment=batch_environment)
        self.evaluator = FoodWebCoevolutionEvaluator(
            evaluator.architecture, evaluator.edge_architecture, batch_task,
        )
        self.evolution, self.config = evolution, config
        self.population_layout = get_batch_population_layout(config.population_mode)
        if evaluator.config.max_steps != config.episode_steps:
            raise ValueError("batch evaluator max_steps must equal episode_steps")
        for genome in (config.initial_genome, config.initial_prey_genome, config.initial_predator_genome):
            if genome is not None and len(genome) != evaluator.codec.dimension:
                raise ValueError("initial genome does not match the joint rule architecture")

    def run(
        self, progress: Callable[[dict[str, object]], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, object]:
        return self._run_shared(progress=progress, should_stop=should_stop)

    def _run_shared(
        self, *, progress: Callable[[dict[str, object]], None] | None,
        should_stop: Callable[[], bool] | None,
    ) -> dict[str, object]:
        """Evolve one update rule across prey and predator embodiments.

        A candidate is evaluated in complete worlds, never against a separate
        role-specific opponent population.  This makes the genome's fitness
        reflect how well its local dynamics adapt fresh random graphs to both
        body roles.
        """
        initial = _shared_initial_genome(
            self.evaluator.codec.dimension,
            initial=(self.config.initial_genome or self.evolution.initial_genome),
            prey_initial=self.config.initial_prey_genome,
            predator_initial=self.config.initial_predator_genome,
        )
        optimizer_config = replace(
            self.evolution, generations=self.config.generations,
            population_size=self.evolution.population_size,
            seed=self.config.seed, initial_genome=initial,
        )
        optimizer = _make_optimizer(
            self.evaluator.codec.dimension, optimizer_config, codec=self.evaluator.codec,
        )
        best = (float("-inf"), initial, _public_behavior(_mean_behavior([])), _public_behavior(_mean_behavior([])))
        history: list[dict[str, object]] = []
        evaluations = validation_evaluations = 0
        validation_seeds = tuple(
            self.config.seed + 90_000_019 + trial * 10_007
            for trial in range(self.config.validation_trials)
        )
        test_seeds = tuple(
            self.config.seed + 190_000_033 + trial * 10_007
            for trial in range(self.config.test_trials)
        )
        for generation in range(1, self.config.generations + 1):
            if should_stop and should_stop():
                raise EvolutionTerminated()
            seeds = tuple(
                self.config.seed + generation * 1_000_003 + trial * 10_007
                for trial in range(self.config.trials)
            )
            population = optimizer.ask()
            rows = tuple(self.evaluator.evaluate_shared(genome, seeds=seeds) for genome in population)
            scores = tuple(self.evaluator.shared_score(row) for row in rows)
            optimizer.tell(population, scores)
            winner_index = max(range(len(population)), key=lambda index: scores[index])
            winner = rows[winner_index]
            evaluations += len(population) * len(seeds)
            validation = self.evaluator.evaluate_shared(population[winner_index], seeds=validation_seeds)
            validation_score = self.evaluator.shared_score(validation)
            validation_evaluations += len(validation_seeds)
            if validation_score > best[0]:
                best = (
                    validation_score, tuple(population[winner_index]),
                    dict(validation.prey_behavior), dict(validation.predator_behavior),
                )
            row = {
                "generation": generation,
                "shared_best_lifetime": scores[winner_index],
                "shared_mean_lifetime": fmean(scores),
                "shared_validation_lifetime": validation_score,
                "prey_best_lifetime": winner.prey_mean_lifetime,
                "prey_mean_lifetime": fmean(item.prey_mean_lifetime for item in rows),
                "prey_validation_lifetime": validation.prey_mean_lifetime,
                "predator_best_lifetime": winner.predator_mean_lifetime if self.evaluator.config.predator_count else 0.0,
                "predator_mean_lifetime": (
                    fmean(item.predator_mean_lifetime for item in rows)
                    if self.evaluator.config.predator_count else 0.0
                ),
                "predator_validation_lifetime": (
                    validation.predator_mean_lifetime if self.evaluator.config.predator_count else 0.0
                ),
                "shared_genotype": population_statistics(
                    population, node_dimension=self.evaluator.architecture.parameter_count,
                ),
                "shared_sampling": getattr(optimizer, "last_sampling_telemetry", {}),
                "shared_normalizations": getattr(optimizer, "last_ask_normalizations", 0),
                "episode_seeds": list(seeds), "validation_seeds": list(validation_seeds),
            }
            history.append(row)
            shared_snapshot = self._snapshot(
                optimizer, (best[0], best[1], {"prey": best[2], "predator": best[3]}),
                evaluations, validation_evaluations,
            )
            prey_snapshot = {**shared_snapshot, "best_lifetime": validation.prey_mean_lifetime, "behavior": best[2]}
            predator_snapshot = {**shared_snapshot, "best_lifetime": validation.predator_mean_lifetime if self.evaluator.config.predator_count else 0.0, "behavior": best[3]}
            if progress:
                progress({
                    "phase": "batch_food_web_shared_rule", "training_mode": "batch",
                    "algorithm": self.evolution.algorithm, "objective": "mean_role_lifetime",
                    "objective_units": "ticks", "generation": generation,
                    "generations": self.config.generations, "workers": 1,
                    "execution_backend": self.evaluator.config.network.execution_backend,
                    "device": _execution_device(self.evaluator.config.network),
                    "shared": shared_snapshot, "prey": prey_snapshot,
                    "predator": predator_snapshot, "history": list(history),
                })
        if should_stop and should_stop():
            raise EvolutionTerminated()
        selected = self.evaluator.evaluate_shared(best[1], seeds=test_seeds)
        zero = self.evaluator.evaluate_shared((0.0,) * self.evaluator.codec.dimension, seeds=test_seeds)
        vision_masked = self.evaluator.evaluate_shared(best[1], seeds=test_seeds, observation_mask="vision")
        shared_snapshot = self._snapshot(
            optimizer, (best[0], best[1], {"prey": best[2], "predator": best[3]}),
            evaluations, validation_evaluations,
        )
        shared_snapshot.update({
            "selection_validation_lifetime": best[0],
            "test_lifetime": self.evaluator.shared_score(selected),
            "test_evaluations": 3 * len(test_seeds),
            "baselines": {
                "zero_rule_lifetime": self.evaluator.shared_score(zero),
                "vision_masked_lifetime": self.evaluator.shared_score(vision_masked),
            },
        })

        def role_snapshot(role: Species) -> dict[str, object]:
            is_prey = role is Species.PREY
            role_selected = selected.prey_mean_lifetime if is_prey else selected.predator_mean_lifetime
            role_zero = zero.prey_mean_lifetime if is_prey else zero.predator_mean_lifetime
            role_masked = vision_masked.prey_mean_lifetime if is_prey else vision_masked.predator_mean_lifetime
            return {
                **shared_snapshot, "best_lifetime": best[0], "behavior": best[2] if is_prey else best[3],
                "test_lifetime": role_selected,
                "test_behavior": dict(selected.prey_behavior if is_prey else selected.predator_behavior),
                "baselines": {
                    "zero_rule_lifetime": role_zero,
                    "vision_masked_lifetime": role_masked,
                    "lifetime_gain_over_zero_rule": role_selected - role_zero,
                    "vision_lifetime_delta": role_selected - role_masked,
                },
            }

        prey_snapshot, predator_snapshot = role_snapshot(Species.PREY), role_snapshot(Species.PREDATOR)
        return {
            "task": "batch_food_web_shared_rule_evolution", "training_mode": "batch",
            "algorithm": self.evolution.algorithm, "objective": "mean_role_lifetime",
            "objective_units": "ticks", "rule_sharing": "one_genome_for_prey_and_predator",
            "population_size": self.evolution.population_size,
            "population_mode": "shared_rule_cohort", "world_count": self.evolution.population_size,
            "generations": self.config.generations, "episode_steps": self.config.episode_steps,
            "trials": self.config.trials, "validation_trials": self.config.validation_trials,
            "test_trials": self.config.test_trials, "test_seeds": list(test_seeds),
            "execution": {"workers": 1, "backend": self.evaluator.config.network.execution_backend, "device": _execution_device(self.evaluator.config.network)},
            "shared": shared_snapshot, "prey": prey_snapshot, "predator": predator_snapshot,
            "history": history, "shared_best_genome": list(best[1]),
            # Compatibility aliases; both deliberately reference the one rule.
            "prey_best_genome": list(best[1]), "predator_best_genome": list(best[1]),
        }

        # Legacy independent-species implementation retained below only to
        # make historical checkpoints readable; it is unreachable for new runs.
        initial = self.config.initial_genome if self.config.initial_genome is not None else self.evolution.initial_genome
        zero = (0.0,) * self.evaluator.codec.dimension
        prey_initial = self.config.initial_prey_genome if self.config.initial_prey_genome is not None else (initial or zero)
        predator_initial = self.config.initial_predator_genome if self.config.initial_predator_genome is not None else (initial or zero)
        prey_config = EmbodiedRuleEvolutionConfig(
            generations=self.config.generations,
            population_size=self.population_layout.genome_population_size(
                self.evolution.population_size, self.evaluator.config.prey_count,
            ),
            initial_sigma=self.evolution.initial_sigma, seed=self.config.seed,
            initial_genome=prey_initial, algorithm=self.evolution.algorithm,
            mutation_sigma=self.evolution.mutation_sigma, elite_fraction=self.evolution.elite_fraction,
            immigrant_fraction=self.evolution.immigrant_fraction, immigrant_sigma=self.evolution.immigrant_sigma,
            immigrant_mode=self.evolution.immigrant_mode, max_genome_norm=self.evolution.max_genome_norm,
            max_parameter_magnitude=self.evolution.max_parameter_magnitude,
            **_multiscale_kwargs(self.evolution),
        )
        predator_config = EmbodiedRuleEvolutionConfig(
            generations=self.config.generations,
            population_size=self.population_layout.genome_population_size(
                self.evolution.population_size, max(1, self.evaluator.config.predator_count),
            ),
            initial_sigma=self.evolution.initial_sigma, seed=self.config.seed + 1,
            initial_genome=predator_initial, algorithm=self.evolution.algorithm,
            mutation_sigma=self.evolution.mutation_sigma, elite_fraction=self.evolution.elite_fraction,
            immigrant_fraction=self.evolution.immigrant_fraction, immigrant_sigma=self.evolution.immigrant_sigma,
            immigrant_mode=self.evolution.immigrant_mode, max_genome_norm=self.evolution.max_genome_norm,
            max_parameter_magnitude=self.evolution.max_parameter_magnitude,
            **_multiscale_kwargs(self.evolution),
        )
        prey_optimizer = _make_optimizer(self.evaluator.codec.dimension, prey_config, codec=self.evaluator.codec)
        predator_optimizer = _make_optimizer(self.evaluator.codec.dimension, predator_config, codec=self.evaluator.codec)
        prey_hall = [(float("-inf"), tuple(prey_initial))]
        predator_hall = [(float("-inf"), tuple(predator_initial))]
        empty_behavior = _public_behavior(_mean_behavior([]))
        prey_best = (float("-inf"), tuple(prey_initial), empty_behavior)
        predator_best = (float("-inf"), tuple(predator_initial), empty_behavior)
        prey_evaluations = predator_evaluations = 0
        prey_validation_evaluations = predator_validation_evaluations = 0
        history: list[dict[str, object]] = []
        validation_seeds = tuple(
            self.config.seed + 90_000_019 + trial * 10_007
            for trial in range(self.config.validation_trials)
        )
        # This bank is never evaluated inside the optimizer/model-selection
        # loop.  It is touched once, after the validation-selected rule is
        # frozen, so the final number is a genuine test estimate.
        test_seeds = tuple(
            self.config.seed + 190_000_033 + trial * 10_007
            for trial in range(self.config.test_trials)
        )
        # Fixed anchor opponents make validation lifetime comparable across
        # generations.  The training pools below can still coevolve.
        prey_validation_opponents = (tuple(predator_initial),)
        predator_validation_opponents = (tuple(prey_initial),)

        workers = self._resolved_workers()
        executor_context = (
            ProcessPoolExecutor(
                max_workers=workers, mp_context=get_context("spawn"),
                initializer=_initialize_evaluation_worker, initargs=(self.evaluator,),
            )
            if workers > 1 else nullcontext(None)
        )
        with executor_context as executor:
            for generation in range(1, self.config.generations + 1):
                if should_stop and should_stop():
                    raise EvolutionTerminated()
                seeds = tuple(self.config.seed + generation * 1_000_003 + trial * 10_007 for trial in range(self.config.trials))
                predator_pool = tuple(genome for _, genome in predator_hall[:self.config.opponent_pool_size])
                prey_population = prey_optimizer.ask()
                prey_rows = self._evaluate_population(executor, prey_population, predator_pool, Species.PREY, seeds)
                prey_scores = tuple(row[0] for row in prey_rows)
                prey_optimizer.tell(prey_population, prey_scores)
                prey_index = max(range(len(prey_population)), key=lambda index: prey_scores[index])
                prey_winner = prey_population[prey_index]
                prey_evaluations += len(prey_population) * len(predator_pool) * len(seeds)
                prey_validation = self.evaluator.evaluate_focal(
                    prey_winner, prey_validation_opponents, Species.PREY, validation_seeds
                )
                prey_validation_evaluations += len(prey_validation_opponents) * len(validation_seeds)
                prey_hall = self._validated_archive(prey_hall, prey_validation[0], prey_winner)
                if prey_validation[0] > prey_best[0]:
                    prey_best = prey_validation[0], prey_winner, prey_validation[2]

                predator_scores: tuple[float, ...] = ()
                if self.evaluator.config.predator_count:
                    prey_pool = tuple(genome for _, genome in prey_hall[:self.config.opponent_pool_size])
                    predator_population = predator_optimizer.ask()
                    predator_rows = self._evaluate_population(
                        executor, predator_population, prey_pool, Species.PREDATOR, seeds,
                    )
                    predator_scores = tuple(row[0] for row in predator_rows)
                    predator_optimizer.tell(predator_population, predator_scores)
                    predator_index = max(range(len(predator_population)), key=lambda index: predator_scores[index])
                    predator_winner = predator_population[predator_index]
                    predator_evaluations += len(predator_population) * len(prey_pool) * len(seeds)
                    predator_validation = self.evaluator.evaluate_focal(
                        predator_winner, predator_validation_opponents, Species.PREDATOR, validation_seeds
                    )
                    predator_validation_evaluations += len(predator_validation_opponents) * len(validation_seeds)
                    predator_hall = self._validated_archive(predator_hall, predator_validation[0], predator_winner)
                    if predator_validation[0] > predator_best[0]:
                        predator_best = predator_validation[0], predator_winner, predator_validation[2]
                else:
                    predator_validation = (0.0, (), empty_behavior)
                    predator_best = 0.0, tuple(predator_initial), empty_behavior

                row = {
                    "generation": generation,
                    "prey_best_lifetime": prey_scores[prey_index], "prey_mean_lifetime": fmean(prey_scores),
                    "prey_validation_lifetime": prey_validation[0],
                    "predator_best_lifetime": max(predator_scores) if predator_scores else 0.0,
                    "predator_mean_lifetime": fmean(predator_scores) if predator_scores else 0.0,
                    "predator_validation_lifetime": predator_validation[0],
                    **{f"prey_{key}": value for key, value in prey_validation[2].items()},
                    **{f"predator_{key}": value for key, value in predator_validation[2].items()},
                    "episode_seeds": list(seeds),
                    "validation_seeds": list(validation_seeds),
                    "prey_genotype": population_statistics(prey_population, node_dimension=self.evaluator.architecture.parameter_count),
                    "predator_genotype": population_statistics(predator_population, node_dimension=self.evaluator.architecture.parameter_count) if predator_scores else {},
                    "prey_normalizations": getattr(prey_optimizer, "last_ask_normalizations", 0),
                    "predator_normalizations": getattr(predator_optimizer, "last_ask_normalizations", 0),
                    "prey_sampling": getattr(prey_optimizer, "last_sampling_telemetry", {}),
                    "predator_sampling": getattr(predator_optimizer, "last_sampling_telemetry", {}) if predator_scores else {},
                    "prey_phenotype_diversity": _phenotype_diversity([item[2] for item in prey_rows]),
                    "predator_phenotype_diversity": _phenotype_diversity([item[2] for item in predator_rows]) if predator_scores else 0.0,
                }
                history.append(row)
                event = {
                    "phase": "batch_food_web", "training_mode": "batch", "algorithm": self.evolution.algorithm,
                    "objective": "restricted_mean_lifetime", "objective_units": "ticks",
                    "generation": generation, "generations": self.config.generations,
                    "workers": workers, "execution_backend": self.evaluator.config.network.execution_backend,
                    "device": _execution_device(self.evaluator.config.network),
                    "prey": self._snapshot(prey_optimizer, prey_best, prey_evaluations, prey_validation_evaluations),
                    "predator": self._snapshot(predator_optimizer, predator_best, predator_evaluations, predator_validation_evaluations, active=bool(self.evaluator.config.predator_count)),
                    "history": list(history),
                }
                if progress:
                    progress(event)

            if should_stop and should_stop():
                raise EvolutionTerminated()
            prey_snapshot = self._snapshot(prey_optimizer, prey_best, prey_evaluations, prey_validation_evaluations)
            predator_snapshot = self._snapshot(predator_optimizer, predator_best, predator_evaluations, predator_validation_evaluations, active=bool(self.evaluator.config.predator_count))
            prey_test = self._final_test(
                prey_best[1], tuple(predator_initial), Species.PREY, test_seeds, zero, executor,
            )
            prey_test["selection_validation_lifetime"] = prey_best[0]
            prey_snapshot.update(prey_test)
            if self.evaluator.config.predator_count:
                predator_test = self._final_test(
                    predator_best[1], tuple(prey_initial), Species.PREDATOR, test_seeds, zero, executor,
                )
                predator_test["selection_validation_lifetime"] = predator_best[0]
                predator_snapshot.update(predator_test)
            else:
                predator_snapshot.update(self._inactive_test_summary())
        return {
            "task": "batch_food_web_coevolution", "training_mode": "batch", "algorithm": self.evolution.algorithm,
            "objective": "restricted_mean_lifetime", "objective_units": "ticks",
            "population_size": self.evolution.population_size,
            "population_mode": self.config.population_mode,
            "world_count": self.evolution.population_size,
            "prey_genome_population_size": prey_config.population_size,
            "predator_genome_population_size": (
                predator_config.population_size if self.evaluator.config.predator_count else 0
            ),
            "initial_sigma": self.evolution.initial_sigma,
            "mutation_sigma": self.evolution.mutation_sigma or self.evolution.initial_sigma,
            "elite_fraction": self.evolution.elite_fraction,
            "immigrant_fraction": self.evolution.immigrant_fraction,
            "immigrant_sigma": self.evolution.immigrant_sigma or max(.05, self.evolution.initial_sigma * 3.0),
            "immigrant_mode": self.evolution.immigrant_mode,
            "local_mutation_sigma": self.evolution.local_mutation_sigma or self.evolution.mutation_sigma or self.evolution.initial_sigma,
            "local_offspring_fraction": self.evolution.local_offspring_fraction,
            "regional_fraction": self.evolution.regional_fraction,
            "regional_scale": self.evolution.regional_scale,
            "regional_min_std": self.evolution.regional_min_std,
            "global_fraction": self.evolution.global_fraction,
            "global_parameter_range": self.evolution.global_parameter_range,
            "global_viability_filter": self.evolution.global_viability_filter,
            "global_max_sampling_attempts": self.evolution.global_max_sampling_attempts,
            "max_genome_norm": self.evolution.max_genome_norm,
            "max_parameter_magnitude": self.evolution.max_parameter_magnitude,
            "generations": self.config.generations, "episode_steps": self.config.episode_steps,
            "trials": self.config.trials, "validation_trials": self.config.validation_trials,
            "test_trials": self.config.test_trials, "test_seeds": list(test_seeds),
            "opponent_pool_size": self.config.opponent_pool_size,
            "execution": {
                "workers": workers, "backend": self.evaluator.config.network.execution_backend,
                "device": _execution_device(self.evaluator.config.network),
            },
            "prey": prey_snapshot, "predator": predator_snapshot, "history": history,
            "prey_best_genome": list(prey_best[1]), "predator_best_genome": list(predator_best[1]),
        }

    def _final_test(
        self, genome: tuple[float, ...], opponent: tuple[float, ...],
        species: Species, seeds: tuple[int, ...], zero: tuple[float, ...],
        executor: ProcessPoolExecutor | None,
    ) -> dict[str, object]:
        selected, neutral, vision_masked = self._evaluate_jobs(
            executor,
            (
                (genome, (opponent,), species, seeds, "none"),
                (zero, (opponent,), species, seeds, "none"),
                (genome, (opponent,), species, seeds, "vision"),
            ),
        )
        return {
            "selection_validation_lifetime": None,
            "test_lifetime": selected[0],
            "test_lifetimes": list(selected[1]),
            "test_behavior": dict(selected[2]),
            "test_evaluations": 3 * len(seeds),
            "baselines": {
                "zero_rule_lifetime": neutral[0],
                "zero_rule_lifetimes": list(neutral[1]),
                "zero_rule_behavior": dict(neutral[2]),
                "vision_masked_lifetime": vision_masked[0],
                "vision_masked_lifetimes": list(vision_masked[1]),
                "vision_masked_behavior": dict(vision_masked[2]),
                "lifetime_gain_over_zero_rule": selected[0] - neutral[0],
                "vision_lifetime_delta": selected[0] - vision_masked[0],
            },
        }

    def _resolved_workers(self) -> int:
        if self.config.workers:
            workers = self.config.workers
        else:
            workers = min(self.evolution.population_size, max(1, (os.cpu_count() or 2) - 1))
        device = self.evaluator.config.network.device
        uses_cuda = (
            self.evaluator.config.network.execution_backend == "torch"
            and resolve_device(device).type == "cuda"
        )
        if uses_cuda:
            # A single CUDA owner avoids duplicated model/state memory and
            # context contention. Candidate-level process parallelism is for CPU.
            return 1
        return workers

    def _evaluate_population(
        self, executor: ProcessPoolExecutor | None, population: Sequence[tuple[float, ...]],
        opponents: tuple[tuple[float, ...], ...], species: Species, seeds: tuple[int, ...],
    ) -> tuple[tuple[float, tuple[float, ...], dict[str, float]], ...]:
        agents_per_world = (
            self.evaluator.config.prey_count if species is Species.PREY
            else self.evaluator.config.predator_count
        )
        indexed_population = list(enumerate(population))
        if self.config.population_mode == "mixed_individual_population":
            # Re-group every generation independently of genome values.  A
            # persistent slot would otherwise couple a genome to one spawn
            # position, graph-seed sequence, and set of competitors.
            Random(seeds[0] + (0x51A7 if species is Species.PREY else 0xA93D)).shuffle(
                indexed_population
            )
        groups = self.population_layout.groups(
            tuple(genome for _, genome in indexed_population), agents_per_world,
        )
        jobs = tuple(
            (self.config.population_mode, group, opponents, species, seeds)
            for group in groups
        )
        if executor is not None:
            rows = executor.map(_evaluate_population_group_worker, jobs, chunksize=1)
        else:
            rows = (
                _evaluate_population_group_worker_with(self.evaluator, job)
                for job in jobs
            )
        ordered_rows = [
            row for group_rows in rows for row in group_rows
        ]
        if self.config.population_mode == "shared_rule_cohort":
            return tuple(ordered_rows)
        restored: list[tuple[float, tuple[float, ...], dict[str, float]] | None] = [
            None
        ] * len(population)
        for (original_index, _), row in zip(indexed_population, ordered_rows, strict=True):
            restored[original_index] = row
        if any(row is None for row in restored):
            raise RuntimeError("mixed population evaluation lost a candidate score")
        return tuple(row for row in restored if row is not None)

    def _evaluate_jobs(
        self, executor: ProcessPoolExecutor | None,
        jobs: Sequence[tuple[tuple[float, ...], tuple[tuple[float, ...], ...], Species, tuple[int, ...], Literal["none", "vision"]]],
    ) -> tuple[tuple[float, tuple[float, ...], dict[str, float]], ...]:
        if executor is not None:
            return tuple(executor.map(_evaluate_focal_worker, jobs, chunksize=1))
        return tuple(
            self.evaluator.evaluate_focal(
                genome, opponents, species, seeds, observation_mask=observation_mask,
            )
            for genome, opponents, species, seeds, observation_mask in jobs
        )

    @staticmethod
    def _inactive_test_summary() -> dict[str, object]:
        return {
            "selection_validation_lifetime": 0.0, "test_lifetime": 0.0,
            "test_lifetimes": [], "test_behavior": {}, "test_evaluations": 0,
            "baselines": {
                "zero_rule_lifetime": 0.0, "zero_rule_lifetimes": [], "zero_rule_behavior": {},
                "vision_masked_lifetime": 0.0, "vision_masked_lifetimes": [], "vision_masked_behavior": {},
                "lifetime_gain_over_zero_rule": 0.0, "vision_lifetime_delta": 0.0,
            },
        }

    @staticmethod
    def _snapshot(
        optimizer: RuleOptimizer, best: tuple[float, tuple[float, ...], Mapping[str, float]],
        evaluations: int, validation_evaluations: int, *, active: bool = True,
    ) -> dict[str, object]:
        snapshot: dict[str, object] = {
            "updates": optimizer.generation if active else 0, "evaluations": evaluations,
            "validation_evaluations": validation_evaluations,
            "best_lifetime": best[0], "best_genome": list(best[1]), "sigma": optimizer.sigma,
            "behavior": dict(best[2]),
        }
        if isinstance(optimizer, GeneticAlgorithm):
            snapshot["elite_archive"] = list(optimizer.elite_archive)
        return snapshot

    @staticmethod
    def _validated_archive(
        archive: list[tuple[float, tuple[float, ...]]], lifetime: float, genome: tuple[float, ...],
    ) -> list[tuple[float, tuple[float, ...]]]:
        rows = [item for item in archive if item[1] != genome]
        rows.append((float(lifetime), genome))
        rows.sort(key=lambda item: item[0], reverse=True)
        return rows[:24]


@dataclass(frozen=True, slots=True)
class OnlineRuleBirth(ControllerBlueprint):
    """One library assignment; graph and node state are still created at birth."""

    blueprint: EmbodiedFoodWebControllerBlueprint
    genome: tuple[float, ...]
    cohort: int

    def build(self, *, seed: int | None = None) -> Controller:
        return self.blueprint.build(seed=seed)


class OnlineRuleLibrary:
    """Per-species steady-state optimizer using completed lifespan alone."""

    evaluation_replicates = 2

    def __init__(
        self, codec: GenomeCodec, config: EmbodiedRuleEvolutionConfig, network: EmbodiedNetworkConfig,
        *, seed: int,
    ) -> None:
        self.codec, self.network, self.random = codec, network, Random(seed)
        self.algorithm = config.algorithm
        self.optimizer = _make_optimizer(codec.dimension, config, seed=seed, codec=codec)
        initial = tuple(config.initial_genome) if config.initial_genome is not None else (0.0,) * codec.dimension
        self.archive: list[tuple[float, tuple[float, ...]]] = [(0.0, initial)]
        self._has_evaluated_archive = False
        self.cohort_index = 0
        self.cohort: list[tuple[float, ...]] = []
        self.assignments: list[int] = []
        self.scores: dict[int, list[float]] = {}
        self.updates = 0
        self.deaths = 0
        self._next_cohort()

    def birth(self) -> OnlineRuleBirth:
        index = min(range(len(self.cohort)), key=lambda item: (self.assignments[item], item))
        self.assignments[index] += 1
        genome = self.cohort[index]
        node_rule, edge_rule = self.codec.decode_groups(genome, output_scale=self.network.rule_output_scale)
        assert node_rule is not None and edge_rule is not None
        blueprint = EmbodiedFoodWebControllerBlueprint(node_rule, edge_rule, self.network, self.random.randrange(2**32))
        return OnlineRuleBirth(blueprint, genome, self.cohort_index)

    def observe(self, birth: OnlineRuleBirth, lifetime: float) -> None:
        self.deaths += 1
        if birth.cohort != self.cohort_index:
            return  # A late death from a closed cohort cannot be compared with the current cohort.
        try:
            index = self.cohort.index(birth.genome)
        except ValueError:
            return
        self.scores.setdefault(index, []).append(float(lifetime))
        if len(self.scores) == len(self.cohort) and all(
            len(self.scores[index]) >= self.evaluation_replicates
            for index in range(len(self.cohort))
        ):
            values = [fmean(self.scores[index]) for index in range(len(self.cohort))]
            # Archive comparable replicated means, never a lucky single life.
            for score, genome in zip(values, self.cohort, strict=True):
                self._archive(score, genome)
            self.optimizer.tell(self.cohort, values)
            if isinstance(self.optimizer, GeneticAlgorithm):
                # The GA archive is the source of truth for both the next
                # elite cohort and the reported best rule: its score is an
                # average across every elite re-evaluation.
                self.archive = [
                    (float(record["mean_score"]), tuple(float(value) for value in record["genome"]))
                    for record in self.optimizer.elite_archive
                ]
            self.updates += 1
            self._next_cohort()

    def snapshot(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm, "cohort": self.cohort_index, "updates": self.updates, "deaths": self.deaths,
            "evaluated": len(self.scores), "library_size": len(self.cohort),
            "best_lifetime": self.archive[0][0], "best_genome": list(self.archive[0][1]),
            "sigma": self.optimizer.sigma, "evaluation_replicates": self.evaluation_replicates,
            "elite_archive": (
                list(self.optimizer.elite_archive)
                if isinstance(self.optimizer, GeneticAlgorithm) else []
            ),
        }

    def _next_cohort(self) -> None:
        # Both optimizers require ``tell`` to receive exactly what ``ask``
        # supplied; the archive remains separate for reporting and warm starts.
        self.cohort = list(self.optimizer.ask())
        self.assignments, self.scores = [0] * len(self.cohort), {}
        self.cohort_index += 1

    def _archive(self, lifetime: float, genome: tuple[float, ...]) -> None:
        if not self._has_evaluated_archive:
            # The initial row is only a pre-evaluation fallback and must not
            # outrank genuinely evaluated lifetimes.
            self.archive.clear()
            self._has_evaluated_archive = True
        self.archive.append((lifetime, genome))
        self.archive.sort(key=lambda item: item[0], reverse=True)
        del self.archive[24:]


@dataclass(frozen=True, slots=True)
class ContinuousFoodWebConfig:
    """Continuous world budget; optimizer updates are triggered by real deaths."""

    ticks: int = 600
    seed: int = 1
    initial_genome: tuple[float, ...] | None = None
    initial_prey_genome: tuple[float, ...] | None = None
    initial_predator_genome: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.ticks < 1:
            raise ValueError("continuous evolution ticks must be positive")


class ContinuousFoodWebCoevolutionRunner:
    """Keep one food web alive while one rule library serves every species."""

    def __init__(
        self, evaluator: FoodWebCoevolutionEvaluator, evolution: EmbodiedRuleEvolutionConfig,
        config: ContinuousFoodWebConfig,
    ) -> None:
        self.evaluator, self.evolution, self.config = evaluator, evolution, config
        for genome in (config.initial_genome, config.initial_prey_genome, config.initial_predator_genome):
            if genome is not None and len(genome) != evaluator.codec.dimension:
                raise ValueError("initial genome does not match the joint rule architecture")

    def run(
        self, progress: Callable[[dict[str, object]], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, object]:
        initial = _shared_initial_genome(
            self.evaluator.codec.dimension,
            initial=(self.config.initial_genome or self.evolution.initial_genome),
            prey_initial=self.config.initial_prey_genome,
            predator_initial=self.config.initial_predator_genome,
        )
        shared_config = EmbodiedRuleEvolutionConfig(
            generations=1, population_size=self.evolution.population_size,
            initial_sigma=self.evolution.initial_sigma, seed=self.config.seed, initial_genome=initial,
            algorithm=self.evolution.algorithm,
            mutation_sigma=self.evolution.mutation_sigma, elite_fraction=self.evolution.elite_fraction,
            immigrant_fraction=self.evolution.immigrant_fraction, immigrant_sigma=self.evolution.immigrant_sigma,
            immigrant_mode=self.evolution.immigrant_mode, max_genome_norm=self.evolution.max_genome_norm,
            max_parameter_magnitude=self.evolution.max_parameter_magnitude,
            **_multiscale_kwargs(self.evolution),
        )
        shared_library = OnlineRuleLibrary(
            self.evaluator.codec, shared_config, self.evaluator.config.network, seed=self.config.seed + 101,
        )
        world = FoodWebEnvironment(self.evaluator.config.environment, seed=self.config.seed)
        agents = make_reference_population(
            prey_count=self.evaluator.config.prey_count, predator_count=self.evaluator.config.predator_count,
            width=self.evaluator.config.environment.width, height=self.evaluator.config.environment.height,
            prey_initial_energy=self.evaluator.config.environment.prey_initial_energy, predator_initial_energy=self.evaluator.config.environment.predator_initial_energy, controller=RandomControllerBlueprint(), seed=self.config.seed,
        )
        # A birth selects from the same optimizer cohort for either species.
        libraries = {Species.PREY: shared_library, Species.PREDATOR: shared_library}
        for organism in agents:
            organism.controller = libraries[organism.species].birth()
            world.add(organism)
        observations = world.reset(seed=self.config.seed)
        random = Random(self.config.seed + 303)
        controllers = {organism.id: organism.controller.build(seed=random.randrange(2**32)) for organism in agents if organism.controller}
        for controller in controllers.values():
            controller.begin_episode(seed=random.randrange(2**32))
        population = world.population()
        lifetimes = {"prey": deque(maxlen=40), "predator": deque(maxlen=40)}
        cumulative_deaths = {"prey": 0, "predator": 0}
        cumulative_meals = {"prey": 0, "predator": 0}
        body_totals = {
            "prey": {"hunger": 0.0, "energy_change": 0.0, "samples": 0},
            "predator": {"hunger": 0.0, "energy_change": 0.0, "samples": 0},
        }

        def library_snapshot(species: Species) -> dict[str, object]:
            name = str(species)
            snapshot = libraries[species].snapshot()
            # The optimizer statistics are shared, while the ecology counters
            # remain role-specific for inspection and plotting.
            snapshot["shared_deaths"] = snapshot["deaths"]
            snapshot["deaths"] = cumulative_deaths[name]
            samples = max(1, int(body_totals[name]["samples"]))
            snapshot["behavior"] = {
                "meals": cumulative_meals[name],
                "meal_rate": cumulative_meals[name] / samples,
                "mean_hunger": float(body_totals[name]["hunger"]) / samples,
                "mean_abs_energy_change": float(body_totals[name]["energy_change"]) / samples,
                "mean_lifetime": fmean(lifetimes[name]) if lifetimes[name] else None,
            }
            if species is Species.PREY:
                environment = self.evaluator.config.environment
                metabolic_cost = environment.prey_metabolism * environment.timestep_seconds
                required_meal_rate = metabolic_cost / max(environment.plant_energy, 1e-12)
                meal_rate = cumulative_meals[name] / samples
                natural_lifetime = environment.prey_initial_energy / max(metabolic_cost, 1e-12)
                mean_lifetime = snapshot["behavior"]["mean_lifetime"]
                snapshot["behavior"].update({
                    "required_meal_rate": required_meal_rate,
                    "meal_rate_coverage": meal_rate / max(required_meal_rate, 1e-12),
                    "natural_lifetime": natural_lifetime,
                    "excess_lifetime": (float(mean_lifetime) - natural_lifetime) if mean_lifetime is not None else None,
                })
            return snapshot

        environment = self.evaluator.config.environment
        prey_demand = self.evaluator.config.prey_count * environment.prey_metabolism
        prey_supply = environment.plant_regrowth * environment.plant_energy
        ecology = {
            "prey_energy_supply_per_second": prey_supply,
            "prey_energy_demand_per_second": prey_demand,
            "prey_energy_supply_ratio": prey_supply / max(prey_demand, 1e-12),
            "population_sustainable_from_regrowth": prey_supply >= prey_demand,
        }

        telemetry: deque[dict[str, object]] = deque(maxlen=240)
        for tick in range(1, self.config.ticks + 1):
            if should_stop and should_stop():
                raise EvolutionTerminated()
            for agent_id, observation in observations.items():
                species = str(population[agent_id].species)
                body_totals[species]["hunger"] += float(observation.get("hunger", 0.0))
                body_totals[species]["energy_change"] += abs(float(observation.get("energy_change", 0.0)))
                body_totals[species]["samples"] += 1
            actions = {agent_id: controllers[agent_id].act(observation, available_actions=world.available_actions(agent_id)) for agent_id, observation in observations.items()}
            result = world.step(actions)
            population = world.population()
            for meal in result.info.get("meals", ()):
                cumulative_meals[str(meal["species"])] += 1
            for record in result.info.get("death_records", ()):
                agent_id = AgentId(str(record["agent_id"]))
                organism = population[agent_id]
                cumulative_deaths[str(organism.species)] += 1
                lifetimes[str(organism.species)].append(float(record["age"]))
                birth = organism.controller
                if not isinstance(birth, OnlineRuleBirth):
                    raise RuntimeError("continuous world lost its inherited rule blueprint")
                libraries[organism.species].observe(birth, float(record["age"]))
                controllers[agent_id].end_episode()
                organism.controller = libraries[organism.species].birth()
                controllers[agent_id] = organism.controller.build(seed=random.randrange(2**32))
                controllers[agent_id].begin_episode(seed=random.randrange(2**32))
            observations = result.observations
            prey_samples = max(1, int(body_totals["prey"]["samples"]))
            predator_samples = max(1, int(body_totals["predator"]["samples"]))
            telemetry.append({
                "tick": tick, "prey_deaths": cumulative_deaths["prey"],
                "predator_deaths": cumulative_deaths["predator"],
                "prey_meals": cumulative_meals["prey"], "predator_meals": cumulative_meals["predator"],
                "prey_mean_lifetime": fmean(lifetimes["prey"]) if lifetimes["prey"] else None,
                "predator_mean_lifetime": fmean(lifetimes["predator"]) if lifetimes["predator"] else None,
                "prey_mean_hunger": float(body_totals["prey"]["hunger"]) / prey_samples,
                "predator_mean_hunger": float(body_totals["predator"]["hunger"]) / predator_samples,
                "prey_mean_abs_energy_change": float(body_totals["prey"]["energy_change"]) / prey_samples,
                "predator_mean_abs_energy_change": float(body_totals["predator"]["energy_change"]) / predator_samples,
                "prey_meal_rate_coverage": (
                    cumulative_meals["prey"] / prey_samples
                    / max(environment.prey_metabolism * environment.timestep_seconds / max(environment.plant_energy, 1e-12), 1e-12)
                ),
            })
            if progress and (tick == 1 or tick % 4 == 0 or tick == self.config.ticks):
                progress({"tick": tick, "ticks": self.config.ticks, "phase": "continuous_food_web_shared_rule", "objective": "completed_lifetime", "objective_units": "ticks", "population": world.snapshot()["population"], "ecology": ecology, "execution_backend": self.evaluator.config.network.execution_backend, "device": _execution_device(self.evaluator.config.network), "shared": shared_library.snapshot(), "prey": library_snapshot(Species.PREY), "predator": library_snapshot(Species.PREDATOR), "telemetry": list(telemetry)})
        for controller in controllers.values():
            controller.end_episode()
        shared_snapshot = shared_library.snapshot()
        return {"task": "continuous_food_web_shared_rule_evolution", "algorithm": self.evolution.algorithm, "objective": "completed_lifetime", "objective_units": "ticks", "rule_sharing": "one_genome_for_prey_and_predator", "population_size": self.evolution.population_size, "initial_sigma": self.evolution.initial_sigma, "ticks": self.config.ticks, "execution": {"workers": 1, "backend": self.evaluator.config.network.execution_backend, "device": _execution_device(self.evaluator.config.network)}, "population": world.snapshot()["population"], "ecology": ecology, "shared": shared_snapshot, "prey": library_snapshot(Species.PREY), "predator": library_snapshot(Species.PREDATOR), "telemetry": list(telemetry), "shared_best_genome": list(shared_library.archive[0][1]), "prey_best_genome": list(shared_library.archive[0][1]), "predator_best_genome": list(shared_library.archive[0][1])}


class FoodWebDemonstration:
    """Mutable visual replay of one shared rule in a fresh ecology."""

    def __init__(
        self, evaluator: FoodWebCoevolutionEvaluator, shared_genome: Sequence[float],
        *, seed: int,
    ) -> None:
        self.evaluator, self.random, self.tick = evaluator, Random(seed + 41), 0
        self.world = FoodWebEnvironment(evaluator.config.environment, seed=seed)
        genome = tuple(float(value) for value in shared_genome)
        prey = evaluator._blueprint(genome, seed, Species.PREY)
        predator = evaluator._blueprint(genome, seed, Species.PREDATOR)
        agents = make_reference_population(
            prey_count=evaluator.config.prey_count, predator_count=evaluator.config.predator_count,
            width=evaluator.config.environment.width, height=evaluator.config.environment.height,
            prey_initial_energy=evaluator.config.environment.prey_initial_energy, predator_initial_energy=evaluator.config.environment.predator_initial_energy, controller=RandomControllerBlueprint(), seed=seed,
        )
        for organism in agents:
            organism.controller = prey if organism.species is Species.PREY else predator
            self.world.add(organism)
        self.observations = self.world.reset(seed=seed)
        self.controllers = {organism.id: organism.controller.build(seed=self.random.randrange(2**32)) for organism in agents if organism.controller}
        for controller in self.controllers.values():
            controller.begin_episode(seed=self.random.randrange(2**32))
        self.last_events: dict[str, object] = {"births": (), "deaths": (), "meals": ()}

    def advance(self, ticks: int = 1) -> dict[str, object]:
        if ticks < 1 or ticks > 128:
            raise ValueError("demo ticks must be between 1 and 128")
        for _ in range(ticks):
            actions = {agent_id: self.controllers[agent_id].act(observation, available_actions=self.world.available_actions(agent_id)) for agent_id, observation in self.observations.items()}
            result = self.world.step(actions)
            population = self.world.population()
            for agent_id in result.info.get("births", ()):
                organism = population[agent_id]
                self.controllers[agent_id].end_episode()
                self.controllers[agent_id] = organism.controller.build(seed=self.random.randrange(2**32))
                self.controllers[agent_id].begin_episode(seed=self.random.randrange(2**32))
            self.observations, self.last_events, self.tick = result.observations, dict(result.info), self.tick + 1
        return self.snapshot()

    def snapshot(self) -> dict[str, object]:
        # The canvas and body telemetry must describe the same ecology tick.
        # Supplying the already-computed observations here avoids a second,
        # potentially stale inspection request during fast playback.
        return {
            "tick": self.tick,
            "state": self.world.snapshot(),
            "observations": {str(agent_id): observation for agent_id, observation in self.observations.items()},
            "events": self.last_events,
        }

    def individual_snapshot(self, agent_id: AgentId) -> dict[str, object]:
        """Return one living organism's private recurrent network for display."""
        controller = self.controllers.get(agent_id)
        if not isinstance(controller, EmbodiedFoodWebController):
            raise KeyError(agent_id)
        population = self.world.population()
        organism = population.get(agent_id)
        if organism is None:
            raise KeyError(agent_id)
        return {
            "tick": self.tick,
            "individual": {
                "id": organism.id,
                "species": organism.species.value,
                "energy": organism.energy,
                "position": {"x": organism.position.x, "y": organism.position.y},
                "heading": organism.heading,
                "age": organism.age,
                "life": organism.life,
            },
            # Use the exact sensory record supplied to this controller for the
            # current world state.  This keeps visual ray debugging faithful to
            # the learned interface rather than reconstructing it in the UI.
            "observation": self.observations.get(agent_id, {}),
            "network": controller.inspection_snapshot(),
        }
