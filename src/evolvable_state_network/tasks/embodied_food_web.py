"""Evolution task: shared local rules learn through random embodied food-web agents."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from random import Random
from statistics import fmean
from typing import Callable, Literal, Mapping, Sequence

from ..embodied import EmbodiedNetwork, EmbodiedNetworkConfig, FoodWebAgentAdapter
from ..environments import (
    Action, AgentId, Controller, ControllerBlueprint, EpisodeRunner, FoodWebConfig,
    FoodWebEnvironment, Observation, RandomControllerBlueprint, Species,
    make_reference_population,
)
from ..evolution.candidate import EdgeArchitecture, MLPEdgeRule, MLPUpdateRule, RuleArchitecture
from ..evolution.cmaes import CMAES, CMAESConfig
from ..evolution.genetic import GeneticAlgorithm, GeneticAlgorithmConfig
from ..evolution.genome import GenomeCodec


BEHAVIOR_KEYS = (
    "meals", "meal_rate", "mean_hunger", "mean_abs_energy_change",
    "mean_action_change", "early_return_rate", "late_return_rate", "adaptation_delta",
    "mean_turn", "mean_abs_turn", "mean_speed", "early_mean_turn", "late_mean_turn",
    "turn_drift", "early_mean_abs_turn", "late_mean_abs_turn", "abs_turn_drift",
    "early_mean_speed", "late_mean_speed", "speed_drift", "turn_saturation_rate",
    "plant_visible_rate", "plant_steering_alignment", "deaths_per_1000_steps",
    "mean_completed_lifetime", "final_energy_fraction",
)


def _network_seed(seed: int, species: Species) -> int:
    """Return scenario randomness that is independent of candidate parameters."""
    species_salt = 0x51A7 if species is Species.PREY else 0xA93D
    return (int(seed) * 1_000_003 + species_salt) % (2**32)


def _mean_behavior(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    return {
        key: fmean(float(row.get(key, 0.0)) for row in rows) if rows else 0.0
        for key in BEHAVIOR_KEYS
    }


def _species_behavior(result: object, ids: Sequence[AgentId]) -> dict[str, float]:
    behavior = getattr(result, "behavior")
    return _mean_behavior([behavior[agent_id] for agent_id in ids])


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
            schema=self._config.observation_schema,
            directional=self._config.nodes >= 15,
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
    """One species-specific task; predator and prey receive separate runs."""

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


@dataclass(frozen=True, slots=True)
class EmbodiedFoodWebEvaluation:
    genome: tuple[float, ...]
    trial_returns: tuple[float, ...]
    fitness: float
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
        node_rule, edge_rule = self.codec.decode_groups(encoded)
        assert node_rule is not None and edge_rule is not None
        trials = tuple(
            self._trial(node_rule, edge_rule, self.config.seed + 10_007 * index)
            for index in range(self.config.trials)
        )
        returns = tuple(item[0] for item in trials)
        return EmbodiedFoodWebEvaluation(encoded, returns, fmean(returns), _mean_behavior([item[1] for item in trials]))

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
        return fmean(result.returns[agent_id] for agent_id in ids), _species_behavior(result, ids)


@dataclass(frozen=True, slots=True)
class FoodWebCoevolutionEvaluation:
    """Matched prey/predator rule evaluations from one shared set of episodes."""

    prey_genome: tuple[float, ...]
    predator_genome: tuple[float, ...]
    prey_trial_returns: tuple[float, ...]
    predator_trial_returns: tuple[float, ...]
    prey_behavior: Mapping[str, float]
    predator_behavior: Mapping[str, float]

    @property
    def prey_fitness(self) -> float:
        return fmean(self.prey_trial_returns)

    @property
    def predator_fitness(self) -> float:
        return fmean(self.predator_trial_returns)


class FoodWebCoevolutionEvaluator:
    """Evaluate two independent rule genomes in the *same* food-web episode."""

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
        trial_returns = tuple(
            self._trial(prey, predator, self.config.seed + 10_007 * index)
            for index in range(self.config.trials)
        )
        return FoodWebCoevolutionEvaluation(
            prey, predator,
            tuple(item[0] for item in trial_returns), tuple(item[1] for item in trial_returns),
            _mean_behavior([item[2] for item in trial_returns]),
            _mean_behavior([item[3] for item in trial_returns]),
        )

    def evaluate_focal(
        self, genome: Sequence[float], opponents: Sequence[Sequence[float]],
        focal_species: Species, seeds: Sequence[int], *, observation_mask: Literal["none", "vision"] = "none",
    ) -> tuple[float, tuple[float, ...], dict[str, float]]:
        """Evaluate one candidate against frozen opponents and matched random networks."""
        focal = tuple(float(value) for value in genome)
        if not opponents or not seeds:
            raise ValueError("batch evaluation needs at least one opponent and seed")
        returns: list[float] = []
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
                returns.append(result[0] if focal_species is Species.PREY else result[1])
                behavior.append(result[2] if focal_species is Species.PREY else result[3])
        return fmean(returns), tuple(returns), _mean_behavior(behavior)

    def _blueprint(
        self, genome: tuple[float, ...], seed: int, species: Species,
        observation_mask: Literal["none", "vision"] = "none",
    ) -> EmbodiedFoodWebControllerBlueprint:
        node_rule, edge_rule = self.codec.decode_groups(genome)
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
        prey_return = fmean(result.returns[agent_id] for agent_id in prey_ids)
        # A prey-only ecology has no predator genome to score.  Keep the
        # evaluator total so shared callers can still construct it safely.
        predator_return = fmean(result.returns[agent_id] for agent_id in predator_ids) if predator_ids else 0.0
        return (
            prey_return, predator_return,
            _species_behavior(result, prey_ids),
            _species_behavior(result, predator_ids) if predator_ids else _mean_behavior([]),
        )


@dataclass(frozen=True, slots=True)
class EmbodiedRuleEvolutionConfig:
    """Optimizer settings for one agent species' local update-rule genome."""

    generations: int = 8
    population_size: int = 8
    initial_sigma: float = .05
    seed: int = 1
    initial_genome: tuple[float, ...] | None = None
    algorithm: Literal["cma_es", "genetic"] = "cma_es"

    def __post_init__(self) -> None:
        if self.generations < 1 or self.population_size < 2 or self.initial_sigma <= 0:
            raise ValueError("evolution configuration is invalid")


RuleOptimizer = CMAES | GeneticAlgorithm


def _make_optimizer(dimension: int, config: EmbodiedRuleEvolutionConfig, *, seed: int | None = None) -> RuleOptimizer:
    optimizer_seed = config.seed if seed is None else seed
    if config.algorithm == "genetic":
        return GeneticAlgorithm(
            GeneticAlgorithmConfig(
                dimension, config.population_size, config.initial_sigma, optimizer_seed,
                elite_fraction=.25, immigrant_fraction=.25,
                immigrant_sigma=max(.05, config.initial_sigma * 3.0),
            ),
            config.initial_genome,
        )
    return CMAES(
        CMAESConfig(dimension, config.population_size, config.initial_sigma, optimizer_seed),
        config.initial_genome,
    )


class EmbodiedRuleEvolutionRunner:
    """Optimises only rule genomes; every evaluation rebuilds random agents."""

    def __init__(self, evaluator: EmbodiedFoodWebEvaluator, config: EmbodiedRuleEvolutionConfig) -> None:
        self.evaluator, self.config = evaluator, config
        if config.initial_genome is not None and len(config.initial_genome) != evaluator.codec.dimension:
            raise ValueError("initial genome does not match the joint rule architecture")

    def run(self, progress: Callable[[dict[str, object]], None] | None = None) -> dict[str, object]:
        optimizer = _make_optimizer(self.evaluator.codec.dimension, self.config)
        history: list[dict[str, object]] = []
        best: EmbodiedFoodWebEvaluation | None = None
        for _ in range(self.config.generations):
            population = optimizer.ask()
            evaluations = self.evaluator.evaluate_batch(population)
            winner = max(evaluations, key=lambda item: item.fitness)
            if best is None or winner.fitness > best.fitness:
                best = winner
            optimizer.tell(population, [item.fitness for item in evaluations])
            row = {"generation": optimizer.generation, "best_fitness": winner.fitness, "mean_fitness": fmean(item.fitness for item in evaluations), "sigma": optimizer.sigma}
            history.append(row)
            if progress:
                progress(row)
        assert best is not None
        return {"task": "food_web", "algorithm": self.config.algorithm, "focal_species": str(self.evaluator.config.focal_species), "best_genome": list(best.genome), "best_fitness": best.fitness, "best_trial_returns": list(best.trial_returns), "history": history}


class FoodWebCoevolutionRunner:
    """Run prey and predator optimizer populations in matched world episodes."""

    def __init__(self, evaluator: FoodWebCoevolutionEvaluator, config: EmbodiedRuleEvolutionConfig) -> None:
        self.evaluator, self.config = evaluator, config
        if config.initial_genome is not None and len(config.initial_genome) != evaluator.codec.dimension:
            raise ValueError("initial genome does not match the joint rule architecture")

    def run(self, progress: Callable[[dict[str, object]], None] | None = None) -> dict[str, object]:
        prey_optimizer = _make_optimizer(self.evaluator.codec.dimension, self.config)
        predator_optimizer = _make_optimizer(self.evaluator.codec.dimension, self.config, seed=self.config.seed + 1)
        history: list[dict[str, object]] = []
        best_prey: FoodWebCoevolutionEvaluation | None = None
        best_predator: FoodWebCoevolutionEvaluation | None = None
        for _ in range(self.config.generations):
            prey_population, predator_population = prey_optimizer.ask(), predator_optimizer.ask()
            evaluations = tuple(self.evaluator.evaluate(prey, predator) for prey, predator in zip(prey_population, predator_population, strict=True))
            prey_winner = max(evaluations, key=lambda item: item.prey_fitness)
            predator_winner = max(evaluations, key=lambda item: item.predator_fitness)
            if best_prey is None or prey_winner.prey_fitness > best_prey.prey_fitness:
                best_prey = prey_winner
            if best_predator is None or predator_winner.predator_fitness > best_predator.predator_fitness:
                best_predator = predator_winner
            prey_optimizer.tell(prey_population, [item.prey_fitness for item in evaluations])
            predator_optimizer.tell(predator_population, [item.predator_fitness for item in evaluations])
            row = {
                "generation": prey_optimizer.generation,
                "prey_best_fitness": prey_winner.prey_fitness,
                "prey_mean_fitness": fmean(item.prey_fitness for item in evaluations),
                "predator_best_fitness": predator_winner.predator_fitness,
                "predator_mean_fitness": fmean(item.predator_fitness for item in evaluations),
                "prey_sigma": prey_optimizer.sigma, "predator_sigma": predator_optimizer.sigma,
            }
            history.append(row)
            if progress:
                progress(row)
        assert best_prey is not None and best_predator is not None
        return {
            "task": "food_web_coevolution",
            "algorithm": self.config.algorithm,
            "prey_best_genome": list(best_prey.prey_genome), "prey_best_fitness": best_prey.prey_fitness,
            "predator_best_genome": list(best_predator.predator_genome), "predator_best_fitness": best_predator.predator_fitness,
            "history": history,
        }


@dataclass(frozen=True, slots=True)
class BatchFoodWebConfig:
    """Comparable episodic batches with frozen opponents and common seeds."""

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

    def __post_init__(self) -> None:
        if (
            self.generations < 1 or self.episode_steps < 1 or self.trials < 1
            or self.validation_trials < 1 or self.test_trials < 1 or self.opponent_pool_size < 1
        ):
            raise ValueError("batch food-web configuration is invalid")


class BatchFoodWebCoevolutionRunner:
    """Alternately optimize each species in complete, directly comparable batches."""

    def __init__(
        self, evaluator: FoodWebCoevolutionEvaluator, evolution: EmbodiedRuleEvolutionConfig,
        config: BatchFoodWebConfig,
    ) -> None:
        self.evaluator, self.evolution, self.config = evaluator, evolution, config
        if evaluator.config.max_steps != config.episode_steps:
            raise ValueError("batch evaluator max_steps must equal episode_steps")
        for genome in (config.initial_genome, config.initial_prey_genome, config.initial_predator_genome):
            if genome is not None and len(genome) != evaluator.codec.dimension:
                raise ValueError("initial genome does not match the joint rule architecture")

    def run(self, progress: Callable[[dict[str, object]], None] | None = None) -> dict[str, object]:
        initial = self.config.initial_genome if self.config.initial_genome is not None else self.evolution.initial_genome
        zero = (0.0,) * self.evaluator.codec.dimension
        prey_initial = self.config.initial_prey_genome if self.config.initial_prey_genome is not None else (initial or zero)
        predator_initial = self.config.initial_predator_genome if self.config.initial_predator_genome is not None else (initial or zero)
        prey_config = EmbodiedRuleEvolutionConfig(
            generations=self.config.generations, population_size=self.evolution.population_size,
            initial_sigma=self.evolution.initial_sigma, seed=self.config.seed,
            initial_genome=prey_initial, algorithm=self.evolution.algorithm,
        )
        predator_config = EmbodiedRuleEvolutionConfig(
            generations=self.config.generations, population_size=self.evolution.population_size,
            initial_sigma=self.evolution.initial_sigma, seed=self.config.seed + 1,
            initial_genome=predator_initial, algorithm=self.evolution.algorithm,
        )
        prey_optimizer = _make_optimizer(self.evaluator.codec.dimension, prey_config)
        predator_optimizer = _make_optimizer(self.evaluator.codec.dimension, predator_config)
        prey_hall = [(float("-inf"), tuple(prey_initial))]
        predator_hall = [(float("-inf"), tuple(predator_initial))]
        empty_behavior = _mean_behavior([])
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
        # Fixed anchor opponents make validation fitness comparable across
        # generations.  The training pools below can still coevolve.
        prey_validation_opponents = (tuple(predator_initial),)
        predator_validation_opponents = (tuple(prey_initial),)

        for generation in range(1, self.config.generations + 1):
            seeds = tuple(self.config.seed + generation * 1_000_003 + trial * 10_007 for trial in range(self.config.trials))
            predator_pool = tuple(genome for _, genome in predator_hall[:self.config.opponent_pool_size])
            prey_population = prey_optimizer.ask()
            prey_rows = tuple(self.evaluator.evaluate_focal(genome, predator_pool, Species.PREY, seeds) for genome in prey_population)
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
                predator_rows = tuple(self.evaluator.evaluate_focal(genome, prey_pool, Species.PREDATOR, seeds) for genome in predator_population)
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
                "prey_best_fitness": prey_scores[prey_index], "prey_mean_fitness": fmean(prey_scores),
                "prey_validation_fitness": prey_validation[0],
                "predator_best_fitness": max(predator_scores) if predator_scores else 0.0,
                "predator_mean_fitness": fmean(predator_scores) if predator_scores else 0.0,
                "predator_validation_fitness": predator_validation[0],
                **{f"prey_{key}": value for key, value in prey_validation[2].items()},
                **{f"predator_{key}": value for key, value in predator_validation[2].items()},
                "episode_seeds": list(seeds),
                "validation_seeds": list(validation_seeds),
            }
            history.append(row)
            event = {
                "phase": "batch_food_web", "training_mode": "batch", "algorithm": self.evolution.algorithm,
                "generation": generation, "generations": self.config.generations,
                "prey": self._snapshot(prey_optimizer, prey_best, prey_evaluations, prey_validation_evaluations),
                "predator": self._snapshot(predator_optimizer, predator_best, predator_evaluations, predator_validation_evaluations, active=bool(self.evaluator.config.predator_count)),
                "history": list(history),
            }
            if progress:
                progress(event)

        prey_snapshot = self._snapshot(prey_optimizer, prey_best, prey_evaluations, prey_validation_evaluations)
        predator_snapshot = self._snapshot(predator_optimizer, predator_best, predator_evaluations, predator_validation_evaluations, active=bool(self.evaluator.config.predator_count))
        prey_test = self._final_test(
            prey_best[1], tuple(predator_initial), Species.PREY, test_seeds, zero
        )
        prey_test["selection_validation_fitness"] = prey_best[0]
        prey_snapshot.update(prey_test)
        if self.evaluator.config.predator_count:
            predator_test = self._final_test(
                predator_best[1], tuple(prey_initial), Species.PREDATOR, test_seeds, zero
            )
            predator_test["selection_validation_fitness"] = predator_best[0]
            predator_snapshot.update(predator_test)
        else:
            predator_snapshot.update(self._inactive_test_summary())
        return {
            "task": "batch_food_web_coevolution", "training_mode": "batch", "algorithm": self.evolution.algorithm,
            "generations": self.config.generations, "episode_steps": self.config.episode_steps,
            "trials": self.config.trials, "validation_trials": self.config.validation_trials,
            "test_trials": self.config.test_trials, "test_seeds": list(test_seeds),
            "opponent_pool_size": self.config.opponent_pool_size,
            "prey": prey_snapshot, "predator": predator_snapshot, "history": history,
            "prey_best_genome": list(prey_best[1]), "predator_best_genome": list(predator_best[1]),
        }

    def _final_test(
        self, genome: tuple[float, ...], opponent: tuple[float, ...],
        species: Species, seeds: tuple[int, ...], zero: tuple[float, ...],
    ) -> dict[str, object]:
        selected = self.evaluator.evaluate_focal(genome, (opponent,), species, seeds)
        neutral = self.evaluator.evaluate_focal(zero, (opponent,), species, seeds)
        vision_masked = self.evaluator.evaluate_focal(
            genome, (opponent,), species, seeds, observation_mask="vision"
        )
        return {
            "selection_validation_fitness": None,
            "test_fitness": selected[0],
            "test_returns": list(selected[1]),
            "test_behavior": dict(selected[2]),
            "test_evaluations": 3 * len(seeds),
            "baselines": {
                "zero_rule_fitness": neutral[0],
                "zero_rule_returns": list(neutral[1]),
                "zero_rule_behavior": dict(neutral[2]),
                "vision_masked_fitness": vision_masked[0],
                "vision_masked_returns": list(vision_masked[1]),
                "vision_masked_behavior": dict(vision_masked[2]),
                "gain_over_zero_rule": selected[0] - neutral[0],
                "vision_ablation_delta": selected[0] - vision_masked[0],
            },
        }

    @staticmethod
    def _inactive_test_summary() -> dict[str, object]:
        return {
            "selection_validation_fitness": 0.0, "test_fitness": 0.0,
            "test_returns": [], "test_behavior": {}, "test_evaluations": 0,
            "baselines": {
                "zero_rule_fitness": 0.0, "zero_rule_returns": [], "zero_rule_behavior": {},
                "vision_masked_fitness": 0.0, "vision_masked_returns": [], "vision_masked_behavior": {},
                "gain_over_zero_rule": 0.0, "vision_ablation_delta": 0.0,
            },
        }

    @staticmethod
    def _snapshot(
        optimizer: RuleOptimizer, best: tuple[float, tuple[float, ...], Mapping[str, float]],
        evaluations: int, validation_evaluations: int, *, active: bool = True,
    ) -> dict[str, object]:
        return {
            "updates": optimizer.generation if active else 0, "evaluations": evaluations,
            "validation_evaluations": validation_evaluations,
            "best_fitness": best[0], "best_genome": list(best[1]), "sigma": optimizer.sigma,
            "behavior": dict(best[2]),
        }

    @staticmethod
    def _validated_archive(
        archive: list[tuple[float, tuple[float, ...]]], fitness: float, genome: tuple[float, ...],
    ) -> list[tuple[float, tuple[float, ...]]]:
        rows = [item for item in archive if item[1] != genome]
        rows.append((float(fitness), genome))
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
    """Per-species steady-state optimizer library with an elite reporting archive."""

    evaluation_replicates = 2

    def __init__(
        self, codec: GenomeCodec, config: EmbodiedRuleEvolutionConfig, network: EmbodiedNetworkConfig,
        *, seed: int,
    ) -> None:
        self.codec, self.network, self.random = codec, network, Random(seed)
        self.algorithm = config.algorithm
        self.optimizer = _make_optimizer(codec.dimension, config, seed=seed)
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
        node_rule, edge_rule = self.codec.decode_groups(genome)
        assert node_rule is not None and edge_rule is not None
        blueprint = EmbodiedFoodWebControllerBlueprint(node_rule, edge_rule, self.network, self.random.randrange(2**32))
        return OnlineRuleBirth(blueprint, genome, self.cohort_index)

    def observe(self, birth: OnlineRuleBirth, fitness: float) -> None:
        self.deaths += 1
        if birth.cohort != self.cohort_index:
            return  # A late death from a closed cohort cannot be compared with the current cohort.
        try:
            index = self.cohort.index(birth.genome)
        except ValueError:
            return
        self.scores.setdefault(index, []).append(float(fitness))
        if len(self.scores) == len(self.cohort) and all(
            len(self.scores[index]) >= self.evaluation_replicates
            for index in range(len(self.cohort))
        ):
            values = [fmean(self.scores[index]) for index in range(len(self.cohort))]
            # Archive comparable replicated means, never a lucky single life.
            for score, genome in zip(values, self.cohort, strict=True):
                self._archive(score, genome)
            self.optimizer.tell(self.cohort, values)
            self.updates += 1
            self._next_cohort()

    def snapshot(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm, "cohort": self.cohort_index, "updates": self.updates, "deaths": self.deaths,
            "evaluated": len(self.scores), "library_size": len(self.cohort),
            "best_fitness": self.archive[0][0], "best_genome": list(self.archive[0][1]),
            "sigma": self.optimizer.sigma, "evaluation_replicates": self.evaluation_replicates,
        }

    def _next_cohort(self) -> None:
        # Both optimizers require ``tell`` to receive exactly what ``ask``
        # supplied; the archive remains separate for reporting and warm starts.
        self.cohort = list(self.optimizer.ask())
        self.assignments, self.scores = [0] * len(self.cohort), {}
        self.cohort_index += 1

    def _archive(self, fitness: float, genome: tuple[float, ...]) -> None:
        if not self._has_evaluated_archive:
            # The initial row is only a pre-evaluation fallback and must not
            # outrank genuinely evaluated negative-fitness rules.
            self.archive.clear()
            self._has_evaluated_archive = True
        self.archive.append((fitness, genome))
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
    """Keep one food web alive while each death replaces exactly one organism."""

    def __init__(
        self, evaluator: FoodWebCoevolutionEvaluator, evolution: EmbodiedRuleEvolutionConfig,
        config: ContinuousFoodWebConfig,
    ) -> None:
        self.evaluator, self.evolution, self.config = evaluator, evolution, config
        for genome in (config.initial_genome, config.initial_prey_genome, config.initial_predator_genome):
            if genome is not None and len(genome) != evaluator.codec.dimension:
                raise ValueError("initial genome does not match the joint rule architecture")

    def run(self, progress: Callable[[dict[str, object]], None] | None = None) -> dict[str, object]:
        initial = self.config.initial_genome if self.config.initial_genome is not None else self.evolution.initial_genome
        prey_initial = self.config.initial_prey_genome if self.config.initial_prey_genome is not None else initial
        predator_initial = self.config.initial_predator_genome if self.config.initial_predator_genome is not None else initial
        prey_config = EmbodiedRuleEvolutionConfig(
            generations=1, population_size=self.evolution.population_size,
            initial_sigma=self.evolution.initial_sigma, seed=self.config.seed, initial_genome=prey_initial,
            algorithm=self.evolution.algorithm,
        )
        predator_config = EmbodiedRuleEvolutionConfig(
            generations=1, population_size=self.evolution.population_size,
            initial_sigma=self.evolution.initial_sigma, seed=self.config.seed, initial_genome=predator_initial,
            algorithm=self.evolution.algorithm,
        )
        prey_library = OnlineRuleLibrary(self.evaluator.codec, prey_config, self.evaluator.config.network, seed=self.config.seed + 101)
        predator_library = OnlineRuleLibrary(self.evaluator.codec, predator_config, self.evaluator.config.network, seed=self.config.seed + 202)
        world = FoodWebEnvironment(self.evaluator.config.environment, seed=self.config.seed)
        agents = make_reference_population(
            prey_count=self.evaluator.config.prey_count, predator_count=self.evaluator.config.predator_count,
            width=self.evaluator.config.environment.width, height=self.evaluator.config.environment.height,
            prey_initial_energy=self.evaluator.config.environment.prey_initial_energy, predator_initial_energy=self.evaluator.config.environment.predator_initial_energy, controller=RandomControllerBlueprint(), seed=self.config.seed,
        )
        libraries = {Species.PREY: prey_library, Species.PREDATOR: predator_library}
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
                libraries[organism.species].observe(birth, float(record["fitness"]))
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
                progress({"tick": tick, "ticks": self.config.ticks, "phase": "continuous_food_web", "population": world.snapshot()["population"], "ecology": ecology, "prey": library_snapshot(Species.PREY), "predator": library_snapshot(Species.PREDATOR), "telemetry": list(telemetry)})
        for controller in controllers.values():
            controller.end_episode()
        return {"task": "continuous_food_web_coevolution", "algorithm": self.evolution.algorithm, "ticks": self.config.ticks, "population": world.snapshot()["population"], "ecology": ecology, "prey": library_snapshot(Species.PREY), "predator": library_snapshot(Species.PREDATOR), "telemetry": list(telemetry), "prey_best_genome": list(prey_library.archive[0][1]), "predator_best_genome": list(predator_library.archive[0][1])}


class FoodWebDemonstration:
    """Mutable visual replay of the two best rule genomes in a fresh ecology."""

    def __init__(
        self, evaluator: FoodWebCoevolutionEvaluator, prey_genome: Sequence[float], predator_genome: Sequence[float],
        *, seed: int,
    ) -> None:
        self.evaluator, self.random, self.tick = evaluator, Random(seed + 41), 0
        self.world = FoodWebEnvironment(evaluator.config.environment, seed=seed)
        prey = evaluator._blueprint(tuple(float(value) for value in prey_genome), seed, Species.PREY)
        predator = evaluator._blueprint(tuple(float(value) for value in predator_genome), seed, Species.PREDATOR)
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
        return {"tick": self.tick, "state": self.world.snapshot(), "events": self.last_events}
