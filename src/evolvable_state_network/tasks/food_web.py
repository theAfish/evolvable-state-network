"""Evaluate ESN update-rule genomes in the predator–prey–plant world.

This explicit bridge keeps ``evolution`` independent of any particular task
and keeps ``environments`` independent of any optimizer or neural substrate.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin, tanh
from statistics import fmean
from typing import Sequence

from ..environments import (
    Action, Controller, ControllerBlueprint, EpisodeRunner, FoodWebConfig,
    FoodWebEnvironment, Observation, RandomControllerBlueprint, Species,
    make_reference_population,
)
from ..evolution.candidate import MLPUpdateRule, RuleArchitecture
from ..evolution.genome import GenomeCodec


def food_web_features(observation: Observation, width: int) -> tuple[float, ...]:
    """Project variable ray observations into a bounded ESN message vector."""
    if width < 1:
        raise ValueError("feature width must be positive")
    raw = [
        min(max(float(observation.get("energy", 0.0)) / 20.0, 0.0), 2.0),
        sin(float(observation.get("heading", 0.0))),
        cos(float(observation.get("heading", 0.0))),
        min(float(observation.get("age", 0.0)) / 400.0, 1.0),
    ]
    for ray in observation.get("vision", ()):
        distance, ray_range = ray.get("distance"), float(ray.get("range", 24.0))
        proximity = 0.0 if distance is None else 1.0 - min(float(distance) / ray_range, 1.0)
        raw.extend((proximity, proximity * float(ray.get("kind") == "plant"), proximity * float(ray.get("kind") == "prey"), proximity * float(ray.get("kind") == "predator")))
    features = [0.0] * width
    for index, value in enumerate(raw):
        features[index % width] += value
    return tuple(tanh(value) for value in features)


class MLPFoodWebController(Controller):
    """Use a local ESN update rule as a recurrent embodied controller."""

    def __init__(self, rule: MLPUpdateRule) -> None:
        self._rule = rule
        self._state = rule.initial_state()

    def begin_episode(self, *, seed: int | None = None) -> None:
        self._state = self._rule.initial_state()

    def act(self, observation: Observation, *, available_actions: Sequence[Action]) -> Action:
        if not available_actions:
            raise ValueError("environment offered no actions")
        self._state = self._rule.update(self._state, food_web_features(observation, self._rule.state_width), .05, .12)
        turn = max(-1.0, min(1.0, self._state[0] / .12))
        speed = max(0.0, min(1.0, (self._state[1] / .12 + 1.0) / 2.0))
        return {"kind": "turn_move", "turn": turn, "speed": speed}


@dataclass(frozen=True, slots=True)
class MLPFoodWebControllerBlueprint(ControllerBlueprint):
    """Immutable controller recipe created from a decoded ESN node rule."""

    rule: MLPUpdateRule

    def __post_init__(self) -> None:
        if self.rule.state_width < 2:
            raise ValueError("food-web control requires an ESN state width of at least two")

    def build(self, *, seed: int | None = None) -> Controller:
        return MLPFoodWebController(self.rule)


@dataclass(frozen=True, slots=True)
class FoodWebTaskConfig:
    environment: FoodWebConfig = FoodWebConfig()
    max_steps: int = 160
    trials: int = 3
    seed: int = 1
    prey_count: int = 5
    predator_count: int = 2
    focal_species: Species = Species.PREY

    def __post_init__(self) -> None:
        if self.max_steps < 1 or self.trials < 1 or self.prey_count < 1 or self.predator_count < 1:
            raise ValueError("task steps, trials, and population counts must be positive")


@dataclass(frozen=True, slots=True)
class FoodWebEvaluation:
    genome: tuple[float, ...]
    trial_returns: tuple[float, ...]
    fitness: float


class FoodWebTaskEvaluator:
    """Deterministically score a node-rule genome on repeated food-web trials."""

    def __init__(self, architecture: RuleArchitecture | None = None, config: FoodWebTaskConfig | None = None) -> None:
        self.architecture = architecture or RuleArchitecture()
        if self.architecture.state_width < 2:
            raise ValueError("food-web task requires a rule architecture with state_width >= 2")
        self.codec = GenomeCodec(self.architecture)
        self.config = config or FoodWebTaskConfig()

    def evaluate(self, genome: Sequence[float]) -> FoodWebEvaluation:
        encoded = tuple(float(value) for value in genome)
        blueprint = MLPFoodWebControllerBlueprint(self.codec.decode(encoded))
        lifetimes = tuple(self._run_trial(blueprint, self.config.seed + trial) for trial in range(self.config.trials))
        return FoodWebEvaluation(encoded, lifetimes, fmean(lifetimes))

    def _run_trial(self, focal_controller: ControllerBlueprint, seed: int) -> float:
        agents = make_reference_population(
            prey_count=self.config.prey_count, predator_count=self.config.predator_count,
            width=self.config.environment.width, height=self.config.environment.height,
            controller=RandomControllerBlueprint(),
        )
        focal = [agent.id for agent in agents if agent.species is self.config.focal_species]
        for agent in agents:
            if agent.species is self.config.focal_species:
                agent.controller = focal_controller
        result = EpisodeRunner(FoodWebEnvironment(self.config.environment, seed=seed)).run(agents, max_steps=self.config.max_steps, seed=seed)
        return fmean(float(result.behavior[agent_id]["restricted_lifetime"]) for agent_id in focal)
