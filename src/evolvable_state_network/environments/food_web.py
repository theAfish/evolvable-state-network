"""A continuous, toroidal predator–prey–plant environment.

The world owns only ecology.  Its agents carry controller blueprints, but it
never selects, mutates, or evaluates them; those concerns live in callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import pi
from random import Random
from typing import Mapping, Sequence

import numpy as np

from .controllers import RandomControllerBlueprint
from .core import Action, AgentId, ControllerBlueprint, Environment, Observation, StepResult
from .sensing import RayHit, SenseTarget, Vec2, Viewport, scan_ray_fan


# Population placement and environment layout receive the same episode seed.
# Keep their pseudo-random streams independent so an organism's index cannot
# reveal (or exactly reproduce) a plant-cluster position.
_PREY_POSITION_SEED_SALT = 0x6A09E667
_PREDATOR_POSITION_SEED_SALT = 0xBB67AE85


class Species(StrEnum):
    PLANT = "plant"
    PREY = "prey"
    PREDATOR = "predator"


@dataclass(slots=True)
class Organism:
    id: AgentId
    species: Species
    position: Vec2
    energy: float
    heading: float = 0.0
    viewport: Viewport = field(default_factory=Viewport)
    radius: float = 1.2
    age: int = 0
    life: int = 0
    alive: bool = True
    traits: dict[str, float] = field(default_factory=dict)
    last_energy_change: float = 0.0
    ate_last_step: bool = False
    controller: ControllerBlueprint | None = None
    spawn_position: Vec2 = field(init=False)
    spawn_energy: float = field(init=False)

    def __post_init__(self) -> None:
        self.spawn_position, self.spawn_energy = self.position, self.energy


@dataclass(frozen=True, slots=True)
class Plant:
    id: str
    position: Vec2
    radius: float = .65


@dataclass(frozen=True, slots=True)
class FoodWebConfig:
    width: float = 100.0
    height: float = 60.0
    initial_plants: int = 24
    max_plants: int = 80
    timestep_seconds: float = .125
    plant_regrowth: float = 24.0
    prey_metabolism: float = 3.6
    predator_metabolism: float = 5.6
    plant_energy: float = 2.0
    prey_energy: float = 4.0
    prey_initial_energy: float = 9.0
    predator_initial_energy: float = 14.0
    max_turn: float = 9.0
    max_speed: float = 10.0
    interaction_range: float = 2.0
    spawn_cluster_radius: float = 12.0
    spawn_candidate_count: int = 48
    plant_cluster_count: int = 4
    plant_cluster_radius: float = 5.0
    respawn_on_death: bool = True

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.timestep_seconds <= 0:
            raise ValueError("world dimensions and timestep must be positive")
        if self.initial_plants < 0 or self.max_plants < self.initial_plants or self.plant_regrowth < 0:
            raise ValueError("plant counts/regrowth are invalid")
        if self.prey_initial_energy <= 0 or self.predator_initial_energy <= 0:
            raise ValueError("initial organism energy must be positive")
        if self.plant_cluster_count < 0 or self.plant_cluster_radius < 0:
            raise ValueError("plant clustering parameters are invalid")


class FoodWebEnvironment(Environment):
    """Synchronous 2D food web with movement, predation, and plant regrowth."""

    ACTIONS: tuple[Action, ...] = tuple(
        {"kind": "turn_move", "turn": turn, "speed": speed}
        for turn in (-1.0, 0.0, 1.0) for speed in (0.0, 1.0)
    )

    def __init__(self, config: FoodWebConfig | None = None, *, seed: int | None = None) -> None:
        self.config = config or FoodWebConfig()
        self._random = Random(seed)
        self._organisms: dict[AgentId, Organism] = {}
        self._plants: dict[str, Plant] = {}
        self._next_plant = 0
        self._plant_cluster_centers: tuple[Vec2, ...] = ()
        self._plant_regrowth_credit = 0.0
        self._elapsed_seconds = 0.0
        self._last_meal_time: dict[AgentId, float] = {}

    def add(self, organism: Organism) -> None:
        if organism.id in self._organisms:
            raise ValueError(f"duplicate organism id: {organism.id}")
        self._organisms[organism.id] = organism

    def reset(self, *, seed: int | None = None) -> Mapping[AgentId, Observation]:
        if seed is not None:
            self._random.seed(seed)
        self._plants, self._next_plant = {}, 0
        self._plant_regrowth_credit, self._elapsed_seconds = 0.0, 0.0
        self._plant_cluster_centers = tuple(
            Vec2(self._random.uniform(0, self.config.width), self._random.uniform(0, self.config.height))
            for _ in range(self.config.plant_cluster_count)
        )
        for _ in range(self.config.initial_plants):
            self._grow_plant()
        for organism in self._organisms.values():
            organism.position, organism.energy, organism.alive, organism.age, organism.life = organism.spawn_position, organism.spawn_energy, True, 0, 0
            organism.last_energy_change, organism.ate_last_step = 0.0, False
        self._last_meal_time = {organism.id: 0.0 for organism in self._organisms.values()}
        return self._observations()

    def available_actions(self, agent_id: AgentId) -> Sequence[Action]:
        return self.ACTIONS if self._organisms[agent_id].alive else ()

    def step(self, actions: Mapping[AgentId, Action]) -> StepResult:
        dt, event_time = self.config.timestep_seconds, self._elapsed_seconds + self.config.timestep_seconds
        living = [organism for organism in self._organisms.values() if organism.alive]
        starting_energy = {organism.id: organism.energy for organism in living}
        for organism in living:
            organism.ate_last_step = False
        # The food web exposes ecological events and body sensations, never a
        # hand-shaped learning reward. Evolution measures lifespan directly.
        rewards = {organism.id: 0.0 for organism in living}
        meals: list[dict[str, object]] = []
        if living:
            turns = np.asarray([float(actions.get(organism.id, {}).get("turn", 0.0)) for organism in living])
            speeds = np.clip(np.asarray([float(actions.get(organism.id, {}).get("speed", 0.0)) for organism in living]), 0.0, 1.0)
            headings = np.asarray([organism.heading for organism in living]) + turns * self.config.max_turn * dt
            positions = np.asarray([(organism.position.x, organism.position.y) for organism in living], dtype=np.float64)
            multipliers = np.asarray([self._trait(organism, "speed_multiplier") for organism in living])
            positions += np.column_stack((np.cos(headings), np.sin(headings))) * (speeds * multipliers * self.config.max_speed * dt)[:, None]
            positions = np.mod(positions, (self.config.width, self.config.height))
            metabolism = np.asarray([self._metabolism(organism) * self._trait(organism, "metabolism_multiplier") for organism in living])
            for organism, heading, position, energy in zip(living, headings, positions, np.asarray([o.energy for o in living]) - metabolism * dt, strict=True):
                organism.heading, organism.position, organism.energy, organism.age = float(heading), Vec2(float(position[0]), float(position[1])), float(energy), organism.age + 1

        prey = [organism for organism in living if organism.species is Species.PREY]
        plants = tuple(self._plants.values())
        if prey and plants:
            nearby = self._periodic_distance_squared(np.asarray([(o.position.x, o.position.y) for o in prey]), np.asarray([(p.position.x, p.position.y) for p in plants])) <= self.config.interaction_range ** 2
            available = np.ones(len(plants), dtype=bool)
            for index, organism in enumerate(prey):
                choices = np.flatnonzero(nearby[index] & available)
                if choices.size:
                    plant = plants[int(choices[0])]
                    available[int(choices[0])] = False
                    del self._plants[plant.id]
                    organism.energy += self.config.plant_energy
                    self._record_meal(organism, event_time, meals)

        deaths: list[AgentId] = []
        predators = [organism for organism in living if organism.species is Species.PREDATOR]
        if predators and prey:
            nearby = self._periodic_distance_squared(np.asarray([(o.position.x, o.position.y) for o in predators]), np.asarray([(o.position.x, o.position.y) for o in prey])) <= self.config.interaction_range ** 2
            prey_alive = np.asarray([organism.alive for organism in prey], dtype=bool)
            for index, predator in enumerate(predators):
                choices = np.flatnonzero(nearby[index] & prey_alive)
                if choices.size:
                    victim = prey[int(choices[0])]
                    prey_alive[int(choices[0])] = False
                    victim.alive = False
                    deaths.append(victim.id)
                    predator.energy += self.config.prey_energy
                    self._record_meal(predator, event_time, meals)

        births: list[AgentId] = []
        for organism in living:
            if organism.alive and organism.energy <= 0:
                organism.alive = False
                deaths.append(organism.id)
        for organism in living:
            organism.last_energy_change = (
                organism.energy - starting_energy[organism.id]
            ) / max(organism.spawn_energy, 1e-12)
        death_records = tuple(
            {"agent_id": str(agent_id), "species": str(self._organisms[agent_id].species),
             "age": self._organisms[agent_id].age}
            for agent_id in dict.fromkeys(deaths)
        )
        if self.config.respawn_on_death:
            for organism in self._organisms.values():
                if not organism.alive:
                    self._respawn(organism)
                    births.append(organism.id)
        self._plant_regrowth_credit += self.config.plant_regrowth * dt
        for _ in range(int(self._plant_regrowth_credit)):
            self._grow_plant()
        self._plant_regrowth_credit %= 1.0
        self._elapsed_seconds += dt
        alive = frozenset(organism.id for organism in self._organisms.values() if organism.alive)
        return StepResult(self._observations(), rewards, not alive, alive, {"plants": len(self._plants), "deaths": tuple(deaths), "births": tuple(births), "death_records": death_records, "meals": tuple(meals)})

    def population(self) -> Mapping[AgentId, Organism]:
        return self._organisms.copy()

    def snapshot(self) -> dict[str, object]:
        return {"time": round(self._elapsed_seconds, 3), "bounds": {"width": self.config.width, "height": self.config.height}, "plant_capacity": self.config.max_plants,
                "plant_clusters": [{"x": round(center.x, 2), "y": round(center.y, 2), "radius": self.config.plant_cluster_radius} for center in self._plant_cluster_centers],
                "plants": [{"id": plant.id, "x": round(plant.position.x, 2), "y": round(plant.position.y, 2), "radius": plant.radius} for plant in self._plants.values()],
                "organisms": [{"id": str(organism.id), "species": str(organism.species), "x": round(organism.position.x, 2), "y": round(organism.position.y, 2), "heading": round(organism.heading, 3), "energy": round(organism.energy, 2), "age": organism.age, "life": organism.life} for organism in self._organisms.values() if organism.alive],
                "population": {species: sum(organism.alive and str(organism.species) == species for organism in self._organisms.values()) for species in (str(Species.PREY), str(Species.PREDATOR))}}

    def _grow_plant(self) -> None:
        if len(self._plants) < self.config.max_plants:
            plant_id = f"plant-{self._next_plant}"
            self._next_plant += 1
            if self._plant_cluster_centers:
                center = self._plant_cluster_centers[self._next_plant % len(self._plant_cluster_centers)]
                position = Vec2(
                    (center.x + self._random.gauss(0.0, self.config.plant_cluster_radius)) % self.config.width,
                    (center.y + self._random.gauss(0.0, self.config.plant_cluster_radius)) % self.config.height,
                )
            else:
                position = Vec2(self._random.uniform(0, self.config.width), self._random.uniform(0, self.config.height))
            self._plants[plant_id] = Plant(plant_id, position)

    def _respawn(self, organism: Organism) -> None:
        organism.position = self._spawn_position(organism)
        organism.heading, organism.energy, organism.age, organism.alive, organism.life = self._random.uniform(-pi, pi), organism.spawn_energy, 0, True, organism.life + 1
        organism.last_energy_change, organism.ate_last_step = 0.0, False
        self._last_meal_time[organism.id] = self._elapsed_seconds + self.config.timestep_seconds

    def _spawn_position(self, newborn: Organism) -> Vec2:
        """Place every replacement uniformly in the full toroidal world.

        Continuous rule learning should expose a newborn to a new local
        ecological context rather than preserving spatial family clusters.
        """
        return Vec2(self._random.uniform(0, self.config.width), self._random.uniform(0, self.config.height))

    def _metabolism(self, organism: Organism) -> float:
        return self.config.prey_metabolism if organism.species is Species.PREY else self.config.predator_metabolism

    @staticmethod
    def _trait(organism: Organism, name: str) -> float:
        return organism.traits.get(name, 1.0)

    def _periodic_distance_squared(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        offsets = first[:, None, :] - second[None, :, :]
        extent = np.asarray((self.config.width, self.config.height))
        offsets -= np.round(offsets / extent) * extent
        return np.sum(offsets ** 2, axis=2)

    def _scan(self, organism: Organism) -> tuple[RayHit, ...]:
        targets = [SenseTarget(plant.id, str(Species.PLANT), plant.position, plant.radius) for plant in self._plants.values()]
        targets.extend(SenseTarget(str(other.id), str(other.species), other.position, other.radius) for other in self._organisms.values() if other.alive and other.id != organism.id)
        return scan_ray_fan(organism.position, organism.viewport, organism.heading, targets, periodic_bounds=(self.config.width, self.config.height))

    @staticmethod
    def _ray_json(ray: RayHit, ray_range: float) -> dict[str, object]:
        return {"angle": ray.angle, "distance": ray.distance, "kind": ray.kind, "target_id": ray.target_id, "range": ray_range}

    def _record_meal(self, organism: Organism, time: float, meals: list[dict[str, object]]) -> None:
        meals.append({"agent_id": str(organism.id), "species": str(organism.species), "interval": time - self._last_meal_time.get(organism.id, time)})
        self._last_meal_time[organism.id] = time
        organism.ate_last_step = True

    def _observations(self) -> Mapping[AgentId, Observation]:
        observations: dict[AgentId, Observation] = {}
        for organism in self._organisms.values():
            if not organism.alive:
                continue
            hunger = max(0.0, min(1.0, 1.0 - organism.energy / max(organism.spawn_energy, 1e-12)))
            natural_lifetime = organism.spawn_energy / max(self._metabolism(organism), 1e-12)
            time_since_meal = max(0.0, self._elapsed_seconds - self._last_meal_time.get(organism.id, 0.0))
            observations[organism.id] = {
                "position": {"x": organism.position.x, "y": organism.position.y},
                "heading": organism.heading,
                "energy": organism.energy,
                "age": organism.age,
                # These are body sensations, not optimizer rewards.  A rule
                # may ignore them, react directly, or retain their history in
                # its recurrent node/edge state.
                "hunger": hunger,
                "energy_change": organism.last_energy_change,
                "ate": organism.ate_last_step,
                "time_since_meal": min(1.0, time_since_meal / max(natural_lifetime, 1e-12)),
                "vision": tuple(self._ray_json(ray, organism.viewport.range) for ray in self._scan(organism)),
            }
        return observations


def make_reference_population(*, prey_count: int = 5, predator_count: int = 2, width: float = 100.0, height: float = 60.0, prey_initial_energy: float = 9.0, predator_initial_energy: float = 14.0, controller: ControllerBlueprint | None = None, seed: int | None = None) -> list[Organism]:
    """Make independently controlled agents with seedable random initial poses."""
    if prey_count < 0 or predator_count < 0 or prey_initial_energy <= 0 or predator_initial_energy <= 0:
        raise ValueError("population counts and initial energy must be positive")
    blueprint = controller or RandomControllerBlueprint()
    def positions(count: int, offset: float, seed_salt: int) -> list[Vec2]:
        if seed is not None:
            random = Random((int(seed) + seed_salt) % (2**32))
            return [Vec2(random.uniform(0, width), random.uniform(0, height)) for _ in range(count)]
        columns = max(1, int(np.ceil(np.sqrt(max(1, count) * width / height))))
        rows = max(1, int(np.ceil(count / columns)))
        return [Vec2(((index % columns + .5) / columns * width + offset) % width, (index // columns + .5) / rows * height) for index in range(count)]
    traits = {"speed_multiplier": 1.0, "metabolism_multiplier": 1.0}
    prey = [Organism(AgentId(f"prey-{index}"), Species.PREY, position, prey_initial_energy, heading=index * .7, traits=dict(traits), controller=blueprint) for index, position in enumerate(positions(prey_count, 0.0, _PREY_POSITION_SEED_SALT))]
    predators = [Organism(AgentId(f"predator-{index}"), Species.PREDATOR, position, predator_initial_energy, heading=pi, viewport=Viewport(range=24, field_of_view=pi * .65, ray_count=11), traits=dict(traits), controller=blueprint) for index, position in enumerate(positions(predator_count, width * .19, _PREDATOR_POSITION_SEED_SALT))]
    return prey + predators
