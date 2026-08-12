"""Episode orchestration shared by any environment and controller family."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Mapping, Protocol, Sequence

from .core import AgentId, ControllerBlueprint, Environment, StepResult, Transition


class EpisodeAgent(Protocol):
    id: AgentId
    controller: ControllerBlueprint | None


class SimulationHook(Protocol):
    def on_step(self, result: StepResult) -> None: ...


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    steps: int
    terminated: bool
    returns: Mapping[AgentId, float]
    final_population: Mapping[str, int]
    behavior: Mapping[AgentId, Mapping[str, float]]


class EpisodeRunner:
    """Keeps controller lifecycles outside of the environment implementation."""

    def __init__(self, environment: Environment, hooks: Sequence[SimulationHook] = ()) -> None:
        self.environment, self.hooks = environment, tuple(hooks)

    def run(self, agents: Sequence[EpisodeAgent], *, max_steps: int, seed: int | None = None) -> EpisodeResult:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        add = getattr(self.environment, "add")
        available_actions = getattr(self.environment, "available_actions")
        for agent in agents:
            add(agent)
        observations = self.environment.reset(seed=seed)
        random = Random(seed)
        blueprints = {agent.id: agent.controller for agent in agents}
        controllers = {agent_id: blueprint.build(seed=random.randrange(2**32)) for agent_id, blueprint in blueprints.items() if blueprint}
        for controller in controllers.values():
            controller.begin_episode(seed=random.randrange(2**32))
        returns = {agent_id: 0.0 for agent_id in controllers}
        behavior = {
            agent_id: {
                "meals": 0.0, "hunger_sum": 0.0, "energy_change_sum": 0.0,
                "samples": 0.0, "action_change_sum": 0.0, "action_changes": 0.0,
                "early_return": 0.0, "late_return": 0.0,
                "turn_sum": 0.0, "abs_turn_sum": 0.0, "speed_sum": 0.0,
                "early_turn": 0.0, "late_turn": 0.0,
                "early_abs_turn": 0.0, "late_abs_turn": 0.0,
                "early_speed": 0.0, "late_speed": 0.0,
                "turn_saturated": 0.0, "plant_visible": 0.0,
                "plant_directional": 0.0, "plant_aligned": 0.0,
                "deaths": 0.0, "completed_lifetime_sum": 0.0,
                "first_death_age": 0.0, "first_death_observed": 0.0,
            }
            for agent_id in controllers
        }
        previous_actions: dict[AgentId, Mapping[str, object]] = {}
        window = max(1, max_steps // 4)
        retired: set[AgentId] = set()
        terminated = False
        for step in range(1, max_steps + 1):
            actions = {agent_id: controllers[agent_id].act(observation, available_actions=available_actions(agent_id))
                       for agent_id, observation in observations.items() if agent_id in controllers}
            for agent_id, action in actions.items():
                row = behavior[agent_id]
                body = observations[agent_id]
                row["hunger_sum"] += float(body.get("hunger", 0.0))
                row["energy_change_sum"] += abs(float(body.get("energy_change", 0.0)))
                row["samples"] += 1.0
                turn, speed = float(action.get("turn", 0.0)), float(action.get("speed", 0.0))
                row["turn_sum"] += turn
                row["abs_turn_sum"] += abs(turn)
                row["speed_sum"] += speed
                row["turn_saturated"] += float(abs(turn) >= .8)
                if step <= window:
                    row["early_turn"] += turn
                    row["early_abs_turn"] += abs(turn)
                    row["early_speed"] += speed
                if step > max_steps - window:
                    row["late_turn"] += turn
                    row["late_abs_turn"] += abs(turn)
                    row["late_speed"] += speed
                rays = tuple(body.get("vision", ()))
                plants = tuple(
                    (index, ray) for index, ray in enumerate(rays)
                    if ray.get("kind") == "plant" and ray.get("distance") is not None
                )
                if plants:
                    row["plant_visible"] += 1.0
                    pixel, _ = min(plants, key=lambda item: float(item[1]["distance"]))
                    direction = pixel - (len(rays) - 1) / 2
                    if abs(direction) >= .5:
                        row["plant_directional"] += 1.0
                        row["plant_aligned"] += float(turn * direction > 0.0)
                previous = previous_actions.get(agent_id)
                if previous is not None:
                    row["action_change_sum"] += abs(float(action.get("turn", 0.0)) - float(previous.get("turn", 0.0)))
                    row["action_change_sum"] += abs(float(action.get("speed", 0.0)) - float(previous.get("speed", 0.0)))
                    row["action_changes"] += 2.0
                previous_actions[agent_id] = action
            result = self.environment.step(actions)
            deaths = set(result.info.get("deaths", ()))
            births = set(result.info.get("births", ()))
            for agent_id, action in actions.items():
                reward = result.rewards.get(agent_id, 0.0)
                returns[agent_id] += reward
                if step <= window:
                    behavior[agent_id]["early_return"] += reward
                if step > max_steps - window:
                    behavior[agent_id]["late_return"] += reward
                controllers[agent_id].learn(Transition(observations[agent_id], action, reward,
                    result.observations.get(agent_id, {}), agent_id in deaths or agent_id not in result.alive, result.info))
            for meal in result.info.get("meals", ()):
                agent_id = AgentId(str(meal["agent_id"]))
                if agent_id in behavior:
                    behavior[agent_id]["meals"] += 1.0
            for record in result.info.get("death_records", ()):
                agent_id = AgentId(str(record["agent_id"]))
                if agent_id in behavior:
                    behavior[agent_id]["deaths"] += 1.0
                    behavior[agent_id]["completed_lifetime_sum"] += float(record["age"])
                    if not behavior[agent_id]["first_death_observed"]:
                        behavior[agent_id]["first_death_age"] = float(record["age"])
                        behavior[agent_id]["first_death_observed"] = 1.0
            # A reused ecology slot is a new organism.  Its controller must
            # start with fresh graph/state randomness, while the blueprint
            # keeps the inherited update rules.  Environments that do not
            # replace slots simply omit ``births`` from their StepResult info.
            for agent_id in deaths - births:
                controller = controllers.pop(agent_id, None)
                if controller is not None:
                    controller.end_episode()
                    retired.add(agent_id)
            for agent_id in births:
                prior = controllers.get(agent_id)
                if prior is not None:
                    prior.end_episode()
                blueprint = blueprints.get(agent_id)
                if blueprint is not None:
                    controller = blueprint.build(seed=random.randrange(2**32))
                    controller.begin_episode(seed=random.randrange(2**32))
                    controllers[agent_id] = controller
                    retired.discard(agent_id)
            for hook in self.hooks:
                hook.on_step(result)
            observations, terminated = result.observations, result.terminated
            if terminated:
                break
        for agent_id, controller in controllers.items():
            if agent_id not in retired:
                controller.end_episode()
        population_by_id = getattr(self.environment, "population")()
        population = population_by_id.values()
        counts: dict[str, int] = {}
        for agent in population:
            if agent.alive:
                species = str(agent.species)
                counts[species] = counts.get(species, 0) + 1
        summaries: dict[AgentId, Mapping[str, float]] = {}
        for agent_id, row in behavior.items():
            samples = max(1.0, row["samples"])
            action_changes = max(1.0, row["action_changes"])
            early_rate, late_rate = row["early_return"] / window, row["late_return"] / window
            early_turn, late_turn = row["early_turn"] / window, row["late_turn"] / window
            early_abs_turn, late_abs_turn = row["early_abs_turn"] / window, row["late_abs_turn"] / window
            early_speed, late_speed = row["early_speed"] / window, row["late_speed"] / window
            organism = population_by_id.get(agent_id)
            final_energy_fraction = (
                float(organism.energy) / max(float(organism.spawn_energy), 1e-12)
                if organism is not None else 0.0
            )
            summaries[agent_id] = {
                "meals": row["meals"],
                "meal_rate": row["meals"] / samples,
                "mean_hunger": row["hunger_sum"] / samples,
                "mean_abs_energy_change": row["energy_change_sum"] / samples,
                "mean_action_change": row["action_change_sum"] / action_changes,
                "early_return_rate": early_rate,
                "late_return_rate": late_rate,
                "adaptation_delta": late_rate - early_rate,
                "mean_turn": row["turn_sum"] / samples,
                "mean_abs_turn": row["abs_turn_sum"] / samples,
                "mean_speed": row["speed_sum"] / samples,
                "early_mean_turn": early_turn,
                "late_mean_turn": late_turn,
                "turn_drift": late_turn - early_turn,
                "early_mean_abs_turn": early_abs_turn,
                "late_mean_abs_turn": late_abs_turn,
                "abs_turn_drift": late_abs_turn - early_abs_turn,
                "early_mean_speed": early_speed,
                "late_mean_speed": late_speed,
                "speed_drift": late_speed - early_speed,
                "turn_saturation_rate": row["turn_saturated"] / samples,
                "plant_visible_rate": row["plant_visible"] / samples,
                "plant_steering_alignment": row["plant_aligned"] / max(1.0, row["plant_directional"]),
                "deaths_per_1000_steps": 1000.0 * row["deaths"] / samples,
                "mean_completed_lifetime": row["completed_lifetime_sum"] / max(1.0, row["deaths"]),
                # First-life restricted survival time is the direct evolution
                # objective. A life still active at the horizon is censored at
                # the observed episode length rather than incorrectly scored 0.
                "restricted_lifetime": (
                    row["first_death_age"] if row["first_death_observed"] else float(step)
                ),
                "survived_horizon": 0.0 if row["first_death_observed"] else 1.0,
                "_death_count": row["deaths"],
                "_completed_lifetime_sum": row["completed_lifetime_sum"],
                "_exposure_steps": samples,
                "_first_lifetime_sum": (
                    row["first_death_age"] if row["first_death_observed"] else float(step)
                ),
                "_first_lifetime_count": 1.0,
                "_horizon_survivors": 0.0 if row["first_death_observed"] else 1.0,
                "final_energy_fraction": final_energy_fraction,
            }
        return EpisodeResult(step, terminated, returns, counts, summaries)
