from __future__ import annotations

import unittest
from random import Random

from evolvable_state_network.environments import (
    Controller, ControllerBlueprint, FoodWebConfig, FoodWebEnvironment,
    EpisodeRunner, Species, make_reference_population,
)
from evolvable_state_network.evolution.candidate import RuleArchitecture
from evolvable_state_network.tasks.food_web import FoodWebTaskConfig, FoodWebTaskEvaluator


class FoodWebTaskTests(unittest.TestCase):
    def test_environment_runs_without_an_evolution_strategy(self) -> None:
        config = FoodWebConfig(initial_plants=4, max_plants=8, plant_regrowth=2.0)
        result = EpisodeRunner(FoodWebEnvironment(config, seed=9)).run(
            make_reference_population(prey_count=2, predator_count=1), max_steps=8, seed=9
        )
        self.assertEqual(result.steps, 8)
        self.assertEqual(set(result.returns), {"prey-0", "prey-1", "predator-0"})
        self.assertTrue(all(value == 0.0 for value in result.returns.values()))
        self.assertEqual(result.final_population, {"prey": 2, "predator": 1})

    def test_esn_food_web_task_is_deterministic(self) -> None:
        architecture = RuleArchitecture(state_width=3, hidden_width=2)
        evaluator = FoodWebTaskEvaluator(
            architecture,
            FoodWebTaskConfig(environment=FoodWebConfig(initial_plants=4, max_plants=8), max_steps=12, trials=2, prey_count=2, predator_count=1, seed=7, focal_species=Species.PREY),
        )
        genome = tuple(.01 * index for index in range(evaluator.codec.dimension))
        self.assertEqual(evaluator.evaluate(genome), evaluator.evaluate(genome))

    def test_environment_package_does_not_import_neural_evolution(self) -> None:
        import evolvable_state_network.environments.food_web as food_web
        self.assertNotIn("evolution", food_web.__dict__)

    def test_training_population_positions_are_seeded_but_not_grid_fixed(self) -> None:
        first = make_reference_population(prey_count=3, predator_count=2, seed=10)
        repeated = make_reference_population(prey_count=3, predator_count=2, seed=10)
        second = make_reference_population(prey_count=3, predator_count=2, seed=11)
        self.assertEqual([agent.position for agent in first], [agent.position for agent in repeated])
        self.assertNotEqual([agent.position for agent in first], [agent.position for agent in second])

    def test_population_seed_stream_is_independent_from_environment_layout(self) -> None:
        for seed in range(32):
            layout_random = Random(seed)
            cluster_centers = {
                (layout_random.uniform(0.0, 100.0), layout_random.uniform(0.0, 60.0))
                for _ in range(4)
            }
            prey = make_reference_population(
                prey_count=8, predator_count=0, width=100.0, height=60.0,
                seed=seed,
            )
            self.assertTrue(all((agent.position.x, agent.position.y) not in cluster_centers for agent in prey))

    def test_initial_energy_is_configurable_for_natural_lifetime_control(self) -> None:
        agents = make_reference_population(prey_count=1, predator_count=1, prey_initial_energy=18.0, predator_initial_energy=28.0)
        self.assertEqual(agents[0].energy, 18.0)
        self.assertEqual(agents[1].energy, 28.0)

    def test_body_observations_report_hunger_energy_change_and_eating_state(self) -> None:
        config = FoodWebConfig(initial_plants=0, max_plants=0, plant_regrowth=0.0)
        world = FoodWebEnvironment(config, seed=3)
        agent = make_reference_population(prey_count=1, predator_count=0, seed=3)[0]
        world.add(agent)
        initial = world.reset(seed=3)[agent.id]
        self.assertEqual(initial["hunger"], 0.0)
        self.assertFalse(initial["ate"])
        result = world.step({agent.id: {"kind": "turn_move", "turn": 0.0, "speed": 0.0}})
        body = result.observations[agent.id]
        self.assertGreater(body["hunger"], 0.0)
        self.assertLess(body["energy_change"], 0.0)
        self.assertFalse(body["ate"])
        self.assertGreater(body["time_since_meal"], 0.0)

    def test_respawn_increments_the_life_marker_used_by_live_trajectories(self) -> None:
        world = FoodWebEnvironment(FoodWebConfig(
            initial_plants=0, max_plants=0, plant_regrowth=0.0,
            prey_initial_energy=.1, respawn_on_death=True,
        ), seed=3)
        agent = make_reference_population(
            prey_count=1, predator_count=0, prey_initial_energy=.1, seed=3,
        )[0]
        world.add(agent)
        world.reset(seed=3)
        world.step({agent.id: {"kind": "turn_move", "turn": 0.0, "speed": 0.0}})
        snapshot = world.snapshot()["organisms"][0]
        self.assertEqual(snapshot["life"], 1)
        self.assertEqual(snapshot["age"], 0)

    def test_episode_reports_behavioral_adaptation_metrics(self) -> None:
        result = EpisodeRunner(FoodWebEnvironment(FoodWebConfig(initial_plants=0, max_plants=0), seed=4)).run(
            make_reference_population(prey_count=1, predator_count=0, seed=4), max_steps=8, seed=4
        )
        metrics = result.behavior[next(iter(result.behavior))]
        self.assertEqual(metrics["meals"], 0.0)
        self.assertIn("mean_hunger", metrics)
        self.assertIn("adaptation_delta", metrics)
        self.assertIn("mean_action_change", metrics)
        self.assertIn("abs_turn_drift", metrics)
        self.assertIn("speed_drift", metrics)
        self.assertIn("plant_visible_rate", metrics)
        self.assertIn("plant_steering_alignment", metrics)
        self.assertIn("deaths_per_1000_steps", metrics)
        self.assertEqual(metrics["restricted_lifetime"], 8.0)
        self.assertEqual(metrics["survived_horizon"], 1.0)
        self.assertIn("final_energy_fraction", metrics)

    def test_restricted_lifetime_scores_survivors_at_horizon_and_deaths_at_age(self) -> None:
        config = FoodWebConfig(
            initial_plants=0, max_plants=0, plant_regrowth=0.0,
            prey_initial_energy=.1,
        )
        result = EpisodeRunner(FoodWebEnvironment(config, seed=4)).run(
            make_reference_population(
                prey_count=1, predator_count=0, prey_initial_energy=.1, seed=4,
            ),
            max_steps=3, seed=4,
        )
        metrics = result.behavior[next(iter(result.behavior))]
        self.assertEqual(metrics["restricted_lifetime"], 1.0)
        self.assertEqual(metrics["survived_horizon"], 0.0)
        self.assertEqual(metrics["mean_completed_lifetime"], 1.0)
        self.assertTrue(all(value == 0.0 for value in result.returns.values()))

    def test_clustered_food_regrows_at_persistent_targetable_patches(self) -> None:
        world = FoodWebEnvironment(FoodWebConfig(
            initial_plants=8, max_plants=8, plant_regrowth=0.0,
            plant_cluster_count=2, plant_cluster_radius=0.0,
        ), seed=12)
        agent = make_reference_population(prey_count=1, predator_count=0, seed=12)[0]
        world.add(agent)
        world.reset(seed=12)
        snapshot = world.snapshot()
        centers = {(item["x"], item["y"]) for item in snapshot["plant_clusters"]}
        positions = {(item["x"], item["y"]) for item in snapshot["plants"]}
        self.assertEqual(len(centers), 2)
        self.assertTrue(positions <= centers)

        uniform = FoodWebEnvironment(FoodWebConfig(
            initial_plants=2, max_plants=2, plant_regrowth=0.0,
            plant_cluster_count=0,
        ), seed=12)
        uniform.add(make_reference_population(prey_count=1, predator_count=0, seed=12)[0])
        uniform.reset(seed=12)
        self.assertEqual(uniform.snapshot()["plant_clusters"], [])

    def test_ray_guided_targeting_beats_blind_sweeping_in_clustered_ecology(self) -> None:
        class PlantSeekingController(Controller):
            def __init__(self, seek: bool) -> None:
                self.seek = seek

            def act(self, observation, *, available_actions):
                rays = tuple(observation.get("vision", ()))
                hits = tuple(
                    (index, ray) for index, ray in enumerate(rays)
                    if ray.get("kind") == "plant" and ray.get("distance") is not None
                )
                if not self.seek or not hits:
                    return {"kind": "turn_move", "turn": 0.0, "speed": 1.0}
                pixel, _ = min(hits, key=lambda item: float(item[1]["distance"]))
                center = (len(rays) - 1) / 2
                turn = max(-1.0, min(1.0, (pixel - center) / max(1.0, center)))
                return {"kind": "turn_move", "turn": turn, "speed": 1.0}

        class Blueprint(ControllerBlueprint):
            def __init__(self, seek: bool) -> None:
                self.seek = seek

            def build(self, *, seed=None):
                return PlantSeekingController(self.seek)

        config = FoodWebConfig(
            initial_plants=24, max_plants=24, plant_regrowth=8.0,
            plant_cluster_count=4, plant_cluster_radius=5.0,
            prey_initial_energy=90.0,
        )

        def meals(seek: bool) -> float:
            total = 0.0
            for seed in range(20, 26):
                result = EpisodeRunner(FoodWebEnvironment(config, seed=seed)).run(
                    make_reference_population(
                        prey_count=1, predator_count=0, prey_initial_energy=90.0,
                        controller=Blueprint(seek), seed=seed,
                    ),
                    max_steps=128, seed=seed,
                )
                total += next(iter(result.behavior.values()))["meals"]
            return total / 6

        self.assertGreater(meals(True), 3.0 * meals(False))
