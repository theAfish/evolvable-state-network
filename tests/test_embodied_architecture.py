from __future__ import annotations

import unittest

from evolvable_state_network.embodied import EmbodiedNetwork, EmbodiedNetworkConfig, FoodWebAgentAdapter
from evolvable_state_network.evolution.candidate import EdgeArchitecture, MLPEdgeRule, MLPUpdateRule, RuleArchitecture
from evolvable_state_network.evolution.cmaes import CMAES, CMAESConfig
from evolvable_state_network.tasks.embodied_food_web import (
    BatchFoodWebCoevolutionRunner,
    BatchFoodWebConfig,
    EmbodiedFoodWebEvaluator,
    EmbodiedFoodWebTaskConfig,
    EmbodiedRuleEvolutionConfig,
    EmbodiedRuleEvolutionRunner,
    FoodWebCoevolutionEvaluator,
    FoodWebCoevolutionRunner,
    ContinuousFoodWebConfig,
    ContinuousFoodWebCoevolutionRunner, EmbodiedFoodWebController, OnlineRuleLibrary,
)
from evolvable_state_network.environments import FoodWebConfig, Species
from evolvable_state_network.environments import AgentId, Controller, ControllerBlueprint, EpisodeRunner, FoodWebEnvironment, Organism
from evolvable_state_network.environments.sensing import Vec2


class EmbodiedArchitectureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.architecture = RuleArchitecture(state_width=2, hidden_width=2)
        self.edge_architecture = EdgeArchitecture(node_state_width=2, latent_width=2, hidden_width=2)
        self.node = MLPUpdateRule(self.architecture, (0.0,) * self.architecture.parameter_count)
        self.edge = MLPEdgeRule(self.edge_architecture, (0.0,) * self.edge_architecture.parameter_count)
        self.config = EmbodiedNetworkConfig(nodes=33, mean_degree=0.0, state_width=2)

    def test_adapter_preserves_ordered_ray_pixels_without_angles(self) -> None:
        values = FoodWebAgentAdapter(vision_pixels=3).encode_observation({
            "hunger": .75, "energy_change": -.1, "ate": True, "time_since_meal": .25,
            "vision": (
                {"kind": "plant", "distance": 2.0, "range": 10.0, "angle": 123.0},
                {"kind": "prey", "distance": 5.0, "range": 10.0, "angle": -456.0},
                {"kind": "predator", "distance": 8.0, "range": 10.0, "angle": 789.0},
            ),
        })
        self.assertEqual(len(values), 13)
        self.assertEqual(values[:4], (.5, -.1, 1.0, -.5))
        expected = (.8, 0.0, 0.0, 0.0, .5, 0.0, 0.0, 0.0, .2)
        for actual, wanted in zip(values[4:], expected, strict=True):
            self.assertAlmostEqual(actual, wanted)

    def test_legacy_adapter_keeps_old_compact_width(self) -> None:
        self.assertEqual(FoodWebAgentAdapter(schema="body_v2", directional=True).input_count, 13)
        self.assertEqual(FoodWebAgentAdapter(schema="body_v2", directional=False).input_count, 7)

    def test_online_library_tells_cma_the_population_it_asked_for(self) -> None:
        config = EmbodiedRuleEvolutionConfig(population_size=2, initial_sigma=.25, seed=13)
        expected = CMAES(CMAESConfig(3, config.population_size, config.initial_sigma, config.seed)).ask()

        class Codec:
            dimension = 3

        library = OnlineRuleLibrary(Codec(), config, self.config, seed=13)
        self.assertEqual(tuple(library.cohort), expected)

    def test_online_library_archives_replicated_means_not_lucky_lives(self) -> None:
        task = EmbodiedFoodWebTaskConfig(network=self.config, max_steps=1, trials=1)
        evaluator = FoodWebCoevolutionEvaluator(self.architecture, self.edge_architecture, task)
        library = OnlineRuleLibrary(
            evaluator.codec,
            EmbodiedRuleEvolutionConfig(population_size=2, initial_sigma=.25, seed=13),
            self.config,
            seed=13,
        )
        first, second, first_again, second_again = (library.birth() for _ in range(4))
        self.assertEqual(first.genome, first_again.genome)
        self.assertEqual(second.genome, second_again.genome)
        library.observe(first, 100.0)
        library.observe(second, 40.0)
        self.assertEqual(library.updates, 0)
        library.observe(first_again, 0.0)
        library.observe(second_again, 40.0)
        self.assertEqual(library.updates, 1)
        self.assertEqual(library.snapshot()["best_fitness"], 50.0)

    def test_adapter_nodes_are_connected_and_actions_are_bounded(self) -> None:
        network = EmbodiedNetwork(self.node, self.edge, FoodWebAgentAdapter(), self.config, seed=5)
        pairs = {(edge.source, edge.target) for edge in network.graph.edges}
        self.assertTrue(all(any(source == node for source, _ in pairs) for node in network.interface.input_nodes))
        self.assertTrue(all(any(target == node for _, target in pairs) for node in network.interface.action_nodes))
        action = network.act({"energy": 9.0, "heading": 0.0, "age": 0, "vision": ()})
        self.assertGreaterEqual(float(action["turn"]), -1.0)
        self.assertLessEqual(float(action["turn"]), 1.0)
        self.assertGreaterEqual(float(action["speed"]), 0.0)
        self.assertLessEqual(float(action["speed"]), 1.0)
        self.assertEqual(
            network.state.node[0][network.interface.input_nodes[0]][-1],
            network.interface.input_tags[0],
        )
        self.assertEqual(
            network.state.node[0][network.interface.action_nodes[0]][-1],
            network.interface.action_tags[0],
        )
        self.assertEqual(EmbodiedFoodWebController.learn, Controller.learn)

    def test_each_network_has_independent_random_node_state(self) -> None:
        first = EmbodiedNetwork(self.node, self.edge, FoodWebAgentAdapter(), self.config, seed=3)
        second = EmbodiedNetwork(self.node, self.edge, FoodWebAgentAdapter(), self.config, seed=4)
        self.assertNotEqual(first.graph, second.graph)
        self.assertNotEqual(first.state.node, second.state.node)

    def test_actuators_start_neutral_while_other_state_remains_random(self) -> None:
        network = EmbodiedNetwork(self.node, self.edge, FoodWebAgentAdapter(), self.config, seed=3)
        for node in network.interface.action_nodes:
            self.assertEqual(network.state.node[0][node], (0.0, 0.0))
        action = network.act({"energy": 9.0, "vision": ()})
        self.assertEqual(action, {"kind": "turn_move", "turn": 0.0, "speed": 0.5})

    def test_competing_genomes_receive_matched_random_networks(self) -> None:
        task = EmbodiedFoodWebTaskConfig(network=self.config, max_steps=1, trials=1)
        evaluator = FoodWebCoevolutionEvaluator(self.architecture, self.edge_architecture, task)
        zero = (0.0,) * evaluator.codec.dimension
        changed = (0.1,) * evaluator.codec.dimension
        first = evaluator._blueprint(zero, 123, Species.PREY).build(seed=17)
        second = evaluator._blueprint(changed, 123, Species.PREY).build(seed=17)
        masked = evaluator._blueprint(zero, 123, Species.PREY, "vision").build(seed=17)
        first.begin_episode(seed=19)
        second.begin_episode(seed=19)
        masked.begin_episode(seed=19)
        self.assertEqual(first._network.graph, second._network.graph)
        self.assertEqual(first._network.state.node, second._network.state.node)
        self.assertEqual(first._network.graph, masked._network.graph)
        self.assertEqual(first._network.state.node, masked._network.state.node)
        different_scenario = evaluator._blueprint(zero, 124, Species.PREY).build(seed=17)
        different_scenario.begin_episode(seed=19)
        self.assertNotEqual(first._network.graph, different_scenario._network.graph)

    def test_joint_rules_are_the_only_evolutionary_genome(self) -> None:
        task = EmbodiedFoodWebTaskConfig(
            network=self.config, environment=FoodWebConfig(initial_plants=3, max_plants=5),
            focal_species=Species.PREY, prey_count=2, predator_count=1, max_steps=8, trials=1, seed=8,
        )
        evaluator = EmbodiedFoodWebEvaluator(self.architecture, self.edge_architecture, task)
        genome = (0.0,) * evaluator.codec.dimension
        self.assertEqual(evaluator.evaluate(genome), evaluator.evaluate(genome))
        report = EmbodiedRuleEvolutionRunner(evaluator, EmbodiedRuleEvolutionConfig(generations=1, population_size=2, seed=2, initial_genome=genome)).run()
        self.assertEqual(len(report["best_genome"]), evaluator.codec.dimension)
        self.assertEqual(report["focal_species"], "prey")

    def test_prey_and_predator_coevolve_in_matched_episodes(self) -> None:
        task = EmbodiedFoodWebTaskConfig(
            network=self.config, environment=FoodWebConfig(initial_plants=3, max_plants=5),
            prey_count=2, predator_count=1, max_steps=8, trials=1, seed=8,
        )
        evaluator = FoodWebCoevolutionEvaluator(self.architecture, self.edge_architecture, task)
        genome = (0.0,) * evaluator.codec.dimension
        result = evaluator.evaluate(genome, genome)
        self.assertEqual(len(result.prey_trial_returns), 1)
        self.assertEqual(len(result.predator_trial_returns), 1)
        report = FoodWebCoevolutionRunner(evaluator, EmbodiedRuleEvolutionConfig(generations=1, population_size=2, seed=2, initial_genome=genome)).run()
        self.assertEqual(len(report["prey_best_genome"]), evaluator.codec.dimension)
        self.assertEqual(len(report["predator_best_genome"]), evaluator.codec.dimension)

    def test_batch_coevolution_uses_complete_common_seed_generations(self) -> None:
        task = EmbodiedFoodWebTaskConfig(
            network=self.config, environment=FoodWebConfig(initial_plants=3, max_plants=5),
            prey_count=2, predator_count=1, max_steps=8, trials=1, seed=8,
        )
        evaluator = FoodWebCoevolutionEvaluator(self.architecture, self.edge_architecture, task)
        genome = (0.0,) * evaluator.codec.dimension
        report = BatchFoodWebCoevolutionRunner(
            evaluator,
            EmbodiedRuleEvolutionConfig(generations=2, population_size=2, seed=8, initial_genome=genome, algorithm="genetic"),
            BatchFoodWebConfig(generations=2, episode_steps=8, trials=2, validation_trials=2, test_trials=2, opponent_pool_size=2, seed=8, initial_genome=genome),
        ).run()
        self.assertEqual(report["training_mode"], "batch")
        self.assertEqual(report["prey"]["updates"], 2)
        self.assertEqual(report["predator"]["updates"], 2)
        self.assertGreaterEqual(report["prey"]["evaluations"], 8)
        self.assertEqual(len(report["history"][0]["episode_seeds"]), 2)
        self.assertEqual(report["history"][0]["validation_seeds"], report["history"][1]["validation_seeds"])
        self.assertIn("prey_validation_fitness", report["history"][0])
        self.assertIn("meal_rate", report["prey"]["behavior"])
        self.assertGreater(report["prey"]["validation_evaluations"], 0)
        training_seeds = {seed for row in report["history"] for seed in row["episode_seeds"]}
        validation_seeds = set(report["history"][0]["validation_seeds"])
        test_seeds = set(report["test_seeds"])
        self.assertTrue(training_seeds.isdisjoint(validation_seeds))
        self.assertTrue(training_seeds.isdisjoint(test_seeds))
        self.assertTrue(validation_seeds.isdisjoint(test_seeds))
        self.assertTrue(all("test_seeds" not in row for row in report["history"]))
        self.assertEqual(report["prey"]["test_evaluations"], 6)
        self.assertIn("test_fitness", report["prey"])
        self.assertIn("zero_rule_fitness", report["prey"]["baselines"])
        self.assertIn("vision_masked_fitness", report["prey"]["baselines"])
        self.assertIn("vision_ablation_delta", report["prey"]["baselines"])
        self.assertIn("abs_turn_drift", report["prey"]["test_behavior"])
        self.assertIn("plant_steering_alignment", report["prey"]["test_behavior"])

    def test_replaced_agent_rebuilds_its_episode_local_controller(self) -> None:
        builds: list[int] = []

        class CountingController(Controller):
            def act(self, observation, *, available_actions):
                return {"kind": "turn_move", "turn": 0.0, "speed": 0.0}

        class CountingBlueprint(ControllerBlueprint):
            def build(self, *, seed=None):
                builds.append(seed or 0)
                return CountingController()

        world = FoodWebEnvironment(FoodWebConfig(initial_plants=0, plant_regrowth=0.0), seed=1)
        prey = Organism(AgentId("prey"), Species.PREY, Vec2(4.0, 4.0), energy=.1, controller=CountingBlueprint())
        EpisodeRunner(world).run([prey], max_steps=2, seed=1)
        self.assertGreaterEqual(len(builds), 3)  # initial controller plus two replacement births

    def test_continuous_coevolution_replaces_deaths_from_online_libraries(self) -> None:
        task = EmbodiedFoodWebTaskConfig(
            network=self.config, environment=FoodWebConfig(initial_plants=0, plant_regrowth=0.0),
            prey_count=2, predator_count=1, max_steps=1, trials=1, seed=3,
        )
        evaluator = FoodWebCoevolutionEvaluator(self.architecture, self.edge_architecture, task)
        genome = (0.0,) * evaluator.codec.dimension
        report = ContinuousFoodWebCoevolutionRunner(
            evaluator, EmbodiedRuleEvolutionConfig(generations=1, population_size=2, seed=3, initial_genome=genome, algorithm="genetic"),
            ContinuousFoodWebConfig(ticks=50, seed=3, initial_genome=genome),
        ).run()
        self.assertGreater(report["prey"]["deaths"], 0)
        self.assertGreater(report["predator"]["deaths"], 0)
        self.assertGreaterEqual(report["prey"]["updates"], 1)
        self.assertEqual(report["algorithm"], "genetic")
        self.assertEqual(report["telemetry"][-1]["tick"], 50)
        self.assertIn("prey_mean_lifetime", report["telemetry"][-1])
        self.assertIn("prey_meals", report["telemetry"][-1])
        self.assertIn("prey_mean_hunger", report["telemetry"][-1])
        self.assertIn("prey_meal_rate_coverage", report["telemetry"][-1])
        self.assertEqual(report["prey"]["evaluation_replicates"], 2)
        self.assertIn("prey_energy_supply_ratio", report["ecology"])

    def test_continuous_coevolution_can_seed_each_species_independently(self) -> None:
        task = EmbodiedFoodWebTaskConfig(
            network=self.config, environment=FoodWebConfig(initial_plants=0, plant_regrowth=0.0),
            prey_count=2, predator_count=1, max_steps=1, trials=1, seed=3,
        )
        evaluator = FoodWebCoevolutionEvaluator(self.architecture, self.edge_architecture, task)
        prey = (0.0,) * evaluator.codec.dimension
        predator = (0.1,) * evaluator.codec.dimension
        report = ContinuousFoodWebCoevolutionRunner(
            evaluator, EmbodiedRuleEvolutionConfig(generations=1, population_size=2, seed=3),
            ContinuousFoodWebConfig(ticks=1, seed=3, initial_prey_genome=prey, initial_predator_genome=predator),
        ).run()
        self.assertEqual(report["prey_best_genome"], list(prey))
        self.assertEqual(report["predator_best_genome"], list(predator))

    def test_continuous_coevolution_supports_prey_only_worlds(self) -> None:
        task = EmbodiedFoodWebTaskConfig(
            network=self.config, environment=FoodWebConfig(initial_plants=0, plant_regrowth=0.0),
            prey_count=2, predator_count=0, max_steps=1, trials=1, seed=3,
        )
        evaluator = FoodWebCoevolutionEvaluator(self.architecture, self.edge_architecture, task)
        report = ContinuousFoodWebCoevolutionRunner(
            evaluator, EmbodiedRuleEvolutionConfig(generations=1, population_size=2, seed=3),
            ContinuousFoodWebConfig(ticks=50, seed=3),
        ).run()
        self.assertEqual(report["population"]["predator"], 0)
        self.assertEqual(report["predator"]["deaths"], 0)
        self.assertGreaterEqual(report["prey"]["updates"], 1)
