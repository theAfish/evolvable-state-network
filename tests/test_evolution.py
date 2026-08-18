from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evolvable_state_network.evolution.candidate import EdgeArchitecture, MLPEdgeRule, MLPUpdateRule, RuleArchitecture
from evolvable_state_network.evolution.cmaes import CMAES, CMAESConfig
from evolvable_state_network.evolution.genetic import (
    GeneticAlgorithm, GeneticAlgorithmConfig, RuleDynamicsViabilityProbe,
    ViabilityResult, population_statistics,
)
from evolvable_state_network.evolution.evaluation import (
    CandidateEvaluator,
    ScenarioConfig,
    ScenarioSuite,
    _failure_report,
)
from evolvable_state_network.evolution import EvolutionConfig, EvolutionRunner
from evolvable_state_network.evolution.genome import GenomeCodec
from evolvable_state_network.graph import generate_random_graph
from evolvable_state_network.inputs import ConstantInput
from evolvable_state_network.metrics import evaluate_metrics
from evolvable_state_network.simulation import Simulation, SimulationConfig


class NonfiniteRule:
    state_width = 1

    def initial_state(self) -> tuple[float, ...]:
        return (0.0,)

    def update(self, state: tuple[float, ...], aggregate: tuple[float, ...], dt: float, max_delta: float) -> tuple[float, ...]:
        return (float("nan"),)


class EvolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.architecture = RuleArchitecture(state_width=2, hidden_width=3)
        self.codec = GenomeCodec(self.architecture)

    def test_genome_round_trip_is_exact_and_deterministic(self) -> None:
        genome = tuple(index / 10 for index in range(self.codec.dimension))
        self.assertEqual(self.codec.encode(self.codec.decode(genome)), genome)

    def test_mlp_input_excludes_redundant_constant_feature(self) -> None:
        node_architecture = RuleArchitecture(state_width=2, hidden_width=1, activation="relu")
        node_rule = MLPUpdateRule(node_architecture, tuple(float(value) for value in range(1, 10)))
        self.assertEqual(node_architecture.input_width, 4)
        self.assertEqual(node_architecture.parameter_count, 9)
        self.assertEqual(node_rule.raw_output((1.0, 2.0), (3.0, 4.0)), (218.0, 254.0))

        edge_architecture = EdgeArchitecture(node_state_width=1, latent_width=1, hidden_width=1, activation="relu")
        edge_rule = MLPEdgeRule(edge_architecture, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0))
        self.assertEqual(edge_architecture.input_width, 4)
        self.assertEqual(edge_architecture.parameter_count, 7)
        self.assertEqual(edge_rule.raw_output((1.0,), (2.0,), (3.0,), (4.0,)), (217.0,))

    def test_candidate_evaluation_is_deterministic_and_batch_consistent(self) -> None:
        evaluator = CandidateEvaluator(self.architecture)
        first = tuple(.03 * index for index in range(self.codec.dimension))
        second = tuple(-.02 * index for index in range(self.codec.dimension))
        self.assertEqual(evaluator.evaluate(first), evaluator.evaluate(first))
        self.assertEqual(evaluator.evaluate_batch((first, second)), (evaluator.evaluate(first), evaluator.evaluate(second)))

    def test_nonfinite_rule_is_hard_failure_even_when_simulator_contains_it(self) -> None:
        result = Simulation(generate_random_graph(3, 1, 2), NonfiniteRule()).run(SimulationConfig(steps=4), ConstantInput(0.0))
        self.assertGreater(result.diagnostics.nonfinite_proposals, 0)
        metrics = evaluate_metrics(result.trajectory)
        self.assertTrue(metrics.boundedness.finite)
        self.assertTrue(_failure_report(metrics, result.diagnostics, 0.0, SimulationConfig(steps=4)).nonfinite)

    def test_cmaes_checkpoint_restores_next_population(self) -> None:
        optimizer = CMAES(CMAESConfig(5, population_size=4, seed=13))
        population = optimizer.ask()
        optimizer.tell(population, (0.4, 0.1, 0.7, 0.3))
        restored = CMAES.from_state_dict(json.loads(json.dumps(optimizer.state_dict())))
        self.assertEqual(optimizer.generation, restored.generation)
        self.assertEqual(optimizer.sigma, restored.sigma)
        self.assertEqual(len(optimizer.ask()), len(restored.ask()))

    def test_genetic_algorithm_is_deterministic_and_preserves_an_elite(self) -> None:
        config = GeneticAlgorithmConfig(3, population_size=4, mutation_sigma=.2, seed=13)
        first, repeated = GeneticAlgorithm(config), GeneticAlgorithm(config)
        population = first.ask()
        self.assertEqual(population, repeated.ask())
        fitnesses = (0.0, 3.0, 1.0, 2.0)
        first.tell(population, fitnesses)
        repeated.tell(population, fitnesses)
        next_population = first.ask()
        self.assertEqual(next_population, repeated.ask())
        self.assertIn(population[1], next_population)
        self.assertEqual(first.generation, 1)

    def test_genetic_controls_bound_mutation_and_support_population_immigrants(self) -> None:
        optimizer = GeneticAlgorithm(GeneticAlgorithmConfig(
            3, population_size=4, mutation_sigma=2.0, seed=4, immigrant_fraction=.5,
            immigrant_mode="population", max_genome_norm=1.0, max_parameter_magnitude=.7,
        ))
        population = optimizer.ask()
        self.assertTrue(all(sum(value * value for value in genome) <= 1.000001 for genome in population))
        optimizer.tell(population, (0.0, .1, .2, .3))
        next_population = optimizer.ask()
        self.assertGreater(optimizer.normalization_count, 0)
        self.assertGreaterEqual(population_statistics(next_population)["population_parameter_diversity"], 0.0)

    def test_multiscale_ga_has_exact_composition_and_global_prior_is_population_independent(self) -> None:
        config = GeneticAlgorithmConfig(
            3, population_size=10, seed=22, elite_fraction=.2,
            regional_fraction=.2, global_fraction=.2, global_parameter_range=.5,
        )
        left, right = GeneticAlgorithm(config, (-10.0,) * 3), GeneticAlgorithm(config, (10.0,) * 3)
        left_population, right_population = left.ask(), right.ask()
        self.assertEqual(len(left_population), config.population_size)
        self.assertEqual(left.pending_sources.count("elite"), 2)
        self.assertEqual(left.pending_sources.count("local_offspring"), 4)
        self.assertEqual(left.pending_sources.count("regional_immigrant"), 2)
        self.assertEqual(left.pending_sources.count("global_immigrant"), 2)
        global_indices = [index for index, source in enumerate(left.pending_sources) if source == "global_immigrant"]
        regional_indices = [index for index, source in enumerate(left.pending_sources) if source == "regional_immigrant"]
        self.assertEqual([left_population[index] for index in global_indices], [right_population[index] for index in global_indices])
        self.assertNotEqual([left_population[index] for index in regional_indices], [right_population[index] for index in regional_indices])

    def test_rule_dynamics_probe_rejects_saturated_rules_and_accepts_active_rules(self) -> None:
        probe = RuleDynamicsViabilityProbe(self.codec)
        saturated = [0.0] * self.codec.dimension
        saturated[-self.architecture.state_width:] = [100.0] * self.architecture.state_width
        active = [0.0] * self.codec.dimension
        active[-self.architecture.state_width:] = [.2] * self.architecture.state_width
        self.assertFalse(probe(saturated).viable)
        self.assertTrue(probe(active).viable)

    def test_global_filter_resamples_then_records_a_fallback_without_fitness(self) -> None:
        calls = []
        def reject(genome: tuple[float, ...]) -> ViabilityResult:
            calls.append(genome)
            return ViabilityResult(False, {"node_raw_saturation_fraction": .9}, ("node_raw_saturation",))
        optimizer = GeneticAlgorithm(
            GeneticAlgorithmConfig(2, population_size=5, seed=4, elite_fraction=.2, global_fraction=.4,
                                   global_viability_filter=True, global_max_sampling_attempts=3),
            global_viability_probe=reject,
        )
        optimizer.ask()
        self.assertEqual(len(calls), 6)
        viability = optimizer.last_sampling_telemetry["global_viability"]
        self.assertEqual(viability["attempts"], 6)
        self.assertEqual(viability["fallbacks"], 2)

    def test_evaluation_reports_raw_rule_outputs_and_applied_updates(self) -> None:
        suite = ScenarioSuite(
            train=(ScenarioConfig("train", 1, 2, 3, nodes=4, mean_degree=2, steps=5, batch_size=1),),
            validation=(), test=(),
        )
        result = CandidateEvaluator(self.architecture, suite, rule_output_scale=.5).evaluate((.1,) * self.codec.dimension)
        dynamics = result.scenario_results[0].diagnostics.dynamics_summary()
        self.assertGreater(dynamics["node_rule_output"]["count"], 0)
        self.assertGreater(dynamics["node_update"]["count"], 0)
        self.assertIn("abs_gt_3_fraction", dynamics["node_rule_output"])
        self.assertIn("near_limit_fraction", dynamics["node_update"])

    def test_exported_best_genome_reproduces_saved_test_evaluation(self) -> None:
        suite = ScenarioSuite(
            train=(ScenarioConfig("train", 1, 2, 3, nodes=5, mean_degree=2, steps=16, batch_size=1),),
            validation=(ScenarioConfig("validation", 11, 12, 13, nodes=7, mean_degree=2, steps=24, batch_size=1, perturbation_strength=.5),),
            test=(ScenarioConfig("test", 21, 22, 23, nodes=9, mean_degree=2, steps=36, batch_size=1, perturbation_strength=.7),),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            runner = EvolutionRunner(EvolutionConfig(generations=1, population_size=2, smoke_samples=4, seed=12, scenarios=suite, architecture=self.architecture))
            runner.run(output)
            exported = json.loads((output / "best_genome.json").read_text(encoding="utf-8"))
            repeated = runner.evaluator.evaluate(exported["genome"], "test")
            self.assertEqual(repeated.to_dict(), exported["test"])
            self.assertTrue((output / "random_search_smoke.json").is_file())
            resumed = EvolutionRunner(EvolutionConfig(generations=2, population_size=2, smoke_samples=4, seed=12, scenarios=suite, architecture=self.architecture)).run(output, resume=True)
            self.assertEqual(len(resumed["history"]), 2)

    def test_completion_reports_export_stages_and_writes_generation_replays(self) -> None:
        suite = ScenarioSuite(
            train=(ScenarioConfig("train", 1, 2, 3, nodes=4, mean_degree=2, steps=8, batch_size=1),),
            validation=(ScenarioConfig("validation", 11, 12, 13, nodes=5, mean_degree=2, steps=10, batch_size=1),),
            test=(ScenarioConfig("test", 21, 22, 23, nodes=6, mean_degree=2, steps=12, batch_size=1),),
        )
        phases: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            EvolutionRunner(EvolutionConfig(generations=1, population_size=2, smoke_samples=4, seed=9, scenarios=suite, architecture=self.architecture)).run(
                output, progress=lambda event: phases.append(str(event["phase"])),
            )
            replays = json.loads((output / "replays" / "index.json").read_text(encoding="utf-8"))["replays"]
        self.assertEqual(phases[-5:], ["writing_analysis", "analysis_summaries", "analysis_charts", "writing_replays", "finalizing"])
        self.assertEqual(sum(entry["split"] == "generation" for entry in replays), 1)
