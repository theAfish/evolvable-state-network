from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evolvable_state_network.candidate import RuleArchitecture
from evolvable_state_network.cmaes import CMAES, CMAESConfig
from evolvable_state_network.evaluation import CandidateEvaluator, ScenarioConfig, ScenarioSuite, _failure_report
from evolvable_state_network.evolution import EvolutionConfig, EvolutionRunner
from evolvable_state_network.genome import GenomeCodec
from evolvable_state_network.graph import generate_random_graph
from evolvable_state_network.inputs import ConstantInput
from evolvable_state_network.metrics import evaluate_metrics
from evolvable_state_network.simulation import Simulation, SimulationConfig


class NonfiniteRule:
    state_width = 1

    def initial_state(self) -> tuple[float, ...]:
        return (0.0,)

    def update(self, state: tuple[float, ...], aggregate: tuple[float, ...], external: tuple[float, ...], dt: float, max_delta: float) -> tuple[float, ...]:
        return (float("nan"),)


class EvolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.architecture = RuleArchitecture(state_width=2, hidden_width=3)
        self.codec = GenomeCodec(self.architecture)

    def test_genome_round_trip_is_exact_and_deterministic(self) -> None:
        genome = tuple(index / 10 for index in range(self.codec.dimension))
        self.assertEqual(self.codec.encode(self.codec.decode(genome)), genome)

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
