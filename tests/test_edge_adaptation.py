from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from evolvable_state_network.analysis import edge_activity_summary, edge_perturbation_recovery_curve
from evolvable_state_network.baselines import FixedRNNRule
from evolvable_state_network.evolution.candidate import (
    EdgeArchitecture,
    MLPEdgeRule,
    RuleArchitecture,
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
from evolvable_state_network.perturbations import EdgeStateImpulse
from evolvable_state_network.simulation import Simulation, SimulationConfig, Trajectory
from evolvable_state_network.simulation.torch_backend import (
    TorchMLPSimulator,
    cuda_available,
    resolve_device,
)


class RelaxingEdgeRule:
    """A test-only local channel rule following the bounded-increment form."""

    state_width = 1

    def initial_state(self) -> tuple[float, ...]:
        return (0.0,)

    def update(self, state, source, target, message, source_external, target_external, edge_step_scale):
        from math import tanh
        return (state[0] + edge_step_scale * tanh(-state[0]),)

    def message(self, state, source):
        return tuple(self.communication_strength(state) * value for value in source)

    def communication_strength(self, state):
        from math import tanh
        return .5 * (1 + tanh(state[0]))


class EdgeAdaptationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = generate_random_graph(4, 2, 11)
        self.edge_architecture = EdgeArchitecture(node_state_width=1, latent_width=2, hidden_width=2)
        self.edge_rule = MLPEdgeRule(self.edge_architecture, tuple(.03 * (index + 1) for index in range(self.edge_architecture.parameter_count)))

    def test_edge_updates_are_deterministic_and_effective_strengths_are_smoothly_bounded(self) -> None:
        simulation = Simulation(self.graph, FixedRNNRule(), self.edge_rule)
        config = SimulationConfig(steps=8, batch_size=2, edge_step_scale=.08)
        first = simulation.run(config, ConstantInput(.15))
        second = simulation.run(config, ConstantInput(.15))
        self.assertEqual(first.trajectory.edge_states, second.trajectory.edge_states)
        values = [value for frame in first.trajectory.effective_edge_strengths for batch in frame for value in batch]
        self.assertTrue(values)
        self.assertTrue(all(0.0 < value < 1.0 for value in values))

    def test_edge_latent_coordinates_gate_message_coordinates_independently(self) -> None:
        architecture = EdgeArchitecture(node_state_width=2, latent_width=2, hidden_width=1)
        rule = MLPEdgeRule(architecture, (0.0,) * architecture.parameter_count)
        message = rule.message((0.0, 2.0), (1.0, 1.0))
        self.assertAlmostEqual(message[0], .5)
        self.assertGreater(message[1], .98)
        self.assertAlmostEqual(rule.communication_strength((0.0, 2.0)), sum(message) / 2)

    def test_joint_codec_exports_and_restores_independent_parameter_groups(self) -> None:
        node_architecture = RuleArchitecture(state_width=1, hidden_width=2)
        codec = GenomeCodec(node_architecture, self.edge_architecture, "joint")
        genome = tuple(index / 13 for index in range(codec.dimension))
        groups = codec.export_groups(genome)
        self.assertEqual(codec.restore_groups(groups), genome)
        node, edge = codec.decode_groups(genome)
        self.assertIsNotNone(node)
        self.assertIsNotNone(edge)
        self.assertEqual(node.parameters, tuple(groups["node"]))
        self.assertEqual(edge.parameters, tuple(groups["edge"]))

    def test_joint_batched_evaluation_is_permutation_consistent(self) -> None:
        node_architecture = RuleArchitecture(state_width=1, hidden_width=2)
        suite = ScenarioSuite(
            train=(ScenarioConfig("tiny", 1, 2, 3, nodes=4, mean_degree=2, steps=8, batch_size=2),),
            validation=(), test=(),
        )
        evaluator = CandidateEvaluator(node_architecture, suite, edge_architecture=self.edge_architecture, target="joint")
        first = tuple(.01 * index for index in range(evaluator.codec.dimension))
        second = tuple(-.02 * index for index in range(evaluator.codec.dimension))
        forward = evaluator.evaluate_batch((first, second))
        reverse = evaluator.evaluate_batch((second, first))
        self.assertEqual(forward, (reverse[1], reverse[0]))

    def test_edge_impulse_has_a_recorded_recovery_path(self) -> None:
        graph = generate_random_graph(3, 2, 4)
        result = Simulation(graph, FixedRNNRule(), RelaxingEdgeRule()).run(
            SimulationConfig(steps=30, edge_step_scale=.2), ConstantInput(.1),
            (EdgeStateImpulse(4, (0,), 3.0),),
        )
        after_impulse = result.trajectory.edge_states[5][0][0][0]
        final = result.trajectory.edge_states[-1][0][0][0]
        self.assertLess(abs(final), abs(after_impulse))
        self.assertTrue(edge_perturbation_recovery_curve(result.trajectory))

    def test_joint_checkpoint_restores_parameter_group_configuration(self) -> None:
        node_architecture = RuleArchitecture(state_width=1, hidden_width=2)
        suite = ScenarioSuite(
            train=(ScenarioConfig("train", 3, 4, 5, nodes=4, mean_degree=2, steps=8, batch_size=1),),
            validation=(ScenarioConfig("validation", 6, 7, 8, nodes=4, mean_degree=2, steps=8, batch_size=1),),
            test=(ScenarioConfig("test", 9, 10, 11, nodes=4, mean_degree=2, steps=8, batch_size=1),),
        )
        config = EvolutionConfig(generations=1, population_size=2, smoke_samples=4, seed=19, architecture=node_architecture, edge_architecture=self.edge_architecture, target="joint", scenarios=suite)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            EvolutionRunner(config).run(output)
            exported = json.loads((output / "best_genome.json").read_text(encoding="utf-8"))
            codec = GenomeCodec(node_architecture, self.edge_architecture, "joint")
            self.assertEqual(codec.restore_groups(exported["parameter_groups"]), tuple(exported["genome"]))
            checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["experiment_config"]["target"], "joint")

    def test_edge_collapse_degeneracy_is_detected(self) -> None:
        trajectory = Trajectory()
        for step in range(6):
            node = [[(0.0,), (0.0,)]]
            edge = [[(-8.0,), (-8.0,)]]
            trajectory.append(step, float(step), node, node, edge, [[.0, .0]])
        report = _failure_report(evaluate_metrics(trajectory), Simulation(graph := generate_random_graph(2, 1, 1), FixedRNNRule()).run(SimulationConfig(steps=1), ConstantInput()).diagnostics, 0.0, SimulationConfig(steps=6), trajectory)
        self.assertTrue(report.edge_collapse)
        self.assertTrue(report.communication_elimination_stability)
        self.assertEqual(edge_activity_summary(trajectory)["fraction_inactive"], 1.0)

    def test_persistent_one_sided_state_drift_is_a_failure(self) -> None:
        trajectory = Trajectory()
        for step in range(8):
            node = [[(.8,), (1.0,), (1.2,)]]
            trajectory.append(step, float(step), node, node)
        report = _failure_report(
            evaluate_metrics(trajectory),
            Simulation(generate_random_graph(3, 1, 8), FixedRNNRule()).run(SimulationConfig(steps=1), ConstantInput()).diagnostics,
            0.0, SimulationConfig(steps=8), trajectory,
        )
        self.assertTrue(report.persistent_state_bias)

    @unittest.skipUnless(cuda_available(), "CUDA Torch is unavailable")
    def test_cuda_backend_matches_mlp_reference_within_float32_tolerance(self) -> None:
        node_architecture = RuleArchitecture(state_width=2, hidden_width=2)
        edge_architecture = EdgeArchitecture(node_state_width=2, latent_width=2, hidden_width=2)
        from evolvable_state_network.evolution.candidate import MLPUpdateRule
        node_rule = MLPUpdateRule(node_architecture, (.01,) * node_architecture.parameter_count)
        edge_rule = MLPEdgeRule(edge_architecture, (.02,) * edge_architecture.parameter_count)
        graph = generate_random_graph(4, 2, 3)
        config = SimulationConfig(steps=5, batch_size=1)
        provider = ConstantInput(.1)
        initial_node = [[(0.0, 0.0) for _ in range(graph.n_nodes)]]
        initial_edge = [[(0.0, 0.0) for _ in graph.edges]]
        reference = Simulation(graph, node_rule, edge_rule).run(config, provider, initial_node_state=initial_node, initial_edge_state=initial_edge)
        accelerated = TorchMLPSimulator(graph, node_rule, edge_rule, resolve_device("cuda")).run(config, provider, (), initial_node, initial_edge)
        differences = [abs(left - right) for frame_left, frame_right in zip(reference.trajectory.node_states, accelerated.trajectory.node_states, strict=True) for batch_left, batch_right in zip(frame_left, frame_right, strict=True) for vector_left, vector_right in zip(batch_left, batch_right, strict=True) for left, right in zip(vector_left, vector_right, strict=True)]
        self.assertLess(max(differences), 1e-6)
