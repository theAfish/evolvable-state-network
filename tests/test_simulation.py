from __future__ import annotations

import unittest

from evolvable_state_network.baselines import FixedRNNRule
from evolvable_state_network.graph import generate_random_graph
from evolvable_state_network.inputs import ConstantInput
from evolvable_state_network.perturbations import ImpulseInjection, NodeLesion
from evolvable_state_network.simulation import Simulation, SimulationConfig


class SimulationTests(unittest.TestCase):
    def test_random_graph_is_seed_deterministic(self) -> None:
        self.assertEqual(generate_random_graph(12, 3, 17), generate_random_graph(12, 3, 17))
        self.assertNotEqual(generate_random_graph(12, 3, 17), generate_random_graph(12, 3, 18))

    def test_random_graph_uses_unit_base_edge_weights(self) -> None:
        graph = generate_random_graph(12, 3, 17)
        self.assertTrue(graph.edges)
        self.assertTrue(all(edge.weight == 1.0 for edge in graph.edges))

    def test_impulse_and_lesion_are_bounded_and_reproducible(self) -> None:
        graph = generate_random_graph(4, 2, 3)
        config = SimulationConfig(steps=5, batch_size=2, max_abs_state=0.5)
        disturbances = (ImpulseInjection(1, (0,), 10.0), NodeLesion(2, (0,)))
        first = Simulation(graph, FixedRNNRule()).run(config, ConstantInput(0.0), disturbances)
        second = Simulation(graph, FixedRNNRule()).run(config, ConstantInput(0.0), disturbances)
        self.assertEqual(first.trajectory.node_states, second.trajectory.node_states)
        self.assertEqual(first.trajectory.node_states[2][0][0], (0.5,))
        self.assertTrue(all(snapshot[0][0] == (0.0,) for snapshot in first.trajectory.node_states[3:]))
