from __future__ import annotations

import unittest

from evolvable_state_network.baselines import FixedRNNRule
from evolvable_state_network.dashboard import dashboard_document
from evolvable_state_network.graph import generate_random_graph
from evolvable_state_network.inputs import ConstantInput
from evolvable_state_network.simulation import Simulation, SimulationConfig


class DashboardSerializationTests(unittest.TestCase):
    def test_replay_document_contains_complete_dynamic_state(self) -> None:
        graph = generate_random_graph(3, 1, seed=4)
        config = SimulationConfig(steps=2)
        trajectory = Simulation(graph, FixedRNNRule()).run(
            config, ConstantInput(0.0)
        ).trajectory
        document = dashboard_document(
            graph, {"fixed": (trajectory, {"ok": True})}, config
        )
        self.assertEqual(document["graph"]["nodes"], 3)
        self.assertEqual(document["runs"]["fixed"]["trajectory"]["steps"], [0, 1, 2])
        self.assertEqual(
            document["runs"]["fixed"]["trajectory"]["effective_edge_strengths"],
            trajectory.effective_edge_strengths,
        )
        self.assertEqual(document["simulation_config"]["steps"], 2)


if __name__ == "__main__":
    unittest.main()
