from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evolvable_state_network.dashboard import write_dashboard_bundle
from evolvable_state_network.graph import generate_random_graph
from evolvable_state_network.inputs import ConstantInput
from evolvable_state_network.simulation import Simulation, SimulationConfig
from evolvable_state_network.baselines import FixedRNNRule


class DashboardExportTests(unittest.TestCase):
    def test_bundle_contains_static_app_and_replay_data(self) -> None:
        graph = generate_random_graph(3, 1, seed=4)
        trajectory = Simulation(graph, FixedRNNRule()).run(SimulationConfig(steps=2), ConstantInput(0.0)).trajectory
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            data_path = write_dashboard_bundle(output, graph, {"fixed": (trajectory, {"ok": True})})
            document = json.loads(data_path.read_text(encoding="utf-8"))
            self.assertEqual(document["graph"]["nodes"], 3)
            self.assertEqual(document["runs"]["fixed"]["trajectory"]["steps"], [0, 1, 2])
            self.assertTrue((output / "dashboard" / "index.html").is_file())
