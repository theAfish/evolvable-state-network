from __future__ import annotations

import unittest

from evolvable_state_network.embodied import EmbodiedNetworkConfig
from evolvable_state_network.environments import FoodWebConfig
from evolvable_state_network.evolution.candidate import EdgeArchitecture, RuleArchitecture
from evolvable_state_network.evolution.genome import GenomeCodec
from evolvable_state_network.prey_diagnostics import PreyDiagnosticConfig, diagnose_prey_genome
from evolvable_state_network.tasks.embodied_food_web import EmbodiedFoodWebTaskConfig


class PreyDiagnosticTests(unittest.TestCase):
    def test_report_contains_rule_probes_ablations_and_separate_node_groups(self) -> None:
        architecture = RuleArchitecture(state_width=2, hidden_layers=(3,))
        edge_architecture = EdgeArchitecture(node_state_width=2, latent_width=2, hidden_layers=(3,))
        codec = GenomeCodec(architecture, edge_architecture, "joint")
        task = EmbodiedFoodWebTaskConfig(
            network=EmbodiedNetworkConfig(nodes=34, state_width=2),
            environment=FoodWebConfig(initial_plants=1, max_plants=2),
            prey_count=1, predator_count=0, max_steps=4, trials=1, seed=7,
        )
        report = diagnose_prey_genome(
            (0.0,) * codec.dimension, architecture, edge_architecture, task,
            config=PreyDiagnosticConfig(episodes=1, sweep_points=9),
        )
        self.assertEqual(set(report["node_groups"]), {"sensory_input", "action_output", "anonymous_hidden"})
        self.assertIn("zero_messages", report["ablations"])
        self.assertIn("zero_final_output_bias", report["ablations"])
        self.assertIn("episode_sample", report["drift_curves_channel_0"])
        self.assertIn("normal_state_zero_aggregate", report["self_state_vs_message"])
        self.assertIn("final_layer_channel_0_bias", report["mlp_bias"])

