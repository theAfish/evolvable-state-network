"""Tests for the boundary between HTTP payloads and domain configuration."""

import unittest

from pydantic import ValidationError

from evolvable_state_network.application.configuration import (
    build_async_training_config,
)
from evolvable_state_network.application.models import (
    AsyncTrainingPayload,
    ExperimentPayload,
)


class ApiConfigurationTests(unittest.TestCase):
    def test_training_default_has_no_tick_cap(self) -> None:
        self.assertIsNone(AsyncTrainingPayload().max_ticks)

    def test_training_payload_is_translated_without_transport_dependencies(self) -> None:
        payload = AsyncTrainingPayload(
            seed=19,
            candidate_budget=24,
            max_ticks=300,
            slots=3,
            replicas=2,
            optimizer_batch=6,
            state_width=3,
            initial_state_scale=.2,
            stage_1_lifetime=20,
            stage_2_lifetime=60,
            stage_1_nodes=7,
            stage_2_nodes=11,
            mean_degree=2.5,
            disturbance_interval=9,
            disturbance_strength=.3,
            fatal_threshold=7,
            node_growth_alert=3,
            one_direction_steps=10,
            probe_interval=6,
        )

        config = build_async_training_config(payload, seed=19)

        self.assertEqual(config.seed, 19)
        self.assertEqual(config.candidate_budget, 24)
        self.assertEqual(config.result_batch_size, 6)
        self.assertEqual(config.architecture.state_width, 3)
        self.assertEqual(config.edge_architecture.latent_width, 3)
        self.assertEqual(config.architecture.hidden_width, 8)
        self.assertEqual(config.edge_architecture.hidden_width, 12)
        self.assertEqual(config.initial_state_scale, .2)
        self.assertEqual(tuple(level.lifetime for level in config.levels), (20, 60))
        self.assertEqual(config.levels[1].disturbance_frequency, 9)
        self.assertEqual(config.pathology.fatal_threshold, 7)
        self.assertEqual(config.probes.interval, 6)

    def test_request_models_reject_cross_field_and_unknown_values(self) -> None:
        with self.assertRaises(ValidationError):
            AsyncTrainingPayload(stage_1_lifetime=40, stage_2_lifetime=20)
        with self.assertRaises(ValidationError):
            ExperimentPayload(nodes=4, mean_degree=5)
        with self.assertRaises(ValidationError):
            ExperimentPayload(unsupported=True)


if __name__ == "__main__":
    unittest.main()
