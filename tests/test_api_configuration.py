"""Tests for the boundary between HTTP payloads and domain configuration."""

import unittest

from pydantic import ValidationError

from evolvable_state_network.application.configuration import (
    build_async_training_config,
)
from evolvable_state_network.application.models import (
    AsyncTrainingPayload,
    EmbodiedDemoPayload,
    EmbodiedFoodWebTrainingPayload,
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

    def test_embodied_food_and_demo_ecology_controls_are_validated(self) -> None:
        training = EmbodiedFoodWebTrainingPayload(
            algorithm="genetic", max_food=120, food_growth_rate=9.5,
            max_speed=12.0, max_turn=4.0,
            network_dt=.05, max_delta=.24, edge_step_scale=.06,
        )
        demo = EmbodiedDemoPayload(run_id="run", network_hidden_nodes=48, network_mean_degree=7.5, prey_count=4, predator_count=3, initial_food=20, max_food=90, food_growth_rate=7.5)
        self.assertEqual(training.max_food, 120)
        self.assertEqual(training.food_growth_rate, 9.5)
        self.assertEqual(training.max_speed, 12.0)
        self.assertEqual(training.max_turn, 4.0)
        self.assertEqual(training.algorithm, "genetic")
        self.assertEqual(training.training_mode, "batch")
        self.assertEqual(training.batch_validation_trials, 8)
        self.assertEqual(training.batch_test_trials, 16)
        self.assertEqual(training.hidden_nodes, 31)
        self.assertEqual(training.population_size, 24)
        self.assertEqual(training.network_dt, .05)
        self.assertEqual(training.max_delta, .24)
        self.assertEqual(training.edge_step_scale, .06)
        self.assertEqual(training.body_inputs, ("hunger",))
        self.assertFalse(training.allow_input_output_connections)
        self.assertTrue(EmbodiedFoodWebTrainingPayload(allow_input_output_connections=True).allow_input_output_connections)
        self.assertTrue(training.enforce_survival_pressure)
        self.assertEqual(training.state_width, 2)
        self.assertEqual(
            EmbodiedFoodWebTrainingPayload(max_food=0, enforce_survival_pressure=False).max_food,
            0,
        )
        self.assertEqual(demo.max_food, 90)
        self.assertEqual(demo.network_hidden_nodes, 48)
        self.assertEqual(demo.network_mean_degree, 7.5)
        with self.assertRaises(ValidationError):
            EmbodiedDemoPayload(run_id="run", initial_food=91, max_food=90)
        with self.assertRaises(ValidationError):
            EmbodiedDemoPayload(run_id="run", network_hidden_nodes=0)
        with self.assertRaises(ValidationError):
            EmbodiedFoodWebTrainingPayload(algorithm="unknown")
        with self.assertRaises(ValidationError):
            EmbodiedFoodWebTrainingPayload(training_mode="unknown")
        with self.assertRaises(ValidationError):
            EmbodiedFoodWebTrainingPayload(state_width=1)
        with self.assertRaises(ValidationError):
            EmbodiedFoodWebTrainingPayload(body_inputs=())
        with self.assertRaises(ValidationError):
            EmbodiedFoodWebTrainingPayload(body_inputs=("hunger", "hunger"))
        with self.assertRaises(ValidationError):
            EmbodiedFoodWebTrainingPayload(max_delta=0)
        with self.assertRaises(ValidationError) as horizon_error:
            EmbodiedFoodWebTrainingPayload(
                training_mode="batch", initial_energy_scale=5,
                batch_episode_steps=128,
            )
        self.assertIn("set batch_episode_steps to at least 300", str(horizon_error.exception))
        with self.assertRaises(ValidationError):
            EmbodiedFoodWebTrainingPayload(prey_count=16, food_growth_rate=24.0)
        ablation = EmbodiedFoodWebTrainingPayload(
            training_mode="batch", initial_energy_scale=10,
            batch_episode_steps=64, enforce_survival_pressure=False,
            plant_cluster_count=0,
        )
        self.assertFalse(ablation.enforce_survival_pressure)
        self.assertEqual(ablation.plant_cluster_count, 0)


if __name__ == "__main__":
    unittest.main()
