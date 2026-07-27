from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from evolvable_state_network.async_evolution import (
    AsyncEvolutionConfig,
    AsyncEvolutionRunner,
    CandidateProposal,
    CandidateSlot,
    CurriculumLevel,
    EliteArchive,
    HealthMonitor,
    PathologyConfig,
    ProbeSummary,
    ScenarioBank,
    SteadyStateCMA,
    diagnostic_reference_genomes,
)
from evolvable_state_network.candidate import EdgeArchitecture, RuleArchitecture
from evolvable_state_network.genome import GenomeCodec
from evolvable_state_network.simulation import NetworkState, TransitionDiagnostics


def state(value: float, nodes: int = 3) -> NetworkState:
    return NetworkState(node=[[(value,) for _ in range(nodes)]], edge=[[(0.0,), (0.0,)]])


def state_with_edges(node_value: float, edge_value: float, nodes: int = 3) -> NetworkState:
    return NetworkState(
        node=[[(node_value,) for _ in range(nodes)]], edge=[[(edge_value,), (edge_value,)]]
    )


def vector_state(values: tuple[float, ...], nodes: int = 3) -> NetworkState:
    return NetworkState(node=[[values for _ in range(nodes)]], edge=[[(0.0,), (0.0,)]])


class AsyncEvolutionTests(unittest.TestCase):
    def test_immediate_numerical_death(self) -> None:
        monitor = HealthMonitor(PathologyConfig())
        cause = monitor.observe(
            1, state(0), state(0), (.5,), TransitionDiagnostics(nonfinite_proposals=1)
        )
        self.assertEqual(cause, "nonfinite")
        self.assertEqual(monitor.death_time, 1)

    def test_accumulated_pathology_death_and_recovery_before_death(self) -> None:
        config = PathologyConfig(
            fatal_threshold=3, increase=1, recovery=1, homogenization_variance=-1
        )
        dying = HealthMonitor(config)
        cause = None
        for step in range(1, 4):
            cause = dying.observe(step, state(0), state(0), (0.0,), TransitionDiagnostics())
        self.assertEqual(cause, "communication_collapse")
        recovering = HealthMonitor(config)
        recovering.observe(1, state(0), state(0), (0.0,), TransitionDiagnostics())
        recovering.observe(2, state(0), state(.1), (0.5,), TransitionDiagnostics())
        self.assertIsNone(recovering.death_cause)
        self.assertEqual(recovering.burdens["communication_collapse"], 0)

    def test_unresponsive_fixed_point_dies_but_responsive_stable_point_survives(self) -> None:
        config = PathologyConfig(fatal_threshold=2, homogenization_variance=-1)
        frozen = HealthMonitor(config)
        bad_probe = ProbeSummary(1, 0, 0, 0, True)
        frozen.observe(1, state(0), state(0), (.5,), TransitionDiagnostics(), bad_probe)
        cause = frozen.observe(2, state(0), state(0), (.5,), TransitionDiagnostics(), bad_probe)
        self.assertIn(cause, {"input_unresponsive", "communication_unresponsive", "trajectory_indistinguishable"})
        responsive = HealthMonitor(config)
        good = ProbeSummary(1, .1, .03, .12, True)
        for step in range(1, 5):
            self.assertIsNone(
                responsive.observe(step, state(.1), state(.11), (.5,), TransitionDiagnostics(), good)
            )

    def test_delayed_one_direction_degeneration(self) -> None:
        monitor = HealthMonitor(
            PathologyConfig(
                fatal_threshold=2,
                one_direction_steps=3,
                node_growth_alert=.1,
                homogenization_variance=-1,
            )
        )
        cause = None
        for step in range(1, 6):
            cause = monitor.observe(
                step, state(step * .1), state((step + 1) * .1), (.5,), TransitionDiagnostics()
            )
        self.assertEqual(cause, "one_direction_degeneration")

    def test_opposing_coordinate_drifts_cannot_cancel_in_health_check(self) -> None:
        monitor = HealthMonitor(
            PathologyConfig(
                fatal_threshold=2,
                one_direction_steps=2,
                node_growth_alert=.1,
                homogenization_variance=-1,
            )
        )
        cause = None
        for step in range(1, 4):
            cause = monitor.observe(
                step,
                vector_state((.1 * step, -.1 * step)),
                vector_state((.1 * (step + 1), -.1 * (step + 1))),
                (.5,),
                TransitionDiagnostics(),
            )
        self.assertEqual(cause, "one_direction_degeneration")

    def test_a_single_unresponsive_coordinate_fails_the_probe(self) -> None:
        monitor = HealthMonitor(PathologyConfig(fatal_threshold=2, homogenization_variance=-1))
        partial = ProbeSummary(
            1,
            .1,
            .03,
            .12,
            True,
            coordinate_response=(.1, 0.0),
            coordinate_propagation=(.03, .03),
            coordinate_distinguishability=(.12, .12),
            coordinate_recovered=(True, True),
        )
        monitor.observe(1, vector_state((.1, -.1)), vector_state((.1, -.1)), (.5,), TransitionDiagnostics(), partial)
        cause = monitor.observe(2, vector_state((.1, -.1)), vector_state((.1, -.1)), (.5,), TransitionDiagnostics(), partial)
        self.assertEqual(cause, "input_unresponsive")

    def test_never_active_edge_dynamics_die_even_when_nodes_are_healthy(self) -> None:
        monitor = HealthMonitor(
            PathologyConfig(
                fatal_threshold=2,
                homogenization_variance=-1,
                edge_activity_grace_steps=0,
            )
        )
        monitor.observe(1, state(0), state(0.1), (.5,), TransitionDiagnostics())
        cause = monitor.observe(2, state(.1), state(.2), (.5,), TransitionDiagnostics())
        self.assertEqual(cause, "edge_dynamics_inactive")

    def test_persistently_growing_edge_latents_die_before_safety_limit(self) -> None:
        monitor = HealthMonitor(
            PathologyConfig(
                fatal_threshold=2,
                homogenization_variance=-1,
                edge_saturation_strength=.60,
                edge_growth_alert=.1,
                edge_growth_steps=2,
            )
        )
        cause = None
        for step, (before, after) in enumerate(((.2, .4), (.4, .6), (.6, .8)), start=1):
            cause = monitor.observe(
                step,
                state_with_edges(.1, before),
                state_with_edges(.2, after),
                (.7, .7),
                TransitionDiagnostics(),
            )
        self.assertEqual(cause, "edge_runaway_growth")

    def test_common_scenario_bank_is_candidate_independent(self) -> None:
        level = CurriculumLevel(10)
        first = ScenarioBank.create(7, 0, 3, level)
        second = ScenarioBank.create(7, 0, 3, level)
        self.assertEqual(first, second)
        self.assertEqual(len({item.input_seed for item in first.scenarios}), 3)

    def test_result_buffered_cma_updates_only_after_batch(self) -> None:
        adapter = SteadyStateCMA(3, 4, .2, 9)
        samples = [adapter.ask() for _ in range(4)]
        for index, asked in enumerate(samples):
            assert asked is not None
            sample_id, genome = asked
            updated = adapter.observe("same-bank", sample_id, genome, (1, index))
            self.assertEqual(updated, index == 3)
        self.assertEqual(adapter.update_count, 1)
        # A later censor record for an already told sample cannot tell it twice.
        sample_id, genome = samples[0]
        self.assertFalse(adapter.observe("same-bank", sample_id, genome, (2, 99)))
        self.assertEqual(adapter.update_count, 1)

    def test_validated_elite_preservation(self) -> None:
        archive = EliteArchive(2)
        strong = {"candidate_id": 1, "genome": [1.0], "rank_key": [2, 1, 10]}
        weak = {"candidate_id": 2, "genome": [2.0], "rank_key": [1, 1, 10]}
        worse = {"candidate_id": 3, "genome": [3.0], "rank_key": [0, 0, 0]}
        archive.consider(strong)
        archive.consider(weak)
        archive.consider(worse)
        self.assertEqual([item["candidate_id"] for item in archive.records], [1, 2])

    def test_async_completion_censoring_graduation_replacement_replica_aggregation_and_replay(self) -> None:
        architecture = RuleArchitecture(state_width=1, hidden_width=3)
        edge = EdgeArchitecture(node_state_width=1, latent_width=2, hidden_width=3)
        codec = GenomeCodec(architecture, edge, "joint")
        references = diagnostic_reference_genomes(codec)
        config = AsyncEvolutionConfig(
            slots=3,
            replicas=2,
            result_batch_size=2,
            max_ticks=20,
            seed=3,
            architecture=architecture,
            edge_architecture=edge,
            levels=(CurriculumLevel(8, input_scale=.1, graph_nodes=5),),
            pathology=PathologyConfig(fatal_threshold=2),
            censor_interval=2,
            initial_genomes=references,
        )
        with tempfile.TemporaryDirectory() as directory:
            runner = AsyncEvolutionRunner(config)
            runner.run(Path(directory))
            self.assertEqual(len(runner.slots), 3)
            self.assertGreater(len(runner.archive), 0)
            self.assertTrue(runner.censored)
            self.assertTrue(any(item["kind"] == "milestone" for item in runner.censored))
            self.assertTrue(any(item["status"] == "graduation" for item in runner.archive))
            self.assertGreater(max(slot.candidate_id for slot in runner.slots), 2)
            record = runner.archive[0]
            self.assertEqual(len(record["per_replica_results"]), 2)
            conservative_age = min(item["age"] for item in record["per_replica_results"])
            self.assertEqual(record["age"], conservative_age)
            replay = AsyncEvolutionRunner.replay_record(record, config)
            self.assertEqual(replay["age"], record["age"])
            self.assertEqual(replay["death_cause"], record["death_cause"])
            self.assertEqual(replay["per_replica_results"], record["per_replica_results"])


if __name__ == "__main__":
    unittest.main()
