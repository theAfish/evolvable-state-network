from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from evolvable_state_network.evolution.asynchronous import (
    AsyncEvolutionConfig,
    AsyncEvolutionRunner,
    CandidateProposal,
    CandidateSlot,
    CurriculumLevel,
    EliteArchive,
    SurvivorArchive,
    HealthMonitor,
    PathologyConfig,
    ProbeSummary,
    ScenarioBank,
    SteadyStateCMA,
    diagnostic_reference_genomes,
)
from evolvable_state_network.evolution.candidate import EdgeArchitecture, RuleArchitecture
from evolvable_state_network.evolution.genome import GenomeCodec
from evolvable_state_network.simulation import NetworkState, TransitionDiagnostics


def state(value: float, nodes: int = 3) -> NetworkState:
    return NetworkState(node=[[(value,) for _ in range(nodes)]], edge=[[(0.0,), (0.0,)]])


def state_with_edges(node_value: float, edge_value: float, nodes: int = 3) -> NetworkState:
    return NetworkState(
        node=[[(node_value,) for _ in range(nodes)]], edge=[[(edge_value,), (edge_value,)]]
    )


def state_with_edge_vector(node_value: float, edge_values: tuple[float, ...], nodes: int = 3) -> NetworkState:
    return NetworkState(
        node=[[(node_value,) for _ in range(nodes)]], edge=[[edge_values, edge_values]]
    )


def vector_state(values: tuple[float, ...], nodes: int = 3) -> NetworkState:
    return NetworkState(node=[[values for _ in range(nodes)]], edge=[[(0.0,), (0.0,)]])


class AsyncEvolutionTests(unittest.TestCase):
    def test_training_defaults_to_no_tick_cap(self) -> None:
        self.assertIsNone(AsyncEvolutionConfig().max_ticks)

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

    def test_resting_homogeneous_fixed_point_dies_even_if_probes_would_break_it(self) -> None:
        monitor = HealthMonitor(
            PathologyConfig(
                fatal_threshold=99,
                homogenization_variance=1e-4,
                rest_homogenization_steps=3,
            )
        )
        for step in range(1, 3):
            self.assertIsNone(
                monitor.observe(step, state(0), state(0), (.5,), TransitionDiagnostics())
            )
        self.assertEqual(
            monitor.observe(3, state(0), state(0), (.5,), TransitionDiagnostics()),
            "rest_state_homogenization",
        )

        interrupted = HealthMonitor(
            PathologyConfig(
                fatal_threshold=99,
                homogenization_variance=1e-4,
                rest_homogenization_steps=3,
            )
        )
        for step in range(1, 7):
            self.assertIsNone(
                interrupted.observe(
                    step, state(0), state(0), (.5,), TransitionDiagnostics(), resting=False
                )
            )
        self.assertIsNone(
            interrupted.observe(7, state(0), state(0), (.5,), TransitionDiagnostics())
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

    def test_second_edge_coordinate_runaway_cannot_be_hidden_by_the_first(self) -> None:
        monitor = HealthMonitor(
            PathologyConfig(
                fatal_threshold=2,
                homogenization_variance=-1,
                edge_saturation_strength=.99,
                edge_growth_alert=.1,
                edge_growth_steps=2,
            )
        )
        cause = None
        for step, (before, after) in enumerate(
            (((.0, .0), (.0, .2)), ((.0, .0), (.0, .4)), ((.0, .0), (.0, .6))),
            start=1,
        ):
            cause = monitor.observe(
                step,
                state_with_edge_vector(.1, before),
                state_with_edge_vector(.2, after),
                (.5, .5),
                TransitionDiagnostics(),
                edge_gates=((.5, .5), (.5, .5)),
            )
        self.assertEqual(cause, "edge_runaway_growth")

    def test_second_gate_coordinate_saturation_cannot_be_hidden_by_mean_strength(self) -> None:
        monitor = HealthMonitor(
            PathologyConfig(
                fatal_threshold=2,
                homogenization_variance=-1,
                edge_saturation_strength=.60,
                edge_saturation_fraction=.80,
                edge_growth_steps=12,
            )
        )
        cause = None
        for step in range(1, 3):
            cause = monitor.observe(
                step,
                state_with_edge_vector(.1, (.0, .1)),
                state_with_edge_vector(.2, (.0, .1)),
                # The scalar diagnostic average is healthy; coordinate 1 is not.
                (.5, .5),
                TransitionDiagnostics(),
                edge_gates=((.5, .7), (.5, .7)),
            )
        self.assertEqual(cause, "edge_gate_saturation")

    def test_persistent_node_boundary_residence_is_not_viable(self) -> None:
        monitor = HealthMonitor(
            PathologyConfig(
                fatal_threshold=2,
                homogenization_variance=-1,
                boundary_fraction=.5,
            )
        )
        cause = None
        for step in range(1, 3):
            cause = monitor.observe(
                step, state(3.9), state(3.9), (.5,), TransitionDiagnostics()
            )
        self.assertEqual(cause, "boundary_saturation")

    def test_training_proposals_are_cma_samples(self) -> None:
        config = AsyncEvolutionConfig(
            slots=1,
            replicas=1,
            result_batch_size=2,
            max_ticks=1,
            seed=17,
        )
        runner = AsyncEvolutionRunner(config)
        proposal = runner._proposal()
        assert proposal is not None
        self.assertEqual(proposal.source, "cma")
        self.assertIsNotNone(proposal.sample_id)

    def test_final_deployment_validation_emits_incremental_progress(self) -> None:
        events: list[dict[str, object]] = []
        runner = AsyncEvolutionRunner(
            AsyncEvolutionConfig(
                slots=1,
                replicas=1,
                result_batch_size=2,
                max_ticks=1,
                seed=41,
                levels=(CurriculumLevel(1, graph_nodes=4, mean_degree=2),),
                deployment_validation_replicas=1,
                deployment_validation_nodes=4,
                deployment_validation_mean_degree=2,
                deployment_autonomous_steps=3,
            )
        )
        runner.run(progress=events.append)
        validation = [event["validation"] for event in events if event.get("validation")]
        self.assertTrue(validation)
        self.assertEqual(validation[0]["phase"], "autonomous")
        self.assertEqual(validation[0]["candidate_id"], 0)
        self.assertEqual(validation[-1]["replica"], 1)

    def test_common_scenario_bank_is_candidate_independent(self) -> None:
        level = CurriculumLevel(10)
        first = ScenarioBank.create(7, 0, 3, level)
        second = ScenarioBank.create(7, 0, 3, level)
        self.assertEqual(first, second)
        self.assertEqual(len({item.input_seed for item in first.scenarios}), 3)

    def test_state_perturbations_are_seeded_random_nodes_with_coordinatewise_gaussians(self) -> None:
        architecture = RuleArchitecture(state_width=2, hidden_width=2)
        edge = EdgeArchitecture(node_state_width=2, latent_width=2, hidden_width=2)
        codec = GenomeCodec(architecture, edge, "joint")
        runner = AsyncEvolutionRunner(
            AsyncEvolutionConfig(
                slots=1,
                replicas=1,
                result_batch_size=2,
                architecture=architecture,
                edge_architecture=edge,
                levels=(CurriculumLevel(12, graph_nodes=5),),
                initial_state_scale=.2,
            )
        )
        proposal = CandidateProposal((0.0,) * codec.dimension, "cma", 0, 0)
        replica = runner._new_slot(0, proposal).replicas[0]
        first = replica._gaussian_state_packet(7, .12, salt=17)
        second = replica._gaussian_state_packet(7, .12, salt=17)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first[0], 0)
        self.assertLess(first[0], replica.simulation.graph.n_nodes)
        self.assertEqual(len(first[1]), 2)
        self.assertNotEqual(first[1][0], first[1][1])

    def test_result_buffered_cma_updates_only_after_batch(self) -> None:
        adapter = SteadyStateCMA(3, 4, .2, 9)
        samples = [adapter.ask() for _ in range(4)]
        for index, asked in enumerate(samples):
            assert asked is not None
            sample_id, genome = asked
            updated = adapter.observe("same-bank", sample_id, genome, float(index))
            self.assertEqual(updated, index == 3)
        self.assertEqual(adapter.update_count, 1)
        # A later censor record for an already told sample cannot tell it twice.
        sample_id, genome = samples[0]
        self.assertFalse(adapter.observe("same-bank", sample_id, genome, 99.0))
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

    def test_harder_stage_functional_parent_replaces_lower_stage_parent(self) -> None:
        archive = SurvivorArchive(12)
        stage_one = {"candidate_id": 1, "genome": [1.0], "rank_key": [1, 1, 40]}
        stage_two = {"candidate_id": 2, "genome": [2.0], "rank_key": [2, 1, 100]}
        archive.consider(stage_one)
        archive.consider(stage_two)
        self.assertEqual([item["candidate_id"] for item in archive.records], [2])

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
            self.assertLessEqual(len(runner.slots), 3)
            self.assertGreater(len(runner.slots), 0)
            self.assertGreater(len(runner.archive), 0)
            self.assertTrue(runner.censored)
            self.assertTrue(all(item["kind"] == "run_stop" for item in runner.censored))
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

    def test_stage_keeps_evolving_past_evidence_checkpoint_until_survivors_are_stable(self) -> None:
        architecture = RuleArchitecture(state_width=1, hidden_width=3)
        edge = EdgeArchitecture(node_state_width=1, latent_width=2, hidden_width=3)
        codec = GenomeCodec(architecture, edge, "joint")
        config = AsyncEvolutionConfig(
            slots=3,
            replicas=1,
            result_batch_size=2,
            candidate_budget=1,
            max_ticks=20,
            seed=3,
            architecture=architecture,
            edge_architecture=edge,
            levels=(CurriculumLevel(8, graph_nodes=5),),
            pathology=PathologyConfig(fatal_threshold=2),
            initial_genomes=diagnostic_reference_genomes(codec),
        )
        runner = AsyncEvolutionRunner(config)
        report = runner.run()
        self.assertEqual(report["stop_reason"], "stage_not_passed_tick_limit")
        self.assertGreater(report["completed_candidates"], config.candidate_budget)
        self.assertTrue(report["candidate_evidence_checkpoint_reached"])

    def test_archived_lives_keep_health_aggregates_but_not_trajectories(self) -> None:
        architecture = RuleArchitecture(state_width=1, hidden_width=3)
        edge = EdgeArchitecture(node_state_width=1, latent_width=2, hidden_width=3)
        config = AsyncEvolutionConfig(
            slots=2,
            replicas=1,
            result_batch_size=2,
            max_ticks=20,
            seed=7,
            architecture=architecture,
            edge_architecture=edge,
            levels=(CurriculumLevel(8, graph_nodes=5),),
            pathology=PathologyConfig(fatal_threshold=2),
        )
        runner = AsyncEvolutionRunner(config)
        runner.run()
        replica = runner.archive[0]["per_replica_results"][0]
        self.assertIn("current_pathology_burdens", replica)
        self.assertIn("pathology_violation_counts", replica)
        self.assertNotIn("pathology_trajectories", replica)
        self.assertNotIn("response_probes", replica)

    def test_stage_change_discards_partial_batch_and_starts_a_fresh_comparable_cohort(self) -> None:
        adapter = SteadyStateCMA(3, 2, .2, 9)
        adapter.begin_stage("stage-1")
        asked = adapter.ask()
        assert asked is not None
        sample_id, genome = asked
        self.assertFalse(adapter.observe("stage-1", sample_id, genome, 1))
        adapter.begin_stage("stage-2")
        samples = [adapter.ask() for _ in range(2)]
        for index, asked in enumerate(samples):
            assert asked is not None
            sample_id, genome = asked
            self.assertEqual(
                adapter.observe("stage-2", sample_id, genome, float(index)), index == 1
            )
        self.assertEqual(adapter.update_count, 1)
        self.assertGreater(adapter.cancelled_samples, 0)

    def test_stage_requires_a_completed_cma_cohort_before_advancing(self) -> None:
        architecture = RuleArchitecture(state_width=1, hidden_width=3)
        edge = EdgeArchitecture(node_state_width=1, latent_width=2, hidden_width=3)
        runner = AsyncEvolutionRunner(
            AsyncEvolutionConfig(
                slots=2,
                replicas=1,
                result_batch_size=2,
                max_ticks=20,
                seed=13,
                architecture=architecture,
                edge_architecture=edge,
                levels=(CurriculumLevel(100, graph_nodes=5),),
                stable_population_size=1,
            )
        )
        record = {"candidate_id": 1, "genome": [1.0], "rank_key": [1, 1, 100]}
        runner.stage_survivors[0].append(record)
        self.assertFalse(runner._maybe_advance_curriculum())
        runner.optimizer.update_count = 1
        self.assertTrue(runner._maybe_advance_curriculum())


if __name__ == "__main__":
    unittest.main()
