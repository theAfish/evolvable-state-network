from __future__ import annotations

import unittest

from evolvable_state_network.metrics import evaluate_metrics
from evolvable_state_network.simulation import Trajectory


def trajectory_from_frames(frames: list[list[float]]) -> Trajectory:
    trajectory = Trajectory()
    for step, values in enumerate(frames):
        snapshot = [[(value,) for value in values]]
        trajectory.append(step, float(step), snapshot, snapshot)
    return trajectory


class PathologyMetricTests(unittest.TestCase):
    def test_dead_network_is_silent_and_not_diverse(self) -> None:
        report = evaluate_metrics(trajectory_from_frames([[0.0, 0.0]] * 8))
        self.assertFalse(report.non_silence.non_silent)
        self.assertFalse(report.activity_diversity.diverse)

    def test_saturated_network_is_detected(self) -> None:
        report = evaluate_metrics(trajectory_from_frames([[0.99, -0.99]] * 8))
        self.assertTrue(report.saturation.saturated)
        self.assertTrue(report.non_silence.non_silent)

    def test_synchronized_network_has_low_activity_diversity(self) -> None:
        report = evaluate_metrics(trajectory_from_frames([[0.3, 0.3, 0.3]] * 8))
        self.assertFalse(report.activity_diversity.diverse)

    def test_exploding_network_is_not_bounded(self) -> None:
        report = evaluate_metrics(trajectory_from_frames([[0.0, 0.0], [100.0, -120.0]]))
        self.assertFalse(report.boundedness.bounded)
