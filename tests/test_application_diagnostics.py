from __future__ import annotations

import math
import unittest

from evolvable_state_network.application.diagnostics import (
    compare_vectors,
    distribution_summary,
    raw_output_summary,
)


class ApplicationDiagnosticTests(unittest.TestCase):
    def test_distribution_summary_interpolates_percentiles(self) -> None:
        summary = distribution_summary((0.0, 10.0))

        self.assertEqual(summary["mean"], 5.0)
        self.assertEqual(summary["p10"], 1.0)
        self.assertEqual(summary["p90"], 9.0)
        self.assertEqual(distribution_summary(()), {})

    def test_vector_comparison_rejects_mismatched_dimensions(self) -> None:
        self.assertEqual(
            compare_vectors((1.0,), (1.0, 2.0)),
            {"compatible": False, "left_dimension": 1, "right_dimension": 2},
        )

    def test_vector_comparison_reports_scale_independent_similarity(self) -> None:
        comparison = compare_vectors((1.0, 2.0), (2.0, 4.0))

        self.assertTrue(comparison["compatible"])
        self.assertAlmostEqual(comparison["cosine_similarity"], 1.0)
        self.assertAlmostEqual(comparison["l2_distance"], math.sqrt(5.0))

    def test_raw_output_summary_tracks_absolute_tail_fractions(self) -> None:
        summary = raw_output_summary((-4.0, -2.0, 0.0, 2.0, 4.0))

        self.assertEqual(summary["mean"], 0.0)
        self.assertEqual(summary["abs_gt_1_fraction"], 0.8)
        self.assertEqual(summary["abs_gt_3_fraction"], 0.4)
        self.assertEqual(raw_output_summary(()), {})


if __name__ == "__main__":
    unittest.main()
