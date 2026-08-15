from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evolvable_state_network.plot_data import embodied_plot_rows, evolution_plot_row, write_plot_table


class PlotDataTests(unittest.TestCase):
    def test_embodied_batch_rows_are_available_before_the_final_report(self) -> None:
        rows = embodied_plot_rows(
            {"training_mode": "batch"},
            ({"history": [{"generation": 1, "prey_best_lifetime": 12.0}]},),
        )
        self.assertEqual(rows, [{"generation": 1, "prey_best_lifetime": 12.0}])

    def test_embodied_continuous_rows_keep_all_ticks_seen_in_overlapping_events(self) -> None:
        rows = embodied_plot_rows(
            {"training_mode": "continuous"},
            (
                {"telemetry": [{"tick": 1, "prey_deaths": 0}, {"tick": 2, "prey_deaths": 1}]},
                {"telemetry": [{"tick": 2, "prey_deaths": 1}, {"tick": 3, "prey_deaths": 2}]},
            ),
        )
        self.assertEqual(rows, [
            {"tick": 1, "prey_deaths": 0},
            {"tick": 2, "prey_deaths": 1},
            {"tick": 3, "prey_deaths": 2},
        ])

    def test_table_is_headered_tsv_and_omits_nested_report_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_curves.txt"
            write_plot_table(path, [{"tick": 1, "fitness": 2.5, "details": ["not a column"]}], description="test curves")
            contents = path.read_text(encoding="utf-8")
        self.assertIn("# test curves", contents)
        self.assertIn("tick\tfitness", contents)
        self.assertIn("1\t2.5", contents)
        self.assertNotIn("details", contents)

    def test_async_progress_becomes_compact_curve_row(self) -> None:
        row = evolution_plot_row({
            "tick": 4,
            "report": {"active_slots": 2, "completed_candidates": 3, "deaths": 1, "graduations": 2, "optimizer_updates": 1, "active_slot_utilization": .75},
        })
        self.assertEqual(row, {"tick": 4, "active_slots": 2, "completed_candidates": 3, "deaths": 1, "graduations": 2, "optimizer_updates": 1, "active_slot_utilization": .75})


if __name__ == "__main__":
    unittest.main()
