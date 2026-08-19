from __future__ import annotations

import unittest

from evolvable_state_network.application.jobs import JobRegistry


class JobRegistryTests(unittest.TestCase):
    def test_progress_events_are_grouped_by_phase(self) -> None:
        registry = JobRegistry()
        job_id = registry.create("experiment", seed=7, total=3)

        registry.publish(job_id, {"phase": "smoke", "sample": 1})
        registry.publish(job_id, {"phase": "generation", "generation": 1})
        registry.publish(job_id, {"phase": "validation", "score": 0.5})

        snapshot = registry.snapshot(job_id)
        self.assertEqual(snapshot["samples"], [{"phase": "smoke", "sample": 1}])
        self.assertEqual(snapshot["generations"], [{"phase": "generation", "generation": 1}])
        self.assertEqual(snapshot["latest"], {"phase": "validation", "score": 0.5})
        self.assertEqual(snapshot["phase"], "validation")

    def test_termination_is_kind_scoped_and_only_requested_once(self) -> None:
        registry = JobRegistry()
        job_id = registry.create("embodied", seed=11, total=None)

        with self.assertRaises(KeyError):
            registry.request_termination(job_id, kind="different")
        self.assertTrue(registry.request_termination(job_id, kind="embodied"))
        self.assertTrue(registry.termination_requested(job_id))

        registry.terminate(job_id)
        self.assertFalse(registry.request_termination(job_id, kind="embodied"))
        self.assertEqual(registry.snapshot(job_id)["status"], "terminated")

    def test_terminal_states_retain_results_and_errors(self) -> None:
        registry = JobRegistry()
        completed = registry.create("complete", seed=1, total=1)
        failed = registry.create("failed", seed=2, total=1)

        registry.finish(completed, {"score": 4.2})
        registry.fail(failed, RuntimeError("boom"))

        self.assertEqual(registry.snapshot(completed)["result"], {"score": 4.2})
        self.assertEqual(registry.snapshot(completed)["phase"], "complete")
        self.assertEqual(registry.snapshot(failed)["error"], "boom")
        self.assertEqual(registry.snapshot(failed)["phase"], "failed")


if __name__ == "__main__":
    unittest.main()
