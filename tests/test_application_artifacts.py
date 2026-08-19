from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from threading import Lock

from evolvable_state_network.application.artifacts import write_json_atomically


class ApplicationArtifactTests(unittest.TestCase):
    def test_atomic_json_write_replaces_content_without_leaving_temporary_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text('{"old": true}', encoding="utf-8")

            write_json_atomically(path, {"new": [1, 2, 3]}, lock=Lock())

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"new": [1, 2, 3]})
            self.assertEqual(list(path.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
