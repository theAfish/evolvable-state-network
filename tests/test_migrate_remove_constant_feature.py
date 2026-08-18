from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


_SCRIPT = Path(__file__).parents[1] / "scripts" / "migrate_remove_constant_feature.py"
_SPEC = importlib.util.spec_from_file_location("migrate_remove_constant_feature", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class RemoveConstantFeatureMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = {
            "architecture": {"state_width": 1, "hidden_width": 1, "hidden_layers": None},
            "edge_architecture": {
                "node_state_width": 1, "latent_width": 1, "hidden_width": 1, "hidden_layers": None,
            },
            "prey_best_genome": [float(value) for value in range(1, 15)],
            "prey": {"best_genome": [float(value) for value in range(1, 15)]},
        }

    def test_folds_constant_weight_into_first_layer_bias(self) -> None:
        migrated = _MODULE.migrate_document(self.document)
        # Node legacy parameters: [W(state, aggregate, constant), b, W, b].
        # Edge follows with the same 3 -> 2 input compaction.
        self.assertEqual(migrated, 2)
        self.assertEqual(
            self.document["prey_best_genome"],
            [1.0, 2.0, 7.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 23.0, 13.0, 14.0],
        )
        self.assertEqual(self.document["prey"]["best_genome"], self.document["prey_best_genome"])
        self.assertEqual(self.document["genome_migration"]["parameter_count"], 12)

    def test_writes_a_sibling_without_changing_source_by_default(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "checkpoint.json"
            source.write_text(json.dumps(self.document), encoding="utf-8")
            self.assertEqual(_MODULE.main([str(source)]), 0)
            migrated = source.with_name("checkpoint.migrated.json")
            self.assertTrue(migrated.is_file())
            self.assertEqual(json.loads(source.read_text(encoding="utf-8")), self.document)

    def test_rejects_serialized_cmaes_checkpoint(self) -> None:
        document = {
            "architecture": {"state_width": 1, "hidden_width": 1},
            "optimizer": {"pycma_pickle": "not-safe-to-transform"},
        }
        with self.assertRaises(_MODULE.MigrationError):
            _MODULE.migrate_document(document)
