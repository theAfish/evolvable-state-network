#!/usr/bin/env python
"""One-off migration for genomes that included a redundant constant input.

Old node rules used ``[state, aggregate, 1]`` and old edge rules used
``[edge_state, source, target, message, 1]`` despite also storing a bias for
every MLP layer.  This script removes the last weight of the *first* layer and
adds it to that layer's explicit bias, preserving the rule exactly.

Examples
--------
Preview one running embodied checkpoint without writing anything::

    .\\.venv\\Scripts\\python.exe scripts\\migrate_remove_constant_feature.py \
        .outputs\\embodied_runs\\<run-id> --dry-run

Write migrated sibling files (``checkpoint.migrated.json`` etc.)::

    .\\.venv\\Scripts\\python.exe scripts\\migrate_remove_constant_feature.py \
        .outputs\\embodied_runs\\<run-id>

Replace files atomically, retaining a ``.legacy-constant-input`` backup::

    .\\.venv\\Scripts\\python.exe scripts\\migrate_remove_constant_feature.py \
        .outputs\\embodied_runs\\<run-id> --in-place

Stop an old-code training process before using ``--in-place``.  It would
otherwise write another legacy checkpoint on its next save.

This intentionally refuses generic CMA-ES checkpoints containing a serialized
optimizer state.  Their internal covariance state has the old dimension and
cannot be safely transformed by a JSON-only migration.  Embodied checkpoints
are safe because continuation seeds a fresh optimizer from the migrated best
genomes.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence


Target = Literal["node", "edge", "joint"]
GenomeDocument = dict[str, Any]
MIGRATION_NAME = "remove_redundant_constant_feature_v1"
SUPPORTED_FILENAMES = frozenset({"checkpoint.json", "report.json", "best_genome.json"})
GENOME_KEYS = frozenset({
    "genome", "best_genome", "prey_best_genome", "predator_best_genome",
    "initial_genome", "initial_prey_genome", "initial_predator_genome",
})


class MigrationError(ValueError):
    """The selected JSON is not a migratable legacy genome document."""


@dataclass(frozen=True)
class MLPShape:
    input_width: int
    widths: tuple[int, ...]

    @property
    def parameter_count(self) -> int:
        previous = self.input_width
        count = 0
        for width in self.widths:
            count += width * (previous + 1)
            previous = width
        return count

    @property
    def legacy_parameter_count(self) -> int:
        # Only the first layer had the extra constant coordinate.
        return self.parameter_count + self.widths[0]


@dataclass(frozen=True)
class GenomeLayout:
    node: MLPShape
    edge: MLPShape | None
    target: Target

    @property
    def node_parameter_count(self) -> int:
        return self.node.parameter_count if self.target in ("node", "joint") else 0

    @property
    def edge_parameter_count(self) -> int:
        return self.edge.parameter_count if self.edge is not None and self.target in ("edge", "joint") else 0

    @property
    def parameter_count(self) -> int:
        return self.node_parameter_count + self.edge_parameter_count

    @property
    def legacy_parameter_count(self) -> int:
        node = self.node.legacy_parameter_count if self.target in ("node", "joint") else 0
        edge = self.edge.legacy_parameter_count if self.edge is not None and self.target in ("edge", "joint") else 0
        return node + edge


def _hidden_widths(architecture: GenomeDocument) -> tuple[int, ...]:
    raw = architecture.get("hidden_layers")
    if raw is None:
        raw = (architecture["hidden_width"],)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise MigrationError("architecture.hidden_layers must be a non-empty sequence")
    widths = tuple(int(width) for width in raw)
    if any(width < 1 for width in widths):
        raise MigrationError("architecture.hidden_layers must contain positive widths")
    return widths


def _layout(document: GenomeDocument) -> GenomeLayout:
    architecture = document.get("architecture")
    if not isinstance(architecture, dict):
        experiment = document.get("experiment_config")
        architecture = experiment.get("architecture") if isinstance(experiment, dict) else None
    if not isinstance(architecture, dict):
        raise MigrationError("document has no architecture metadata")

    edge_architecture = document.get("edge_architecture")
    target: object = document.get("target")
    experiment = document.get("experiment_config")
    if isinstance(experiment, dict):
        edge_architecture = experiment.get("edge_architecture", edge_architecture)
        target = experiment.get("target", target)
    if target is None:
        target = "joint" if edge_architecture is not None else "node"
    if target not in ("node", "edge", "joint"):
        raise MigrationError(f"unsupported evolution target: {target!r}")

    state_width = int(architecture["state_width"])
    node = MLPShape(2 * state_width, _hidden_widths(architecture) + (state_width,))
    if target in ("edge", "joint"):
        if not isinstance(edge_architecture, dict):
            raise MigrationError("edge/joint genome has no edge_architecture metadata")
        node_state_width = int(edge_architecture["node_state_width"])
        latent_width = int(edge_architecture["latent_width"])
        edge = MLPShape(
            latent_width + 3 * node_state_width,
            _hidden_widths(edge_architecture) + (latent_width,),
        )
    else:
        edge = None
    return GenomeLayout(node, edge, target)


def _fold_first_layer_constant(parameters: Sequence[float], shape: MLPShape) -> list[float]:
    """Fold the legacy first-layer constant column into its explicit bias."""
    if len(parameters) != shape.legacy_parameter_count:
        raise MigrationError(
            f"legacy parameter vector has length {len(parameters)}, expected {shape.legacy_parameter_count}"
        )
    first_width = shape.widths[0]
    legacy_input_width = shape.input_width + 1
    weights_end = first_width * legacy_input_width
    old_weights = [float(value) for value in parameters[:weights_end]]
    old_bias = [float(value) for value in parameters[weights_end: weights_end + first_width]]
    result: list[float] = []
    for row in range(first_width):
        start = row * legacy_input_width
        result.extend(old_weights[start: start + shape.input_width])
    result.extend(
        old_bias[row] + old_weights[row * legacy_input_width + shape.input_width]
        for row in range(first_width)
    )
    result.extend(float(value) for value in parameters[weights_end + first_width:])
    if len(result) != shape.parameter_count:
        raise AssertionError("internal migration dimension mismatch")
    return result


def migrate_genome(values: Sequence[float], layout: GenomeLayout) -> tuple[list[float], bool]:
    """Return a compact genome and whether a legacy vector was converted."""
    numeric = [float(value) for value in values]
    if len(numeric) == layout.parameter_count:
        return numeric, False
    if len(numeric) != layout.legacy_parameter_count:
        raise MigrationError(
            f"genome has length {len(numeric)}; expected legacy {layout.legacy_parameter_count} "
            f"or current {layout.parameter_count}"
        )
    cursor = 0
    result: list[float] = []
    if layout.target in ("node", "joint"):
        end = cursor + layout.node.legacy_parameter_count
        result.extend(_fold_first_layer_constant(numeric[cursor:end], layout.node))
        cursor = end
    if layout.target in ("edge", "joint"):
        assert layout.edge is not None
        end = cursor + layout.edge.legacy_parameter_count
        result.extend(_fold_first_layer_constant(numeric[cursor:end], layout.edge))
    return result, True


def _migrate_parameter_groups(groups: dict[str, Any], layout: GenomeLayout) -> int:
    migrated = 0
    if layout.target in ("node", "joint") and isinstance(groups.get("node"), list):
        compact, changed = migrate_genome(groups["node"], GenomeLayout(layout.node, None, "node"))
        groups["node"] = compact
        migrated += int(changed)
    if layout.target in ("edge", "joint") and isinstance(groups.get("edge"), list):
        assert layout.edge is not None
        compact, changed = migrate_genome(groups["edge"], GenomeLayout(layout.node, layout.edge, "edge"))
        groups["edge"] = compact
        migrated += int(changed)
    return migrated


def _walk_and_migrate(value: Any, layout: GenomeLayout) -> int:
    migrated = 0
    if isinstance(value, dict):
        groups = value.get("parameter_groups")
        if isinstance(groups, dict):
            migrated += _migrate_parameter_groups(groups, layout)
        for key, child in tuple(value.items()):
            if key in GENOME_KEYS and isinstance(child, list):
                compact, changed = migrate_genome(child, layout)
                value[key] = compact
                migrated += int(changed)
            elif key == "initial_genomes" and isinstance(child, list):
                converted = []
                for genome in child:
                    if not isinstance(genome, list):
                        raise MigrationError("initial_genomes must contain numeric lists")
                    compact, changed = migrate_genome(genome, layout)
                    converted.append(compact)
                    migrated += int(changed)
                value[key] = converted
            elif key != "parameter_groups":
                migrated += _walk_and_migrate(child, layout)
    elif isinstance(value, list):
        for child in value:
            migrated += _walk_and_migrate(child, layout)
    return migrated


def migrate_document(document: GenomeDocument) -> int:
    """Migrate all rule-genome vectors in one JSON document in place."""
    optimizer = document.get("optimizer")
    if isinstance(optimizer, dict) and "pycma_pickle" in optimizer:
        raise MigrationError(
            "generic CMA-ES checkpoint contains a serialized optimizer; migrate its exported best_genome.json "
            "instead, then restart evolution from that genome"
        )
    layout = _layout(document)
    migrated = _walk_and_migrate(document, layout)
    document["genome_migration"] = {
        "name": MIGRATION_NAME,
        "legacy_parameter_count": layout.legacy_parameter_count,
        "parameter_count": layout.parameter_count,
        "migrated_vectors": migrated,
    }
    return migrated


def _files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        raise MigrationError(f"path does not exist: {path}")
    yield from sorted(candidate for candidate in path.rglob("*.json") if candidate.name in SUPPORTED_FILENAMES)


def _output_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.migrated{path.suffix}")


def _backup_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.legacy-constant-input{path.suffix}")


def _write_json(path: Path, document: GenomeDocument) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="a JSON artifact or a run directory containing checkpoint/report files")
    parser.add_argument("--in-place", action="store_true", help="replace each source after writing a .legacy-constant-input backup")
    parser.add_argument("--dry-run", action="store_true", help="validate and report migrations without writing files")
    arguments = parser.parse_args(argv)
    if arguments.in_place and arguments.dry_run:
        parser.error("--in-place and --dry-run cannot be combined")

    try:
        candidates = tuple(_files(arguments.path))
        if not candidates:
            raise MigrationError("no checkpoint.json, report.json, or best_genome.json found")
        for source in candidates:
            document = json.loads(source.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                raise MigrationError("top-level JSON must be an object")
            migrated = migrate_document(document)
            if arguments.dry_run:
                print(f"would migrate {source} ({migrated} genome vector(s))")
                continue
            destination = source if arguments.in_place else _output_path(source)
            if arguments.in_place:
                backup = _backup_path(source)
                if backup.exists():
                    raise MigrationError(f"refusing to overwrite existing backup: {backup}")
                backup.write_bytes(source.read_bytes())
            _write_json(destination, document)
            print(f"migrated {source} -> {destination} ({migrated} genome vector(s))")
    except (OSError, json.JSONDecodeError, MigrationError) as error:
        print(f"migration failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
