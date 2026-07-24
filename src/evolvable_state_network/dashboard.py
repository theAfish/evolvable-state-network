"""Export recorded experiments and bundle the dependency-free browser dashboard."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

from .graph import Graph
from .simulation import SimulationConfig, Trajectory


def trajectory_to_dict(trajectory: Trajectory) -> dict[str, object]:
    """Serialize every recorded state required for replay and inspection."""
    return {
        "times": trajectory.times,
        "steps": trajectory.steps,
        "node_states": trajectory.node_states,
        "edge_states": trajectory.edge_states,
        "inputs": trajectory.inputs,
        "events": [asdict(event) for event in trajectory.events],
    }


def write_dashboard_bundle(
    output: Path,
    graph: Graph,
    runs: Mapping[str, tuple[Trajectory, Mapping[str, object]]],
    config: SimulationConfig | None = None,
) -> Path:
    """Write replay data and copy the static dashboard beneath ``output``."""
    copy_dashboard_assets(output)
    document = dashboard_document(graph, runs, config)
    data_path = output / "dashboard_data.json"
    data_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return data_path


def copy_dashboard_assets(output: Path) -> Path:
    """Copy the browser application without requiring an existing experiment."""
    source = Path(__file__).parent / "web"
    destination = output / "dashboard"
    shutil.copytree(source, destination, dirs_exist_ok=True)
    return destination


def dashboard_document(
    graph: Graph,
    runs: Mapping[str, tuple[Trajectory, Mapping[str, object]]],
    config: SimulationConfig | None = None,
) -> dict[str, object]:
    """Build the in-memory dashboard response used by the local UI server."""
    document: dict[str, object] = {
        "schema_version": 1,
        "graph": {
            "nodes": graph.n_nodes,
            "edges": [asdict(edge) for edge in graph.edges],
        },
        "runs": {
            name: {"trajectory": trajectory_to_dict(trajectory), "metrics": dict(metrics)}
            for name, (trajectory, metrics) in runs.items()
        },
    }
    if config is not None:
        document["simulation_config"] = asdict(config)
    return document
