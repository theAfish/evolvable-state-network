"""Serialize recorded experiments for the FastAPI replay interface."""

from __future__ import annotations

from dataclasses import asdict
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
        "effective_edge_strengths": trajectory.effective_edge_strengths,
        "inputs": trajectory.inputs,
        "events": [asdict(event) for event in trajectory.events],
    }


def dashboard_document(
    graph: Graph,
    runs: Mapping[str, tuple[Trajectory, Mapping[str, object]]],
    config: SimulationConfig | None = None,
) -> dict[str, object]:
    """Build the replay response returned directly by FastAPI."""
    document: dict[str, object] = {
        "schema_version": 2,
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
