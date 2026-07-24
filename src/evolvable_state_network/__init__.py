"""A small, deterministic foundation for generic stateful graph dynamics."""

from .baselines import FixedRNNRule, HomeostaticRule
from .graph import Graph, generate_random_graph
from .metrics import MetricReport, evaluate_metrics
from .simulation import Simulation, SimulationConfig, SimulationResult

__all__ = [
    "FixedRNNRule",
    "Graph",
    "HomeostaticRule",
    "MetricReport",
    "Simulation",
    "SimulationConfig",
    "SimulationResult",
    "evaluate_metrics",
    "generate_random_graph",
]
