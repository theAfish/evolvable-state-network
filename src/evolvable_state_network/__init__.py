"""A small, deterministic foundation for generic stateful graph dynamics."""

from .baselines import FixedRNNRule, HomeostaticRule
from .graph import Graph, generate_random_graph
from .metrics import MetricReport, evaluate_metrics
from .simulation import Simulation, SimulationConfig, SimulationResult
from .evolution import (
    AsyncEvolutionConfig,
    AsyncEvolutionRunner,
    CandidateEvaluator,
    EdgeArchitecture,
    EvaluationResult,
    GenomeCodec,
    MLPEdgeRule,
    MLPUpdateRule,
    RuleArchitecture,
    ScenarioConfig,
    ScenarioSuite,
)

__all__ = [
    "FixedRNNRule",
    "Graph",
    "HomeostaticRule",
    "EdgeArchitecture",
    "MLPEdgeRule",
    "MLPUpdateRule",
    "MetricReport",
    "CandidateEvaluator",
    "AsyncEvolutionConfig",
    "AsyncEvolutionRunner",
    "EvaluationResult",
    "GenomeCodec",
    "RuleArchitecture",
    "ScenarioConfig",
    "ScenarioSuite",
    "Simulation",
    "SimulationConfig",
    "SimulationResult",
    "evaluate_metrics",
    "generate_random_graph",
]
