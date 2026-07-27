"""A small, deterministic foundation for generic stateful graph dynamics."""

from .baselines import FixedRNNRule, HomeostaticRule
from .graph import Graph, generate_random_graph
from .metrics import MetricReport, evaluate_metrics
from .simulation import Simulation, SimulationConfig, SimulationResult
from .candidate import EdgeArchitecture, MLPEdgeRule, MLPUpdateRule, RuleArchitecture
from .evaluation import CandidateEvaluator, EvaluationResult, ScenarioConfig, ScenarioSuite
from .genome import GenomeCodec
from .async_evolution import AsyncEvolutionConfig, AsyncEvolutionRunner

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
