"""Shared construction of the fixed comparison experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .baselines import FixedRNNRule, HomeostaticRule
from .graph import Graph, generate_random_graph
from .inputs import GaussianInput
from .metrics import evaluate_metrics
from .perturbations import ImpulseInjection, InputDistributionShift, NodeLesion, Perturbation, WeightNoise
from .simulation import Simulation, SimulationConfig, Trajectory


@dataclass(frozen=True, slots=True)
class ExperimentRequest:
    """The small, safe parameter surface exposed by the CLI and local UI."""

    seed: int = 7
    nodes: int = 24
    mean_degree: float = 5.0
    steps: int = 300
    batch_size: int = 4
    dt: float = 0.05
    baseline: str = "both"

    def __post_init__(self) -> None:
        if not 2 <= self.nodes <= 200:
            raise ValueError("nodes must be between 2 and 200")
        if not 0 <= self.mean_degree <= self.nodes - 1:
            raise ValueError("mean_degree must be between 0 and nodes - 1")
        if not 1 <= self.steps <= 5_000 or not 1 <= self.batch_size <= 64 or not 0 < self.dt <= 1:
            raise ValueError("steps, batch_size, or dt is outside the supported range")
        if self.baseline not in {"both", "fixed_rnn", "homeostatic"}:
            raise ValueError("baseline must be both, fixed_rnn, or homeostatic")


@dataclass(frozen=True, slots=True)
class ExperimentData:
    graph: Graph
    config: SimulationConfig
    disturbances: tuple[Perturbation, ...]
    runs: Mapping[str, tuple[Trajectory, Mapping[str, object]]]


def run_experiment(request: ExperimentRequest) -> ExperimentData:
    """Run one deterministic reference experiment without writing artifacts."""
    graph = generate_random_graph(request.nodes, request.mean_degree, request.seed)
    config = SimulationConfig(steps=request.steps, dt=request.dt, batch_size=request.batch_size)
    quarter, half = request.steps // 4, request.steps // 2
    disturbances: tuple[Perturbation, ...] = (
        InputDistributionShift(quarter, quarter + max(2, request.steps // 12), offset=0.35, scale=1.35),
        ImpulseInjection(half, (0,), 1.2),
        NodeLesion(half + max(2, request.steps // 12), (1,), half + max(4, request.steps // 6)),
        WeightNoise(half, min(request.steps - 1, half + max(2, request.steps // 10)), 0.12, request.seed + 19),
    )
    candidates = {"fixed_rnn": FixedRNNRule(), "homeostatic": HomeostaticRule()}
    if request.baseline != "both":
        candidates = {request.baseline: candidates[request.baseline]}
    input_provider = GaussianInput(request.seed + 101, standard_deviation=0.28)
    runs: dict[str, tuple[Trajectory, Mapping[str, object]]] = {}
    for name, rule in candidates.items():
        result = Simulation(graph, rule).run(config, input_provider, disturbances)
        runs[name] = (result.trajectory, evaluate_metrics(result.trajectory, safety_bound=config.max_abs_state).to_dict())
    return ExperimentData(graph, config, disturbances, runs)
