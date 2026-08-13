"""Random embodied networks whose only inherited state is their local rules."""

from __future__ import annotations

from dataclasses import dataclass
from math import tanh
from random import Random
from typing import Literal

import torch

from ..graph import Edge, Graph
from ..rules import EdgeRule, NodeRule
from ..simulation import NetworkState, Simulation, SimulationConfig, TransitionDiagnostics
from ..simulation.torch_backend import TorchMLPSimulator, resolve_device
from ..types import zeros
from ..evolution.candidate import MLPEdgeRule, MLPUpdateRule
from .adapters import AgentAdapter, bounded


@dataclass(frozen=True, slots=True)
class EmbodiedNetworkConfig:
    """Topology and interface layout for one newly born agent network."""

    nodes: int = 34
    mean_degree: float = 4.0
    state_width: int = 2
    initial_state_scale: float = .12
    dt: float = .05
    max_delta: float = .12
    max_abs_state: float = 4.0
    edge_step_scale: float = .06
    vision_pixels: int = 9
    execution_backend: Literal["python", "torch"] = "python"
    device: Literal["auto", "cpu", "cuda"] = "cpu"

    def __post_init__(self) -> None:
        if self.nodes < 3 or self.mean_degree < 0 or self.state_width < 1 or self.vision_pixels < 1:
            raise ValueError("network shape is invalid")
        if self.execution_backend not in {"python", "torch"} or self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("unknown embodied execution backend or device")
        if self.execution_backend == "python" and self.device == "cuda":
            raise ValueError("the reference Python backend cannot run on CUDA")
        if self.initial_state_scale < 0 or self.dt <= 0 or self.max_delta <= 0 or self.max_abs_state <= 0:
            raise ValueError("network integration parameters are invalid")


@dataclass(frozen=True, slots=True)
class NetworkInterface:
    """Fixed boundary-node allocation; all remaining nodes stay anonymous."""

    input_nodes: tuple[int, ...]
    action_nodes: tuple[int, ...]

    @classmethod
    def allocate(cls, nodes: int, adapter: AgentAdapter) -> "NetworkInterface":
        required = adapter.input_count + adapter.action_count
        if nodes < required:
            raise ValueError(f"network needs at least {required} nodes for this agent adapter")
        return cls(
            tuple(range(adapter.input_count)),
            tuple(range(adapter.input_count, required)),
        )


def generate_embodied_graph(config: EmbodiedNetworkConfig, interface: NetworkInterface, seed: int) -> Graph:
    """Sample a directed sensory → recurrent → action graph.

    Boundary nodes are ports, not recurrent tissue: sensory nodes only emit,
    action nodes only receive, and every path crosses at least one anonymous
    recurrent node.  This intentionally rules out input-input, output-output,
    output-input, and direct input-output connections.
    """
    inputs, actions = set(interface.input_nodes), set(interface.action_nodes)
    hidden = tuple(node for node in range(config.nodes) if node not in inputs | actions)
    if not hidden:
        raise ValueError("embodied network needs at least one hidden node between inputs and actions")
    candidates = [
        (source, target)
        for source in (*interface.input_nodes, *hidden)
        for target in (*hidden, *interface.action_nodes)
        if source != target
    ]
    random = Random(seed + 7_919)
    probability = min(1.0, config.mean_degree * config.nodes / max(1, len(candidates)))
    pairs = {pair for pair in candidates if random.random() < probability}
    for node in interface.input_nodes:
        if not any(source == node for source, _ in pairs):
            pairs.add((node, random.choice(hidden)))
    for node in interface.action_nodes:
        if not any(target == node for _, target in pairs):
            pairs.add((random.choice(hidden), node))
    return Graph(config.nodes, tuple(Edge(source, target, 1.0) for source, target in sorted(pairs)))


class EmbodiedNetwork:
    """One random graph/initial state controlled by shared node and edge rules."""

    def __init__(self, node_rule: NodeRule, edge_rule: EdgeRule, adapter: AgentAdapter, config: EmbodiedNetworkConfig, *, seed: int) -> None:
        if node_rule.state_width != config.state_width:
            raise ValueError("node rule width must match embodied network configuration")
        if max(adapter.input_signal_channels, default=0) >= config.state_width:
            raise ValueError("node state width cannot represent the adapter's sensory signal channels")
        self.adapter, self.config = adapter, config
        self.interface = NetworkInterface.allocate(config.nodes, adapter)
        self.graph = generate_embodied_graph(config, self.interface, seed)
        self._simulation = Simulation(self.graph, node_rule, edge_rule)
        self._integration = SimulationConfig(steps=1, dt=config.dt, batch_size=1, max_delta=config.max_delta, max_abs_state=config.max_abs_state, edge_step_scale=config.edge_step_scale)
        self.diagnostics = TransitionDiagnostics()
        self._step = 0
        self.state = self._random_initial_state(seed)
        self._torch_simulator: TorchMLPSimulator | None = None
        self._torch_node: torch.Tensor | None = None
        self._torch_edge: torch.Tensor | None = None
        self._torch_input_nodes: torch.Tensor | None = None
        self._torch_action_nodes: torch.Tensor | None = None
        if config.execution_backend == "torch":
            if not isinstance(node_rule, MLPUpdateRule) or not isinstance(edge_rule, MLPEdgeRule):
                raise TypeError("the Torch embodied backend requires the standard MLP node and edge rules")
            self._torch_simulator = TorchMLPSimulator(self.graph, node_rule, edge_rule, resolve_device(config.device))
            self._torch_node = torch.tensor(self.state.node, dtype=torch.float32, device=self._torch_simulator.device)
            self._torch_edge = torch.tensor(self.state.edge, dtype=torch.float32, device=self._torch_simulator.device)
            self._torch_input_nodes = torch.tensor(self.interface.input_nodes, dtype=torch.long, device=self._torch_simulator.device)
            self._torch_action_nodes = torch.tensor(self.interface.action_nodes, dtype=torch.long, device=self._torch_simulator.device)

    def _random_initial_state(self, seed: int) -> NetworkState:
        random = Random(seed + 1_093)
        node = [[tuple(random.gauss(0.0, self.config.initial_state_scale) for _ in range(self.config.state_width)) for _ in range(self.config.nodes)]]
        # Boundary nodes are ports rather than persistent recurrent tissue.
        # Inputs are overwritten from the body every tick and actions start at
        # their neutral readout; all unused chemical coordinates stay zero.
        for boundary_node in (*self.interface.input_nodes, *self.interface.action_nodes):
            node[0][boundary_node] = zeros(self.config.state_width)
        return NetworkState(node=node, edge=self._simulation.initial_state(1).edge)

    def inspection_snapshot(self) -> dict[str, object]:
        """Return the current network state in a small JSON-ready form.

        This is intentionally an observation-only view for the ecology demo;
        it never exposes a controller mutation path.
        """
        if self._torch_node is not None and self._torch_edge is not None:
            node_state = self._torch_node[0].detach().cpu().tolist()
            edge_state = self._torch_edge[0].detach().cpu().tolist()
        else:
            node_state = [list(vector) for vector in self.state.node[0]]
            edge_state = [list(vector) for vector in self.state.edge[0]]
        return {
            "step": self._step,
            "state_width": self.config.state_width,
            "nodes": self.config.nodes,
            "vision_pixels": self.config.vision_pixels,
            "input_signal_channels": list(self.adapter.input_signal_channels),
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "state": edge_state[index],
                    "communication_strength": self._simulation.edge_rule.communication_strength(
                        tuple(edge_state[index])
                    ),
                }
                for index, edge in enumerate(self.graph.edges)
            ],
            "node_state": node_state,
            "input_nodes": list(self.interface.input_nodes),
            "action_nodes": list(self.interface.action_nodes),
        }

    def act(self, observation: object) -> dict[str, object]:
        """Inject one observation, run one synchronous graph tick, emit action."""
        values = self.adapter.encode_observation(observation)  # type: ignore[arg-type]
        if len(values) != len(self.interface.input_nodes):
            raise ValueError("adapter returned the wrong number of input channels")
        if self._torch_simulator is not None:
            return self._act_torch(values)
        for node, value, channel in zip(
            self.interface.input_nodes, values, self.adapter.input_signal_channels, strict=True
        ):
            vector = [0.0] * self.config.state_width
            vector[channel] = bounded(value)
            self.state.node[0][node] = tuple(vector)
        external = [[zeros(self.config.state_width) for _ in range(self.config.nodes)]]
        self.state = self._simulation._step(self.state, external, self._step, self._integration, (), self.diagnostics, None)
        # Input values are pure current sensations, not recurrent state; action
        # ports retain their readout coordinate only.  All other channels are
        # zeroed on every tick in both kinds of boundary node.
        for node, value, channel in zip(
            self.interface.input_nodes, values, self.adapter.input_signal_channels, strict=True
        ):
            vector = [0.0] * self.config.state_width
            vector[channel] = bounded(value)
            self.state.node[0][node] = tuple(vector)
        for node in self.interface.action_nodes:
            self.state.node[0][node] = (self.state.node[0][node][0],) + (0.0,) * (self.config.state_width - 1)
        self._step += 1
        outputs = tuple(tanh(self.state.node[0][node][0]) for node in self.interface.action_nodes)
        return dict(self.adapter.decode_action(outputs))

    def _act_torch(self, values: tuple[float, ...]) -> dict[str, object]:
        """Advance a persistent tensor state without trajectory construction or host copies."""
        assert (
            self._torch_simulator is not None and self._torch_node is not None and self._torch_edge is not None
            and self._torch_input_nodes is not None and self._torch_action_nodes is not None
        )
        device = self._torch_simulator.device
        with torch.inference_mode():
            vectors = torch.zeros((len(values), self.config.state_width), dtype=torch.float32, device=device)
            vectors[
                torch.arange(len(values), device=device),
                torch.tensor(self.adapter.input_signal_channels, dtype=torch.long, device=device),
            ] = torch.tensor(values, dtype=torch.float32, device=device).clamp(-1.0, 1.0)
            self._torch_node[0, self._torch_input_nodes] = vectors
            external = torch.zeros_like(self._torch_node)
            self._torch_node, self._torch_edge = self._torch_simulator.step_state(
                self._torch_node, self._torch_edge, external, self._step, self._integration, self.diagnostics,
            )
            self._torch_node[0, self._torch_input_nodes] = vectors
            if self.config.state_width > 1:
                self._torch_node[0, self._torch_action_nodes, 1:] = 0.0
            self._step += 1
            outputs = torch.tanh(
                self._torch_node[0, self._torch_action_nodes, 0]
            ).cpu().tolist()
        return dict(self.adapter.decode_action(tuple(float(value) for value in outputs)))
