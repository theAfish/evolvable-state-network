"""Random embodied networks whose only inherited state is their local rules."""

from __future__ import annotations

from dataclasses import dataclass
from math import tanh
from random import Random

from ..graph import Edge, Graph, generate_random_graph
from ..rules import EdgeRule, NodeRule
from ..simulation import NetworkState, Simulation, SimulationConfig, TransitionDiagnostics
from ..types import zeros
from .adapters import AgentAdapter, bounded


@dataclass(frozen=True, slots=True)
class EmbodiedNetworkConfig:
    """Topology and interface layout for one newly born agent network."""

    nodes: int = 24
    mean_degree: float = 4.0
    state_width: int = 2
    initial_state_scale: float = .12
    dt: float = .05
    max_delta: float = .12
    max_abs_state: float = 4.0
    edge_step_scale: float = .06
    observation_schema: str = "ray_image_v3"
    vision_pixels: int = 9

    def __post_init__(self) -> None:
        if self.nodes < 3 or self.mean_degree < 0 or self.state_width < 1 or self.vision_pixels < 1:
            raise ValueError("network shape is invalid")
        if self.observation_schema not in {"ray_image_v3", "body_v2"}:
            raise ValueError("unknown embodied observation schema")
        if self.initial_state_scale < 0 or self.dt <= 0 or self.max_delta <= 0 or self.max_abs_state <= 0:
            raise ValueError("network integration parameters are invalid")


@dataclass(frozen=True, slots=True)
class NetworkInterface:
    """Fixed boundary-node allocation; all remaining nodes stay anonymous."""

    input_nodes: tuple[int, ...]
    action_nodes: tuple[int, ...]
    input_tags: tuple[float, ...]
    action_tags: tuple[float, ...]

    @classmethod
    def allocate(cls, nodes: int, adapter: AgentAdapter) -> "NetworkInterface":
        required = adapter.input_count + adapter.action_count
        if nodes < required:
            raise ValueError(f"network needs at least {required} nodes for this agent adapter")
        tags = tuple(-1.0 + 2.0 * (index + 1) / (required + 1) for index in range(required))
        return cls(
            tuple(range(adapter.input_count)),
            tuple(range(adapter.input_count, required)),
            tags[:adapter.input_count],
            tags[adapter.input_count:],
        )


def generate_embodied_graph(config: EmbodiedNetworkConfig, interface: NetworkInterface, seed: int) -> Graph:
    """Sample a graph while guaranteeing every boundary node participates."""
    base = generate_random_graph(config.nodes, config.mean_degree, seed)
    pairs = {(edge.source, edge.target) for edge in base.edges}
    random = Random(seed + 7_919)
    for node in interface.input_nodes:
        if not any(source == node for source, _ in pairs):
            target = random.randrange(config.nodes - 1)
            pairs.add((node, target if target < node else target + 1))
    for node in interface.action_nodes:
        if not any(target == node for _, target in pairs):
            source = random.randrange(config.nodes - 1)
            pairs.add((source if source < node else source + 1, node))
    return Graph(config.nodes, tuple(Edge(source, target, 1.0) for source, target in sorted(pairs)))


class EmbodiedNetwork:
    """One random graph/initial state controlled by shared node and edge rules."""

    def __init__(self, node_rule: NodeRule, edge_rule: EdgeRule, adapter: AgentAdapter, config: EmbodiedNetworkConfig, *, seed: int) -> None:
        if node_rule.state_width != config.state_width:
            raise ValueError("node rule width must match embodied network configuration")
        self.adapter, self.config = adapter, config
        self.interface = NetworkInterface.allocate(config.nodes, adapter)
        self.graph = generate_embodied_graph(config, self.interface, seed)
        self._simulation = Simulation(self.graph, node_rule, edge_rule)
        self._integration = SimulationConfig(steps=1, dt=config.dt, batch_size=1, max_delta=config.max_delta, max_abs_state=config.max_abs_state, edge_step_scale=config.edge_step_scale)
        self.diagnostics = TransitionDiagnostics()
        self._step = 0
        self.state = self._random_initial_state(seed)

    def _random_initial_state(self, seed: int) -> NetworkState:
        random = Random(seed + 1_093)
        node = [[tuple(random.gauss(0.0, self.config.initial_state_scale) for _ in range(self.config.state_width)) for _ in range(self.config.nodes)]]
        # Actuator nodes are interface buffers, not anonymous recurrent tissue.
        # Giving their readout coordinate a random initial value creates a
        # constant motor command before the rule has processed any sensation:
        # random turn + positive throttle is a circular-motion prior.  Keep all
        # non-actuator state random, but start motors at the neutral command.
        for action_node in self.interface.action_nodes:
            node[0][action_node] = zeros(self.config.state_width)
        return NetworkState(node=node, edge=self._simulation.initial_state(1).edge)

    def act(self, observation: object) -> dict[str, object]:
        """Inject one observation, run one synchronous graph tick, emit action."""
        values = self.adapter.encode_observation(observation)  # type: ignore[arg-type]
        if len(values) != len(self.interface.input_nodes):
            raise ValueError("adapter returned the wrong number of input channels")
        for node, value, tag in zip(self.interface.input_nodes, values, self.interface.input_tags, strict=True):
            vector = [0.0] * self.config.state_width
            vector[0] = bounded(value)
            if self.config.state_width > 1:
                vector[-1] = tag
            self.state.node[0][node] = tuple(vector)
        # Action nodes keep their recurrent output coordinate.  Only the last
        # coordinate is refreshed with an immutable actuator-role marker, so
        # the shared rule can distinguish turn from throttle without inheriting
        # a topology or a conventional learned readout.
        if self.config.state_width > 1:
            for node, tag in zip(self.interface.action_nodes, self.interface.action_tags, strict=True):
                vector = list(self.state.node[0][node])
                vector[-1] = tag
                self.state.node[0][node] = tuple(vector)
        external = [[zeros(self.config.state_width) for _ in range(self.config.nodes)]]
        self.state = self._simulation._step(self.state, external, self._step, self._integration, (), self.diagnostics, None)
        self._step += 1
        outputs = tuple(tanh(self.state.node[0][node][0]) for node in self.interface.action_nodes)
        return dict(self.adapter.decode_action(outputs))
