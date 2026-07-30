"""Torch execution backend for the fixed MLP local-rule experiment family.

This module deliberately implements only the configured MLP node/edge rules.
Arbitrary user-defined rules continue through :mod:`simulation`, whose generic
Python protocol cannot be safely or meaningfully tensorized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from ..evolution.candidate import FixedEdgeRule, MLPEdgeRule, MLPUpdateRule
from ..graph import Graph
from ..inputs import InputProvider
from ..perturbations import (
    EdgeStateImpulse,
    ImpulseInjection,
    InputDistributionShift,
    NodeLesion,
    Perturbation,
    WeightNoise,
)
from .engine import (
    EventWindow,
    NetworkState,
    Simulation,
    SimulationConfig,
    SimulationResult,
    Trajectory,
    TransitionDiagnostics,
)

_DTYPE = torch.float32

# The evolutionary objective must be repeatable.  Tensor cores and atomic
# scatter reductions can change low-order bits between otherwise identical
# CUDA launches, so use reproducible dense reductions for this small fixed
# graph family.
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)


def cuda_available() -> bool:
    return torch.cuda.is_available()


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if cuda_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not cuda_available():
        raise RuntimeError("Torch CUDA backend was requested but no CUDA device is available")
    return device


@dataclass
class TorchMLPSimulator:
    """Synchronous tensor implementation of the standard MLP rule family."""

    graph: Graph
    node_rule: MLPUpdateRule
    edge_rule: MLPEdgeRule | FixedEdgeRule | None
    device: torch.device

    def __post_init__(self) -> None:
        if self.edge_rule is not None and self.edge_rule.architecture.node_state_width != self.node_rule.state_width:
            raise ValueError("node and edge rules must agree on node-state width")
        self.width = self.node_rule.state_width
        self.edge_width = self.edge_rule.state_width if self.edge_rule is not None else 0
        self.source = torch.tensor([edge.source for edge in self.graph.edges], dtype=torch.long, device=self.device)
        self.target = torch.tensor([edge.target for edge in self.graph.edges], dtype=torch.long, device=self.device)
        self.base_weight = torch.tensor([edge.weight for edge in self.graph.edges], dtype=_DTYPE, device=self.device)
        self.target_matrix = torch.zeros((len(self.graph.edges), self.graph.n_nodes), dtype=_DTYPE, device=self.device)
        if len(self.graph.edges):
            self.target_matrix[torch.arange(len(self.graph.edges), device=self.device), self.target] = 1.0
        self._node_parameters = self._parameters(self.node_rule.parameters, self.node_rule.architecture.hidden_width, self.node_rule.architecture.input_width, self.width)
        if self.edge_rule is not None:
            self._edge_parameters = self._parameters(self.edge_rule.parameters, self.edge_rule.architecture.hidden_width, self.edge_rule.architecture.input_width, self.edge_width)
            self.projection = torch.tensor(self.edge_rule.architecture.projection, dtype=_DTYPE, device=self.device)

    def _parameters(self, values: tuple[float, ...], hidden: int, inputs: int, outputs: int) -> tuple[torch.Tensor, ...]:
        flat = torch.tensor(values, dtype=_DTYPE, device=self.device)
        cursor = 0
        first = flat[cursor : cursor + hidden * inputs].reshape(hidden, inputs)
        cursor += hidden * inputs
        hidden_bias = flat[cursor : cursor + hidden]
        cursor += hidden
        second = flat[cursor : cursor + outputs * hidden].reshape(outputs, hidden)
        cursor += outputs * hidden
        return first, hidden_bias, second, flat[cursor : cursor + outputs]

    @staticmethod
    def _mlp(features: torch.Tensor, parameters: tuple[torch.Tensor, ...]) -> torch.Tensor:
        first, hidden_bias, second, output_bias = parameters
        return torch.tanh(torch.matmul(torch.tanh(torch.matmul(features, first.T) + hidden_bias), second.T) + output_bias)

    def run(
        self,
        config: SimulationConfig,
        provider: InputProvider,
        disturbances: Iterable[Perturbation],
        initial_node: list[list[tuple[float, ...]]],
        initial_edge: list[list[tuple[float, ...]]],
    ) -> SimulationResult:
        disturbances = tuple(disturbances)
        node = torch.tensor(initial_node, dtype=_DTYPE, device=self.device)
        edge = torch.tensor(initial_edge, dtype=_DTYPE, device=self.device)
        batch = config.batch_size
        diagnostics = TransitionDiagnostics()
        trajectory = Trajectory(events=Simulation._event_windows(disturbances, config.steps))
        node_records: list[torch.Tensor] = []
        edge_records: list[torch.Tensor] = []
        input_records: list[torch.Tensor] = []
        strength_records: list[torch.Tensor] = []
        steps: list[int] = []

        def record(step: int, external: torch.Tensor) -> None:
            steps.append(step)
            node_records.append(node.detach().clone())
            edge_records.append(edge.detach().clone())
            input_records.append(external.detach().clone())
            strength_records.append(self._strength(edge).detach().clone())

        external = self._external(provider, 0, batch, disturbances)
        record(0, external)
        for step in range(config.steps):
            external = self._external(provider, step, batch, disturbances)
            node, edge = self._step(node, edge, external, step, config, disturbances, diagnostics)
            if (step + 1) % config.record_every == 0 or step + 1 == config.steps:
                record(step + 1, external)

        # One bulk device-to-host transfer avoids synchronizing every frame.
        node_host = torch.stack(node_records).cpu().tolist()
        edge_host = torch.stack(edge_records).cpu().tolist()
        input_host = torch.stack(input_records).cpu().tolist()
        strength_host = torch.stack(strength_records).cpu().tolist()
        for index, step in enumerate(steps):
            trajectory.append(
                step, step * config.dt, node_host[index], input_host[index], edge_host[index], strength_host[index],
            )
        return SimulationResult(NetworkState(node=node.cpu().tolist(), edge=edge.cpu().tolist()), trajectory, diagnostics)

    def _external(self, provider: InputProvider, step: int, batch: int, disturbances: tuple[Perturbation, ...]) -> torch.Tensor:
        value = torch.tensor(provider.sample(step, batch, self.graph.n_nodes, self.width), dtype=_DTYPE, device=self.device)
        shifts = [item for item in disturbances if isinstance(item, InputDistributionShift) and item.active(step)]
        if shifts:
            scale = 1.0
            for shift in shifts:
                scale *= shift.scale
            value = value * scale + sum(shift.offset for shift in shifts)
        return value

    def _strength(self, edge: torch.Tensor) -> torch.Tensor:
        """Return the mean gate for scalar diagnostics and graph styling."""
        if self.edge_rule is None:
            return torch.ones(edge.shape[:2], dtype=_DTYPE, device=self.device)
        return self._gates(edge).mean(dim=-1)

    def _gates(self, edge: torch.Tensor) -> torch.Tensor:
        """Return one learned gate for every node-state coordinate."""
        if self.edge_rule is None:
            return torch.ones((*edge.shape[:2], self.width), dtype=_DTYPE, device=self.device)
        indices = (
            self.edge_rule.architecture.gate_index
            + torch.arange(self.width, device=self.device)
        ) % self.edge_width
        return .5 * (1.0 + torch.tanh(edge[..., indices]))

    def _step(
        self, node: torch.Tensor, edge: torch.Tensor, external: torch.Tensor, step: int,
        config: SimulationConfig, disturbances: tuple[Perturbation, ...], diagnostics: TransitionDiagnostics,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lesions = {node_index for item in disturbances if isinstance(item, NodeLesion) and item.active(step) for node_index in item.nodes}
        valid_edge = torch.ones(len(self.graph.edges), dtype=torch.bool, device=self.device)
        if lesions:
            for node_index in lesions:
                valid_edge &= (self.source != node_index) & (self.target != node_index)
        source_state = node[:, self.source] if len(self.graph.edges) else node[:, :0]
        target_state = node[:, self.target] if len(self.graph.edges) else node[:, :0]
        source_external = external[:, self.source] if len(self.graph.edges) else external[:, :0]
        target_external = external[:, self.target] if len(self.graph.edges) else external[:, :0]
        if self.edge_rule is None:
            messages = source_state
            next_edge = edge
        else:
            current_message = self._edge_message(edge, source_state)
            if isinstance(self.edge_rule, FixedEdgeRule):
                next_edge = edge
            else:
                features = torch.cat((edge, source_state, target_state, current_message, torch.ones((*edge.shape[:2], 1), dtype=_DTYPE, device=self.device)), dim=-1)
                next_edge = edge + config.edge_step_scale * (config.dt / .05) * self._mlp(features, self._edge_parameters)
            next_edge = torch.where(valid_edge[None, :, None], next_edge, edge)
            messages = self._edge_message(next_edge, source_state)
        messages = torch.where(valid_edge[None, :, None], messages, torch.zeros_like(messages))
        weight = self.base_weight[None, :].expand(node.shape[0], -1)
        for noise in (item for item in disturbances if isinstance(item, WeightNoise) and item.active(step)):
            noise_values = [[noise.sample(step, edge_index, batch_index) for edge_index in range(len(self.graph.edges))] for batch_index in range(node.shape[0])]
            weight = weight + torch.tensor(noise_values, dtype=_DTYPE, device=self.device)
        weighted = messages * weight[..., None]
        aggregate = torch.zeros_like(node)
        if len(self.graph.edges):
            target_matrix = self.target_matrix * valid_edge[:, None].to(_DTYPE)
            aggregate = torch.einsum("bew,en->bnw", weighted, target_matrix)
            counts = target_matrix.sum(dim=0)
            aggregate = aggregate / counts.clamp_min(1.0)[None, :, None]
        features = torch.cat((node, aggregate, torch.ones((*node.shape[:2], 1), dtype=_DTYPE, device=self.device)), dim=-1)
        proposed = node + config.max_delta * self.node_rule.architecture.increment_fraction * (config.dt / .05) * self._mlp(features, self._node_parameters)
        delta = proposed - node
        next_node = torch.clamp(node + torch.clamp(delta, -config.max_delta, config.max_delta), -config.max_abs_state, config.max_abs_state)
        if lesions:
            lesion_tensor = torch.tensor(sorted(lesions), dtype=torch.long, device=self.device)
            next_node[:, lesion_tensor] = 0.0
        for impulse in (item for item in disturbances if isinstance(item, ImpulseInjection) and item.active(step)):
            amount = (float(impulse.amount),) * self.width if isinstance(impulse.amount, (int, float)) else impulse.amount
            next_node[:, list(impulse.nodes)] = torch.clamp(next_node[:, list(impulse.nodes)] + torch.tensor(amount, dtype=_DTYPE, device=self.device), -config.max_abs_state, config.max_abs_state)
        for impulse in (item for item in disturbances if isinstance(item, EdgeStateImpulse) and item.active(step)):
            if self.edge_width:
                amount = (float(impulse.amount),) * self.edge_width if isinstance(impulse.amount, (int, float)) else impulse.amount
                next_edge[:, list(impulse.edges)] += torch.tensor(amount, dtype=_DTYPE, device=self.device)
        components = node.numel() + edge.numel()
        diagnostics.components += components
        diagnostics.components_per_step.append(components)
        diagnostics.clipped_components_per_step.append(0)
        return next_node, next_edge

    def _edge_message(self, edge: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        if self.edge_rule is None:
            return source
        projected = torch.matmul(source, self.projection.T)
        return projected * self._gates(edge)
