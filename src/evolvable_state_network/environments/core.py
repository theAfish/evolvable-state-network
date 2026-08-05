"""Small, optimizer-free contracts for embodied environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, NewType, Sequence

AgentId = NewType("AgentId", str)
Observation = Mapping[str, Any]
Action = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Transition:
    """Experience emitted by an episode runner after an environment step."""

    observation: Observation
    action: Action
    reward: float
    next_observation: Observation
    terminated: bool
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StepResult:
    observations: Mapping[AgentId, Observation]
    rewards: Mapping[AgentId, float]
    terminated: bool
    alive: frozenset[AgentId]
    info: Mapping[str, Any] = field(default_factory=dict)


class Environment(ABC):
    """A simultaneous-action episodic world; it never owns an optimizer."""

    @abstractmethod
    def reset(self, *, seed: int | None = None) -> Mapping[AgentId, Observation]:
        """Reset the world and return observations for its living agents."""

    @abstractmethod
    def step(self, actions: Mapping[AgentId, Action]) -> StepResult:
        """Advance one tick; absent actions must have deterministic semantics."""


class Controller(ABC):
    """Episode-local policy, optionally with within-lifetime learning."""

    @abstractmethod
    def act(self, observation: Observation, *, available_actions: Sequence[Action]) -> Action:
        """Choose an environment action."""

    def begin_episode(self, *, seed: int | None = None) -> None:
        pass

    def learn(self, transition: Transition) -> None:
        pass

    def end_episode(self) -> None:
        pass


class ControllerBlueprint(ABC):
    """A serialisable/heritable recipe that creates independent controllers."""

    @abstractmethod
    def build(self, *, seed: int | None = None) -> Controller:
        """Create one episode-local controller."""
