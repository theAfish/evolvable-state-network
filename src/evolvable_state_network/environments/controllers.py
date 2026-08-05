"""Small environment-only controller baselines."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Sequence

from .core import Action, Controller, ControllerBlueprint, Observation


class RandomController(Controller):
    def __init__(self, *, seed: int | None = None) -> None:
        self._random = Random(seed)

    def act(self, observation: Observation, *, available_actions: Sequence[Action]) -> Action:
        if not available_actions:
            raise ValueError("environment offered no actions")
        return self._random.choice(available_actions)


@dataclass(frozen=True, slots=True)
class RandomControllerBlueprint(ControllerBlueprint):
    def build(self, *, seed: int | None = None) -> Controller:
        return RandomController(seed=seed)
