"""Environment-specific normalisation at the boundary of a generic network."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Sequence

from ..environments import Action, Observation


def bounded(value: float) -> float:
    """Normalise a scalar contractually represented in ``[-1, 1]``."""
    return max(-1.0, min(1.0, float(value)))


class AgentAdapter(ABC):
    """Maps one environment agent's observations/actions to unnamed channels.

    This is intentionally the only place where node positions receive semantic
    meaning.  Local node and edge update rules never receive these labels.
    """

    @property
    @abstractmethod
    def input_count(self) -> int: ...

    @property
    @abstractmethod
    def action_count(self) -> int: ...

    @property
    @abstractmethod
    def input_signal_channels(self) -> tuple[int, ...]:
        """Chemical-state coordinate assigned to each sensory port."""

    @abstractmethod
    def encode_observation(self, observation: Observation) -> tuple[float, ...]:
        """Return exactly ``input_count`` bounded input-node values."""

    @abstractmethod
    def decode_action(self, outputs: Sequence[float]) -> Action:
        """Map exactly ``action_count`` bounded network outputs to an action."""


class FoodWebAgentAdapter(AgentAdapter):
    """Food-web body plus an ordered, ego-relative one-dimensional ray image."""

    # Selected body channels followed by three proximity channels per ray pixel:
    # plant, prey, predator.  Pixel order is left-to-right in the agent's
    # current field of view, so no absolute or relative angle scalar is needed.
    BODY_INPUTS = ("hunger", "energy_change", "ate", "time_since_meal")
    # turn in [-1, 1], throttle translated from [-1, 1] to [0, 1].
    action_count = 2

    def __init__(
        self,
        *,
        vision_pixels: int = 9,
        body_inputs: Sequence[Literal["hunger", "energy_change", "ate", "time_since_meal"]] = ("hunger",),
    ) -> None:
        if vision_pixels < 1:
            raise ValueError("vision_pixels must be positive")
        if not body_inputs or any(item not in self.BODY_INPUTS for item in body_inputs) or len(set(body_inputs)) != len(body_inputs):
            raise ValueError("body_inputs must be a non-empty selection of unique known body channels")
        self.vision_pixels = vision_pixels
        self.body_inputs = tuple(body_inputs)

    @property
    def input_count(self) -> int:
        return len(self.body_inputs) + 3 * self.vision_pixels

    @property
    def input_signal_channels(self) -> tuple[int, ...]:
        # Interoception uses a distinct chemical signal from exteroceptive ray
        # vision.  The port remains sparse: only this coordinate is nonzero.
        return (1,) * len(self.body_inputs) + (0,) * (3 * self.vision_pixels)

    def encode_observation(self, observation: Observation) -> tuple[float, ...]:
        fallback_hunger = 1.0 - float(observation.get("energy", 0.0)) / 9.0
        hunger = bounded(2.0 * float(observation.get("hunger", fallback_hunger)) - 1.0)
        energy_change = bounded(float(observation.get("energy_change", 0.0)))
        ate = 1.0 if bool(observation.get("ate", False)) else -1.0
        time_since_meal = bounded(2.0 * float(observation.get("time_since_meal", 0.0)) - 1.0)
        values = {"hunger": hunger, "energy_change": energy_change, "ate": ate, "time_since_meal": time_since_meal}
        body = tuple(values[name] for name in self.body_inputs)
        return body + self._ray_image(observation)

    def _ray_image(self, observation: Observation) -> tuple[float, ...]:
        rays = tuple(observation.get("vision", ()))
        image = [0.0] * (3 * self.vision_pixels)
        channels = {"plant": 0, "prey": 1, "predator": 2}
        for source_index, ray in enumerate(rays):
            distance, ray_range = ray.get("distance"), float(ray.get("range", 1.0))
            kind = str(ray.get("kind"))
            if distance is None or ray_range <= 0 or kind not in channels:
                continue
            # Ray order, not RayHit.angle, defines ego-relative pixel position.
            pixel = 0 if len(rays) <= 1 else round(source_index * (self.vision_pixels - 1) / (len(rays) - 1))
            offset = 3 * pixel + channels[kind]
            image[offset] = max(image[offset], bounded(1.0 - float(distance) / ray_range))
        return tuple(image)

    def decode_action(self, outputs: Sequence[float]) -> Action:
        if len(outputs) != self.action_count:
            raise ValueError("food-web adapter needs turn and throttle outputs")
        return {"kind": "turn_move", "turn": bounded(outputs[0]), "speed": (bounded(outputs[1]) + 1.0) / 2.0}
