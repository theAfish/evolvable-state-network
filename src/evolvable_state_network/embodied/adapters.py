"""Environment-specific normalisation at the boundary of a generic network."""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import cos, sin
from typing import Sequence

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

    @abstractmethod
    def encode_observation(self, observation: Observation) -> tuple[float, ...]:
        """Return exactly ``input_count`` bounded input-node values."""

    @abstractmethod
    def decode_action(self, outputs: Sequence[float]) -> Action:
        """Map exactly ``action_count`` bounded network outputs to an action."""


class FoodWebAgentAdapter(AgentAdapter):
    """Food-web body plus an ordered, ego-relative one-dimensional ray image."""

    # Four body channels followed by three proximity channels per ray pixel:
    # plant, prey, predator.  Pixel order is left-to-right in the agent's
    # current field of view, so no absolute or relative angle scalar is needed.
    input_count = 31
    # turn in [-1, 1], throttle translated from [-1, 1] to [0, 1].
    action_count = 2

    def __init__(
        self, *, vision_pixels: int = 9, schema: str = "ray_image_v3",
        directional: bool = True,
    ) -> None:
        if vision_pixels < 1:
            raise ValueError("vision_pixels must be positive")
        if schema not in {"ray_image_v3", "body_v2"}:
            raise ValueError(f"unknown food-web observation schema: {schema}")
        self.vision_pixels, self.schema = vision_pixels, schema
        self.directional = directional
        # body_v2 is retained only so old saved 7/13-channel controllers can
        # still be demonstrated. New evolution always uses ray_image_v3.
        self.input_count = 4 + 3 * vision_pixels if schema == "ray_image_v3" else (13 if directional else 7)

    def encode_observation(self, observation: Observation) -> tuple[float, ...]:
        fallback_hunger = 1.0 - float(observation.get("energy", 0.0)) / 9.0
        hunger = bounded(2.0 * float(observation.get("hunger", fallback_hunger)) - 1.0)
        energy_change = bounded(float(observation.get("energy_change", 0.0)))
        ate = 1.0 if bool(observation.get("ate", False)) else -1.0
        time_since_meal = bounded(2.0 * float(observation.get("time_since_meal", 0.0)) - 1.0)
        body = (hunger, energy_change, ate, time_since_meal)
        if self.schema == "ray_image_v3":
            return body + self._ray_image(observation)

        kinds = {"plant": (0.0, 0.0, 0.0), "prey": (0.0, 0.0, 0.0), "predator": (0.0, 0.0, 0.0)}
        for ray in observation.get("vision", ()):
            distance, ray_range = ray.get("distance"), float(ray.get("range", 1.0))
            if distance is None or ray_range <= 0:
                continue
            kind = str(ray.get("kind"))
            proximity = bounded(1.0 - float(distance) / ray_range)
            if kind in kinds and proximity > kinds[kind][0]:
                angle = float(ray.get("angle", 0.0))
                kinds[kind] = (proximity, proximity * bounded(sin(angle)), proximity * bounded(cos(angle)))
        if not self.directional:
            return body + (kinds["plant"][0], kinds["prey"][0], kinds["predator"][0])
        return body + (*kinds["plant"], *kinds["prey"], *kinds["predator"])

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
