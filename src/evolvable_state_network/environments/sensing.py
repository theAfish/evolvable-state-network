"""Continuous-space geometry and vectorised ray-fan sensing primitives."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Iterable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Vec2:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class Viewport:
    range: float = 18.0
    field_of_view: float = pi * 0.8
    ray_count: int = 9

    def __post_init__(self) -> None:
        if self.range <= 0 or not 0 < self.field_of_view <= 2 * pi or self.ray_count < 1:
            raise ValueError("viewport requires positive range, valid FOV, and at least one ray")

    def ray_angles(self, heading: float) -> tuple[float, ...]:
        if self.ray_count == 1:
            return (heading,)
        return tuple(float(angle) for angle in np.linspace(
            heading - self.field_of_view / 2, heading + self.field_of_view / 2, self.ray_count
        ))


@dataclass(frozen=True, slots=True)
class SenseTarget:
    id: str
    kind: str
    position: Vec2
    radius: float


@dataclass(frozen=True, slots=True)
class RayHit:
    angle: float
    distance: float | None
    kind: str | None
    target_id: str | None


def scan_ray_fan(
    origin: Vec2, viewport: Viewport, heading: float, targets: Iterable[SenseTarget],
    *, periodic_bounds: tuple[float, float] | None = None,
) -> tuple[RayHit, ...]:
    """Return the nearest circular target intersected by every finite ray."""
    target_list = tuple(targets)
    angles = np.asarray(viewport.ray_angles(heading), dtype=np.float64)
    if not target_list:
        return tuple(RayHit(float(angle), None, None, None) for angle in angles)
    positions: NDArray[np.float64] = np.asarray([(target.position.x, target.position.y) for target in target_list])
    radii: NDArray[np.float64] = np.asarray([target.radius for target in target_list])
    target_indices = np.arange(len(target_list))
    if periodic_bounds is not None:
        width, height = periodic_bounds
        if width <= 0 or height <= 0:
            raise ValueError("periodic bounds must be positive")
        x_tiles, y_tiles = int(np.ceil(viewport.range / width)) + 1, int(np.ceil(viewport.range / height)) + 1
        translations = np.asarray([(x * width, y * height) for x in range(-x_tiles, x_tiles + 1) for y in range(-y_tiles, y_tiles + 1)])
        positions = (positions[None, :, :] + translations[:, None, :]).reshape(-1, 2)
        radii, target_indices = np.tile(radii, len(translations)), np.tile(target_indices, len(translations))
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    offsets = positions[None, :, :] - np.asarray((origin.x, origin.y))[None, None, :]
    projected = np.einsum("rnk,rk->rn", offsets, directions)
    perpendicular_squared = np.einsum("rnk,rnk->rn", offsets, offsets) - projected**2
    root = np.sqrt(np.maximum(radii[None, :] ** 2 - perpendicular_squared, 0.0))
    entry, exit_distance = projected - root, projected + root
    distances = np.where(entry >= 0.0, entry, exit_distance)
    visible = (radii[None, :] ** 2 >= perpendicular_squared) & (distances >= 0.0) & (distances <= viewport.range)
    candidates = np.where(visible, distances, np.inf)
    nearest_indices = np.argmin(candidates, axis=1)
    nearest_distances = candidates[np.arange(len(angles)), nearest_indices]
    return tuple(RayHit(float(angle), None if np.isinf(distance) else float(distance),
                        None if np.isinf(distance) else target_list[index].kind,
                        None if np.isinf(distance) else target_list[index].id)
                 for angle, index, distance in zip(angles, target_indices[nearest_indices], nearest_distances, strict=True))
