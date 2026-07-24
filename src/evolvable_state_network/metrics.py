"""Metrics for stability and response of generic state-vector trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from statistics import fmean

from .simulation import EventWindow, Trajectory


@dataclass(frozen=True, slots=True)
class Boundedness:
    finite: bool
    maximum_absolute_value: float
    bounded: bool


@dataclass(frozen=True, slots=True)
class NonSilence:
    rms: float
    non_silent: bool


@dataclass(frozen=True, slots=True)
class Saturation:
    fraction: float
    saturated: bool


@dataclass(frozen=True, slots=True)
class ActivityDiversity:
    node_time_mean_std: float
    diverse: bool


@dataclass(frozen=True, slots=True)
class PerturbationResponse:
    magnitude: float
    responsive: bool
    measured_events: int


@dataclass(frozen=True, slots=True)
class Recovery:
    error: float
    recovered: bool
    measured_events: int


@dataclass(frozen=True, slots=True)
class MetricReport:
    boundedness: Boundedness
    non_silence: NonSilence
    saturation: Saturation
    activity_diversity: ActivityDiversity
    perturbation_response: PerturbationResponse
    recovery: Recovery

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_metrics(
    trajectory: Trajectory,
    *,
    safety_bound: float = 4.0,
    silence_rms: float = 1e-3,
    saturation_level: float = 0.95,
    diversity_threshold: float = 0.02,
    response_threshold: float = 0.02,
) -> MetricReport:
    """Measure the first coordinate consistently across arbitrary rule widths.

    This is an observation convention, not an interpretation of that coordinate.
    It keeps early experiments comparable; later work can register other readout
    functions without changing the simulator.
    """
    if not trajectory.node_states:
        raise ValueError("cannot evaluate an empty trajectory")
    values_by_frame = [_first_coordinates(snapshot) for snapshot in trajectory.node_states]
    values = [value for frame in values_by_frame for value in frame]
    finite = all(isfinite(value) for value in values)
    maximum = max((abs(value) for value in values if isfinite(value)), default=float("inf"))
    boundedness = Boundedness(finite, maximum, finite and maximum <= safety_bound)
    rms = sqrt(fmean(value * value for value in values if isfinite(value))) if finite else float("inf")
    non_silence = NonSilence(rms, finite and rms > silence_rms)
    saturation_fraction = (
        sum(abs(value) >= saturation_level for value in values if isfinite(value)) / len(values) if finite else 1.0
    )
    saturation = Saturation(saturation_fraction, saturation_fraction >= 0.5)
    node_means = _node_time_means(trajectory)
    diversity = _population_std(node_means)
    activity_diversity = ActivityDiversity(diversity, diversity >= diversity_threshold)
    response, recovery = _event_metrics(trajectory, values_by_frame, response_threshold)
    return MetricReport(boundedness, non_silence, saturation, activity_diversity, response, recovery)


def _first_coordinates(snapshot: list[list[tuple[float, ...]]]) -> list[float]:
    return [vector[0] for batch in snapshot for vector in batch]


def _node_time_means(trajectory: Trajectory) -> list[float]:
    batch_count = len(trajectory.node_states[0])
    node_count = len(trajectory.node_states[0][0])
    return [
        fmean(snapshot[batch][node][0] for snapshot in trajectory.node_states)
        for batch in range(batch_count)
        for node in range(node_count)
    ]


def _population_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = fmean(values)
    return sqrt(fmean((value - mean) ** 2 for value in values))


def _event_metrics(
    trajectory: Trajectory, frames: list[list[float]], response_threshold: float
) -> tuple[PerturbationResponse, Recovery]:
    signal = [fmean(abs(value) for value in frame) for frame in frames]
    magnitudes: list[float] = []
    recovery_errors: list[float] = []
    for event in trajectory.events:
        pre = [signal[index] for index, step in enumerate(trajectory.steps) if step <= event.start]
        active = [signal[index] for index, step in enumerate(trajectory.steps) if event.start < step <= event.end + 1]
        post = [signal[index] for index, step in enumerate(trajectory.steps) if step > event.end + 1]
        if not pre or not active:
            continue
        reference = fmean(pre)
        magnitude = max(abs(value - reference) for value in active)
        magnitudes.append(magnitude)
        if post:
            recovery_errors.append(abs(fmean(post[-min(10, len(post)) :]) - reference))
    response_value = fmean(magnitudes) if magnitudes else 0.0
    recovery_error = fmean(recovery_errors) if recovery_errors else 0.0
    tolerance = max(0.05, response_value * 0.5)
    return (
        PerturbationResponse(response_value, bool(magnitudes) and response_value >= response_threshold, len(magnitudes)),
        Recovery(recovery_error, not recovery_errors or recovery_error <= tolerance, len(recovery_errors)),
    )
