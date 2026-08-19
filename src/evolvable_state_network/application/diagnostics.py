"""Reusable numeric summaries and synthetic probes for rule diagnostics."""

from __future__ import annotations

from math import sqrt
from random import Random
from statistics import fmean, median, pstdev
from typing import Sequence

from ..evolution.candidate import EdgeArchitecture, RuleArchitecture, _forward
from ..evolution.genome import GenomeCodec


def interpolated_percentile(ordered_values: Sequence[float], fraction: float) -> float:
    """Linearly interpolate a percentile from an already sorted sequence."""
    position = fraction * (len(ordered_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered_values) - 1)
    return ordered_values[lower] + (
        ordered_values[upper] - ordered_values[lower]
    ) * (position - lower)


def distribution_summary(values: Sequence[float]) -> dict[str, float]:
    """Return stable JSON-friendly descriptive statistics for trial values."""
    ordered = sorted(values)
    if not ordered:
        return {}
    standard_deviation = pstdev(ordered)
    return {
        "mean": fmean(ordered),
        "standard_deviation": standard_deviation,
        "variance": standard_deviation**2,
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "median": median(ordered),
        "p10": interpolated_percentile(ordered, 0.10),
        "p25": interpolated_percentile(ordered, 0.25),
        "p75": interpolated_percentile(ordered, 0.75),
        "p90": interpolated_percentile(ordered, 0.90),
    }


def compare_vectors(
    left: Sequence[float], right: Sequence[float]
) -> dict[str, float | int | bool]:
    """Compare vectors without silently aligning incompatible dimensions."""
    if len(left) != len(right):
        return {
            "compatible": False,
            "left_dimension": len(left),
            "right_dimension": len(right),
        }
    squared_distance = sum(
        (left_value - right_value) ** 2
        for left_value, right_value in zip(left, right, strict=True)
    )
    l2_distance = sqrt(squared_distance)
    left_norm = sqrt(sum(value**2 for value in left))
    right_norm = sqrt(sum(value**2 for value in right))
    dimension_scale = sqrt(max(1, len(left)))
    return {
        "compatible": True,
        "left_dimension": len(left),
        "right_dimension": len(right),
        "l2_distance": l2_distance,
        "rms_distance": l2_distance / dimension_scale,
        "cosine_similarity": sum(
            left_value * right_value
            for left_value, right_value in zip(left, right, strict=True)
        )
        / max(left_norm * right_norm, 1e-12),
        "left_rms": left_norm / dimension_scale,
        "right_rms": right_norm / dimension_scale,
    }


def raw_output_summary(values: Sequence[float]) -> dict[str, float]:
    """Summarize signed rule outputs and their absolute saturation tails."""
    if not values:
        return {}
    absolute = tuple(abs(value) for value in values)
    ordered_absolute = sorted(absolute)
    return {
        "mean": fmean(values),
        "standard_deviation": pstdev(values),
        "rms": sqrt(fmean(value**2 for value in values)),
        "minimum": min(values),
        "maximum": max(values),
        "abs_p50": interpolated_percentile(ordered_absolute, 0.50),
        "abs_p90": interpolated_percentile(ordered_absolute, 0.90),
        "abs_p99": interpolated_percentile(ordered_absolute, 0.99),
        "abs_gt_1_fraction": sum(value > 1 for value in absolute) / len(absolute),
        "abs_gt_2_fraction": sum(value > 2 for value in absolute) / len(absolute),
        "abs_gt_3_fraction": sum(value > 3 for value in absolute) / len(absolute),
    }


def synthetic_rule_outputs(
    genome: Sequence[float],
    architecture: RuleArchitecture,
    edge_architecture: EdgeArchitecture,
    *,
    state_limit: float,
    probe_count: int,
    seed: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Evaluate raw node and edge MLP outputs on common bounded inputs.

    The probes bypass downstream ``tanh`` and update scaling. They are a quick
    saturation signal, not a replacement for traces from the learned policy's
    real state distribution.
    """
    node_rule, edge_rule = GenomeCodec(
        architecture, edge_architecture, "joint"
    ).decode_groups(genome)
    assert node_rule is not None and edge_rule is not None
    random = Random(seed)
    node_values: list[float] = []
    edge_values: list[float] = []
    for _ in range(probe_count):
        node_features = tuple(
            random.uniform(-state_limit, state_limit)
            for _ in range(2 * architecture.state_width)
        )
        node_values.extend(
            _forward(node_features, node_rule._layers, architecture.activation)
        )
        edge_features = (
            tuple(
                random.uniform(-1.0, 1.0)
                for _ in range(edge_architecture.latent_width)
            )
            + tuple(
                random.uniform(-state_limit, state_limit)
                for _ in range(3 * architecture.state_width)
            )
        )
        edge_values.extend(
            _forward(edge_features, edge_rule._layers, edge_architecture.activation)
        )
    return tuple(node_values), tuple(edge_values)
