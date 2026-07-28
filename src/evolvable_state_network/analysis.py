"""Dependency-free analysis summaries and SVG plots for viability experiments."""

from __future__ import annotations

import json
from html import escape
from math import sqrt
from pathlib import Path
from statistics import fmean
from typing import Callable, Iterable, Sequence

from .evolution.evaluation import EvaluationResult, ScenarioResult
from .simulation import Trajectory


def trajectory_plot_data(trajectory: Trajectory, coordinate: int = 0, batch: int = 0) -> dict[str, object]:
    return {"times": trajectory.times, "series": [[frame[batch][node][coordinate] for frame in trajectory.node_states] for node in range(len(trajectory.node_states[0][batch]))]}


def state_distribution(trajectory: Trajectory, coordinate: int = 0) -> dict[str, object]:
    values = [vector[coordinate] for frame in trajectory.node_states for batch in frame for vector in batch]
    return {"count": len(values), "mean": fmean(values), "standard_deviation": _std(values), "minimum": min(values), "maximum": max(values)}


def node_correlation_matrix(trajectory: Trajectory, coordinate: int = 0, batch: int = 0) -> list[list[float]]:
    series = [[frame[batch][node][coordinate] for frame in trajectory.node_states] for node in range(len(trajectory.node_states[0][batch]))]
    return [[_correlation(left, right) for right in series] for left in series]


def perturbation_recovery_curve(trajectory: Trajectory, coordinate: int = 0) -> list[dict[str, float | str]]:
    signal = [fmean(abs(vector[coordinate]) for batch in frame for vector in batch) for frame in trajectory.node_states]
    output: list[dict[str, float | str]] = []
    for event in trajectory.events:
        reference = fmean(signal[index] for index, step in enumerate(trajectory.steps) if step <= event.start)
        for index, step in enumerate(trajectory.steps):
            if step >= event.start:
                output.append({"event": event.kind, "step_since_event": float(step - event.start), "deviation": abs(signal[index] - reference)})
    return output


def update_magnitude_distribution(trajectory: Trajectory) -> dict[str, object]:
    magnitudes = [abs(current - previous) for before, after in zip(trajectory.node_states, trajectory.node_states[1:]) for batch_before, batch_after in zip(before, after, strict=True) for node_before, node_after in zip(batch_before, batch_after, strict=True) for previous, current in zip(node_before, node_after, strict=True)]
    return {"count": len(magnitudes), "mean": fmean(magnitudes), "standard_deviation": _std(magnitudes), "minimum": min(magnitudes), "maximum": max(magnitudes)}


def edge_state_distribution(trajectory: Trajectory, coordinate: int = 0) -> dict[str, object]:
    values = [vector[coordinate] for frame in trajectory.edge_states for batch in frame for vector in batch]
    return _distribution(values)


def effective_connection_strength_distribution(trajectory: Trajectory) -> dict[str, object]:
    return _distribution([value for frame in trajectory.effective_edge_strengths for batch in frame for value in batch])


def edge_update_magnitude_distribution(trajectory: Trajectory) -> dict[str, object]:
    values = [abs(current - previous) for before, after in zip(trajectory.edge_states, trajectory.edge_states[1:]) for batches_before, batches_after in zip(before, after, strict=True) for vector_before, vector_after in zip(batches_before, batches_after, strict=True) for previous, current in zip(vector_before, vector_after, strict=True)]
    return _distribution(values)


def edge_activity_summary(trajectory: Trajectory, *, inactive_threshold: float = .02, bound_threshold: float = .98) -> dict[str, float]:
    values = [value for frame in trajectory.effective_edge_strengths for batch in frame for value in batch]
    if not values:
        return {"fraction_inactive": 0.0, "fraction_near_bounds": 0.0}
    return {"fraction_inactive": sum(value <= inactive_threshold for value in values) / len(values), "fraction_near_bounds": sum(value <= 1 - bound_threshold or value >= bound_threshold for value in values) / len(values)}


def final_edge_state_sensitivity(first: Trajectory, second: Trajectory) -> dict[str, float]:
    """Compare final channel vectors from two otherwise matched input histories."""
    if not first.edge_states or not second.edge_states:
        raise ValueError("both trajectories must contain edge states")
    left, right = first.edge_states[-1], second.edge_states[-1]
    differences = [abs(a - b) for batch_left, batch_right in zip(left, right, strict=True) for edge_left, edge_right in zip(batch_left, batch_right, strict=True) for a, b in zip(edge_left, edge_right, strict=True)]
    return {"mean_absolute_difference": fmean(differences) if differences else 0.0, "maximum_absolute_difference": max(differences, default=0.0)}


def edge_perturbation_recovery_curve(trajectory: Trajectory, coordinate: int = 0) -> list[dict[str, float | str]]:
    signal = [fmean(abs(vector[coordinate]) for batch in frame for vector in batch) for frame in trajectory.edge_states]
    output: list[dict[str, float | str]] = []
    for event in trajectory.events:
        if event.kind not in {"edge_impulse", "impulse", "lesion", "InputDistributionShift"}:
            continue
        reference = fmean(signal[index] for index, step in enumerate(trajectory.steps) if step <= event.start)
        for index, step in enumerate(trajectory.steps):
            if step >= event.start:
                output.append({"event": event.kind, "step_since_event": float(step - event.start), "deviation": abs(signal[index] - reference)})
    return output


def train_validation_gap(train: EvaluationResult, validation: EvaluationResult) -> dict[str, float]:
    return {"train_fitness": train.fitness, "validation_fitness": validation.fitness, "gap": train.fitness - validation.fitness}


def scale_generalization(results: Iterable[ScenarioResult]) -> list[dict[str, float | str]]:
    return [{"scenario": result.scenario.name, "nodes": float(result.scenario.nodes), "steps": float(result.scenario.steps), "fitness": result.score, "viable": float(not result.failures.failed)} for result in results]


def long_horizon_stability(results: Iterable[ScenarioResult]) -> list[dict[str, float | str]]:
    return [{"scenario": result.scenario.name, "steps": float(result.scenario.steps), "max_abs_state": result.metrics.boundedness.maximum_absolute_value, "raw_max_abs": result.diagnostics.raw_maximum_absolute_value, "viable": float(not result.failures.failed)} for result in results]


def write_analysis_bundle(
    output: Path,
    train: EvaluationResult,
    validation: EvaluationResult,
    test: EvaluationResult,
    *,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Export compact diagnostic analysis without delaying experiment completion.

    Fitness metrics and replay files remain exact.  The distribution and
    correlation diagnostics use an event-preserving frame sample because their
    value is visual inspection, not re-scoring the candidate.
    """
    output.mkdir(parents=True, exist_ok=True)
    chosen = next((item for item in test.scenario_results if item.trajectory is not None), None)
    document: dict[str, object] = {
        "train_validation_gap": train_validation_gap(train, validation),
        "scale_generalization": scale_generalization(test.scenario_results),
        "long_horizon_stability": long_horizon_stability(test.scenario_results),
    }
    if chosen and chosen.trajectory:
        diagnostic = _diagnostic_sample(chosen.trajectory)
        document["analysis_sampling"] = {
            "source_frames": len(chosen.trajectory.steps),
            "diagnostic_frames": len(diagnostic.steps),
            "mode": "event-preserving uniform sample",
        }
        if progress:
            progress("analysis_summaries")
        document.update({"state_distribution": state_distribution(diagnostic), "node_correlation_matrix": node_correlation_matrix(diagnostic), "perturbation_recovery_curve": perturbation_recovery_curve(diagnostic), "update_magnitude_distribution": update_magnitude_distribution(diagnostic)})
        if diagnostic.edge_states and diagnostic.edge_states[0] and diagnostic.edge_states[0][0] and diagnostic.edge_states[0][0][0]:
            document.update({"edge_state_distribution": edge_state_distribution(diagnostic), "effective_connection_strength_distribution": effective_connection_strength_distribution(diagnostic), "edge_update_magnitude_distribution": edge_update_magnitude_distribution(diagnostic), "edge_activity": edge_activity_summary(diagnostic), "edge_perturbation_recovery_curve": edge_perturbation_recovery_curve(diagnostic)})
        if progress:
            progress("analysis_charts")
        _write_line_svg(output / "trajectory.svg", trajectory_plot_data(diagnostic)["series"], "Trajectory: coordinate 0")
        recovery = perturbation_recovery_curve(diagnostic)
        _write_line_svg(output / "recovery.svg", [[float(point["deviation"]) for point in recovery]], "Perturbation recovery")
    path = output / "analysis.json"
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


def _diagnostic_sample(trajectory: Trajectory, maximum_frames: int = 64) -> Trajectory:
    """Return an evenly spaced sample that always includes event boundaries."""
    total = len(trajectory.steps)
    if total <= maximum_frames:
        return trajectory
    stride = max(1, (total - 1 + maximum_frames - 2) // (maximum_frames - 1))
    indexes = set(range(0, total, stride))
    indexes.add(total - 1)
    for event in trajectory.events:
        for index, step in enumerate(trajectory.steps):
            if step in {event.start, event.end, event.end + 1}:
                indexes.add(index)
    chosen = sorted(indexes)
    return Trajectory(
        times=[trajectory.times[index] for index in chosen],
        steps=[trajectory.steps[index] for index in chosen],
        node_states=[trajectory.node_states[index] for index in chosen],
        edge_states=[trajectory.edge_states[index] for index in chosen],
        effective_edge_strengths=[trajectory.effective_edge_strengths[index] for index in chosen],
        inputs=[trajectory.inputs[index] for index in chosen],
        events=trajectory.events,
    )


def _write_line_svg(path: Path, series: Sequence[Sequence[float]], title: str) -> None:
    """Write a compact, dependency-free line-chart SVG.

    Analysis is part of the interactive job's completion path.  These two
    charts are intentionally simple, so avoiding Matplotlib's renderer and
    font handling keeps report export fast and predictable on local Windows
    installations.
    """
    width, height = 1000, 480
    left, right, top, bottom = 72, 28, 54, 58
    chart_width, chart_height = width - left - right, height - top - bottom
    visible = [tuple(float(value) for value in row) for row in series[:8] if row]
    values = [value for row in visible for value in row]
    minimum, maximum = (min(values), max(values)) if values else (-1.0, 1.0)
    if minimum == maximum:
        minimum -= 1.0
        maximum += 1.0
    span = maximum - minimum
    longest = max((len(row) for row in visible), default=1)

    def x(index: int, length: int) -> float:
        return left + chart_width * index / max(1, length - 1)

    def y(value: float) -> float:
        return top + chart_height * (maximum - value) / span

    colors = ("#3b82f6", "#ef4444", "#10b981", "#a855f7", "#f59e0b", "#06b6d4", "#ec4899", "#64748b")
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="30" font-family="system-ui, sans-serif" font-size="18" font-weight="600">{escape(title)}</text>',
    ]
    for tick in range(5):
        value = minimum + span * tick / 4
        py = y(value)
        elements.append(f'<line x1="{left}" y1="{py:.2f}" x2="{width - right}" y2="{py:.2f}" stroke="#d9e1ec" stroke-width="1"/>')
        elements.append(f'<text x="{left - 10}" y="{py + 4:.2f}" text-anchor="end" font-family="system-ui, sans-serif" font-size="11" fill="#475569">{value:.3g}</text>')
    elements.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#64748b"/>')
    elements.append(f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#64748b"/>')
    for index, row in enumerate(visible):
        points = " ".join(f"{x(step, len(row)):.2f},{y(value):.2f}" for step, value in enumerate(row))
        elements.append(f'<polyline points="{points}" fill="none" stroke="{colors[index]}" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>')
    elements.append(f'<text x="{left + chart_width / 2:.2f}" y="{height - 20}" text-anchor="middle" font-family="system-ui, sans-serif" font-size="12" fill="#334155">recorded step (0–{longest - 1})</text>')
    elements.append(f'<text x="18" y="{top + chart_height / 2:.2f}" transform="rotate(-90 18 {top + chart_height / 2:.2f})" text-anchor="middle" font-family="system-ui, sans-serif" font-size="12" fill="#334155">value</text>')
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def _std(values: Sequence[float]) -> float:
    return sqrt(fmean((value - fmean(values)) ** 2 for value in values)) if values else 0.0


def _distribution(values: Sequence[float]) -> dict[str, object]:
    return {"count": len(values), "mean": fmean(values) if values else 0.0, "standard_deviation": _std(values), "minimum": min(values, default=0.0), "maximum": max(values, default=0.0)}


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    lm, rm = fmean(left), fmean(right)
    numerator = sum((a - lm) * (b - rm) for a, b in zip(left, right, strict=True))
    denominator = sqrt(sum((a - lm) ** 2 for a in left) * sum((b - rm) ** 2 for b in right))
    return numerator / denominator if denominator > 1e-12 else 1.0
