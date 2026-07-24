"""Dependency-free SVG trajectory plots for experiment artifacts."""

from __future__ import annotations

from pathlib import Path

from .simulation import Trajectory


_PALETTE = ("#0b7285", "#d9480f", "#5f3dc4", "#2b8a3e", "#c2255c", "#1971c2", "#e67700", "#495057")


def write_trajectory_svg(trajectory: Trajectory, path: Path, title: str) -> None:
    """Write a compact plot of coordinate zero for up to eight nodes in batch 0."""
    if not trajectory.node_states:
        raise ValueError("cannot plot an empty trajectory")
    width, height, margin = 920, 480, 58
    node_count = len(trajectory.node_states[0][0])
    series = [
        [snapshot[0][node][0] for snapshot in trajectory.node_states]
        for node in range(min(node_count, len(_PALETTE)))
    ]
    all_values = [value for values in series for value in values]
    low, high = min(all_values), max(all_values)
    if low == high:
        low, high = low - 1.0, high + 1.0
    padding = (high - low) * 0.08
    low, high = low - padding, high + padding
    start, finish = trajectory.times[0], trajectory.times[-1]
    if start == finish:
        finish = start + 1.0

    def x(value: float) -> float:
        return margin + (value - start) / (finish - start) * (width - 2 * margin)

    def y(value: float) -> float:
        return height - margin - (value - low) / (high - low) * (height - 2 * margin)

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin}" y="28" font-family="sans-serif" font-size="18">{_xml(title)}</text>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#343a40"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#343a40"/>',
        f'<text x="{margin}" y="{height - 18}" font-family="sans-serif" font-size="12">time</text>',
        f'<text x="8" y="{margin}" font-family="sans-serif" font-size="12">coordinate 0</text>',
        f'<text x="{margin}" y="{height - margin + 18}" font-family="sans-serif" font-size="11">{start:.2f}</text>',
        f'<text x="{width - margin - 32}" y="{height - margin + 18}" font-family="sans-serif" font-size="11">{finish:.2f}</text>',
        f'<text x="{margin - 48}" y="{margin + 4}" font-family="sans-serif" font-size="11">{high:.2f}</text>',
        f'<text x="{margin - 48}" y="{height - margin + 4}" font-family="sans-serif" font-size="11">{low:.2f}</text>',
    ]
    for index, values in enumerate(series):
        points = " ".join(f"{x(time):.2f},{y(value):.2f}" for time, value in zip(trajectory.times, values, strict=True))
        color = _PALETTE[index]
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.4" points="{points}"/>')
        parts.append(
            f'<text x="{width - margin - 84}" y="{margin + 18 * (index + 1)}" fill="{color}" '
            f'font-family="sans-serif" font-size="12">node {index}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _xml(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
