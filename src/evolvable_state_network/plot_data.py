"""Plain-text, tabular exports intended for plotting CLI training runs."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def write_plot_table(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    *,
    description: str,
) -> None:
    """Write scalar records as a headered TSV file that plotting tools can read.

    Nested data is deliberately omitted: it belongs in the JSON report, while this
    companion file is designed for ``pandas.read_csv(..., sep="\\t", comment="#")``
    and spreadsheet import.
    """
    materialized = [
        {key: value for key, value in row.items() if _is_scalar(value)}
        for row in rows
    ]
    columns = _columns(materialized)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# {description}\n")
        handle.write("# Tab-separated values; blank cells represent unavailable values.\n")
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in materialized:
            writer.writerow({column: _format(row.get(column)) for column in columns})


def embodied_plot_rows(
    report: Mapping[str, object], events: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return complete, de-duplicated curve records from an embodied CLI run."""
    if report.get("training_mode") == "batch":
        history = report.get("history", [])
        return [dict(row) for row in history if isinstance(row, Mapping)]

    by_tick: dict[int, dict[str, object]] = {}
    for event in events:
        telemetry = event.get("telemetry", [])
        if not isinstance(telemetry, list):
            continue
        for row in telemetry:
            if isinstance(row, Mapping) and isinstance(row.get("tick"), int):
                by_tick[row["tick"]] = dict(row)
    if not by_tick:
        telemetry = report.get("telemetry", [])
        if isinstance(telemetry, list):
            for row in telemetry:
                if isinstance(row, Mapping) and isinstance(row.get("tick"), int):
                    by_tick[row["tick"]] = dict(row)
    return [by_tick[tick] for tick in sorted(by_tick)]


def evolution_plot_row(event: Mapping[str, object]) -> dict[str, object] | None:
    """Extract one compact curve sample from a generic evolution progress event."""
    if event.get("phase") == "generation":
        return {key: value for key, value in event.items() if key != "phase"}
    if "tick" not in event or not isinstance(event.get("report"), Mapping):
        return None
    report = event["report"]
    assert isinstance(report, Mapping)
    return {
        "tick": event["tick"],
        "active_slots": report.get("active_slots"),
        "completed_candidates": report.get("completed_candidates"),
        "deaths": report.get("deaths"),
        "graduations": report.get("graduations"),
        "optimizer_updates": report.get("optimizer_updates"),
        "active_slot_utilization": report.get("active_slot_utilization"),
    }


def _columns(rows: Iterable[Mapping[str, object]]) -> list[str]:
    seen: dict[str, None] = {}
    for row in rows:
        for key in row:
            seen.setdefault(key, None)
    # Index-like columns first makes the files friendlier in both Excel and gnuplot.
    priority = ("generation", "tick")
    return [key for key in priority if key in seen] + sorted(key for key in seen if key not in priority)


def _is_scalar(value: object) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
