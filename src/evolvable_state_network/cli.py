"""Command-line comparison experiment for the two fixed reference rules."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .dashboard import write_dashboard_bundle
from .experiment import ExperimentRequest, run_experiment
from .plotting import write_trajectory_svg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare fixed local dynamics rules under common disturbances.")
    parser.add_argument("--output", type=Path, default=Path("experiment_output"), help="artifact directory")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--nodes", type=int, default=24)
    parser.add_argument("--mean-degree", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.05)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.nodes < 2:
        raise SystemExit("--nodes must be at least 2")
    args.output.mkdir(parents=True, exist_ok=True)
    experiment = run_experiment(ExperimentRequest(args.seed, args.nodes, args.mean_degree, args.steps, args.batch_size, args.dt))
    graph, config, disturbances = experiment.graph, experiment.config, experiment.disturbances
    report: dict[str, object] = {
        "schema_version": 1,
        "config": asdict(config),
        "graph": {"nodes": graph.n_nodes, "edges": len(graph.edges), "seed": args.seed, "topology": "erdos_renyi"},
        "perturbations": [
            {"type": type(disturbance).__name__, **asdict(disturbance)} for disturbance in disturbances
        ],
        "baselines": {},
    }
    baselines: dict[str, object] = {}
    dashboard_runs = experiment.runs
    for name, (trajectory, metric_dict) in dashboard_runs.items():
        plot_path = args.output / f"{name}_trajectory.svg"
        write_trajectory_svg(trajectory, plot_path, f"{name}: coordinate 0, batch 0")
        baselines[name] = {"metrics": metric_dict, "trajectory_plot": plot_path.name}
    report["baselines"] = baselines
    report_path = args.output / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    dashboard_path = write_dashboard_bundle(args.output, graph, dashboard_runs, config)
    print(f"Wrote {report_path}")
    print(f"Wrote {dashboard_path} and dashboard/")
    for name, outcome in baselines.items():
        print(f"{name}: {json.dumps(outcome['metrics'], sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
