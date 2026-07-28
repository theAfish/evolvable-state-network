"""Command line entry point for asynchronous survival evolution."""

from __future__ import annotations

import argparse
from typing import Sequence

from .evolution import (
    AsyncEvolutionConfig,
    AsyncEvolutionRunner,
    EdgeArchitecture,
    EvolutionConfig,
    EvolutionRunner,
    RuleArchitecture,
    run_diagnostic_experiment,
)
from .storage import new_run_directory


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run asynchronous death-driven evolution of shared local rules.")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--diagnostic", action="store_true", help="run the required short reference experiment")
    parser.add_argument("--ticks", type=int, default=200)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--result-batch", type=int, default=8)
    parser.add_argument("--legacy-generational", action="store_true", help="use the old fixed-horizon runner")
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--smoke-samples", type=int, default=16)
    parser.add_argument("--evolve", choices=("node", "edge", "joint"), default="joint")
    args = parser.parse_args(argv)
    if args.diagnostic:
        output = new_run_directory("async_runs")
        report = run_diagnostic_experiment(output, args.seed)
        print(f"Wrote {output / 'diagnostic_report.json'}")
        print(f"Completed candidates: {report['completed_candidates']}")
        return 0
    architecture = RuleArchitecture()
    edge_architecture = EdgeArchitecture(node_state_width=architecture.state_width) if args.evolve in {"edge", "joint"} else None
    if not args.legacy_generational:
        output = new_run_directory("async_runs")
        report = AsyncEvolutionRunner(
            AsyncEvolutionConfig(
                seed=args.seed,
                max_ticks=args.ticks,
                slots=args.slots,
                replicas=args.replicas,
                result_batch_size=args.result_batch,
                architecture=architecture,
                edge_architecture=edge_architecture,
                target=args.evolve,
            )
        ).run(output)
        print(f"Wrote {output / 'candidate_archive.json'}")
        print(f"Wrote {output / 'living_censored.json'}")
        print(f"Completed candidates: {report['completed_candidates']}")
        return 0
    output = new_run_directory("evolution_runs")
    runner = EvolutionRunner(EvolutionConfig(seed=args.seed, generations=args.generations, population_size=args.population, smoke_samples=args.smoke_samples, architecture=architecture, edge_architecture=edge_architecture, target=args.evolve))
    report = runner.run(output)
    print(f"Wrote {output / 'random_search_smoke.json'}")
    print(f"Wrote {output / 'best_genome.json'}")
    print(f"Best train fitness: {report['best']['fitness']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
