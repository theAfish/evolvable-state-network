"""Smoke-gated Phase 1A evolutionary experiment orchestration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from statistics import fmean
from typing import Callable, Sequence

from .candidate import EdgeArchitecture, RuleArchitecture
from .cmaes import CMAES, CMAESConfig
from .evaluation import (
    CandidateEvaluator,
    EvaluationResult,
    ScenarioConfig,
    ScenarioSuite,
    default_scenario_suite,
)
from .genome import EvolutionTarget
from ..analysis import write_analysis_bundle
from ..dashboard import dashboard_document
from ..graph import generate_random_graph
from ..simulation import SimulationConfig


@dataclass(frozen=True, slots=True)
class EvolutionConfig:
    seed: int = 41
    generations: int = 4
    population_size: int = 8
    initial_sigma: float = .35
    smoke_samples: int = 16
    checkpoint_every: int = 1
    architecture: RuleArchitecture = RuleArchitecture()
    edge_architecture: EdgeArchitecture | None = None
    target: EvolutionTarget = "node"
    scenarios: ScenarioSuite | None = None

    def __post_init__(self) -> None:
        if self.generations < 1 or self.smoke_samples < 4 or self.checkpoint_every < 1:
            raise ValueError("generations, smoke_samples, and checkpoint_every must be positive")


@dataclass(frozen=True, slots=True)
class SmokeReport:
    seed: int
    samples: int
    fitnesses: tuple[float, ...]
    viable_fraction: float
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float
    meaningful: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _smoke_report(seed: int, results: Sequence[EvaluationResult]) -> SmokeReport:
    fitnesses = tuple(item.fitness for item in results)
    mean = fmean(fitnesses)
    deviation = fmean((value - mean) ** 2 for value in fitnesses) ** .5
    return SmokeReport(
        seed, len(results), fitnesses, fmean(item.viable_fraction for item in results), min(fitnesses), max(fitnesses),
        mean, deviation, deviation > 1e-6 and len({round(value, 8) for value in fitnesses}) > 1,
    )


def random_search_smoke_test(
    evaluator: CandidateEvaluator, *, samples: int = 32, seed: int = 41,
    on_sample: Callable[[dict[str, float | int]], None] | None = None,
) -> SmokeReport:
    rng = Random(seed)
    results = []
    for index in range(samples):
        result = evaluator.evaluate(tuple(rng.gauss(0, .45) for _ in range(evaluator.codec.dimension)))
        results.append(result)
        if on_sample is not None:
            on_sample({
                "sample": index + 1, "fitness": result.fitness, "mean_score": result.mean_score,
                "viable_fraction": result.viable_fraction,
                "failed_scenarios": sum(item.failures.failed for item in result.scenario_results),
            })
    fitnesses = tuple(item.fitness for item in results)
    mean = fmean(fitnesses)
    deviation = fmean((value - mean) ** 2 for value in fitnesses) ** .5
    viable_fraction = fmean(item.viable_fraction for item in results)
    # More than one observable value confirms that the suite is not constant.
    meaningful = deviation > 1e-6 and len({round(value, 8) for value in fitnesses}) > 1
    return SmokeReport(seed, samples, fitnesses, viable_fraction, min(fitnesses), max(fitnesses), mean, deviation, meaningful)


class EvolutionRunner:
    def __init__(self, config: EvolutionConfig) -> None:
        self.config = config
        self.evaluator = CandidateEvaluator(
            config.architecture, config.scenarios or default_scenario_suite(),
            edge_architecture=config.edge_architecture, target=config.target,
        )

    def run(
        self, output: Path, *, resume: bool = False,
        progress: Callable[[dict[str, object]], None] | None = None,
    ) -> dict[str, object]:
        output.mkdir(parents=True, exist_ok=True)
        smoke_path = output / "random_search_smoke.json"
        checkpoint_path = output / "checkpoint.json"
        if resume:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self._assert_compatible(checkpoint["experiment_config"])
            optimizer = CMAES.from_state_dict(checkpoint["optimizer"])
            history = list(checkpoint["history"])
            best = checkpoint.get("best")
            smoke = checkpoint["smoke_report"]
            generation_archive = list(checkpoint.get("generation_archive", []))
        else:
            # The initial CMA-ES population is generation 0 and the required
            # smoke test. It is not an unrelated random search.
            optimizer = CMAES(CMAESConfig(self.evaluator.codec.dimension, self.config.population_size, self.config.initial_sigma, self.config.seed))
            initial_population = optimizer.ask()
            initial_scenarios = self.evaluator.training_scenarios(0, self.config.seed)
            initial_evaluations = self.evaluator.evaluate_batch(initial_population, "train", scenarios=initial_scenarios)
            for index, evaluation in enumerate(initial_evaluations):
                if progress:
                    progress({"phase": "smoke", "sample": index + 1, "fitness": evaluation.fitness, "mean_score": evaluation.mean_score, "viable_fraction": evaluation.viable_fraction, "failed_scenarios": sum(item.failures.failed for item in evaluation.scenario_results)})
            smoke_result = _smoke_report(self.config.seed, initial_evaluations)
            smoke = smoke_result.to_dict()
            smoke_path.write_text(json.dumps(smoke, indent=2, sort_keys=True), encoding="utf-8")
            if progress:
                progress({"phase": "smoke_complete", "meaningful": smoke_result.meaningful, "mean": smoke_result.mean, "standard_deviation": smoke_result.standard_deviation})
            if not smoke_result.meaningful and self.config.population_size >= 4:
                raise RuntimeError("random-search smoke test did not produce a non-degenerate fitness distribution")
            history: list[dict[str, object]] = []
            best: dict[str, object] | None = None
            generation_archive: list[dict[str, object]] = []
            self._record_generation(optimizer, initial_evaluations, history, generation_archive, best, initial_scenarios)
            best = generation_archive[-1]["global_best"]
            optimizer.tell(initial_population, [item.fitness for item in initial_evaluations])
            if progress:
                progress({"phase": "generation", **history[-1]})
        while optimizer.generation < self.config.generations:
            population = optimizer.ask()
            generation_scenarios = self.evaluator.training_scenarios(optimizer.generation, self.config.seed)
            evaluations = self.evaluator.evaluate_batch(population, "train", scenarios=generation_scenarios)
            fitnesses = [item.fitness for item in evaluations]
            winner_index = max(range(len(evaluations)), key=lambda index: fitnesses[index])
            winner = evaluations[winner_index]
            previous_best = best
            self._record_generation(optimizer, evaluations, history, generation_archive, previous_best, generation_scenarios)
            best = generation_archive[-1]["global_best"]
            optimizer.tell(population, fitnesses)
            if progress:
                progress({"phase": "generation", **history[-1]})
            if optimizer.generation % self.config.checkpoint_every == 0:
                self._write_checkpoint(checkpoint_path, optimizer, history, best, smoke, generation_archive)
        assert best is not None
        genome = tuple(float(value) for value in best["genome"])
        train_final = self.evaluator.evaluate(genome, "train", retain_trajectories=True, scenarios=tuple(ScenarioConfig(**item) for item in generation_archive[-1]["training_scenarios"]))
        validation = self.evaluator.evaluate(genome, "validation", retain_trajectories=True)
        if progress:
            progress({"phase": "validation", "fitness": validation.fitness, "viable_fraction": validation.viable_fraction})
        test = self.evaluator.evaluate(genome, "test", retain_trajectories=True)
        if progress:
            progress({"phase": "test", "fitness": test.fitness, "viable_fraction": test.viable_fraction})
            progress({"phase": "writing_reports"})
        exported = {"schema_version": 2, "architecture": asdict(self.config.architecture), "edge_architecture": asdict(self.config.edge_architecture) if self.config.edge_architecture else None, "target": self.config.target, "genome": list(genome), "parameter_groups": self.evaluator.codec.export_groups(genome), "train": train_final.to_dict(), "validation": validation.to_dict(), "test": test.to_dict()}
        (output / "best_genome.json").write_text(json.dumps(exported, indent=2, sort_keys=True), encoding="utf-8")
        report = {"schema_version": 1, "experiment_config": self._config_dict(), "smoke_report": smoke, "history": history, "generation_archive": generation_archive, "best": best, "validation": validation.to_dict(), "test": test.to_dict()}
        (output / "evolution_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        if progress:
            progress({"phase": "writing_analysis"})
        write_analysis_bundle(
            output / "analysis",
            train_final,
            validation,
            test,
            progress=(lambda stage: progress({"phase": stage})) if progress else None,
        )
        if progress:
            progress({"phase": "writing_replays"})
        self._write_replay_documents(output, train_final, validation, test, generation_archive)
        if progress:
            progress({"phase": "finalizing"})
        self._write_checkpoint(checkpoint_path, optimizer, history, best, smoke, generation_archive)
        return report

    @staticmethod
    def _record_generation(optimizer: CMAES, evaluations: Sequence[EvaluationResult], history: list[dict[str, object]], archive: list[dict[str, object]], prior_best: dict[str, object] | None, scenarios: Sequence[ScenarioConfig]) -> None:
        fitnesses = [item.fitness for item in evaluations]
        winner = evaluations[max(range(len(evaluations)), key=lambda index: fitnesses[index])]
        winner_record = {"generation": optimizer.generation, "fitness": winner.fitness, "genome": list(winner.genome), "train": winner.to_dict()}
        best = winner_record if prior_best is None or winner.fitness > float(prior_best["fitness"]) else prior_best
        history.append({"generation": optimizer.generation, "best_fitness": winner.fitness, "mean_fitness": fmean(fitnesses), "sigma": optimizer.sigma})
        archive.append({"generation": optimizer.generation, "best": winner_record, "global_best": best, "training_scenarios": [asdict(item) for item in scenarios]})

    def _write_replay_documents(
        self, output: Path, train: EvaluationResult, validation: EvaluationResult, test: EvaluationResult,
        generation_archive: Sequence[dict[str, object]],
    ) -> None:
        """Export each evaluated graph trajectory for the interactive Replay view."""
        replay_directory = output / "replays"
        replay_directory.mkdir(parents=True, exist_ok=True)
        index: list[dict[str, object]] = []
        for split, evaluation in (("train", train), ("validation", validation), ("test", test)):
            for item in evaluation.scenario_results:
                if item.trajectory is None:
                    continue
                scenario = item.scenario
                graph = generate_random_graph(
                    scenario.nodes, scenario.mean_degree, scenario.graph_seed, scenario.topology
                )
                configuration = SimulationConfig(steps=scenario.steps, batch_size=scenario.batch_size)
                filename = f"{split}-{scenario.name}.json"
                document = dashboard_document(
                    graph,
                    {"evolved_mlp": (item.trajectory, item.metrics.to_dict())},
                    configuration,
                )
                (replay_directory / filename).write_text(json.dumps(document), encoding="utf-8")
                index.append(
                    {
                        "file": filename,
                        "label": f"{split}: {scenario.name}",
                        "split": split,
                        "nodes": scenario.nodes,
                        "steps": scenario.steps,
                        "fitness": item.score,
                        "viable": not item.failures.failed,
                    }
                )
        # Every generation is demonstrated on exactly the same held-out graph,
        # making its dynamics directly comparable across evolutionary time.
        demo_scenario = self.evaluator.suite.validation[0]
        demo_graph = generate_random_graph(demo_scenario.nodes, demo_scenario.mean_degree, demo_scenario.graph_seed, demo_scenario.topology)
        demo_config = SimulationConfig(steps=demo_scenario.steps, batch_size=demo_scenario.batch_size)
        for record in generation_archive:
            generation = int(record["generation"])
            # Replay the global best available at this generation, not a
            # transient generation winner that can be nonviable.
            genome = record["global_best"]["genome"]
            # This is deliberately one fixed graph for every generation.
            # Evaluating the whole validation suite here repeated work already
            # completed above and made the UI appear stuck in the test phase.
            evaluated = self.evaluator.evaluate(
                genome,
                "validation",
                retain_trajectories=True,
                scenarios=(demo_scenario,),
                independently_seed_candidate=False,
            )
            item = evaluated.scenario_results[0]
            filename = f"generation-{generation}-demo.json"
            document = dashboard_document(demo_graph, {f"generation_{generation}_best": (item.trajectory, item.metrics.to_dict())}, demo_config)
            (replay_directory / filename).write_text(json.dumps(document), encoding="utf-8")
            index.append({"file": filename, "label": f"generation {generation} global best: fixed validation demo", "split": "generation", "generation": generation, "nodes": demo_scenario.nodes, "steps": demo_scenario.steps, "fitness": float(record["global_best"]["fitness"]), "viable": not item.failures.failed})
        (replay_directory / "index.json").write_text(json.dumps({"replays": index}, indent=2), encoding="utf-8")

    def _write_checkpoint(self, path: Path, optimizer: CMAES, history: list[dict[str, object]], best: dict[str, object] | None, smoke: dict[str, object], generation_archive: list[dict[str, object]]) -> None:
        data = {"schema_version": 1, "experiment_config": self._config_dict(), "optimizer": optimizer.state_dict(), "history": history, "best": best, "smoke_report": smoke, "generation_archive": generation_archive}
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    def _config_dict(self) -> dict[str, object]:
        return {"seed": self.config.seed, "generations": self.config.generations, "population_size": self.config.population_size, "initial_sigma": self.config.initial_sigma, "smoke_samples": self.config.smoke_samples, "checkpoint_every": self.config.checkpoint_every, "architecture": asdict(self.config.architecture), "edge_architecture": asdict(self.config.edge_architecture) if self.config.edge_architecture else None, "target": self.config.target, "scenarios": asdict(self.evaluator.suite)}

    def _assert_compatible(self, saved: dict[str, object]) -> None:
        current = self._config_dict()
        for key in ("seed", "population_size", "initial_sigma", "architecture", "edge_architecture", "target", "scenarios"):
            if json.dumps(saved.get(key), sort_keys=True) != json.dumps(current.get(key), sort_keys=True):
                raise ValueError(f"checkpoint is incompatible with current {key}")
