"""FastAPI application for the state-network research workspace."""

from __future__ import annotations

import json
import secrets
from pathlib import Path
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .async_evolution import run_diagnostic_experiment
from .candidate import EdgeArchitecture, FixedEdgeRule, MLPUpdateRule, RuleArchitecture
from .dashboard import dashboard_document
from .evaluation import CandidateEvaluator
from .evolution import EvolutionConfig, EvolutionRunner, random_search_smoke_test
from .experiment import ExperimentRequest, run_experiment
from .genome import GenomeCodec
from .graph import generate_random_graph
from .inputs import GaussianInput
from .simulation import Simulation, SimulationConfig, TransitionDiagnostics
from .storage import application_data_dir


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentPayload(StrictModel):
    seed: int = Field(7, ge=0, lt=2**32)
    nodes: int = Field(24, ge=2, le=200)
    mean_degree: float = Field(5.0, ge=0)
    steps: int = Field(300, ge=1, le=5000)
    batch_size: int = Field(4, ge=1, le=64)
    dt: float = Field(.05, gt=0, le=1)
    baseline: Literal["both", "fixed_rnn", "homeostatic"] = "both"

    @model_validator(mode="after")
    def validate_degree(self) -> "ExperimentPayload":
        if self.mean_degree > self.nodes - 1:
            raise ValueError("mean_degree cannot exceed nodes - 1")
        return self


class EvolutionPayload(StrictModel):
    seed: int | None = Field(None, ge=0, lt=2**32)
    samples: int = Field(16, ge=4, le=64)
    generations: int = Field(4, ge=1, le=20)
    population: int = Field(8, ge=2, le=32)


class AsyncDiagnosticPayload(StrictModel):
    seed: int | None = Field(None, ge=0, lt=2**32)


class LiveSessionPayload(StrictModel):
    model_id: str
    seed: int = Field(7, ge=0, lt=2**32)
    nodes: int = Field(24, ge=2, le=200)
    mean_degree: float = Field(5.0, ge=0)
    batch_size: int = Field(1, ge=1, le=64)
    dt: float = Field(.05, gt=0, le=1)
    topology: Literal["erdos_renyi", "ring"] = "erdos_renyi"
    input_seed: int = Field(108, ge=0, lt=2**32)
    input_standard_deviation: float = Field(.28, ge=0)

    @model_validator(mode="after")
    def validate_degree(self) -> "LiveSessionPayload":
        if self.mean_degree > self.nodes - 1:
            raise ValueError("mean_degree cannot exceed nodes - 1")
        return self


class LiveStepPayload(StrictModel):
    steps: int = Field(1, ge=1, le=8)


class ApplicationState:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.jobs: dict[str, dict[str, object]] = {}
        self.jobs_lock = Lock()
        self.live_sessions: dict[str, dict[str, object]] = {}
        self.live_lock = Lock()

    def ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def artifact_url(self, path: Path) -> str:
        return f"/artifacts/{path.relative_to(self.root).as_posix()}"

    def update_job(self, job_id: str, event: dict[str, object]) -> None:
        with self.jobs_lock:
            job = self.jobs[job_id]
            phase = str(event.get("phase", "running"))
            job["phase"] = phase
            if phase == "smoke":
                job["samples"].append(event)
            elif phase == "generation":
                job["generations"].append(event)
            else:
                job["latest"] = event

    def job_snapshot(self, job_id: str) -> dict[str, object]:
        with self.jobs_lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return dict(job)

    def async_run_summary(self, run_directory: Path) -> dict[str, object]:
        report_path = run_directory / "diagnostic_report.json"
        archive_path = run_directory / "candidate_archive.json"
        censored_path = run_directory / "living_censored.json"
        if not report_path.is_file():
            raise ValueError("asynchronous diagnostic report is unavailable")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        archive = json.loads(archive_path.read_text(encoding="utf-8")) if archive_path.is_file() else []
        censored = json.loads(censored_path.read_text(encoding="utf-8")) if censored_path.is_file() else []
        living = [item for item in censored if item.get("kind") == "living_at_stop"]
        candidates = [
            {
                "candidate_id": item["candidate_id"],
                "status": item["status"],
                "level": item["level"],
                "age": item["age"],
                "death_cause": item["death_cause"],
                "source": item["sampling"]["source"],
                "optimizer_update": item["sampling"]["optimizer_update"],
                "rank_key": item["rank_key"],
                "replicas": [
                    {
                        "age": replica["age"],
                        "death_cause": replica["death_cause"],
                        "burden": replica["normalized_pathology_burden"],
                        "responsiveness": replica["responsiveness"],
                        "propagation": replica["propagation"],
                        "distinguishability": replica["distinguishability"],
                        "recovered": replica["recovered"],
                    }
                    for replica in item["per_replica_results"]
                ],
            }
            for item in archive
        ]
        return {
            "run_id": run_directory.name,
            "report": report,
            "candidates": candidates,
            "censored_tail": censored[-40:],
            "slots": [
                {
                    "slot": index,
                    "candidate_id": item["candidate_id"],
                    "age": item["age"],
                    "level": item["level"],
                    "milestone": item.get("milestone", item["age"]),
                    "source": item.get("source", "living"),
                    "worst_burden": (
                        max(0.0, -float(item["rank_key"][3]))
                        if len(item.get("rank_key", ())) >= 4 else 0.0
                    ),
                }
                for index, item in enumerate(living)
            ],
            "artifacts": {
                "report": self.artifact_url(report_path),
                "archive": self.artifact_url(archive_path),
                "censored": self.artifact_url(censored_path),
                "config": self.artifact_url(run_directory / "diagnostic_config.json"),
            },
        }

    def latest_async_summary(self) -> dict[str, object]:
        root = self.root / "async_runs"
        candidates = (
            [
                path for path in root.iterdir()
                if path.is_dir() and (path / "diagnostic_report.json").is_file()
            ]
            if root.is_dir() else []
        )
        if not candidates:
            return {"available": False}
        latest = max(candidates, key=lambda path: (path / "diagnostic_report.json").stat().st_mtime)
        return {"available": True, **self.async_run_summary(latest)}

    def available_live_models(self) -> list[dict[str, object]]:
        root = self.root / "evolution_runs"
        if not root.is_dir():
            return []
        models: list[dict[str, object]] = []
        for path in root.iterdir():
            genome_path = path / "best_genome.json"
            if not path.is_dir() or not genome_path.is_file():
                continue
            try:
                document = json.loads(genome_path.read_text(encoding="utf-8"))
                models.append(
                    {
                        "id": path.name,
                        "target": document.get("target", "node"),
                        "parameters": len(document.get("genome", ())),
                        "validation_fitness": document.get("validation", {}).get("fitness"),
                        "test_fitness": document.get("test", {}).get("fitness"),
                    }
                )
            except (OSError, ValueError, TypeError):
                continue
        return sorted(models, key=lambda item: str(item["id"]), reverse=True)

    def create_live_session(self, payload: LiveSessionPayload) -> dict[str, object]:
        model_path = self.root / "evolution_runs" / payload.model_id / "best_genome.json"
        if not payload.model_id or not model_path.is_file() or model_path.parent.name != payload.model_id:
            raise ValueError("select a completed evolution run with exported best parameters")
        document = json.loads(model_path.read_text(encoding="utf-8"))
        architecture = RuleArchitecture(**document["architecture"])
        edge_data = document.get("edge_architecture")
        edge_architecture = EdgeArchitecture(**edge_data) if edge_data else None
        target = str(document.get("target", "node"))
        if target not in {"node", "edge", "joint"}:
            raise ValueError("exported model has an unsupported evolution target")
        codec = GenomeCodec(architecture, edge_architecture, target)  # type: ignore[arg-type]
        node_rule, edge_rule = codec.decode_groups(document["genome"])
        node_rule = node_rule or MLPUpdateRule(architecture, (0.0,) * architecture.parameter_count)
        if edge_rule is None and edge_architecture is not None:
            edge_rule = FixedEdgeRule(edge_architecture)
        graph = generate_random_graph(payload.nodes, payload.mean_degree, payload.seed, payload.topology)
        config = SimulationConfig(steps=1, batch_size=payload.batch_size, dt=payload.dt)
        simulator = Simulation(graph, node_rule, edge_rule)
        session_id = uuid4().hex
        provider = GaussianInput(payload.input_seed, standard_deviation=payload.input_standard_deviation)
        session: dict[str, object] = {
            "id": session_id,
            "model_id": payload.model_id,
            "graph": graph,
            "config": config,
            "simulator": simulator,
            "state": simulator.initial_state(payload.batch_size),
            "provider": provider,
            "step": 0,
            "input": provider.sample(0, payload.batch_size, payload.nodes, node_rule.state_width),
            "diagnostics": TransitionDiagnostics(),
            "topology": payload.topology,
            "seed": payload.seed,
        }
        with self.live_lock:
            self.live_sessions[session_id] = session
        return self.live_snapshot(session)

    @staticmethod
    def live_snapshot(session: dict[str, object]) -> dict[str, object]:
        simulator = session["simulator"]
        state = session["state"]
        config = session["config"]
        assert isinstance(simulator, Simulation) and isinstance(config, SimulationConfig)
        return {
            "session_id": session["id"],
            "model_id": session["model_id"],
            "step": session["step"],
            "time": float(session["step"]) * config.dt,
            "graph": {
                "nodes": simulator.graph.n_nodes,
                "edges": [
                    {"source": edge.source, "target": edge.target, "weight": edge.weight}
                    for edge in simulator.graph.edges
                ],
            },
            "simulation_config": {"dt": config.dt, "batch_size": config.batch_size},
            "node_state": state.node,
            "edge_state": state.edge,
            "effective_edge_strengths": simulator._effective_strengths(state.edge),
            "inputs": session["input"],
            "topology": session["topology"],
            "graph_seed": session["seed"],
        }

    def advance_live_session(self, session_id: str, count: int) -> dict[str, object]:
        with self.live_lock:
            session = self.live_sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            simulator = session["simulator"]
            state = session["state"]
            provider = session["provider"]
            config = session["config"]
            diagnostics = session["diagnostics"]
            assert isinstance(simulator, Simulation)
            assert isinstance(provider, GaussianInput)
            assert isinstance(config, SimulationConfig)
            assert isinstance(diagnostics, TransitionDiagnostics)
            for _ in range(count):
                step = int(session["step"])
                external = provider.sample(step, config.batch_size, simulator.graph.n_nodes, simulator.node_rule.state_width)
                state = simulator._step(state, external, step, config, (), diagnostics, None)
                session["state"] = state
                session["input"] = external
                session["step"] = step + 1
            return self.live_snapshot(session)

    def new_job(self, kind: str, seed: int, total: int) -> str:
        job_id = uuid4().hex
        with self.jobs_lock:
            self.jobs[job_id] = {
                "id": job_id,
                "kind": kind,
                "status": "running",
                "phase": "queued",
                "seed": seed,
                "samples_total": total,
                "samples": [],
                "generations": [],
                "latest": {},
                "result": None,
            }
        return job_id

    def finish_job(self, job_id: str, result: dict[str, object]) -> None:
        with self.jobs_lock:
            self.jobs[job_id].update(
                {"status": "complete", "phase": "complete", "result": result}
            )

    def fail_job(self, job_id: str, error: Exception) -> None:
        with self.jobs_lock:
            self.jobs[job_id].update(
                {"status": "failed", "phase": "failed", "error": str(error)}
            )


def _seed(value: int | None) -> int:
    return secrets.randbelow(2**32) if value is None else value


def create_app(data_dir: Path | None = None) -> FastAPI:
    root = (data_dir or application_data_dir()).resolve()
    runtime = ApplicationState(root)
    application = FastAPI(
        title="Evolvable State Network",
        summary="Asynchronous survival evolution and replay API",
        version="0.2.0",
    )
    application.state.runtime = runtime

    @application.get("/api/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "storage": str(runtime.root)}

    @application.post("/api/experiment")
    def experiment(payload: ExperimentPayload) -> dict[str, object]:
        request = ExperimentRequest(**payload.model_dump())
        result = run_experiment(request)
        return dashboard_document(result.graph, result.runs, result.config)

    @application.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict[str, object]:
        try:
            return runtime.job_snapshot(job_id)
        except KeyError as error:
            raise HTTPException(404, "unknown job") from error

    @application.post("/api/async/diagnostic")
    def start_async(
        payload: AsyncDiagnosticPayload, background_tasks: BackgroundTasks
    ) -> dict[str, str]:
        runtime.ensure_root()
        seed = _seed(payload.seed)
        job_id = runtime.new_job("async_diagnostic", seed, 80)
        run_directory = runtime.root / "async_runs" / job_id

        def worker() -> None:
            try:
                report = run_diagnostic_experiment(
                    run_directory,
                    seed,
                    progress=lambda event: runtime.update_job(
                        job_id, {"phase": "asynchronous", **event}
                    ),
                )
                result = runtime.async_run_summary(run_directory)
                result["report"] = report
                runtime.finish_job(job_id, result)
            except Exception as error:
                runtime.fail_job(job_id, error)

        background_tasks.add_task(worker)
        return {"job_id": job_id}

    @application.get("/api/async/latest")
    def latest_async() -> dict[str, object]:
        return runtime.latest_async_summary()

    def start_evolution_job(
        kind: Literal["random_search", "search"],
        payload: EvolutionPayload,
        background_tasks: BackgroundTasks,
    ) -> dict[str, str]:
        runtime.ensure_root()
        seed = _seed(payload.seed)
        total = payload.population if kind == "search" else payload.samples
        job_id = runtime.new_job(kind, seed, total)

        def worker() -> None:
            try:
                architecture = RuleArchitecture()
                edge_architecture = EdgeArchitecture(node_state_width=architecture.state_width)
                if kind == "random_search":
                    report = random_search_smoke_test(
                        CandidateEvaluator(
                            architecture,
                            edge_architecture=edge_architecture,
                            target="joint",
                        ),
                        samples=payload.samples,
                        seed=seed,
                        on_sample=lambda event: runtime.update_job(
                            job_id, {"phase": "smoke", **event}
                        ),
                    )
                    runtime.finish_job(job_id, {"smoke_report": report.to_dict()})
                    return
                run_directory = runtime.root / "evolution_runs" / job_id
                runner = EvolutionRunner(
                    EvolutionConfig(
                        seed=seed,
                        generations=payload.generations,
                        population_size=payload.population,
                        smoke_samples=payload.samples,
                        architecture=architecture,
                        edge_architecture=edge_architecture,
                        target="joint",
                    )
                )
                report = runner.run(
                    run_directory,
                    progress=lambda event: runtime.update_job(job_id, event),
                )
                runtime.finish_job(
                    job_id,
                    {
                        "best_fitness": report["best"]["fitness"],
                        "validation_fitness": report["validation"]["fitness"],
                        "test_fitness": report["test"]["fitness"],
                        "output_url": runtime.artifact_url(run_directory / "evolution_report.json"),
                        "best_genome_url": runtime.artifact_url(run_directory / "best_genome.json"),
                        "analysis_url": runtime.artifact_url(run_directory / "analysis" / "analysis.json"),
                        "trajectory_svg_url": runtime.artifact_url(run_directory / "analysis" / "trajectory.svg"),
                        "recovery_svg_url": runtime.artifact_url(run_directory / "analysis" / "recovery.svg"),
                        "replay_index_url": runtime.artifact_url(run_directory / "replays" / "index.json"),
                    },
                )
            except Exception as error:
                runtime.fail_job(job_id, error)

        background_tasks.add_task(worker)
        return {"job_id": job_id}

    @application.post("/api/evolution/random-search")
    def random_search(
        payload: EvolutionPayload, background_tasks: BackgroundTasks
    ) -> dict[str, str]:
        return start_evolution_job("random_search", payload, background_tasks)

    @application.post("/api/evolution/search")
    def evolution_search(
        payload: EvolutionPayload, background_tasks: BackgroundTasks
    ) -> dict[str, str]:
        return start_evolution_job("search", payload, background_tasks)

    @application.get("/api/live/models")
    def live_models() -> dict[str, object]:
        return {"models": runtime.available_live_models()}

    @application.post("/api/live/sessions")
    def create_live(payload: LiveSessionPayload) -> dict[str, object]:
        try:
            return runtime.create_live_session(payload)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    @application.post("/api/live/sessions/{session_id}/step")
    def step_live(session_id: str, payload: LiveStepPayload) -> dict[str, object]:
        try:
            return runtime.advance_live_session(session_id, payload.steps)
        except KeyError as error:
            raise HTTPException(404, "live session is unavailable") from error

    @application.get("/dashboard", include_in_schema=False)
    @application.get("/dashboard/", include_in_schema=False)
    def old_dashboard_url() -> RedirectResponse:
        return RedirectResponse("/", status_code=307)

    application.mount(
        "/artifacts",
        StaticFiles(directory=str(runtime.root), check_dir=False),
        name="artifacts",
    )
    web = Path(__file__).with_name("web")
    application.mount("/", StaticFiles(directory=str(web), html=True), name="frontend")
    return application


app = create_app()
