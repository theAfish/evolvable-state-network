"""FastAPI application for the state-network research workspace."""

from __future__ import annotations

import json
import secrets
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .async_evolution import (
    AsyncEvolutionConfig,
    CurriculumLevel,
    HealthMonitor,
    PathologyConfig,
    ProbeConfig,
    async_config_from_dict,
    replay_archived_candidate,
    run_async_experiment,
    run_diagnostic_experiment,
)
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


class AsyncTrainingPayload(StrictModel):
    seed: int | None = Field(None, ge=0, lt=2**32)
    candidate_budget: int = Field(200, ge=8, le=5000)
    max_ticks: int = Field(5000, ge=20, le=100_000)
    slots: int = Field(8, ge=1, le=32)
    replicas: int = Field(3, ge=1, le=8)
    optimizer_batch: int = Field(8, ge=2, le=32)
    state_width: int = Field(2, ge=1, le=8)
    stage_1_lifetime: int = Field(40, ge=4, le=2000)
    stage_2_lifetime: int = Field(100, ge=8, le=5000)
    stage_1_nodes: int = Field(8, ge=3, le=100)
    stage_2_nodes: int = Field(12, ge=3, le=200)
    mean_degree: float = Field(3.0, ge=0.5, le=20)
    input_scale: float = Field(.12, gt=0, le=2)
    disturbance_interval: int = Field(10, ge=2, le=1000)
    disturbance_strength: float = Field(.12, ge=0, le=2)
    fatal_threshold: float = Field(8.0, gt=0, le=100)
    node_growth_alert: float = Field(3.2, gt=0, lt=4)
    one_direction_steps: int = Field(12, ge=2, le=1000)
    probe_interval: int = Field(8, ge=2, le=1000)

    @model_validator(mode="after")
    def validate_training_shape(self) -> "AsyncTrainingPayload":
        if self.stage_2_lifetime <= self.stage_1_lifetime:
            raise ValueError("stage 2 lifetime must exceed stage 1 lifetime")
        if self.mean_degree > min(self.stage_1_nodes, self.stage_2_nodes) - 1:
            raise ValueError("mean_degree must fit both curriculum graph sizes")
        if self.optimizer_batch > self.candidate_budget:
            raise ValueError("optimizer_batch cannot exceed candidate_budget")
        return self


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
        config_path = run_directory / "diagnostic_config.json"
        if not report_path.is_file():
            raise ValueError("asynchronous diagnostic report is unavailable")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        archive = json.loads(archive_path.read_text(encoding="utf-8")) if archive_path.is_file() else []
        censored = json.loads(censored_path.read_text(encoding="utf-8")) if censored_path.is_file() else []
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        completed = int(report.get("completed_candidates", len(archive)))
        replicas = int(config.get("replicas", 0))
        report.setdefault("ticks_elapsed", config.get("max_ticks"))
        report.setdefault("tick_limit", config.get("max_ticks"))
        report.setdefault("stop_reason", "tick_limit_reached")
        report.setdefault("candidate_budget", config.get("candidate_budget"))
        report.setdefault("candidates_started", len({item.get("candidate_id") for item in archive}) + int(config.get("slots", 0)))
        report.setdefault("completed_replica_lives", completed * replicas)
        report.setdefault("active_replica_lives", int(config.get("slots", 0)) * replicas)
        report.setdefault("deaths", sum(item.get("status") == "death" for item in archive))
        report.setdefault("graduations", sum(item.get("status") == "graduation" for item in archive))
        report.setdefault(
            "proposals_by_source",
            {
                source: sum(item.get("sampling", {}).get("source") == source for item in archive)
                for source in {item.get("sampling", {}).get("source") for item in archive}
                if source is not None
            },
        )
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
                "functional": bool(item.get("functional", False)),
                "live_eligible": bool(item.get("live_eligible", False)),
                "replicas": [
                    {
                        "index": index,
                        "age": replica["age"],
                        "death_cause": replica["death_cause"],
                        "burden": replica["normalized_pathology_burden"],
                        "responsiveness": replica["responsiveness"],
                        "propagation": replica["propagation"],
                        "distinguishability": replica["distinguishability"],
                        "recovered": replica["recovered"],
                        "coordinate_responsiveness": replica.get("coordinate_responsiveness", []),
                        "coordinate_propagation": replica.get("coordinate_propagation", []),
                        "coordinate_distinguishability": replica.get("coordinate_distinguishability", []),
                        "coordinate_recovered": replica.get("coordinate_recovered", []),
                        "replay_url": (
                            f"/api/async/replays/{run_directory.name}/{item['candidate_id']}/{index}"
                        ),
                    }
                    for index, replica in enumerate(item["per_replica_results"])
                ],
            }
            for item in archive
        ]
        return {
            "run_id": run_directory.name,
            "run_kind": "training" if config.get("candidate_budget") is not None else "diagnostic",
            "settings": {
                "candidate_budget": config.get("candidate_budget"),
                "max_ticks": config.get("max_ticks"),
                "slots": config.get("slots"),
                "replicas": config.get("replicas"),
                "optimizer_batch": config.get("result_batch_size"),
                "node_state_width": config.get("architecture", {}).get("state_width"),
                "levels": config.get("levels", []),
                "fatal_threshold": config.get("pathology", {}).get("fatal_threshold"),
                "node_growth_alert": config.get("pathology", {}).get("node_growth_alert"),
                "one_direction_steps": config.get("pathology", {}).get("one_direction_steps"),
                "probe_interval": config.get("probes", {}).get("interval"),
            },
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
                **(
                    {"elites": self.artifact_url(run_directory / "elite_archive.json")}
                    if (run_directory / "elite_archive.json").is_file()
                    else {}
                ),
                "censored": self.artifact_url(censored_path),
                "config": self.artifact_url(config_path),
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

    def async_replay_document(
        self, run_id: str, candidate_id: int, replica: int
    ) -> dict[str, object]:
        run_directory = (self.root / "async_runs" / run_id).resolve()
        async_root = (self.root / "async_runs").resolve()
        if async_root not in run_directory.parents or not (run_directory / "candidate_archive.json").is_file():
            raise KeyError(run_id)
        archive = json.loads((run_directory / "candidate_archive.json").read_text(encoding="utf-8"))
        record = next((item for item in archive if int(item["candidate_id"]) == candidate_id), None)
        if record is None:
            raise ValueError("candidate is not present in this asynchronous archive")
        config = async_config_from_dict(
            json.loads((run_directory / "diagnostic_config.json").read_text(encoding="utf-8"))
        )
        graph, trajectory, simulation_config, metrics = replay_archived_candidate(
            record, config, replica
        )
        return dashboard_document(
            graph,
            {f"survival candidate {candidate_id} · replica {replica}": (trajectory, metrics)},
            simulation_config,
        )

    @staticmethod
    def _survival_elites(run_directory: Path, config: dict[str, object]) -> list[dict[str, object]]:
        elite_path = run_directory / "elite_archive.json"
        archive_path = run_directory / "candidate_archive.json"
        records = json.loads(
            (elite_path if elite_path.is_file() else archive_path).read_text(encoding="utf-8")
        )
        final_level = len(config.get("levels", ())) - 1
        def deployable(item: dict[str, object]) -> bool:
            if "live_eligible" in item:
                return bool(item["live_eligible"])
            rank_key = list(item.get("rank_key", ()))
            replicas = list(item.get("per_replica_results", ()))
            return (
                item.get("status") == "graduation"
                and int(item.get("level", -1)) == final_level
                and len(rank_key) > 1
                and bool(rank_key[1])
                and max(
                    (float(replica.get("normalized_pathology_burden", 1.0)) for replica in replicas),
                    default=1.0,
                ) <= 1e-12
            )
        records = [item for item in records if deployable(item)]
        records.sort(key=lambda item: tuple(item.get("rank_key", ())), reverse=True)
        unique: list[dict[str, object]] = []
        genomes: set[tuple[float, ...]] = set()
        for record in records:
            genome = tuple(float(value) for value in record.get("genome", ()))
            if genome and genome not in genomes:
                genomes.add(genome)
                unique.append(record)
            if len(unique) >= int(config.get("elite_size", 4)):
                break
        return unique

    def available_live_models(self) -> list[dict[str, object]]:
        models: list[dict[str, object]] = []
        legacy_root = self.root / "evolution_runs"
        if legacy_root.is_dir():
            for path in legacy_root.iterdir():
                genome_path = path / "best_genome.json"
                if not path.is_dir() or not genome_path.is_file():
                    continue
                try:
                    document = json.loads(genome_path.read_text(encoding="utf-8"))
                    edge_document = document.get("edge_architecture") or {}
                    models.append(
                        {
                            "id": f"legacy:{path.name}",
                            "source": "legacy",
                            "run_id": path.name,
                            "target": document.get("target", "node"),
                            "node_state_width": int(
                                document.get("architecture", {}).get("state_width", 1)
                            ),
                            "edge_state_width": int(
                                edge_document.get("latent_width", 0)
                            ),
                            "parameters": len(document.get("genome", ())),
                            "validation_fitness": document.get("validation", {}).get("fitness"),
                            "test_fitness": document.get("test", {}).get("fitness"),
                            "modified": genome_path.stat().st_mtime,
                        }
                    )
                except (OSError, ValueError, TypeError):
                    continue
        async_root = self.root / "async_runs"
        if async_root.is_dir():
            for path in async_root.iterdir():
                config_path = path / "diagnostic_config.json"
                archive_path = path / "candidate_archive.json"
                if not path.is_dir() or not config_path.is_file() or not archive_path.is_file():
                    continue
                try:
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                    for rank, record in enumerate(self._survival_elites(path, config), start=1):
                        replicas = list(record.get("per_replica_results", ()))
                        rank_key = [float(value) for value in record.get("rank_key", ())]
                        minimum = lambda key: min(
                            (float(item.get(key, 0.0)) for item in replicas), default=0.0
                        )
                        models.append(
                            {
                                "id": f"survival:{path.name}:{record['candidate_id']}",
                                "source": "survival",
                                "run_id": path.name,
                                "candidate_id": int(record["candidate_id"]),
                                "elite_rank": rank,
                                "run_kind": "training" if config.get("candidate_budget") is not None else "smoke_test",
                                "target": config.get("target", "joint"),
                                "node_state_width": int(
                                    config.get("architecture", {}).get("state_width", 1)
                                ),
                                "edge_state_width": int(
                                    config.get("edge_architecture", {}).get("latent_width", 0)
                                ),
                                "parameters": len(record.get("genome", ())),
                                "stage": int(record.get("level", 0)) + 1,
                                "lifetime": int(record.get("age", 0)),
                                "functional": bool(rank_key[1]) if len(rank_key) > 1 else False,
                                "worst_pathology_burden": max(
                                    (float(item.get("normalized_pathology_burden", 0.0)) for item in replicas),
                                    default=0.0,
                                ),
                                "minimum_response": minimum("responsiveness"),
                                "minimum_propagation": minimum("propagation"),
                                "minimum_distinguishability": minimum("distinguishability"),
                                "recovered_across_replicas": all(
                                    bool(item.get("recovered")) for item in replicas
                                ),
                                "selection_key": rank_key,
                                "_selection_key": tuple(rank_key),
                                "modified": archive_path.stat().st_mtime,
                            }
                        )
                except (OSError, ValueError, TypeError, KeyError):
                    continue
        survival_models = [item for item in models if item["source"] == "survival"]
        legacy_models = [item for item in models if item["source"] == "legacy"]
        survival_models.sort(
            key=lambda item: (
                item["run_kind"] == "training",
                tuple(item["_selection_key"]),
                float(item["modified"]),
            ),
            reverse=True,
        )
        legacy_models.sort(key=lambda item: float(item["modified"]), reverse=True)
        for global_rank, model in enumerate(survival_models, start=1):
            model["global_rank"] = global_rank
        models = survival_models + legacy_models
        for model in models:
            model.pop("modified", None)
            model.pop("_selection_key", None)
        return models

    def _load_live_model(self, model_id: str) -> dict[str, object]:
        if model_id.startswith("survival:"):
            parts = model_id.split(":")
            if len(parts) != 3 or not parts[1].isalnum() or not parts[2].isdigit():
                raise ValueError("invalid survival model identifier")
            run_directory = (self.root / "async_runs" / parts[1]).resolve()
            async_root = (self.root / "async_runs").resolve()
            config_path = run_directory / "diagnostic_config.json"
            if async_root not in run_directory.parents or not config_path.is_file():
                raise ValueError("selected survival run is unavailable")
            config = json.loads(config_path.read_text(encoding="utf-8"))
            candidate_id = int(parts[2])
            record = next(
                (item for item in self._survival_elites(run_directory, config) if int(item["candidate_id"]) == candidate_id),
                None,
            )
            if record is None:
                raise ValueError("selected survival elite is unavailable")
            return {
                "architecture": config["architecture"],
                "edge_architecture": config.get("edge_architecture"),
                "target": config.get("target", "joint"),
                "genome": record["genome"],
                "pathology": config.get("pathology", {}),
            }
        legacy_id = model_id.removeprefix("legacy:")
        model_path = self.root / "evolution_runs" / legacy_id / "best_genome.json"
        if not legacy_id or not model_path.is_file() or model_path.parent.name != legacy_id:
            raise ValueError("select a trained survival elite or legacy parameter export")
        return json.loads(model_path.read_text(encoding="utf-8"))

    def create_live_session(self, payload: LiveSessionPayload) -> dict[str, object]:
        document = self._load_live_model(payload.model_id)
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
            "monitor": HealthMonitor(PathologyConfig(**dict(document.get("pathology", {})))),
            # Sandbox is intentionally observational: training decides survival;
            # Live replay keeps running so a failure can be inspected.
            "last_warning": None,
            "last_safety_event": None,
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
        monitor = session.get("monitor")
        assert isinstance(monitor, HealthMonitor)
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
            "status": "running",
            "last_warning": session.get("last_warning"),
            "last_safety_event": session.get("last_safety_event"),
            "normalized_pathology_burden": monitor.normalized_burden,
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
            monitor = session["monitor"]
            assert isinstance(simulator, Simulation)
            assert isinstance(provider, GaussianInput)
            assert isinstance(config, SimulationConfig)
            assert isinstance(diagnostics, TransitionDiagnostics)
            assert isinstance(monitor, HealthMonitor)
            for _ in range(count):
                step = int(session["step"])
                previous = deepcopy(state)
                prior_nonfinite = diagnostics.nonfinite_proposals
                prior_clipped = diagnostics.state_clipped
                external = provider.sample(step, config.batch_size, simulator.graph.n_nodes, simulator.node_rule.state_width)
                state = simulator._step(state, external, step, config, (), diagnostics, None)
                session["state"] = state
                session["input"] = external
                session["step"] = step + 1
                step_diagnostics = TransitionDiagnostics(
                    nonfinite_proposals=diagnostics.nonfinite_proposals - prior_nonfinite,
                    state_clipped=diagnostics.state_clipped - prior_clipped,
                    last_state_clip=(
                        diagnostics.last_state_clip
                        if diagnostics.state_clipped > prior_clipped
                        else None
                    ),
                )
                if step_diagnostics.state_clipped:
                    session["last_safety_event"] = {
                        "step": step + 1,
                        "kind": "node_state_clipped",
                        "details": step_diagnostics.last_state_clip,
                    }
                strengths = [
                    simulator.edge_rule.communication_strength(vector)
                    for row in state.edge
                    for vector in row
                ]
                cause = monitor.observe(
                    step + 1, previous, state, strengths, step_diagnostics
                )
                if cause:
                    session["last_warning"] = {"step": step + 1, "cause": cause}
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

    @application.post("/api/async/train")
    def start_async_training(
        payload: AsyncTrainingPayload, background_tasks: BackgroundTasks
    ) -> dict[str, str]:
        runtime.ensure_root()
        seed = _seed(payload.seed)
        job_id = runtime.new_job("async_training", seed, payload.candidate_budget)
        run_directory = runtime.root / "async_runs" / job_id
        architecture = RuleArchitecture(state_width=payload.state_width, hidden_width=3)
        edge_architecture = EdgeArchitecture(
            node_state_width=payload.state_width,
            latent_width=payload.state_width,
            hidden_width=3,
        )
        config = AsyncEvolutionConfig(
            slots=payload.slots,
            replicas=payload.replicas,
            result_batch_size=payload.optimizer_batch,
            max_ticks=payload.max_ticks,
            candidate_budget=payload.candidate_budget,
            seed=seed,
            architecture=architecture,
            edge_architecture=edge_architecture,
            target="joint",
            levels=(
                CurriculumLevel(
                    payload.stage_1_lifetime,
                    input_scale=payload.input_scale,
                    graph_nodes=payload.stage_1_nodes,
                    mean_degree=payload.mean_degree,
                ),
                CurriculumLevel(
                    payload.stage_2_lifetime,
                    payload.disturbance_interval,
                    payload.disturbance_strength,
                    True,
                    payload.input_scale * 1.5,
                    payload.stage_2_nodes,
                    payload.mean_degree,
                ),
            ),
            pathology=PathologyConfig(
                fatal_threshold=payload.fatal_threshold,
                node_growth_alert=payload.node_growth_alert,
                one_direction_steps=payload.one_direction_steps,
            ),
            probes=ProbeConfig(interval=payload.probe_interval, duration=2, amplitude=.10),
            curriculum_window=max(8, payload.optimizer_batch),
            censor_interval=max(2, payload.stage_1_lifetime // 5),
        )

        def worker() -> None:
            try:
                report = run_async_experiment(
                    run_directory,
                    config,
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

    @application.get("/api/async/replays/{run_id}/{candidate_id}/{replica}")
    def async_replay(run_id: str, candidate_id: int, replica: int) -> dict[str, object]:
        try:
            return runtime.async_replay_document(run_id, candidate_id, replica)
        except KeyError as error:
            raise HTTPException(404, "asynchronous run is unavailable") from error
        except (IndexError, ValueError) as error:
            raise HTTPException(404, str(error)) from error

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
        return {
            "models": runtime.available_live_models(),
            "latest_survival": runtime.latest_async_summary(),
        }

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
