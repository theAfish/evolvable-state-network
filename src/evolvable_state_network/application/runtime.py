"""Stateful services used by the HTTP API.

This module owns jobs, persisted asynchronous-run artifacts, and incremental
live sessions.  Keeping those concerns out of :mod:`api` leaves route handlers
responsible only for transport-level validation and response mapping.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from random import Random
from threading import Lock
from uuid import uuid4

from ..evolution.asynchronous import (
    HealthMonitor,
    PathologyConfig,
    async_config_from_dict,
    replay_archived_candidate,
)
from ..evolution.candidate import (
    EdgeArchitecture,
    FixedEdgeRule,
    MLPUpdateRule,
    RuleArchitecture,
)
from ..evolution.genome import GenomeCodec
from ..dashboard import dashboard_document
from ..graph import generate_random_graph
from ..simulation import Simulation, SimulationConfig, TransitionDiagnostics
from .models import LiveSessionPayload


class ApplicationRuntime:
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
        # These are the only true right-censored records: lives still running
        # when the external run limit interrupted a stage.
        living = [item for item in censored if item.get("kind") == "run_stop"]
        candidates = [
            {
                "candidate_id": item["candidate_id"],
                "status": item["status"],
                "level": item["level"],
                "age": item["age"],
                "cma_lifetime": item.get("cma_lifetime", item["age"]),
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
                        "debug_replay_url": (
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
                "stable_population_size": config.get("stable_population_size"),
                "node_state_width": config.get("architecture", {}).get("state_width"),
                "initial_state_scale": config.get("initial_state_scale", .12),
                "levels": config.get("levels", []),
                "fatal_threshold": config.get("pathology", {}).get("fatal_threshold"),
                "node_growth_alert": config.get("pathology", {}).get("node_growth_alert"),
                "one_direction_steps": config.get("pathology", {}).get("one_direction_steps"),
                "probe_interval": config.get("probes", {}).get("interval"),
                "deployment_validation": {
                    "replicas": config.get("deployment_validation_replicas", 3),
                    "nodes": config.get("deployment_validation_nodes", 24),
                    "mean_degree": config.get("deployment_validation_mean_degree", 5.0),
                    "autonomous_steps": config.get("deployment_autonomous_steps", 200),
                    "workers": config.get("deployment_validation_workers"),
                },
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
                "right_censored": self.artifact_url(censored_path),
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
            # No compatibility path for records produced before the fresh-graph
            # viability screen existed: those candidates may have converged on
            # their small training graph and must not be offered to Live.
            validation = item.get("deployment_validation")
            if (
                not isinstance(validation, dict)
                or not bool(validation.get("passed"))
                or not isinstance(validation.get("autonomous"), list)
                or not isinstance(validation.get("perturbed"), list)
            ):
                return False
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
        models.sort(
            key=lambda item: (
                item["run_kind"] == "training",
                tuple(item["_selection_key"]),
                float(item["modified"]),
            ),
            reverse=True,
        )
        for global_rank, model in enumerate(models, start=1):
            model["global_rank"] = global_rank
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
        raise ValueError("select a trained survival elite")

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
        # Match Survival evaluation: every node-coordinate begins as a small,
        # independent, zero-mean perturbation.  It is deterministic for a
        # chosen Live seed, but does not inject a signal during evolution.
        state = simulator.initial_state(payload.batch_size)
        initial_rng = Random(payload.seed)
        state.node = [
            [
                tuple(
                    initial_rng.gauss(0.0, payload.initial_state_scale)
                    for _ in range(node_rule.state_width)
                )
                for _ in range(payload.nodes)
            ]
            for _ in range(payload.batch_size)
        ]
        session_id = uuid4().hex
        zero_input = [[(0.0,) * node_rule.state_width for _ in range(payload.nodes)] for _ in range(payload.batch_size)]
        session: dict[str, object] = {
            "id": session_id,
            "model_id": payload.model_id,
            "graph": graph,
            "config": config,
            "simulator": simulator,
            "state": state,
            "step": 0,
            "input": zero_input,
            "diagnostics": TransitionDiagnostics(),
            "monitor": HealthMonitor(PathologyConfig(**dict(document.get("pathology", {})))),
            # Sandbox is intentionally observational: training decides survival;
            # Live replay keeps running so a failure can be inspected.
            "last_warning": None,
            "last_safety_event": None,
            "topology": payload.topology,
            "seed": payload.seed,
            "initial_state_scale": payload.initial_state_scale,
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
            "initial_state_scale": session["initial_state_scale"],
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
            config = session["config"]
            diagnostics = session["diagnostics"]
            monitor = session["monitor"]
            assert isinstance(simulator, Simulation)
            assert isinstance(config, SimulationConfig)
            assert isinstance(diagnostics, TransitionDiagnostics)
            assert isinstance(monitor, HealthMonitor)
            for _ in range(count):
                step = int(session["step"])
                previous = deepcopy(state)
                prior_nonfinite = diagnostics.nonfinite_proposals
                prior_clipped = diagnostics.state_clipped
                external = [[(0.0,) * simulator.node_rule.state_width for _ in range(simulator.graph.n_nodes)] for _ in range(config.batch_size)]
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

    def new_job(self, kind: str, seed: int, total: int | None) -> str:
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
