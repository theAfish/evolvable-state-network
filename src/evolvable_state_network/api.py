"""FastAPI route composition for the state-network research workspace."""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from .application.configuration import build_async_training_config
from .application.models import (
    AsyncDiagnosticPayload,
    AsyncTrainingPayload,
    EvolutionPayload,
    ExperimentPayload,
    LiveSessionPayload,
    LiveStepPayload,
)
from .application.runtime import ApplicationRuntime
from .evolution.asynchronous import run_async_experiment, run_diagnostic_experiment
from .evolution.candidate import EdgeArchitecture, RuleArchitecture
from .dashboard import dashboard_document
from .evolution.evaluation import CandidateEvaluator
from .evolution import EvolutionConfig, EvolutionRunner, random_search_smoke_test
from .experiment import ExperimentRequest, run_experiment
from .storage import application_data_dir


def _seed(value: int | None) -> int:
    return secrets.randbelow(2**32) if value is None else value


def create_app(data_dir: Path | None = None) -> FastAPI:
    """Create an isolated application instance rooted at ``data_dir``."""

    root = (data_dir or application_data_dir()).resolve()
    runtime = ApplicationRuntime(root)
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
        config = build_async_training_config(payload, seed)

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
                        "output_url": runtime.artifact_url(
                            run_directory / "evolution_report.json"
                        ),
                        "best_genome_url": runtime.artifact_url(
                            run_directory / "best_genome.json"
                        ),
                        "analysis_url": runtime.artifact_url(
                            run_directory / "analysis" / "analysis.json"
                        ),
                        "trajectory_svg_url": runtime.artifact_url(
                            run_directory / "analysis" / "trajectory.svg"
                        ),
                        "recovery_svg_url": runtime.artifact_url(
                            run_directory / "analysis" / "recovery.svg"
                        ),
                        "replay_index_url": runtime.artifact_url(
                            run_directory / "replays" / "index.json"
                        ),
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
