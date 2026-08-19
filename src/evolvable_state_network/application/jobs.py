"""Thread-safe lifecycle management for background application jobs.

The registry is transport-agnostic: API routes and other application services
can publish progress without knowing how jobs are stored or synchronized.
"""

from __future__ import annotations

from threading import Event, Lock
from uuid import uuid4


class JobRegistry:
    """Own mutable background-job state behind one synchronization boundary."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, object]] = {}
        self._termination_events: dict[str, Event] = {}
        self._lock = Lock()

    def create(self, kind: str, seed: int, total: int | None) -> str:
        job_id = uuid4().hex
        with self._lock:
            self._jobs[job_id] = {
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
                "termination_requested": False,
            }
            self._termination_events[job_id] = Event()
        return job_id

    def publish(self, job_id: str, event: dict[str, object]) -> None:
        """Record progress in the appropriate bounded top-level job field."""
        with self._lock:
            job = self._jobs[job_id]
            phase = str(event.get("phase", "running"))
            job["phase"] = phase
            if phase == "smoke":
                job["samples"].append(event)
            elif phase == "generation":
                job["generations"].append(event)
            else:
                job["latest"] = event

    def snapshot(self, job_id: str) -> dict[str, object]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return dict(job)

    def request_termination(self, job_id: str, *, kind: str) -> bool:
        """Request cooperative termination of a running job of ``kind``."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["kind"] != kind:
                raise KeyError(job_id)
            if job["status"] != "running":
                return False
            job["termination_requested"] = True
            job["phase"] = "termination_requested"
            self._termination_events[job_id].set()
            return True

    def termination_requested(self, job_id: str) -> bool:
        with self._lock:
            event = self._termination_events.get(job_id)
            return event.is_set() if event is not None else False

    def finish(self, job_id: str, result: dict[str, object]) -> None:
        self._set_terminal_state(job_id, "complete", result=result)

    def fail(self, job_id: str, error: Exception) -> None:
        self._set_terminal_state(job_id, "failed", error=str(error))

    def terminate(self, job_id: str) -> None:
        self._set_terminal_state(job_id, "terminated", termination_requested=True)

    def _set_terminal_state(self, job_id: str, status: str, **fields: object) -> None:
        with self._lock:
            self._jobs[job_id].update({"status": status, "phase": status, **fields})
