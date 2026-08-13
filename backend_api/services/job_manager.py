"""In-memory + disk-backed async job store.

Kaggle / long FLUX jobs can outlive a uvicorn process restart (OOM kill, accidental
``--reload``, or run_kaggle replacing a stale PID). Pure in-memory storage then
returns HTTP 404 \"Job not found\" after the UI already showed real progress.

Jobs are persisted as JSON under FABRICVISION_JOB_DIR (default: outputs/jobs).
A single process is still required for BackgroundTasks workers; persistence only
keeps status readable after a restart and surfaces a clear \"backend restarted\"
failure instead of a mysterious missing job.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from backend_api.schemas.response_models import JobStatusResponse
from backend_api.services.job_stages import map_step_to_stage

logger = logging.getLogger("fabricvision.job_manager")


def _default_persist_dir() -> Path:
    raw = (os.environ.get("FABRICVISION_JOB_DIR") or "").strip()
    if raw:
        return Path(raw)
    try:
        from backend_api.config.settings import settings

        return Path(settings.OUTPUT_DIR) / "jobs"
    except Exception:
        return Path("outputs") / "jobs"


class JobManager:
    """
    Job store for async AI tasks.

    Memory is authoritative while this process lives. Disk is the recovery path
    when GET /status hits a restarted process (classic Kaggle \"Job not found\").
    """

    def __init__(self, persist_dir: Optional[Path] = None):
        self._jobs: Dict[str, JobStatusResponse] = {}
        self._lock = threading.RLock()
        self._persist_dir = Path(persist_dir) if persist_dir else _default_persist_dir()
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._pid = os.getpid()
        self._load_existing()

    def _job_path(self, job_id: str) -> Path:
        safe = "".join(c for c in job_id if c.isalnum() or c in ("-", "_"))
        return self._persist_dir / f"{safe}.json"

    def _to_dict(self, job: JobStatusResponse) -> Dict[str, Any]:
        if hasattr(job, "model_dump"):
            data = job.model_dump()
        else:
            data = job.dict()
        meta = dict(data.get("metadata") or {})
        meta.setdefault("server_pid", self._pid)
        data["metadata"] = meta
        return data

    def _from_dict(self, data: Dict[str, Any]) -> JobStatusResponse:
        return JobStatusResponse(**data)

    def _persist(self, job: JobStatusResponse) -> None:
        path = self._job_path(job.job_id)
        tmp = path.with_suffix(".json.tmp")
        payload = self._to_dict(job)
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except Exception as exc:
            logger.warning("Failed to persist job %s: %s", job.job_id, exc)

    def _fail_orphaned_inflight(self, job: JobStatusResponse) -> JobStatusResponse:
        """If this in-flight job belongs to a previous PID, mark BACKEND_RESTARTED."""
        meta = dict(job.metadata or {})
        prev_pid = meta.get("server_pid")
        if (
            job.status in ("queued", "processing")
            and prev_pid is not None
            and int(prev_pid) != int(self._pid)
        ):
            job.status = "failed"
            job.error = (
                "Backend process restarted while this job was running "
                f"(old_pid={prev_pid}, new_pid={self._pid}). "
                "Please click Generate again."
            )
            job.error_type = "BACKEND_RESTARTED"
            job.failed_stage = job.stage or job.current_step or "processing"
            job.current_step = "Failed (backend restarted)"
            job.stage = "failed"
            meta["server_pid"] = self._pid
            meta["restarted"] = True
            job.metadata = meta
            self._persist(job)
        return job

    def _load_existing(self) -> None:
        try:
            files = list(self._persist_dir.glob("*.json"))
        except Exception:
            return
        for path in files:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                job = self._fail_orphaned_inflight(self._from_dict(data))
                self._jobs[job.job_id] = job
            except Exception as exc:
                logger.warning("Skipping corrupt job file %s: %s", path, exc)

    def create_job(self) -> str:
        """Create a new job and return its UUID."""
        with self._lock:
            job_id = uuid.uuid4().hex
            job = JobStatusResponse(
                job_id=job_id,
                status="queued",
                progress=0,
                current_step="Waiting for worker...",
                stage="queued",
                metadata={"server_pid": self._pid},
            )
            self._jobs[job_id] = job
            self._persist(job)
            logger.info(
                "job_created job_id=%s pid=%s persist=%s",
                job_id,
                self._pid,
                self._job_path(job_id),
            )
            return job_id

    def get_job(self, job_id: str) -> Optional[JobStatusResponse]:
        """Retrieve job status (memory, then disk)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                path = self._job_path(job_id)
                if path.is_file():
                    try:
                        job = self._fail_orphaned_inflight(
                            self._from_dict(
                                json.loads(path.read_text(encoding="utf-8"))
                            )
                        )
                        self._jobs[job_id] = job
                        logger.info(
                            "job_loaded_from_disk job_id=%s status=%s pid=%s",
                            job_id,
                            job.status,
                            self._pid,
                        )
                    except Exception as exc:
                        logger.warning(
                            "job_disk_load_failed job_id=%s err=%s", job_id, exc
                        )
                        return None
                else:
                    logger.warning(
                        "job_not_found job_id=%s pid=%s known=%s",
                        job_id,
                        self._pid,
                        len(self._jobs),
                    )
                    return None
            return job

    def update_job(self, job_id: str, **kwargs):
        """Update job fields; stage is always derived from status + current_step."""
        with self._lock:
            if job_id not in self._jobs:
                # Recover from disk if memory was wiped mid-flight.
                recovered = self.get_job(job_id)
                if recovered is None:
                    logger.warning(
                        "job_update_missing job_id=%s keys=%s pid=%s",
                        job_id,
                        sorted(kwargs.keys()),
                        self._pid,
                    )
                    return
            job = self._jobs[job_id]
            for key, value in kwargs.items():
                if key == "stage":
                    continue  # derived below — callers must not invent stages
                if hasattr(job, key):
                    setattr(job, key, value)
            job.stage = map_step_to_stage(job.current_step, job.status)
            meta = dict(job.metadata or {})
            meta["server_pid"] = self._pid
            job.metadata = meta
            self._persist(job)


job_manager = JobManager()
