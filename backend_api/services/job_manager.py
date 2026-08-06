import uuid
from typing import Dict, Any

from backend_api.schemas.response_models import JobStatusResponse

class JobManager:
    """
    In-memory job store for async AI tasks.
    In a real production environment with multiple workers, this would be backed by Redis + Celery.
    """
    def __init__(self):
        self._jobs: Dict[str, JobStatusResponse] = {}

    def create_job(self) -> str:
        """Create a new job and return its UUID."""
        job_id = uuid.uuid4().hex
        self._jobs[job_id] = JobStatusResponse(
            job_id=job_id,
            status="queued",
            progress=0,
            current_step="Waiting for worker..."
        )
        return job_id

    def get_job(self, job_id: str) -> JobStatusResponse:
        """Retrieve job status."""
        return self._jobs.get(job_id)

    def update_job(self, job_id: str, **kwargs):
        """Update job fields."""
        if job_id in self._jobs:
            job = self._jobs[job_id]
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)

job_manager = JobManager()
