"""Unit tests for authoritative job stage mapping."""

from backend_api.services.job_stages import map_step_to_stage
from backend_api.services.job_manager import JobManager


def test_map_step_to_stage_generation_lifecycle():
    assert map_step_to_stage("Waiting for worker...", "queued") == "queued"
    assert map_step_to_stage("Loading model", "processing") == "loading_model"
    assert map_step_to_stage("Encoding prompt", "processing") == "encoding_prompt"
    assert map_step_to_stage("Generating (step 2/3)", "processing") == "generating"
    assert map_step_to_stage("Decoding image", "processing") == "decoding"
    assert map_step_to_stage("Saving result", "processing") == "saving"
    assert map_step_to_stage("Completed", "completed") == "completed"
    assert map_step_to_stage("Failed", "failed") == "failed"


def test_job_manager_derives_stage_on_update():
    jm = JobManager()
    job_id = jm.create_job()
    job = jm.get_job(job_id)
    assert job.stage == "queued"

    jm.update_job(job_id, status="processing", progress=92, current_step="Saving result")
    job = jm.get_job(job_id)
    assert job.stage == "saving"
    assert job.status == "processing"

    jm.update_job(job_id, status="completed", progress=100, current_step="Completed")
    job = jm.get_job(job_id)
    assert job.stage == "completed"
