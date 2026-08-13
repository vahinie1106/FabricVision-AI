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


def test_job_manager_survives_process_restart(tmp_path):
    """Disk persistence: status remains after a new JobManager PID (Kaggle restart)."""
    store = tmp_path / "jobs"
    jm1 = JobManager(persist_dir=store)
    job_id = jm1.create_job()
    jm1.update_job(
        job_id,
        status="processing",
        progress=45,
        current_step="Encoding prompt",
    )
    assert jm1.get_job(job_id).progress == 45

    # Simulate a different uvicorn PID loading the same on-disk job.
    jm2 = JobManager(persist_dir=store)
    jm2._pid = jm1._pid + 99999
    # Force re-load path (empty memory, re-read disk with new pid).
    jm2._jobs.clear()
    jm2._load_existing()
    job = jm2.get_job(job_id)
    assert job is not None
    assert job.job_id == job_id
    # In-flight job from another PID must not look like a live success.
    assert job.status == "failed"
    assert job.error_type == "BACKEND_RESTARTED"
    assert "restarted" in (job.error or "").lower()


def test_job_manager_get_job_marks_orphan_without_preload(tmp_path):
    """get_job disk path must apply BACKEND_RESTARTED even if _load_existing was skipped."""
    store = tmp_path / "jobs"
    jm1 = JobManager(persist_dir=store)
    job_id = jm1.create_job()
    jm1.update_job(
        job_id,
        status="processing",
        progress=45,
        current_step="Encoding prompt",
    )

    jm2 = JobManager(persist_dir=store)
    jm2._pid = jm1._pid + 4242
    jm2._jobs.clear()
    job = jm2.get_job(job_id)
    assert job is not None
    assert job.status == "failed"
    assert job.error_type == "BACKEND_RESTARTED"


def test_job_manager_completed_survives_restart(tmp_path):
    store = tmp_path / "jobs"
    jm1 = JobManager(persist_dir=store)
    job_id = jm1.create_job()
    jm1.update_job(
        job_id,
        status="completed",
        progress=100,
        current_step="Completed",
        result_url="/outputs/demo.png",
    )
    jm2 = JobManager(persist_dir=store)
    job = jm2.get_job(job_id)
    assert job is not None
    assert job.status == "completed"
    assert job.result_url == "/outputs/demo.png"
