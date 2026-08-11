"""Unit tests: generation job progress stays pollable; metadata on completion."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from backend_api.services.job_manager import JobManager
from backend_api.services.job_stages import map_step_to_stage


def test_map_step_does_not_treat_upload_as_loading_model():
    # Regression: bare "load" matched "uploading" / similar substrings.
    assert map_step_to_stage("Uploading Fabric", "processing") != "loading_model"
    assert map_step_to_stage("Loading model", "processing") == "loading_model"
    assert map_step_to_stage("Preparing fabric", "processing") == "preparing_fabric"
    assert map_step_to_stage("Preparing garment conditioning", "processing") == (
        "preparing_conditioning"
    )
    assert map_step_to_stage("Encoding prompt", "processing") == "encoding_prompt"
    assert map_step_to_stage("Generating (step 3/28)", "processing") == "generating"
    assert map_step_to_stage("Decoding image", "processing") == "decoding"
    assert map_step_to_stage("Saving result", "processing") == "saving"


def test_job_manager_preserves_metadata_on_status():
    jm = JobManager()
    job_id = jm.create_job()
    meta = {
        "category": "kurti",
        "fabric": "Cotton",
        "styleAffinity": "casual",
        "confidenceScore": 0.95,
        "model": "FLUX.1-Kontext",
        "generation_mode": "standard",
        "height": 768,
        "width": 768,
        "num_inference_steps": 28,
        "has_image": True,
    }
    jm.update_job(
        job_id,
        status="completed",
        progress=100,
        current_step="Completed",
        result_url="/outputs/generated_garments/images/g.png",
        metadata=meta,
    )
    job = jm.get_job(job_id)
    assert job.status == "completed"
    assert job.stage == "completed"
    assert job.metadata is not None
    assert job.metadata["category"] == "kurti"
    assert job.metadata["height"] == 768
    assert job.result_url == "/outputs/generated_garments/images/g.png"


def test_process_generation_offloads_blocking_work(monkeypatch, tmp_path: Path):
    """
    Model load must not block the asyncio event loop — otherwise GET /status hangs
    and the UI stays at 5% Uploading Fabric for minutes.
    """
    import threading

    from backend_api.services import generation_service as gs
    import backend_api.services.flux_warmup as warm

    # Isolate job store + warmup module state (other tests may leave state=failed).
    jm = JobManager()
    monkeypatch.setattr(gs, "job_manager", jm)
    warm.reset_warmup_state()

    fabric = tmp_path / "fabric.png"
    Image.new("RGB", (64, 64), (200, 100, 50)).save(fabric)

    load_entered = threading.Event()
    load_release = threading.Event()

    class FakeFluxManager:
        loader = None
        allow_fallback = False
        model_path = None
        hf_model_id = None

        def recover_after_oom(self):
            return None

    class FakeModelManager:
        active_model = None
        flux_manager = FakeFluxManager()

        def switch_to(self, name: str):
            # Signal that blocking work started, then wait — must run in a worker thread.
            load_entered.set()
            if not load_release.wait(timeout=5):
                raise TimeoutError("test load_release not signaled")
            pipe = MagicMock()
            loader = MagicMock()
            loader.pipeline = pipe
            loader._reuse_count = 0
            loader._offload_strategy = "model_cpu_offload"
            loader._attention_backend = "memory_efficient_sdpa"
            loader._torch_compile_enabled = False
            loader._used_bnb_4bit = True
            FakeFluxManager.loader = loader
            self.flux_manager.loader = loader
            self.active_model = "flux"

        def clear_vram(self):
            return None

    monkeypatch.setattr(gs, "model_manager", FakeModelManager())

    out_img = tmp_path / "out.png"
    Image.new("RGB", (64, 64), (10, 20, 30)).save(out_img)

    class FakePipeline:
        def __init__(self, *args, **kwargs):
            self.config = MagicMock(
                generation_mode="standard",
                height=768,
                width=768,
                num_inference_steps=28,
                guidance_scale=2.5,
            )
            self.inference_engine = MagicMock()
            self.inference_engine.last_execution_stats = {
                "was_fallback_used": False,
                "was_real_flux_used": True,
                "generation_time_s": 1.23,
                "peak_vram_mb": 1000.0,
                "model_reused": False,
            }

        def run(self, **kwargs):
            cb = kwargs.get("progress_callback")
            if cb:
                cb("Preparing fabric", 22)
                cb("Preparing garment conditioning", 32)
                cb("Encoding prompt", 45)
                cb("Generating (step 1/2)", 55)
                cb("Decoding image", 88)
                cb("Saving result", 94)
                cb("Completed", 100)
            return {
                "image_path": str(out_img),
                "output_path": str(out_img),
                "metadata": {
                    "positive_prompt": "test prompt",
                    "negative_prompt": "",
                    "prompt_stats": {"token_count": 12},
                },
            }

    monkeypatch.setattr(
        "src.features.custom_generator.pipeline.garment_generation_pipeline.GarmentGenerationPipeline",
        FakePipeline,
    )
    monkeypatch.setattr(
        "src.features.custom_generator.pipeline.garment_generation_pipeline.GarmentGenerationConfig",
        MagicMock,
    )
    monkeypatch.setattr(
        "src.features.custom_generator.pipeline.garment_generation_pipeline.normalize_generation_mode",
        lambda m: "standard",
    )

    # OUTPUT_DIR relative resolution for result_url
    monkeypatch.setattr(gs.settings, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(gs.settings, "BASE_DIR", tmp_path)

    job_id = jm.create_job()

    async def _run():
        task = asyncio.create_task(
            gs.process_generation(
                job_id,
                str(fabric),
                "kurti",
                "regular",
                "casual",
                generation_mode="standard",
            )
        )

        # While "model load" is blocked in the worker thread, status must still be readable
        # on the event loop (this is the bugfix for stuck 5% UI).
        for _ in range(100):
            if load_entered.is_set():
                break
            await asyncio.sleep(0.05)
        assert load_entered.is_set(), "worker never entered model load"
        job_mid = jm.get_job(job_id)
        assert job_mid.status == "processing"
        assert job_mid.progress >= 5
        assert job_mid.stage == "loading_model"
        assert "Initializing FLUX" in (job_mid.current_step or "") or "Loading" in (
            job_mid.current_step or ""
        ) or "API-process" in (job_mid.current_step or "") or "FLUX" in (
            job_mid.current_step or ""
        )

        load_release.set()
        await asyncio.wait_for(task, timeout=10)

    asyncio.run(_run())

    job = jm.get_job(job_id)
    assert job.status == "completed", job.error
    assert job.progress == 100
    assert job.stage == "completed"
    assert job.metadata is not None
    assert job.metadata.get("category") == "kurti"
    assert job.metadata.get("model") == "FLUX.1-Kontext"
    assert job.metadata.get("has_image") is True
    assert job.result_url
    assert job.result_url.startswith("/outputs/")


def test_progress_callback_updates_stage_sequence():
    jm = JobManager()
    job_id = jm.create_job()
    steps = [
        ("Loading model", 12, "loading_model"),
        ("Preparing fabric", 22, "preparing_fabric"),
        ("Preparing garment conditioning", 32, "preparing_conditioning"),
        ("Encoding prompt", 45, "encoding_prompt"),
        ("Generating (step 1/28)", 55, "generating"),
        ("Decoding image", 88, "decoding"),
        ("Saving result", 94, "saving"),
    ]
    for step, pct, expected_stage in steps:
        jm.update_job(job_id, status="processing", progress=pct, current_step=step)
        job = jm.get_job(job_id)
        assert job.stage == expected_stage, (step, job.stage)
        assert job.progress == pct
