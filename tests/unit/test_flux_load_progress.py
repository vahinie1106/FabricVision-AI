"""Unit tests for FLUX load progress / cache status plumbing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from backend_api.services.job_stages import map_step_to_stage
from src.features.custom_generator.model.flux_model_loader import FLUXModelLoader


def test_download_step_maps_to_loading_model():
    assert map_step_to_stage("Downloading FLUX weights (CACHE MISS)", "processing") == (
        "loading_model"
    )
    assert map_step_to_stage("FLUX READY", "processing") == "loading_model"
    assert map_step_to_stage(
        "Initializing FLUX (cache check / dependencies)", "processing"
    ) == "loading_model"
    assert map_step_to_stage(
        "Loading FluxKontextPipeline (T5/CLIP/VAE) - may take several minutes",
        "processing",
    ) == "loading_model"
    assert map_step_to_stage(
        "FLUX READY (in-memory hit) — skipping reload", "processing"
    ) == "loading_model"


def test_loader_reuses_pipeline_without_reload(tmp_path: Path):
    loader = FLUXModelLoader(model_path=tmp_path / "flux-kontext", allow_fallback=True)
    sentinel = MagicMock(name="pipeline")
    loader._pipeline = sentinel
    calls = []
    loader.set_progress_callback(lambda step, pct: calls.append((step, pct)))

    out = loader.load()
    assert out is sentinel
    assert loader._reuse_count == 1
    assert any("Reusing" in c[0] for c in calls)


def test_high_vram_standard_defaults(monkeypatch):
    from src.features.custom_generator.pipeline.garment_generation_pipeline import (
        GarmentGenerationConfig,
        GarmentGenerationPipeline,
    )

    cfg = GarmentGenerationConfig(
        height=512,
        width=512,
        num_inference_steps=3,
        guidance_scale=2.5,
        generation_mode="standard",
        allow_fallback=True,
    )
    # Avoid real weight load in ctor
    pipe = GarmentGenerationPipeline.__new__(GarmentGenerationPipeline)
    pipe.config = cfg
    pipe.logger = MagicMock()
    monkeypatch.setattr(pipe, "_gpu_vram_mb", lambda: 15109.0)
    monkeypatch.delenv("FLUX_GENERATION_RESOLUTION", raising=False)
    monkeypatch.delenv("FLUX_PRODUCTION_SIZE", raising=False)
    monkeypatch.delenv("FLUX_STANDARD_STEPS", raising=False)

    pipe._apply_high_vram_standard_defaults("standard")
    assert pipe.config.height == 768
    assert pipe.config.width == 768
    assert pipe.config.num_inference_steps == 12
    assert pipe.config.guidance_scale == 3.0


def test_low_vram_keeps_standard_preset(monkeypatch):
    from src.features.custom_generator.pipeline.garment_generation_pipeline import (
        GarmentGenerationConfig,
        GarmentGenerationPipeline,
    )

    cfg = GarmentGenerationConfig(
        height=512,
        width=512,
        num_inference_steps=3,
        guidance_scale=3.0,
        generation_mode="standard",
        allow_fallback=True,
    )
    pipe = GarmentGenerationPipeline.__new__(GarmentGenerationPipeline)
    pipe.config = cfg
    pipe.logger = MagicMock()
    monkeypatch.setattr(pipe, "_gpu_vram_mb", lambda: 6144.0)

    pipe._apply_high_vram_standard_defaults("standard")
    assert pipe.config.height == 512
    assert pipe.config.num_inference_steps == 3
