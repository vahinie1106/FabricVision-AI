"""Preset mapping: Preview / Standard / Production must not collapse together."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from src.features.custom_generator.inference import flux_vram_policy as pol
from src.features.custom_generator.pipeline.garment_generation_pipeline import (
    GarmentGenerationConfig,
    GarmentGenerationPipeline,
    normalize_generation_mode,
)


def _t4_diag(*, free_mb: float = 3000.0) -> pol.VramDiagnostics:
    return pol.VramDiagnostics(
        gpu_name="Tesla T4[cuda:0]",
        physical_total_mb=15109.0,
        allocated_mb=4000.0,
        reserved_mb=15109.0 - float(free_mb),
        free_mb=float(free_mb),
        max_allocated_mb=4000.0,
        max_reserved_mb=15109.0 - float(free_mb),
        cuda_available=True,
    )


def _pipe_for_mode(monkeypatch, mode: str, *, free_mb: float = 3000.0):
    monkeypatch.setenv("FLUX_PRODUCTION_RESOLUTION", "768")
    monkeypatch.setenv("FLUX_PRODUCTION_STEPS", "12")
    monkeypatch.setenv("FLUX_PRODUCTION_GUIDANCE", "3.0")
    monkeypatch.setenv("FLUX_PRODUCTION_NO_OOM_FALLBACK", "1")
    monkeypatch.delenv("FLUX_GENERATION_RESOLUTION", raising=False)
    monkeypatch.delenv("FLUX_PRODUCTION_SIZE", raising=False)
    monkeypatch.delenv("FLUX_ALLOW_HIGH_RES", raising=False)
    monkeypatch.delenv("FLUX_STANDARD_STEPS", raising=False)
    monkeypatch.delenv("FLUX_PREVIEW_STEPS", raising=False)
    monkeypatch.setattr(pol, "collect_vram_diagnostics", lambda: _t4_diag(free_mb=free_mb))
    cfg = GarmentGenerationConfig(
        config_dir="configs",
        config_path="configs/custom_generator/flux_config.yaml",
        generation_mode=mode,
        allow_fallback=False,
    )
    return GarmentGenerationPipeline(
        config=cfg,
        model_loader=MagicMock(),
        inference_engine=MagicMock(),
    )


def test_normalize_ui_labels():
    assert normalize_generation_mode("Preview") == "preview"
    assert normalize_generation_mode("Standard") == "standard"
    assert normalize_generation_mode("Production") == "production"
    assert normalize_generation_mode("production") == "production"
    assert normalize_generation_mode("high quality") == "production"


def test_preview_preset_stays_preview(monkeypatch):
    pipe = _pipe_for_mode(monkeypatch, "Preview")
    assert pipe.config.mode_key == "preview"
    assert pipe.config.generation_mode == "Preview"
    assert pipe.config.height == 384
    assert pipe.config.width == 384
    assert pipe.config.num_inference_steps == 4


def test_standard_preset_on_t4_is_not_production(monkeypatch):
    pipe = _pipe_for_mode(monkeypatch, "Standard")
    assert pipe.config.mode_key == "standard"
    assert pipe.config.generation_mode == "Standard"
    # T4 completion-first Standard is 512×8, never the 3050 YAML 512×3,
    # and never silent Production 768×12.
    assert pipe.config.height == 512
    assert pipe.config.width == 512
    assert pipe.config.num_inference_steps == 8
    assert (pipe.config.height, pipe.config.num_inference_steps) != (512, 3)
    assert (pipe.config.height, pipe.config.num_inference_steps) != (768, 12)


def test_production_preset_on_t4_is_768x12(monkeypatch):
    pipe = _pipe_for_mode(monkeypatch, "Production", free_mb=2500.0)
    assert pipe.config.mode_key == "production"
    assert pipe.config.generation_mode == "Production"
    assert pipe.config.height == 768
    assert pipe.config.width == 768
    assert pipe.config.num_inference_steps == 12
    assert float(pipe.config.guidance_scale) == 3.0
    # Must not collapse to the Standard 512×3 YAML preset.
    assert (pipe.config.height, pipe.config.num_inference_steps) != (512, 3)


def test_production_no_oom_fallback_does_not_downgrade_resolution(monkeypatch):
    monkeypatch.setenv("FLUX_PRODUCTION_NO_OOM_FALLBACK", "1")
    pipe = _pipe_for_mode(monkeypatch, "Production", free_mb=1200.0)
    assert pipe.config.height == 768
    assert pipe.config.width == 768
    assert pipe.config.num_inference_steps == 12
    src = Path("src/features/custom_generator/inference/flux_inference.py").read_text(
        encoding="utf-8"
    )
    assert "FLUX_PRODUCTION_NO_OOM_FALLBACK" in src
    assert "no silent resize" in src


def test_frontend_exposes_all_three_modes():
    page = Path("frontend/src/app/studio/custom-garment/page.tsx").read_text(
        encoding="utf-8"
    )
    assert 'options={["Preview", "Standard", "Production"]}' in page
    svc = Path("frontend/src/services/generationService.ts").read_text(encoding="utf-8")
    assert 'formData.append("generation_mode", req.generationMode || "standard")' in svc
