"""Preset mapping: Preview / Standard / Production must not collapse together."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from src.features.custom_generator.inference import flux_vram_policy as pol
from src.features.custom_generator.pipeline.garment_generation_pipeline import (
    GarmentGenerationConfig,
    GarmentGenerationPipeline,
    assert_production_config_lock,
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
    # T4 Standard is 712×8 / guidance 3.0, never the 3050 YAML 512×3,
    # and never silent Production 768×12.
    assert pipe.config.height == 712
    assert pipe.config.width == 712
    assert pipe.config.num_inference_steps == 8
    assert float(pipe.config.guidance_scale) == 3.0
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


def test_production_lock_rejects_three_steps():
    try:
        assert_production_config_lock(
            mode_key="production",
            height=512,
            width=512,
            steps=3,
            guidance=3.0,
            physical_vram_mb=15109.0,
        )
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "12" in str(exc)
        assert "3" in str(exc)


def test_production_lock_rejects_t4_512():
    try:
        assert_production_config_lock(
            mode_key="production",
            height=512,
            width=512,
            steps=12,
            guidance=3.0,
            physical_vram_mb=15109.0,
        )
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "768" in str(exc)


def test_production_lock_allows_t4_768x12():
    assert_production_config_lock(
        mode_key="production",
        height=768,
        width=768,
        steps=12,
        guidance=3.0,
        physical_vram_mb=15109.0,
    )


def test_production_lock_rejects_512_even_on_local():
    try:
        assert_production_config_lock(
            mode_key="production",
            height=512,
            width=512,
            steps=12,
            guidance=3.0,
            physical_vram_mb=6144.0,
        )
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "768" in str(exc)
        assert "512" in str(exc)


def test_standard_three_steps_does_not_trip_production_lock():
    assert_production_config_lock(
        mode_key="standard",
        height=512,
        width=512,
        steps=3,
        guidance=3.0,
        physical_vram_mb=15109.0,
    )


def test_frontend_exposes_all_three_modes():
    page = Path("frontend/src/app/studio/custom-garment/page.tsx").read_text(
        encoding="utf-8"
    )
    assert 'options={["Preview", "Standard", "Production"]}' in page
    assert "NEXT_PUBLIC_DEFAULT_GENERATION_MODE" in page
    assert "[FRONTEND QUALITY DEBUG]" in page
    assert "selected_generation_mode=" in page
    svc = Path("frontend/src/services/generationService.ts").read_text(encoding="utf-8")
    assert 'formData.append("generation_mode", selectedMode)' in svc
    assert 'req.generationMode || "standard"' not in svc
    assert "[FRONTEND QUALITY DEBUG]" in svc
    assert "/generate${qs}" in svc or "generation_mode=${encodeURIComponent" in svc
    assert "X-Fabricvision-Generation-Mode" in svc
    api = Path("backend_api/routes/generation.py").read_text(encoding="utf-8")
    assert 'Form("standard")' not in api
    assert "[API QUALITY DEBUG]" in api


def test_production_cannot_resolve_to_three_steps(monkeypatch):
    pipe = _pipe_for_mode(monkeypatch, "Production")
    assert pipe.config.num_inference_steps == 12
    assert pipe.config.num_inference_steps != 3
    try:
        assert_production_config_lock(
            mode_key="production",
            height=int(pipe.config.height),
            width=int(pipe.config.width),
            steps=3,
            guidance=3.0,
            physical_vram_mb=15109.0,
        )
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "12" in str(exc)


def test_production_cannot_resolve_to_512(monkeypatch):
    pipe = _pipe_for_mode(monkeypatch, "Production")
    assert pipe.config.height == 768
    assert pipe.config.width == 768
    try:
        assert_production_config_lock(
            mode_key="production",
            height=512,
            width=512,
            steps=12,
            guidance=3.0,
            physical_vram_mb=15109.0,
        )
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "768" in str(exc)


def test_production_ignores_standard_env_knobs(monkeypatch):
    monkeypatch.setenv("FLUX_GENERATION_RESOLUTION", "512")
    monkeypatch.setenv("FLUX_STANDARD_STEPS", "3")
    monkeypatch.setenv("FLUX_PRODUCTION_RESOLUTION", "768")
    monkeypatch.setenv("FLUX_PRODUCTION_SIZE", "768")
    monkeypatch.setenv("FLUX_PRODUCTION_STEPS", "12")
    monkeypatch.setenv("FLUX_PRODUCTION_GUIDANCE", "3.0")
    monkeypatch.setenv("FLUX_PRODUCTION_NO_OOM_FALLBACK", "1")
    monkeypatch.delenv("FLUX_ALLOW_HIGH_RES", raising=False)
    monkeypatch.delenv("FLUX_PREVIEW_STEPS", raising=False)
    monkeypatch.setattr(pol, "collect_vram_diagnostics", lambda: _t4_diag(free_mb=3000.0))
    cfg = GarmentGenerationConfig(
        config_dir="configs",
        config_path="configs/custom_generator/flux_config.yaml",
        generation_mode="Production",
        allow_fallback=False,
    )
    pipe = GarmentGenerationPipeline(
        config=cfg,
        model_loader=MagicMock(),
        inference_engine=MagicMock(),
    )
    assert pipe.config.mode_key == "production"
    assert pipe.config.height == 768
    assert pipe.config.width == 768
    assert pipe.config.num_inference_steps == 12
    assert float(pipe.config.guidance_scale) == 3.0
    assert pipe.config.num_inference_steps != 3


def test_production_on_rtx3050_still_resolves_768x12(monkeypatch):
    monkeypatch.setenv("FLUX_PRODUCTION_RESOLUTION", "768")
    monkeypatch.setenv("FLUX_PRODUCTION_STEPS", "12")
    monkeypatch.setenv("FLUX_PRODUCTION_GUIDANCE", "3.0")
    monkeypatch.delenv("FLUX_GENERATION_RESOLUTION", raising=False)
    monkeypatch.delenv("FLUX_STANDARD_STEPS", raising=False)
    monkeypatch.setattr(
        pol,
        "collect_vram_diagnostics",
        lambda: pol.VramDiagnostics(
            gpu_name="NVIDIA GeForce RTX 3050 6GB Laptop GPU",
            physical_total_mb=6144.0,
            allocated_mb=3000.0,
            reserved_mb=3500.0,
            free_mb=2644.0,
            max_allocated_mb=3000.0,
            max_reserved_mb=3500.0,
            cuda_available=True,
        ),
    )
    cfg = GarmentGenerationConfig(
        config_dir="configs",
        config_path="configs/custom_generator/flux_config.yaml",
        generation_mode="Production",
        allow_fallback=False,
    )
    pipe = GarmentGenerationPipeline(
        config=cfg,
        model_loader=MagicMock(),
        inference_engine=MagicMock(),
    )
    assert pipe.config.mode_key == "production"
    assert pipe.config.height == 768
    assert pipe.config.width == 768
    assert pipe.config.num_inference_steps == 12
    assert float(pipe.config.guidance_scale) == 3.0
    assert_production_config_lock(
        mode_key=pipe.config.mode_key,
        height=pipe.config.height,
        width=pipe.config.width,
        steps=pipe.config.num_inference_steps,
        guidance=pipe.config.guidance_scale,
        physical_vram_mb=6144.0,
    )
