"""Unit tests for completion-first FLUX VRAM policy."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.features.custom_generator.inference.flux_vram_policy import (
    recommend_oom_fallback,
    select_standard_generation_policy,
)


def test_t4_class_defaults_to_safe_512_without_headroom(monkeypatch):
    monkeypatch.delenv("FLUX_GENERATION_RESOLUTION", raising=False)
    monkeypatch.delenv("FLUX_PRODUCTION_SIZE", raising=False)
    monkeypatch.delenv("FLUX_STANDARD_STEPS", raising=False)
    monkeypatch.delenv("FLUX_ALLOW_HIGH_RES", raising=False)
    monkeypatch.delenv("FLUX_MODEL_CPU_OFFLOAD", raising=False)

    policy = select_standard_generation_policy(
        physical_mb=15109.0,
        free_mb=3200.0,
        offload_strategy="gpu_resident",
    )
    assert policy.height == 512
    assert policy.width == 512
    assert policy.num_inference_steps == 8
    assert policy.profile == "standard_t4_safe"
    assert policy.enable_vae_tiling is True


def test_high_res_requires_real_headroom(monkeypatch):
    monkeypatch.delenv("FLUX_GENERATION_RESOLUTION", raising=False)
    monkeypatch.delenv("FLUX_STANDARD_STEPS", raising=False)
    monkeypatch.delenv("FLUX_ALLOW_HIGH_RES", raising=False)

    policy = select_standard_generation_policy(
        physical_mb=15109.0,
        free_mb=8500.0,
        offload_strategy="gpu_resident",
    )
    assert policy.height == 768
    assert policy.num_inference_steps == 12
    assert policy.profile == "standard_high_res"


def test_forced_resolution_env_wins(monkeypatch):
    monkeypatch.setenv("FLUX_GENERATION_RESOLUTION", "768")
    monkeypatch.setenv("FLUX_STANDARD_STEPS", "10")
    policy = select_standard_generation_policy(
        physical_mb=15109.0,
        free_mb=1000.0,
        offload_strategy="gpu_resident",
    )
    assert policy.height == 768
    assert policy.num_inference_steps == 10


def test_low_vram_keeps_conservative_steps(monkeypatch):
    monkeypatch.delenv("FLUX_GENERATION_RESOLUTION", raising=False)
    monkeypatch.delenv("FLUX_STANDARD_STEPS", raising=False)
    policy = select_standard_generation_policy(
        physical_mb=6144.0,
        free_mb=500.0,
        offload_strategy="model_cpu_offload",
    )
    assert policy.height == 512
    assert policy.num_inference_steps == 3
    assert policy.prefer_model_cpu_offload is True
    assert policy.profile == "standard_low_vram"


def test_oom_fallback_ladder():
    assert recommend_oom_fallback(height=768, width=768, num_inference_steps=12) == {
        "height": 512,
        "width": 512,
        "num_inference_steps": 8,
    }
    assert recommend_oom_fallback(height=512, width=512, num_inference_steps=8) == {
        "height": 512,
        "width": 512,
        "num_inference_steps": 4,
    }
    assert recommend_oom_fallback(height=384, width=384, num_inference_steps=3) is None


def test_pipeline_uses_safe_policy_on_t4_class(monkeypatch):
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
    pipe = GarmentGenerationPipeline.__new__(GarmentGenerationPipeline)
    pipe.config = cfg
    pipe.logger = MagicMock()
    monkeypatch.setattr(pipe, "_gpu_vram_mb", lambda: 15109.0)
    monkeypatch.delenv("FLUX_GENERATION_RESOLUTION", raising=False)
    monkeypatch.delenv("FLUX_PRODUCTION_SIZE", raising=False)
    monkeypatch.delenv("FLUX_STANDARD_STEPS", raising=False)
    monkeypatch.delenv("FLUX_ALLOW_HIGH_RES", raising=False)

    # Simulate low free headroom after GPU-resident NF4 load.
    from src.features.custom_generator.inference import flux_vram_policy as pol

    monkeypatch.setattr(
        pol,
        "collect_vram_diagnostics",
        lambda: pol.VramDiagnostics(
            gpu_name="Tesla T4",
            physical_total_mb=15109.0,
            allocated_mb=11000.0,
            reserved_mb=12000.0,
            free_mb=3109.0,
            max_allocated_mb=11000.0,
            max_reserved_mb=12000.0,
            cuda_available=True,
        ),
    )

    pipe._apply_high_vram_standard_defaults("standard")
    assert pipe.config.height == 512
    assert pipe.config.width == 512
    assert pipe.config.num_inference_steps == 8
    assert pipe.config.guidance_scale == 3.0
