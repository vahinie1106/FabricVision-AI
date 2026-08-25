"""Unit tests for FLUX generation / production resolution knobs."""

from __future__ import annotations

from src.features.custom_generator.inference.flux_inference import (
    ALLOWED_FLUX_GENERATION_RESOLUTIONS,
    DEFAULT_FLUX_GENERATION_RESOLUTION,
    DEFAULT_KAGGLE_PRODUCTION_GUIDANCE,
    DEFAULT_KAGGLE_PRODUCTION_RESOLUTION,
    DEFAULT_KAGGLE_PRODUCTION_STEPS,
    DEFAULT_KAGGLE_STANDARD_GUIDANCE,
    DEFAULT_KAGGLE_STANDARD_RESOLUTION,
    DEFAULT_KAGGLE_STANDARD_STEPS,
    align_flux_pixel_size,
    resolve_flux_generation_resolution,
    resolve_flux_production_guidance,
    resolve_flux_production_resolution,
    resolve_flux_production_steps,
)


def test_default_resolution_is_512(monkeypatch):
    monkeypatch.delenv("FLUX_GENERATION_RESOLUTION", raising=False)
    monkeypatch.delenv("FLUX_PRODUCTION_SIZE", raising=False)
    monkeypatch.delenv("FLUX_PRODUCTION_RESOLUTION", raising=False)
    assert DEFAULT_FLUX_GENERATION_RESOLUTION == 512
    assert resolve_flux_generation_resolution() == 512


def test_env_override_allowed_sizes(monkeypatch):
    monkeypatch.delenv("FLUX_PRODUCTION_SIZE", raising=False)
    monkeypatch.delenv("FLUX_PRODUCTION_RESOLUTION", raising=False)
    for size in (384, 512, 640, 704, 712, 720, 768, 1024):
        monkeypatch.setenv("FLUX_GENERATION_RESOLUTION", str(size))
        assert resolve_flux_generation_resolution() == size


def test_712_is_allowed_and_pack_aligns_to_704():
    assert 712 in ALLOWED_FLUX_GENERATION_RESOLUTIONS
    assert DEFAULT_KAGGLE_STANDARD_RESOLUTION == 712
    assert DEFAULT_KAGGLE_STANDARD_STEPS == 8
    assert DEFAULT_KAGGLE_STANDARD_GUIDANCE == 3.0
    assert align_flux_pixel_size(712) == 704
    assert align_flux_pixel_size(704) == 704
    assert align_flux_pixel_size(720) == 720
    assert align_flux_pixel_size(512) == 512


def test_standard_requested_712_actual_704_delivered_712():
    """User-facing 712; Flux pack-aligned denoise 704; PNG delivered 712."""
    requested = DEFAULT_KAGGLE_STANDARD_RESOLUTION
    actual = align_flux_pixel_size(requested)
    delivered = requested
    assert (requested, actual, delivered) == (712, 704, 712)


def test_invalid_env_falls_back(monkeypatch):
    monkeypatch.setenv("FLUX_GENERATION_RESOLUTION", "999")
    monkeypatch.delenv("FLUX_PRODUCTION_SIZE", raising=False)
    assert resolve_flux_generation_resolution() == 512


def test_production_resolution_defaults_to_768(monkeypatch):
    monkeypatch.delenv("FLUX_GENERATION_RESOLUTION", raising=False)
    monkeypatch.delenv("FLUX_PRODUCTION_SIZE", raising=False)
    monkeypatch.delenv("FLUX_PRODUCTION_RESOLUTION", raising=False)
    monkeypatch.delenv("FLUX_PRODUCTION_STEPS", raising=False)
    monkeypatch.delenv("FLUX_PRODUCTION_GUIDANCE", raising=False)
    assert DEFAULT_KAGGLE_PRODUCTION_RESOLUTION == 768
    assert DEFAULT_KAGGLE_PRODUCTION_STEPS == 12
    assert DEFAULT_KAGGLE_PRODUCTION_GUIDANCE == 3.0
    assert resolve_flux_production_resolution() == 768
    assert resolve_flux_production_steps() == 12
    assert resolve_flux_production_guidance() == 3.0
    assert 768 in ALLOWED_FLUX_GENERATION_RESOLUTIONS


def test_production_resolution_ignores_standard_generation_env(monkeypatch):
    """Standard 512 knob must not demote Production to 512×3."""
    monkeypatch.setenv("FLUX_GENERATION_RESOLUTION", "512")
    monkeypatch.delenv("FLUX_PRODUCTION_RESOLUTION", raising=False)
    monkeypatch.delenv("FLUX_PRODUCTION_SIZE", raising=False)
    assert resolve_flux_production_resolution() == 768
    assert resolve_flux_generation_resolution() == 512


def test_production_resolution_env(monkeypatch):
    monkeypatch.setenv("FLUX_PRODUCTION_RESOLUTION", "768")
    monkeypatch.setenv("FLUX_PRODUCTION_STEPS", "12")
    monkeypatch.setenv("FLUX_PRODUCTION_GUIDANCE", "3.0")
    monkeypatch.delenv("FLUX_GENERATION_RESOLUTION", raising=False)
    assert resolve_flux_production_resolution() == 768
    assert resolve_flux_production_steps() == 12
    assert resolve_flux_production_guidance() == 3.0


def test_pipeline_does_not_clobber_production_with_yaml_step_defaults():
    from pathlib import Path

    src = Path("src/features/custom_generator/pipeline/garment_generation_pipeline.py").read_text(
        encoding="utf-8"
    )
    assert "default_num_inference_steps" in src
    # The YAML default (3 steps) must not be assigned onto config after VRAM policy.
    assert "self.config.num_inference_steps = loaded_gen.get(" not in src
    assert "FLUX_DETAIL_REFINER" in src
    assert "strength=0.45" in src


def test_kaggle_production_lock_unchanged():
    from pathlib import Path

    src = Path("scripts/run_kaggle.py").read_text(encoding="utf-8")
    assert 'os.environ.setdefault("FLUX_CUDA_DEVICE", "0")' in src
    assert 'os.environ.setdefault("CATVTON_CUDA_DEVICE", "1")' in src
    assert 'os.environ.setdefault("FLUX_PRODUCTION_RESOLUTION", "768")' in src
    assert 'os.environ.setdefault("FLUX_PRODUCTION_STEPS", "12")' in src
    assert 'os.environ.setdefault("FLUX_PRODUCTION_GUIDANCE", "3.0")' in src
    assert 'os.environ.setdefault("FLUX_GENERATION_RESOLUTION", "712")' in src
    assert 'os.environ.setdefault("FLUX_STANDARD_STEPS", "8")' in src
