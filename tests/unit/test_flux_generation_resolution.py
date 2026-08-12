"""Unit tests for FLUX generation / production resolution knobs."""

from __future__ import annotations

from src.features.custom_generator.inference.flux_inference import (
    ALLOWED_FLUX_GENERATION_RESOLUTIONS,
    DEFAULT_FLUX_GENERATION_RESOLUTION,
    DEFAULT_KAGGLE_PRODUCTION_RESOLUTION,
    resolve_flux_generation_resolution,
    resolve_flux_production_resolution,
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
    for size in (384, 512, 640, 704, 720, 768, 1024):
        monkeypatch.setenv("FLUX_GENERATION_RESOLUTION", str(size))
        assert resolve_flux_generation_resolution() == size


def test_invalid_env_falls_back(monkeypatch):
    monkeypatch.setenv("FLUX_GENERATION_RESOLUTION", "999")
    monkeypatch.delenv("FLUX_PRODUCTION_SIZE", raising=False)
    assert resolve_flux_generation_resolution() == 512


def test_production_resolution_defaults_700plus(monkeypatch):
    monkeypatch.delenv("FLUX_GENERATION_RESOLUTION", raising=False)
    monkeypatch.delenv("FLUX_PRODUCTION_SIZE", raising=False)
    monkeypatch.delenv("FLUX_PRODUCTION_RESOLUTION", raising=False)
    assert DEFAULT_KAGGLE_PRODUCTION_RESOLUTION >= 700
    assert resolve_flux_production_resolution() >= 700
    assert 704 in ALLOWED_FLUX_GENERATION_RESOLUTIONS
    assert 720 in ALLOWED_FLUX_GENERATION_RESOLUTIONS


def test_production_resolution_env(monkeypatch):
    monkeypatch.setenv("FLUX_PRODUCTION_RESOLUTION", "768")
    monkeypatch.delenv("FLUX_GENERATION_RESOLUTION", raising=False)
    assert resolve_flux_production_resolution() == 768
