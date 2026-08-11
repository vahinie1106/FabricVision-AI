"""Unit tests for FLUX_GENERATION_RESOLUTION resolution knob."""

from __future__ import annotations

from src.features.custom_generator.inference.flux_inference import (
    DEFAULT_FLUX_GENERATION_RESOLUTION,
    resolve_flux_generation_resolution,
)


def test_default_resolution_is_512(monkeypatch):
    monkeypatch.delenv("FLUX_GENERATION_RESOLUTION", raising=False)
    monkeypatch.delenv("FLUX_PRODUCTION_SIZE", raising=False)
    assert DEFAULT_FLUX_GENERATION_RESOLUTION == 512
    assert resolve_flux_generation_resolution() == 512


def test_env_override_allowed_sizes(monkeypatch):
    monkeypatch.delenv("FLUX_PRODUCTION_SIZE", raising=False)
    for size in (384, 512, 640, 768, 1024):
        monkeypatch.setenv("FLUX_GENERATION_RESOLUTION", str(size))
        assert resolve_flux_generation_resolution() == size


def test_invalid_env_falls_back(monkeypatch):
    monkeypatch.setenv("FLUX_GENERATION_RESOLUTION", "999")
    monkeypatch.delenv("FLUX_PRODUCTION_SIZE", raising=False)
    assert resolve_flux_generation_resolution() == 512
