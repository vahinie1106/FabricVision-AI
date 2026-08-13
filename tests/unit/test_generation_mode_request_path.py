"""Request-path: UI Production must not silently become Standard / 3 steps."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend_api.services.generation_service import resolve_incoming_generation_mode
from src.features.custom_generator.pipeline.garment_generation_pipeline import (
    normalize_generation_mode,
)


def test_resolve_incoming_prefers_form_then_header_then_query():
    assert resolve_incoming_generation_mode("Production", "Standard", "Preview") == "Production"
    assert resolve_incoming_generation_mode(None, "Production", "Standard") == "Production"
    assert resolve_incoming_generation_mode("", None, "Production") == "Production"
    assert resolve_incoming_generation_mode("  Production  ", None, None) == "Production"


def test_resolve_incoming_rejects_missing_mode():
    with pytest.raises(ValueError, match="required"):
        resolve_incoming_generation_mode(None, None, None)
    with pytest.raises(ValueError, match="required"):
        resolve_incoming_generation_mode("", "  ", None)


def test_production_label_normalizes_and_not_standard():
    assert normalize_generation_mode("Production") == "production"
    assert normalize_generation_mode("Production") != "standard"


def test_api_route_has_no_silent_standard_form_default():
    src = Path("backend_api/routes/generation.py").read_text(encoding="utf-8")
    assert 'generation_mode: str = Form("standard")' not in src
    assert "Form(None)" in src
    assert "[API QUALITY DEBUG]" in src
    assert "x-fabricvision-generation-mode" in src.lower()


def test_generation_service_logs_quality_debug_fields():
    src = Path("backend_api/services/generation_service.py").read_text(encoding="utf-8")
    for needle in (
        "api_received_generation_mode",
        "requested_mode",
        "normalized_mode",
        "resolved_mode",
        "resolved_resolution",
        "resolved_steps",
        "resolved_guidance",
        'generation_mode: str = "standard"',
    ):
        if needle == 'generation_mode: str = "standard"':
            assert needle not in src
        else:
            assert needle in src


def test_flux_actual_config_logs_exist():
    src = Path("src/features/custom_generator/inference/flux_inference.py").read_text(
        encoding="utf-8"
    )
    assert "[FLUX ACTUAL CONFIG]" in src
    assert "[FLUX ACTUAL OUTPUT]" in src
    assert "Production configuration invalid: expected 12 steps" in src
    assert "Production configuration invalid: expected 768x768" in src
    inf = Path(
        "src/features/custom_generator/pipeline/garment_generation_pipeline.py"
    ).read_text(encoding="utf-8")
    assert "[PIPELINE QUALITY DEBUG]" in src or "[PIPELINE QUALITY DEBUG]" in inf
