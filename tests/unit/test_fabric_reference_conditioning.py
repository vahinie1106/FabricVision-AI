"""Uploaded fabric must be the visual reference passed into FLUX Kontext."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from src.features.custom_generator.inference.fabric_conditioning import (
    build_garment_conditioning_image,
)
from src.features.custom_generator.pipeline.garment_generation_pipeline import (
    GarmentGenerationConfig,
    GarmentGenerationPipeline,
)
from src.features.custom_generator.prompting.garment_prompt_builder import (
    CLIP_MAX_TOKENS,
    GarmentPromptBuilder,
)
from src.features.custom_generator.validation.garment_validator import GarmentValidator


MAGENTA = (220, 20, 180)
YELLOW = (240, 220, 40)


def _striped_fabric(size: int = 256) -> Image.Image:
    """Recognizable printed swatch — not a generic solid."""
    img = Image.new("RGB", (size, size), YELLOW)
    draw = ImageDraw.Draw(img)
    for x in range(0, size, 16):
        if (x // 16) % 2 == 0:
            draw.rectangle([x, 0, x + 8, size], fill=MAGENTA)
    return img


def test_pipeline_requires_uploaded_fabric_reference():
    pipe = GarmentGenerationPipeline.__new__(GarmentGenerationPipeline)
    pipe.config = GarmentGenerationConfig(generation_mode="standard")
    pipe.config.mode_key = "standard"
    pipe.logger = logging.getLogger("test.fabric_ref")
    with pytest.raises(RuntimeError, match="Fabric reference image is required"):
        pipe.run(fabric_metadata={}, user_customization={})


def test_match_fabric_conditioning_keeps_uploaded_print_pixels():
    fabric = _striped_fabric(128)
    out = build_garment_conditioning_image(
        fabric,
        garment_type="dress",
        width=128,
        height=128,
        sleeve="short_sleeve",
        target_color="match_fabric",
    )
    assert out.size == (128, 128)
    assert getattr(build_garment_conditioning_image, "last_recolor_audit", None) is None
    # Studio corners stay white; garment interior keeps the uploaded print.
    assert out.getpixel((2, 2)) == (255, 255, 255)
    interior = {out.getpixel((x, 64)) for x in range(40, 90, 4)}
    assert MAGENTA in interior
    assert YELLOW in interior
    # Not replaced by a generic gray/blue reference.
    assert (128, 128, 128) not in interior
    assert (0, 0, 255) not in interior


def test_pipeline_passes_uploaded_fabric_into_flux_generate(tmp_path: Path):
    captured: dict = {}

    class _Engine:
        last_execution_stats = {
            "was_real_flux_used": True,
            "height": 256,
            "width": 256,
            "num_inference_steps": 3,
            "guidance_scale": 3.0,
        }

        def generate(self, **kwargs):
            captured.update(kwargs)
            img = Image.new("RGB", (256, 256), (40, 90, 130))
            img.putpixel((8, 8), (200, 30, 40))
            return img

    cfg = GarmentGenerationConfig(
        output_root=str(tmp_path / "out"),
        experiments_root=str(tmp_path / "exp"),
        height=256,
        width=256,
        num_inference_steps=3,
        guidance_scale=3.0,
        generation_mode="standard",
        allow_fallback=False,
        config_path="configs/custom_generator/flux_config.yaml",
    )
    pipe = GarmentGenerationPipeline.__new__(GarmentGenerationPipeline)
    pipe.config = cfg
    pipe.config.mode_key = "standard"
    pipe.logger = logging.getLogger("test.fabric_ref")
    pipe.prompt_builder = GarmentPromptBuilder("configs")
    pipe.validator = GarmentValidator(min_resolution=64)
    pipe.inference_engine = _Engine()
    pipe._vram_policy = None
    pipe._apply_high_vram_standard_defaults = lambda *_a, **_k: None
    pipe._apply_production_vram_defaults = lambda *_a, **_k: None

    fabric = _striped_fabric(256)
    result = pipe.run(
        fabric_metadata={"material": "cotton", "pattern": "striped"},
        user_customization={
            "gender": "women",
            "garment_type": "dress",
            "sleeve": "short_sleeve",
            "neckline": "Round Neck",
            "fit": "slim",
            "style": "casual",
            "force_recolor": False,
        },
        reference_image=fabric,
        output_filename="fabric_ref_test",
    )

    ref = captured.get("reference_image")
    prompt = str(captured.get("prompt") or "").lower()
    assert ref is not None
    assert getattr(ref, "size", None) == (256, 256)
    interior = {ref.getpixel((x, 128)) for x in range(80, 180, 4)}
    assert MAGENTA in interior
    assert YELLOW in interior
    assert ref.getpixel((2, 2)) == (255, 255, 255)

    assert "fabric-filled" in prompt or "mockup" in prompt
    assert "print" in prompt and "motifs" in prompt
    assert "dress" in prompt
    assert "round crew neckline" in prompt
    assert "do not invent" in prompt
    assert "no person" in prompt and "no model" in prompt
    assert "standalone" in prompt
    assert "wearable women" not in prompt
    assert "wearable" not in prompt
    assert "women's dress" not in prompt
    # Exact image passed to FLUX is the garment-only mockup, not the raw swatch.
    for x in range(0, 256, 16):
        assert ref.getpixel((x, 1)) == (255, 255, 255)
    assert captured.get("negative_prompt")
    assert "invented print" in str(captured.get("negative_prompt") or "").lower()
    assert Path(result["image_path"]).exists()
    assert result["metadata"]["user_customization"]["neckline"] == "Round Neck" or (
        pipe.prompt_builder.last_prompt_stats.get("neckline_token") == "round_neck"
    )


def test_match_fabric_prompt_is_not_color_only():
    builder = GarmentPromptBuilder("configs")
    pos, neg = builder.build_kontext_prompt(
        {
            "material": "cotton",
            "pattern": "striped",
            "dominant_colors": ["yellow", "magenta"],
            "color_source": "fabric_pixels",
            "fabric_appearance": "yellow magenta striped textile",
        },
        {
            "gender": "women",
            "garment_type": "dress",
            "sleeve": "short_sleeve",
            "neckline": "round_neck",
            "fit": "slim",
        },
    )
    lower = pos.lower()
    assert "fabric-filled" in lower or "mockup" in lower
    assert "print" in lower
    assert "motifs" in lower
    assert "texture" in lower or "weave" in lower
    assert "round crew neckline" in lower
    assert "dress" in lower
    assert "do not invent" in lower
    assert "no person" in lower and "standalone" in lower
    assert "match fabric" not in lower  # never emit the UI label as a color
    assert "invented print" in neg.lower()
    assert builder.last_prompt_stats["token_count"] <= CLIP_MAX_TOKENS
    assert builder.last_prompt_stats.get("color_mode") == "match_fabric"
    # Match Fabric is textile identity, not a color-name approximation.
    assert "print" in lower and "motifs" in lower
    assert "yellow" in lower and "magenta" in lower
    assert "do not invent a new design" in lower
    assert "do not recolor" in lower
    assert not pos.rstrip().endswith("Do not")


def test_dress_conditioning_is_garment_silhouette_not_person():
    """Studio corners and the top strip stay white — no head / face / body."""
    fabric = _striped_fabric(128)
    out = build_garment_conditioning_image(
        fabric,
        garment_type="dress",
        width=128,
        height=128,
        sleeve="short_sleeve",
        target_color="match_fabric",
    )
    assert out.size == (128, 128)
    assert out.mode == "RGB"
    for x in range(0, 128, 8):
        assert out.getpixel((x, 1)) == (255, 255, 255)
    assert out.getpixel((2, 2)) == (255, 255, 255)
    assert out.getpixel((125, 2)) == (255, 255, 255)
    assert out.getpixel((2, 125)) == (255, 255, 255)
    interior = {out.getpixel((x, 64)) for x in range(40, 90, 4)}
    assert MAGENTA in interior
    assert YELLOW in interior


def test_pipeline_passes_garment_conditioning_not_raw_swatch(tmp_path: Path):
    captured: dict = {}

    class _Engine:
        last_execution_stats = {
            "was_real_flux_used": True,
            "height": 128,
            "width": 128,
            "num_inference_steps": 3,
            "guidance_scale": 3.0,
        }

        def generate(self, **kwargs):
            captured.update(kwargs)
            img = Image.new("RGB", (128, 128), (40, 90, 130))
            img.putpixel((8, 8), (200, 30, 40))
            return img

    cfg = GarmentGenerationConfig(
        output_root=str(tmp_path / "out"),
        experiments_root=str(tmp_path / "exp"),
        height=128,
        width=128,
        num_inference_steps=3,
        guidance_scale=3.0,
        generation_mode="standard",
        allow_fallback=False,
        config_path="configs/custom_generator/flux_config.yaml",
    )
    pipe = GarmentGenerationPipeline.__new__(GarmentGenerationPipeline)
    pipe.config = cfg
    pipe.config.mode_key = "standard"
    pipe.logger = logging.getLogger("test.garment_only")
    pipe.prompt_builder = GarmentPromptBuilder("configs")
    pipe.validator = GarmentValidator(min_resolution=64)
    pipe.inference_engine = _Engine()
    pipe._vram_policy = None
    pipe._apply_high_vram_standard_defaults = lambda *_a, **_k: None
    pipe._apply_production_vram_defaults = lambda *_a, **_k: None

    fabric = _striped_fabric(128)
    pipe.run(
        fabric_metadata={"material": "cotton", "pattern": "striped"},
        user_customization={
            "gender": "women",
            "garment_type": "dress",
            "sleeve": "short_sleeve",
            "neckline": "Round Neck",
            "fit": "slim",
            "style": "casual",
            "force_recolor": False,
        },
        reference_image=fabric,
        output_filename="garment_only_ref",
    )
    ref = captured["reference_image"]
    # Raw swatch is full-bleed print; FLUX must receive the white-studio mockup.
    assert ref.getpixel((2, 2)) == (255, 255, 255)
    assert fabric.getpixel((2, 2)) != (255, 255, 255)
    assert MAGENTA in {ref.getpixel((x, 64)) for x in range(40, 90, 4)}

