"""Uploaded fabric must be the visual reference passed into FLUX Kontext."""

from __future__ import annotations

import logging
import math
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
    assert getattr(build_garment_conditioning_image, "last_neckline_key", "") == "round_neck"

    assert "fabric-filled" in prompt or "mockup" in prompt
    assert "preserve source fabric look" in prompt or "same print scale" in prompt
    assert "dress" in prompt
    assert "round crew neckline" in prompt
    assert "no person" in prompt and "no model" in prompt
    assert "standalone" in prompt
    assert "wearable women" not in prompt
    assert "wearable" not in prompt
    assert "women's dress" not in prompt
    assert "not a v-neck" not in prompt
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
    assert "preserve source fabric look" in lower
    assert "same print scale" in lower
    assert "round crew neckline" in lower
    assert "dress" in lower
    assert "no person" in lower and "standalone" in lower
    assert "match fabric" not in lower  # never emit the UI label as a color
    assert "invented print" in neg.lower()
    assert builder.last_prompt_stats["token_count"] <= CLIP_MAX_TOKENS
    assert builder.last_prompt_stats.get("color_mode") == "match_fabric"
    assert "yellow" in lower and "magenta" in lower
    assert "do not recolor" in lower
    assert "not a v-neck" not in lower


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
    assert getattr(build_garment_conditioning_image, "last_neckline_key", "") == "round_neck"


def _first_garment_y(img: Image.Image, x: int):
    white = (255, 255, 255)
    w, h = img.size
    x = max(0, min(w - 1, x))
    for y in range(h):
        if img.getpixel((x, y)) != white:
            return y
    return None


def _last_garment_y(img: Image.Image, x: int):
    white = (255, 255, 255)
    w, h = img.size
    x = max(0, min(w - 1, x))
    last = None
    for y in range(h):
        if img.getpixel((x, y)) != white:
            last = y
    return last


def _opening_angle_deg(img: Image.Image, cx: int, offset: int) -> float:
    y_c = _first_garment_y(img, cx)
    y_l = _first_garment_y(img, cx - offset)
    y_r = _first_garment_y(img, cx + offset)
    assert y_c is not None and y_l is not None and y_r is not None
    v1 = ((cx - offset) - cx, y_l - y_c)
    v2 = ((cx + offset) - cx, y_r - y_c)
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    dot = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(dot))


def test_round_neck_conditioning_is_curved_not_pointed():
    fabric = _striped_fabric(256)
    out = build_garment_conditioning_image(
        fabric,
        garment_type="dress",
        width=256,
        height=256,
        sleeve="short_sleeve",
        neckline="Round Neck",
        target_color="match_fabric",
    )
    cx = 128
    y_c = _first_garment_y(out, cx)
    y_l = _first_garment_y(out, cx - 36)
    y_r = _first_garment_y(out, cx + 36)
    assert y_c is not None and y_l is not None and y_r is not None
    assert y_c >= y_l and y_c >= y_r
    ang = _opening_angle_deg(out, cx, 36)
    assert ang > 125
    y_near = _first_garment_y(out, cx - 10)
    assert y_near is not None
    assert abs(y_c - y_near) <= 8
    interior = {out.getpixel((x, 140)) for x in range(80, 180, 4)}
    assert MAGENTA in interior and YELLOW in interior


def test_v_neck_conditioning_is_pointed():
    fabric = _striped_fabric(256)
    out = build_garment_conditioning_image(
        fabric,
        garment_type="dress",
        width=256,
        height=256,
        sleeve="short_sleeve",
        neckline="V Neck",
        target_color="match_fabric",
    )
    cx = 128
    y_c = _first_garment_y(out, cx)
    y_l = _first_garment_y(out, cx - 36)
    y_r = _first_garment_y(out, cx + 36)
    assert y_c is not None and y_l is not None and y_r is not None
    assert y_c > y_l and y_c > y_r
    ang = _opening_angle_deg(out, cx, 36)
    assert ang < 125
    y_near = _first_garment_y(out, cx - 10)
    assert y_near is not None
    assert (y_c - y_near) >= 6
    interior = {out.getpixel((x, 160)) for x in range(80, 180, 4)}
    assert MAGENTA in interior and YELLOW in interior


def test_round_vs_v_neck_keeps_body_and_fabric_pixels():
    fabric = _striped_fabric(256)
    kwargs = dict(
        fabric_image=fabric,
        garment_type="dress",
        width=256,
        height=256,
        sleeve="short_sleeve",
        target_color="match_fabric",
    )
    rnd = build_garment_conditioning_image(neckline="round_neck", **kwargs)
    vee = build_garment_conditioning_image(neckline="v_neck", **kwargs)
    # Hem / body extent unchanged; only the neck opening differs.
    assert _last_garment_y(rnd, 128) == _last_garment_y(vee, 128)
    assert _first_garment_y(rnd, 128) != _first_garment_y(vee, 128)
    assert rnd.getpixel((2, 2)) == (255, 255, 255)
    assert MAGENTA in {rnd.getpixel((x, 140)) for x in range(80, 180, 4)}
    assert YELLOW in {vee.getpixel((x, 160)) for x in range(80, 180, 4)}


def test_round_neck_is_shallow_crew_not_deep_u():
    """Crew opening must stay curved and shallower than the previous deep U."""
    fabric = _striped_fabric(256)
    kwargs = dict(
        fabric_image=fabric,
        garment_type="dress",
        width=256,
        height=256,
        sleeve="short_sleeve",
        target_color="match_fabric",
    )
    rnd = build_garment_conditioning_image(neckline="Round Neck", **kwargs)
    vee = build_garment_conditioning_image(neckline="V Neck", **kwargs)
    cx = 128
    y_c = _first_garment_y(rnd, cx)
    y_l = _first_garment_y(rnd, cx - 24)
    y_r = _first_garment_y(rnd, cx + 24)
    y_v = _first_garment_y(vee, cx)
    assert y_c is not None and y_l is not None and y_r is not None and y_v is not None
    assert y_c >= y_l and y_c >= y_r
    depth = y_c - min(y_l, y_r)
    assert 3 <= depth <= int(256 * 0.055)
    # Pointed V remains deeper than the crew curve.
    assert y_c < y_v
    ang = _opening_angle_deg(rnd, cx, 24)
    assert ang > 130



