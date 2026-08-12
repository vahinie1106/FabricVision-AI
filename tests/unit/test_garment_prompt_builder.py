"""Unit tests for CLIP-safe garment prompt building and mode normalization."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.features.custom_generator.inference.fabric_conditioning import (
    build_garment_conditioning_image,
    is_match_fabric_color,
    normalize_color_key,
    tint_fabric_preserving_texture,
)
from src.features.custom_generator.pipeline.garment_generation_pipeline import (
    normalize_generation_mode,
)
from src.features.custom_generator.prompting.garment_prompt_builder import (
    CLIP_MAX_TOKENS,
    GarmentPromptBuilder,
)


def _builder() -> GarmentPromptBuilder:
    return GarmentPromptBuilder(Path("configs"))


def test_normalize_generation_mode_aliases():
    assert normalize_generation_mode("Preview") == "preview"
    assert normalize_generation_mode("Fast Preview") == "preview"
    assert normalize_generation_mode("standard") == "standard"
    assert normalize_generation_mode("Standard") == "standard"
    assert normalize_generation_mode("Production") == "production"
    assert normalize_generation_mode("High Quality") == "production"
    assert normalize_generation_mode(None) == "standard"
    assert normalize_generation_mode("") == "standard"
    assert normalize_generation_mode("quality_15") == "quality_15"
    assert normalize_generation_mode("Quality 20") == "quality_20"
    assert normalize_generation_mode("quality") == "quality_20"
    assert normalize_generation_mode("quality_768") == "quality_768"
    assert normalize_generation_mode("low_vram") == "preview"


def test_match_fabric_color_helpers():
    assert is_match_fabric_color("Match Fabric") is True
    assert is_match_fabric_color("match_fabric") is True
    assert is_match_fabric_color("") is True
    assert is_match_fabric_color(None) is True
    assert is_match_fabric_color("Blue") is False
    assert is_match_fabric_color("black") is False
    assert normalize_color_key("Navy Blue") == "navy_blue"


def test_kontext_prompt_match_fabric_preserves_source_colors():
    builder = _builder()
    pos, _ = builder.build_kontext_prompt(
        {
            "material": "cotton",
            "pattern": "floral",
            "dominant_colors": ["white", "red"],
            "color_source": "fabric_pixels",
            "fabric_appearance": "white, red floral textile",
        },
        {
            "gender": "women",
            "garment_type": "dress",
            "sleeve": "short_sleeve",
            "neckline": "round_neck",
            "force_recolor": False,
        },
    )
    lower = pos.lower()
    assert "do not recolor" in lower
    assert "preserve" in lower
    assert "white" in lower and "red" in lower
    assert "apply blue color" not in lower
    assert builder.last_prompt_stats.get("color_mode") == "match_fabric"
    assert "do not recolor" in (builder.last_prompt_stats.get("color_instruction") or "").lower()


def test_kontext_prompt_explicit_blue_applies_target_color():
    builder = _builder()
    pos, _ = builder.build_kontext_prompt(
        {
            "material": "cotton",
            "pattern": "floral",
            "dominant_colors": ["blue"],
            "color_source": "ui_recolor",
            "fabric_appearance": "floral textile texture",
            "source_palette": ["white", "red"],
        },
        {
            "gender": "women",
            "garment_type": "dress",
            "sleeve": "short_sleeve",
            "neckline": "round_neck",
            "color": "blue",
            "force_recolor": True,
        },
    )
    lower = pos.lower()
    assert "blue" in lower
    assert "apply" in lower and "color" in lower
    assert "do not recolor" not in lower
    assert "preserve source fabric look" not in lower
    # Source palette must not become the authoritative color instruction.
    assert "white, red" not in lower
    assert builder.last_prompt_stats.get("color_mode") == "explicit"
    assert "blue" in (builder.last_prompt_stats.get("dominant_colors") or "").lower()
    instruction = (builder.last_prompt_stats.get("color_instruction") or "").lower()
    assert "do not recolor" not in instruction
    assert "blue" in instruction


def test_kontext_prompt_explicit_black_no_contradiction():
    builder = _builder()
    pos, _ = builder.build_kontext_prompt(
        {
            "material": "cotton",
            "pattern": "printed",
            "dominant_colors": ["black"],
            "color_source": "ui_recolor",
        },
        {
            "gender": "women",
            "garment_type": "shirt",
            "sleeve": "short_sleeve",
            "neckline": "round_neck",
            "color": "black",
            "force_recolor": True,
        },
    )
    lower = pos.lower()
    assert "black" in lower
    assert "do not recolor" not in lower
    assert "preserve source fabric print/colors" not in lower
    assert builder.last_prompt_stats["token_count"] <= CLIP_MAX_TOKENS


def test_explicit_color_not_silently_match_fabric():
    builder = _builder()
    ctx = builder._build_context(
        {
            "material": "cotton",
            "pattern": "floral",
            "dominant_colors": ["green"],
            "color_source": "ui_recolor",
        },
        {"color": "green", "force_recolor": True, "garment_type": "dress"},
    )
    assert ctx["color_mode"] == "explicit"
    assert "green" in ctx["dominant_colors"]
    assert "match fabric" not in ctx["dominant_colors"]


def test_tint_fabric_preserving_texture_changes_mean_toward_target():
    # White/red checkerboard-like: bright base + red accents
    img = Image.new("RGB", (64, 64), (240, 240, 240))
    for y in range(64):
        for x in range(0, 64, 8):
            if (x // 8 + y // 8) % 2 == 0:
                img.putpixel((x, y), (200, 30, 40))
    blue = tint_fabric_preserving_texture(img, "blue")
    assert blue.size == img.size
    # Mean should shift toward blue channel dominance vs original red-ish accents
    orig_px = list(img.getdata())
    tint_px = list(blue.getdata())
    orig_r = sum(p[0] for p in orig_px) / len(orig_px)
    tint_b = sum(p[2] for p in tint_px) / len(tint_px)
    tint_r = sum(p[0] for p in tint_px) / len(tint_px)
    assert tint_b > tint_r
    assert orig_r > tint_r  # red accents suppressed after blue tint


def test_conditioning_recolors_only_for_explicit_color():
    fabric = Image.new("RGB", (128, 128), (220, 40, 50))  # red fabric
    match = build_garment_conditioning_image(
        fabric, garment_type="dress", width=64, height=64, target_color=None
    )
    blue = build_garment_conditioning_image(
        fabric, garment_type="dress", width=64, height=64, target_color="blue"
    )
    # Sample center pixel inside silhouette — should differ after tint
    assert match.getpixel((32, 32)) != blue.getpixel((32, 32))
    br, bg, bb = blue.getpixel((32, 32))
    assert bb >= br  # blue-ish after tint


def test_kontext_prompt_within_clip_budget():
    builder = _builder()
    fabric_metadata = {
        "material": "cotton",
        "pattern": "floral",
        "texture": "soft",
        "dominant_colors": ["royal_blue", "cream"],
        "style": "casual",
    }
    user_customization = {
        "gender": "women",
        "garment_type": "dress",
        "sleeve": "short_sleeve",
        "neckline": "round_neck",
        "fit": "slim",
        "style": "casual",
        "occasion": "party",
        "season": "summer",
    }
    pos, neg = builder.build_kontext_prompt(fabric_metadata, user_customization)
    stats = builder.last_prompt_stats

    assert "dress" in pos.lower()
    assert "women" in pos.lower()
    assert "fabric" in pos.lower()
    assert "blurry" in neg.lower() or "swatch" in neg.lower()
    assert stats["token_count"] <= CLIP_MAX_TOKENS
    assert stats["truncated"] is False


def test_kontext_prompt_preserves_structure_attributes():
    builder = _builder()
    pos, _ = builder.build_kontext_prompt(
        {"material": "silk", "pattern": "printed", "dominant_colors": ["navy"]},
        {
            "gender": "women",
            "garment_type": "shirt",
            "sleeve": "puff_sleeve",
            "neckline": "sweetheart_neck",
            "fit": "regular",
        },
    )
    lower = pos.lower()
    assert "shirt" in lower
    assert "sweetheart" in lower or "neckline" in lower
    assert "puff" in lower or "sleeve" in lower
    assert builder.last_prompt_stats["token_count"] <= CLIP_MAX_TOKENS


def test_classic_prompt_builder_basic():
    builder = _builder()
    fabric_metadata = {
        "material": "cotton",
        "pattern": "floral",
        "texture": "soft",
        "dominant_colors": ["royal_blue"],
        "style": "casual",
    }
    user_customization = {
        "gender": "women",
        "garment_type": "kurti",
        "sleeve": "three_quarter_sleeve",
        "neckline": "round_neck",
        "size": "M",
    }
    pos_prompt, neg_prompt = builder.build_prompts(fabric_metadata, user_customization)
    assert "women" in pos_prompt
    assert "kurti" in pos_prompt
    assert "cotton" in pos_prompt
    assert "floral" in pos_prompt
    assert "royal blue" in pos_prompt
    assert "three-quarter" in pos_prompt
    assert "round" in pos_prompt
    assert "blurry" in neg_prompt
    assert builder.last_prompt_stats["token_count"] <= CLIP_MAX_TOKENS
