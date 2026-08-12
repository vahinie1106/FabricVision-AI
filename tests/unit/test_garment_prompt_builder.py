"""Unit tests for CLIP-safe garment prompt building and mode normalization."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from src.features.custom_generator.inference.fabric_conditioning import (
    build_garment_conditioning_image,
    is_match_fabric_color,
    normalize_color_key,
    recolor_fabric_base_preserving_motifs,
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
            "motif_colors": ["red"],
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
    assert "base fabric color" in lower
    assert "printed floral motifs" in lower or "printed motifs" in lower or "print colors" in lower
    assert "preserve source fabric look" not in lower
    assert "recolor to blue" not in lower
    assert "apply blue color" not in lower
    # Must not say whole-fabric "do not recolor" without motif scope.
    assert "do not recolor." not in lower or "motif" in lower or "printed" in lower
    assert "red" in lower  # motif color called out
    assert builder.last_prompt_stats.get("color_mode") == "explicit"
    assert "blue" in (builder.last_prompt_stats.get("dominant_colors") or "").lower()
    instruction = (builder.last_prompt_stats.get("color_instruction") or "").lower()
    assert "base fabric color" in instruction
    assert "blue" in instruction


def test_kontext_prompt_explicit_black_no_contradiction():
    builder = _builder()
    pos, _ = builder.build_kontext_prompt(
        {
            "material": "cotton",
            "pattern": "printed",
            "dominant_colors": ["black"],
            "color_source": "ui_recolor",
            "motif_colors": ["red"],
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
    assert "base fabric color" in lower
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
            "motif_colors": ["blue"],
        },
        {"color": "green", "force_recolor": True, "garment_type": "dress"},
    )
    assert ctx["color_mode"] == "explicit"
    assert "green" in ctx["dominant_colors"]
    assert "match fabric" not in ctx["dominant_colors"]
    assert "blue" in ctx["motif_colors"]


def _make_white_red_floral(size: int = 128) -> Image.Image:
    """Synthetic white base + red floral-like blobs for recolor tests."""
    img = Image.new("RGB", (size, size), (242, 242, 242))
    draw = ImageDraw.Draw(img)
    for cx, cy, r in (
        (32, 32, 14),
        (90, 40, 12),
        (50, 95, 16),
        (100, 100, 10),
        (70, 60, 11),
    ):
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(200, 30, 40))
        # Small darker core for motif interior
        draw.ellipse((cx - r // 2, cy - r // 2, cx + r // 2, cy + r // 2), fill=(170, 20, 30))
    return img


def test_match_fabric_leaves_image_unchanged():
    img = _make_white_red_floral()
    match = build_garment_conditioning_image(
        img, garment_type="dress", width=64, height=64, target_color=None
    )
    # Without target_color, fabric fill is original (no base recolor audit).
    assert getattr(build_garment_conditioning_image, "last_recolor_audit", None) is None
    # Silhouette center should still be near-white or red from original fabric.
    px = match.getpixel((32, 32))
    assert px != (0, 0, 0)


def test_blue_recolors_base_but_keeps_red_motifs(tmp_path):
    img = _make_white_red_floral(160)
    audit = recolor_fabric_base_preserving_motifs(img, "blue")
    out = audit.image
    # Persist synthetic audits for visual QA.
    img.save(tmp_path / "synthetic_original.png")
    out.save(tmp_path / "synthetic_blue.png")
    arr_o = list(img.getdata())
    arr_n = list(out.getdata())
    # Classify original red motif pixels and white-ish base pixels.
    motif_idx = [
        i
        for i, (r, g, b) in enumerate(arr_o)
        if r > 140 and r > g + 40 and r > b + 40
    ]
    base_idx = [
        i
        for i, (r, g, b) in enumerate(arr_o)
        if r > 200 and g > 200 and b > 200
    ]
    assert len(motif_idx) > 50
    assert len(base_idx) > 200

    # Base should move toward blue (B channel rises vs R on former white).
    base_new = [arr_n[i] for i in base_idx]
    mean_b = sum(p[2] for p in base_new) / len(base_new)
    mean_r = sum(p[0] for p in base_new) / len(base_new)
    assert mean_b > mean_r + 15

    # Motifs remain substantially red.
    motif_new = [arr_n[i] for i in motif_idx]
    red_kept = sum(1 for r, g, b in motif_new if r > g + 25 and r > b + 25 and r > 100)
    assert red_kept / len(motif_new) > 0.85

    # Not a solid blue image.
    all_blueish = sum(1 for r, g, b in arr_n if b > r + 20 and b > g)
    assert all_blueish / len(arr_n) < 0.95
    vals = [p[0] for p in arr_n]
    assert max(vals) - min(vals) > 40


def test_black_recolors_base_but_keeps_red_motifs(tmp_path):
    img = _make_white_red_floral(160)
    audit = recolor_fabric_base_preserving_motifs(img, "black")
    audit.image.save(tmp_path / "synthetic_black.png")
    arr_o = list(img.getdata())
    arr_n = list(audit.image.getdata())
    motif_idx = [
        i
        for i, (r, g, b) in enumerate(arr_o)
        if r > 140 and r > g + 40 and r > b + 40
    ]
    base_idx = [
        i
        for i, (r, g, b) in enumerate(arr_o)
        if r > 200 and g > 200 and b > 200
    ]
    base_new = [arr_n[i] for i in base_idx]
    mean_lum = sum(sum(p) / 3 for p in base_new) / len(base_new)
    assert mean_lum < 95  # darkened base (black target)

    motif_new = [arr_n[i] for i in motif_idx]
    red_kept = sum(1 for r, g, b in motif_new if r > g + 25 and r > b + 25 and r > 100)
    assert red_kept / len(motif_new) > 0.85

    # Not solid black
    assert max(max(p) for p in arr_n) > 100


def test_post_recolor_protects_studio_background():
    # White canvas with a white+red patch in the center (garment-like).
    canvas = Image.new("RGB", (128, 128), (255, 255, 255))
    floral = _make_white_red_floral(64)
    canvas.paste(floral, (32, 32))
    out = recolor_fabric_base_preserving_motifs(
        canvas, "blue", protect_studio_background=True
    ).image
    # Corner studio background must stay near-white (not blue).
    cr, cg, cb = out.getpixel((2, 2))
    assert cr > 220 and cg > 220 and cb > 220
    # Interior former-white fabric should be bluish.
    # Sample a known white pixel inside the pasted patch away from red circles.
    ir, ig, ib = out.getpixel((36, 36))
    # Either still motif-protected or base-recolored; at least one interior base
    # sample across the patch should be blue-dominant.
    blues = 0
    reds = 0
    for y in range(40, 80, 4):
        for x in range(40, 80, 4):
            r, g, b = out.getpixel((x, y))
            if b > r + 10:
                blues += 1
            if r > g + 30 and r > b + 30:
                reds += 1
    assert blues > 0
    assert reds > 0


def test_conditioning_recolors_only_for_explicit_color():
    fabric = _make_white_red_floral(128)
    match = build_garment_conditioning_image(
        fabric, garment_type="dress", width=64, height=64, target_color=None
    )
    blue = build_garment_conditioning_image(
        fabric, garment_type="dress", width=64, height=64, target_color="blue"
    )
    audit = build_garment_conditioning_image.last_recolor_audit
    assert audit is not None
    assert audit.base_coverage > 0.3
    # Compare mean of high-alpha (base) pixels on the recolored fabric, not a
    # single silhouette sample that may land on a protected motif.
    base = audit.recolored_base
    mask = audit.base_mask
    base_px = []
    blueish = 0
    for y in range(0, base.size[1], 4):
        for x in range(0, base.size[0], 4):
            if mask.getpixel((x, y)) > 180:
                r, g, b = base.getpixel((x, y))
                base_px.append((r, g, b))
                if b >= r:
                    blueish += 1
    assert len(base_px) > 20
    assert blueish / len(base_px) > 0.7
    assert match.size == blue.size


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
