"""Unit tests for CLIP-safe garment prompt building and mode normalization."""

from __future__ import annotations

from pathlib import Path

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
