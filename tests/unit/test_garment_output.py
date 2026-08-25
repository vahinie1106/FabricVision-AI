"""Tests for final garment PNG persistence and verification."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.features.custom_generator.inference.garment_output import (
    persist_and_verify_garment_png,
)


def test_persist_and_verify_writes_valid_png(tmp_path: Path):
    img = Image.new("RGB", (64, 64), (40, 120, 200))
    # Add some variance so STD > 0
    img.putpixel((10, 10), (200, 30, 40))
    dest = tmp_path / "images" / "garment_testjob.png"
    stats = persist_and_verify_garment_png(img, dest, expected_size=(64, 64))
    assert dest.exists()
    assert stats["file_size"] > 0
    assert stats["width"] == 64 and stats["height"] == 64
    assert stats["mode"] == "RGB"
    assert stats["max"] > 0
    assert stats["std"] > 0
    reopened = Image.open(dest)
    assert reopened.size == (64, 64)


def test_persist_and_verify_rejects_black_image(tmp_path: Path):
    img = Image.new("RGB", (32, 32), (0, 0, 0))
    dest = tmp_path / "garment_black.png"
    with pytest.raises(RuntimeError, match="black"):
        persist_and_verify_garment_png(img, dest)


def test_kontext_prompt_mentions_realistic_construction():
    from src.features.custom_generator.prompting.garment_prompt_builder import (
        CLIP_MAX_TOKENS,
        GarmentPromptBuilder,
    )

    builder = GarmentPromptBuilder(Path("configs"))
    pos, neg = builder.build_kontext_prompt(
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
    assert "standalone" in lower or "natural drape" in lower or "folds" in lower
    assert "do not recolor" in lower
    assert "no person" in lower
    assert "plastic" in neg.lower() or "cgi" in neg.lower()
    assert builder.last_prompt_stats["token_count"] <= CLIP_MAX_TOKENS
