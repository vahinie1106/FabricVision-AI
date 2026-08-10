"""Unit tests for CatVTON conditioning semantics (no GPU inference)."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from src.features.virtual_tryon.catvton_conditioning import (
    analyze_mask,
    attn_version_for_resolution,
    prepare_catvton_person_conditioning,
    resize_garment_condition,
    validate_clothing_mask,
)
from src.features.virtual_tryon.tryon_pipeline import VirtualTryOnPipeline


def test_mask_polarity_white_means_replace():
    person = Image.new("RGB", (64, 64), (10, 20, 30))
    mask = Image.new("L", (64, 64), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([16, 16, 48, 48], fill=255)
    masked = prepare_catvton_person_conditioning(person, mask)
    arr = np.asarray(masked)
    # Replaced (white mask) region must be zeroed — matches image * (mask < 0.5)
    assert arr[32, 32].tolist() == [0, 0, 0]
    # Preserved (black mask) region keeps person color
    assert arr[0, 0].tolist() == [10, 20, 30]


def test_garment_padding_preserves_aspect(tmp_path):
    # Tall thin garment — padding must not stretch to fill canvas.
    g = Image.new("RGB", (40, 120), (200, 50, 50))
    out = resize_garment_condition(g, (96, 128), catvton_root=__import__("pathlib").Path("models/CatVTON"))
    assert out.size == (96, 128)
    # Corners should be white padding
    assert out.getpixel((0, 0)) == (255, 255, 255)


def test_attn_version_selects_vitonhd_for_512():
    assert attn_version_for_resolution(512, 384) == "vitonhd"
    assert attn_version_for_resolution(1024, 768) == "mix"


def test_cloth_type_dress_and_fullbody_default():
    assert VirtualTryOnPipeline._cloth_type_for_garment("dress") == "overall"
    assert VirtualTryOnPipeline._cloth_type_for_garment("shirt") == "upper"
    tall = Image.new("RGB", (400, 800), (1, 2, 3))
    assert VirtualTryOnPipeline._cloth_type_for_garment("garment", tall) == "overall"


def test_validate_mask_rejects_empty():
    empty = Image.new("L", (128, 128), 0)
    ok, reason = validate_clothing_mask(empty, cloth_type="upper")
    assert ok is False
    assert "small" in reason


def test_validate_overall_rejects_upper_only_mask():
    mask = Image.new("L", (200, 400), 0)
    draw = ImageDraw.Draw(mask)
    # torso only — stops at mid-body
    draw.rectangle([50, 40, 150, 200], fill=255)
    ok, reason = validate_clothing_mask(mask, cloth_type="overall")
    assert ok is False
    assert "prematurely" in reason or "skirt" in reason or "thigh" in reason


def test_validate_overall_rejects_pant_split():
    mask = Image.new("L", (200, 400), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([50, 40, 150, 220], fill=255)
    draw.rectangle([55, 220, 90, 360], fill=255)
    draw.rectangle([110, 220, 145, 360], fill=255)
    ok, reason = validate_clothing_mask(mask, cloth_type="overall")
    assert ok is False
    assert "split" in reason


def test_analyze_mask_reports_leg_ratio():
    mask = Image.new("L", (100, 200), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([20, 20, 80, 100], fill=255)  # upper torso only
    stats = analyze_mask(mask)
    assert stats["leg_fill_ratio"] == 0.0
    assert stats["mask_area_ratio"] > 0.05
    assert stats["mask_bbox"] is not None
