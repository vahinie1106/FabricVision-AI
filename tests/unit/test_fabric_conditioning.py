"""Unit tests for fabric conditioning print-scale preservation (no FLUX inference)."""

from PIL import Image, ImageDraw

from src.features.custom_generator.inference.fabric_conditioning import _cover_fabric


def test_cover_fabric_center_crops_large_swatch_without_downscale_smear():
    # Unique pixel pattern — center crop must keep native values (no LANCZOS blur).
    fabric = Image.new("RGB", (1024, 1024), (10, 20, 30))
    draw = ImageDraw.Draw(fabric)
    draw.rectangle([400, 400, 624, 624], fill=(200, 50, 50))
    covered = _cover_fabric(fabric, 512, 512)
    assert covered.size == (512, 512)
    # Center of crop should still be the red square (exact RGB, not blurred neighbors).
    px = covered.getpixel((256, 256))
    assert px == (200, 50, 50)


def test_cover_fabric_tiles_small_swatch():
    swatch = Image.new("RGB", (64, 64), (0, 128, 255))
    covered = _cover_fabric(swatch, 256, 256)
    assert covered.size == (256, 256)
    assert covered.getpixel((0, 0)) == (0, 128, 255)
    assert covered.getpixel((128, 128)) == (0, 128, 255)


def test_cover_fabric_does_not_upscale_when_one_axis_is_short():
    """A 400×800 strip on 768² must tile, not LANCZOS-stretch motifs."""
    fabric = Image.new("RGB", (400, 800), (10, 20, 30))
    draw = ImageDraw.Draw(fabric)
    draw.rectangle([0, 0, 40, 40], fill=(200, 10, 10))
    covered = _cover_fabric(fabric, 768, 768)
    assert covered.size == (768, 768)
    # Native red square stays 40×40 at origin — stretch would enlarge it.
    assert covered.getpixel((20, 20)) == (200, 10, 10)
    assert covered.getpixel((60, 20)) == (10, 20, 30)
    # Horizontal tile repeat of the 400px-wide strip.
    assert covered.getpixel((400 + 20, 20)) == (200, 10, 10)


def test_cover_fabric_near_canvas_uses_modest_cover_not_tile_seam():
    """512×512 on 768 should cover-scale (~1.5×), not hard-tile a seam at x=512."""
    fabric = Image.new("RGB", (512, 512), (10, 20, 30))
    draw = ImageDraw.Draw(fabric)
    draw.rectangle([0, 0, 8, 8], fill=(200, 10, 10))
    covered = _cover_fabric(fabric, 768, 768)
    assert covered.size == (768, 768)
    # Tile would copy the red corner again at (512, 0). Modest cover must not.
    assert covered.getpixel((512, 0)) != (200, 10, 10)


def test_match_fabric_conditioning_skips_recolor():
    from src.features.custom_generator.inference.fabric_conditioning import (
        build_garment_conditioning_image,
    )

    fabric = Image.new("RGB", (128, 128), (180, 40, 40))
    out = build_garment_conditioning_image(
        fabric,
        garment_type="shirt",
        width=128,
        height=128,
        target_color="match_fabric",
    )
    assert out.size == (128, 128)
    assert getattr(build_garment_conditioning_image, "last_recolor_audit", None) is None
    # Garment interior keeps original fabric RGB (not a recolor target).
    assert out.getpixel((64, 64)) == (180, 40, 40)
