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
