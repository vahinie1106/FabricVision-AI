import pytest
from PIL import Image, ImageDraw
from src.virtual_tryon.tryon_validator import TryOnValidator

def create_non_flat_image(width=512, height=512):
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, width - 50, height - 50], fill=(240, 180, 120))
    return img

def test_tryon_validator_success():
    validator = TryOnValidator(min_resolution=256)
    img = create_non_flat_image(512, 512)
    res = validator.validate(img)
    assert res["valid"] is True
    assert res["resolution"] == (512, 512)
    assert res["mode"] == "RGB"

def test_tryon_validator_invalid_mode():
    validator = TryOnValidator(min_resolution=256)
    img = Image.new("L", (512, 512), color=128)
    res = validator.validate(img)
    assert res["valid"] is False
    assert any("Invalid try-on image mode" in issue for issue in res["issues"])
