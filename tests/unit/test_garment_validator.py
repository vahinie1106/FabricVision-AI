import pytest
from PIL import Image, ImageDraw
from src.garment_generation.validation.garment_validator import GarmentValidator

def test_garment_validator_valid_image():
    validator = GarmentValidator(min_resolution=256)
    img = Image.new("RGB", (512, 512), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, 400, 400], fill=(50, 100, 150))
    
    result = validator.validate(img, target_garment="kurti", target_color="royal_blue")
    assert result["valid"] is True
    assert result["is_valid"] is True
    assert result["garment"] == "kurti"
    assert result["human_detected"] is False
    assert result["mannequin_detected"] is False
    assert result["confidence"] >= 0.80

def test_garment_validator_low_resolution():
    validator = GarmentValidator(min_resolution=256)
    img = Image.new("RGB", (128, 128), color=(200, 200, 200))
    
    result = validator.validate(img)
    assert result["valid"] is False
    assert result["is_valid"] is False
    assert any("below minimum threshold" in issue for issue in result["issues"])
