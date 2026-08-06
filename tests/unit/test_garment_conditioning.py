import pytest
from PIL import Image
from src.virtual_tryon.models import GarmentConditioningInput
from src.virtual_tryon.garment_conditioning import GarmentConditioner

def test_garment_conditioning_preparation():
    conditioner = GarmentConditioner(target_resolution=(512, 512))
    img = Image.new("RGB", (800, 800), color=(255, 255, 255))
    inp = GarmentConditioningInput(garment_image=img, garment_type="kurti")
    
    out = conditioner.prepare_garment_condition(inp)
    assert out.garment_image.size == (512, 512)
    assert out.garment_mask.size == (512, 512)
    assert out.garment_image.mode == "RGB"
    assert out.garment_mask.mode == "L"
