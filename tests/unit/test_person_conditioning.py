import pytest
from PIL import Image
from src.virtual_tryon.models import PersonConditioningInput
from src.virtual_tryon.person_conditioning import PersonConditioner

def test_person_conditioning_preparation():
    conditioner = PersonConditioner(target_resolution=(512, 512))
    img = Image.new("RGB", (800, 600), color=(200, 180, 160))
    inp = PersonConditioningInput(person_image=img)
    
    out = conditioner.prepare_person_condition(inp)
    assert out.person_image.size == (512, 512)
    assert out.agnostic_mask.size == (512, 512)
    assert out.person_image.mode == "RGB"
    assert out.agnostic_mask.mode == "L"
