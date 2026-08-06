import json
from pathlib import Path
from src.garment_generation.prompting.garment_prompt_builder import GarmentPromptBuilder

def test_material_vocabulary_normalization():
    builder = GarmentPromptBuilder(Path("configs"))
    mat_path = Path("configs/material_vocabulary.json")
    assert mat_path.exists()
    
    with open(mat_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "natural" in data
    assert "synthetic" in data
    assert "blended" in data
    assert "cotton" in data["natural"]
    assert "polyester" in data["synthetic"]
    
    assert builder.validate_and_normalize_material("Cotton Blend") == "cotton_blend"
