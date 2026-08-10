import json
from pathlib import Path
from src.features.custom_generator.prompting.garment_prompt_builder import GarmentPromptBuilder

def test_color_vocabulary_normalization():
    builder = GarmentPromptBuilder(Path("configs"))
    color_path = Path("configs/semantic_analysis/color_vocabulary.json")
    assert color_path.exists()
    
    with open(color_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "basic_colors" in data
    assert "fashion_colors" in data
    assert "navy_blue" in data["fashion_colors"]
    assert "black" in data["basic_colors"]
    
    assert builder.validate_and_normalize_color("Navy Blue") == "navy_blue"
    assert builder.validate_and_normalize_color("royal_blue") == "royal_blue"
