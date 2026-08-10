import json
from pathlib import Path
from src.features.custom_generator.prompting.garment_prompt_builder import GarmentPromptBuilder

def test_sleeve_vocabulary_validation():
    builder = GarmentPromptBuilder(Path("configs"))
    sleeve_path = Path("configs/semantic_analysis/sleeve_vocabulary.json")
    assert sleeve_path.exists()
    
    with open(sleeve_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "allowed_sleeves" in data
    assert "three_quarter_sleeve" in data["allowed_sleeves"]
    
    assert builder.validate_and_normalize_sleeve("Three Quarter Sleeve") == "three_quarter_sleeve"
    # Empty/missing input falls back to the documented default.
    assert builder.validate_and_normalize_sleeve(None) == "short_sleeve"
    # Unknown-but-descriptive values are sanitized and kept (not silently
    # collapsed to a default) — see garment_prompt_builder.validate_and_normalize_sleeve.
    assert builder.validate_and_normalize_sleeve("invalid_random_sleeve") == "invalid_random_sleeve"
