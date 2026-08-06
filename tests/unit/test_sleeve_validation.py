import json
from pathlib import Path
from src.garment_generation.prompting.garment_prompt_builder import GarmentPromptBuilder

def test_sleeve_vocabulary_validation():
    builder = GarmentPromptBuilder(Path("configs"))
    sleeve_path = Path("configs/sleeve_vocabulary.json")
    assert sleeve_path.exists()
    
    with open(sleeve_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "allowed_sleeves" in data
    assert "three_quarter_sleeve" in data["allowed_sleeves"]
    
    assert builder.validate_and_normalize_sleeve("Three Quarter Sleeve") == "three_quarter_sleeve"
    assert builder.validate_and_normalize_sleeve("invalid_random_sleeve") == "short_sleeve"
