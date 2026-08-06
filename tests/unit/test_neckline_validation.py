import json
from pathlib import Path
from src.garment_generation.prompting.garment_prompt_builder import GarmentPromptBuilder

def test_neckline_vocabulary_validation():
    builder = GarmentPromptBuilder(Path("configs"))
    neck_path = Path("configs/neckline_vocabulary.json")
    assert neck_path.exists()
    
    with open(neck_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "allowed_necklines" in data
    assert "round_neck" in data["allowed_necklines"]
    assert "mandarin_collar" in data["allowed_necklines"]
    
    assert builder.validate_and_normalize_neckline("Mandarin Collar") == "mandarin_collar"
    assert builder.validate_and_normalize_neckline("unknown_neck") == "round_neck"
