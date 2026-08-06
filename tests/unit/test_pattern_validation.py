import json
from pathlib import Path
from src.garment_generation.prompting.garment_prompt_builder import GarmentPromptBuilder

def test_pattern_vocabulary_validation():
    builder = GarmentPromptBuilder(Path("configs"))
    pattern_path = Path("configs/pattern_vocabulary.json")
    assert pattern_path.exists()
    
    with open(pattern_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "allowed_patterns" in data
    assert "floral" in data["allowed_patterns"]
    assert "traditional_motifs" in data["allowed_patterns"]
    assert "polka_dot" in data["allowed_patterns"]
    
    assert builder.validate_and_normalize_pattern("Floral") == "floral"
    assert builder.validate_and_normalize_pattern("Traditional Motifs") == "traditional_motifs"
    assert builder.validate_and_normalize_pattern("unrestricted_random_text") == "solid"
