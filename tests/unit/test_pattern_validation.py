import json
from pathlib import Path
from src.features.custom_generator.prompting.garment_prompt_builder import GarmentPromptBuilder

def test_pattern_vocabulary_validation():
    builder = GarmentPromptBuilder(Path("configs"))
    pattern_path = Path("configs/semantic_analysis/pattern_vocabulary.json")
    assert pattern_path.exists()
    
    with open(pattern_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "allowed_patterns" in data
    assert "floral" in data["allowed_patterns"]
    assert "traditional_motifs" in data["allowed_patterns"]
    assert "polka_dot" in data["allowed_patterns"]
    
    assert builder.validate_and_normalize_pattern("Floral") == "floral"
    assert builder.validate_and_normalize_pattern("Traditional Motifs") == "traditional_motifs"
    # Empty/missing input falls back to the documented default.
    assert builder.validate_and_normalize_pattern(None) == "solid"
    # Unknown-but-descriptive values are sanitized and kept (not silently
    # collapsed to "solid") — see garment_prompt_builder.validate_and_normalize_pattern.
    assert builder.validate_and_normalize_pattern("unrestricted_random_text") == "unrestricted_random_text"
