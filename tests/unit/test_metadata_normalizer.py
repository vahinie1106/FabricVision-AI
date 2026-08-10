import pytest
from pathlib import Path
from src.features.semantic_analysis.validation.metadata_normalizer import MetadataNormalizer

def test_normalization_space_to_snake():
    normalizer = MetadataNormalizer(Path("configs"))
    metadata = {
        "classification": {
            "subcategory": "tank top"
        }
    }
    normalized = normalizer.normalize(metadata)
    assert normalized["classification"]["subcategory"] == "tank_top"

def test_normalization_synonym_mapping():
    normalizer = MetadataNormalizer(Path("configs"))
    metadata = {
        "visual_attributes": {
            "colors": ["Navy Blue"]
        }
    }
    normalized = normalizer.normalize(metadata)
    assert normalized["visual_attributes"]["colors"] == ["navy_blue"]
    
def test_normalization_unknown_fallback():
    normalizer = MetadataNormalizer(Path("configs"))
    metadata = {
        "classification": {
            "subcategory": "unknown random fashion word"
        }
    }
    normalized = normalizer.normalize(metadata)
    assert normalized["classification"]["subcategory"] == "unknown"
