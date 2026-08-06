import pytest
from pathlib import Path
from src.garment_generation.prompting.garment_prompt_builder import GarmentPromptBuilder

def test_garment_prompt_builder_basic():
    builder = GarmentPromptBuilder(Path("configs"))
    fabric_metadata = {
        "material": "cotton",
        "pattern": "floral",
        "texture": "soft",
        "dominant_colors": ["royal_blue"],
        "style": "casual"
    }
    user_customization = {
        "gender": "women",
        "garment_type": "kurti",
        "sleeve": "three_quarter_sleeve",
        "neckline": "round_neck",
        "size": "M"
    }
    
    pos_prompt, neg_prompt = builder.build_prompts(fabric_metadata, user_customization)
    assert "women" in pos_prompt
    assert "kurti" in pos_prompt
    assert "cotton" in pos_prompt
    assert "floral" in pos_prompt
    assert "royal blue" in pos_prompt
    assert "three-quarter-sleeves" in pos_prompt
    assert "round neckline" in pos_prompt
    assert "size M" in pos_prompt
    assert "blurry" in neg_prompt

def test_garment_prompt_builder_fallbacks():
    builder = GarmentPromptBuilder(Path("configs"))
    fabric_metadata = {}
    user_customization = {}
    
    pos_prompt, neg_prompt = builder.build_prompts(fabric_metadata, user_customization)
    assert "women" in pos_prompt
    assert "kurti" in pos_prompt
    assert "cotton" in pos_prompt
    assert isinstance(pos_prompt, str)
    assert isinstance(neg_prompt, str)
