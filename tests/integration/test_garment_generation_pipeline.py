import pytest
from pathlib import Path
from src.garment_generation.pipeline.garment_generation_pipeline import GarmentGenerationConfig, GarmentGenerationPipeline

def test_garment_generation_pipeline_dry_run(tmp_path):
    output_dir = tmp_path / "outputs" / "generated_garments"
    exp_dir = tmp_path / "experiments"
    
    config = GarmentGenerationConfig(
        config_dir="configs",
        output_root=str(output_dir),
        experiments_root=str(exp_dir),
        height=512,
        width=512,
        allow_fallback=True,
    )
    
    pipeline = GarmentGenerationPipeline(config=config)
    
    fabric_metadata = {
        "material": "cotton",
        "pattern": "floral",
        "texture": "smooth",
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
    
    result = pipeline.run(
        fabric_metadata=fabric_metadata,
        user_customization=user_customization,
        output_filename="test_kurti_001"
    )
    
    assert result["status"] == "completed"
    assert Path(result["image_path"]).exists()
    assert Path(result["metadata_path"]).exists()
    assert (exp_dir / "generation_results" / "test_kurti_001_exp.json").exists()
    
    meta = result["metadata"]
    assert meta["gender"] == "women"
    assert meta["garment_type"] == "kurti"
    assert meta["material"] == "cotton"
    assert meta["color"] == "royal_blue"
    assert meta["sleeve"] == "three_quarter_sleeve"
    assert meta["neckline"] == "round_neck"
    assert meta["size"] == "M"
    assert meta["model"] == "FLUX.1-schnell"

    assert meta["prompt_version"] == "v1.0"
