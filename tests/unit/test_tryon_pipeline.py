import pytest
from pathlib import Path
from PIL import Image, ImageDraw
from src.virtual_tryon.models import PersonConditioningInput, GarmentConditioningInput
from src.virtual_tryon.tryon_pipeline import TryOnConfig, VirtualTryOnPipeline

def create_non_flat_image(color1, color2, width=512, height=512):
    img = Image.new("RGB", (width, height), color=color1)
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, width - 50, height - 50], fill=color2)
    return img

def test_virtual_tryon_pipeline_dry_run(tmp_path):
    out_dir = tmp_path / "outputs" / "virtual_tryon"
    exp_dir = tmp_path / "experiments"
    
    config = TryOnConfig(
        config_dir="configs",
        output_root=str(out_dir),
        experiments_root=str(exp_dir),
        height=512,
        width=512,
        allow_fallback=True,
    )
    
    pipeline = VirtualTryOnPipeline(config=config)
    
    person_img = create_non_flat_image((220, 200, 180), (180, 140, 100))
    garment_img = create_non_flat_image((50, 100, 200), (200, 220, 250))
    
    person_input = PersonConditioningInput(person_image=person_img)
    garment_input = GarmentConditioningInput(garment_image=garment_img, garment_type="kurti")
    
    result = pipeline.run(person_input, garment_input, output_filename="test_tryon_001")
    
    assert result.status == "completed"
    assert Path(result.image_path).exists()
    assert Path(result.metadata_path).exists()
    assert (exp_dir / "tryon_results" / "test_tryon_001_exp.json").exists()
    assert result.validation["valid"] is True
