import pytest
from pathlib import Path
from PIL import Image, ImageDraw

from src.virtual_tryon.models import PersonConditioningInput, GarmentConditioningInput
from src.virtual_tryon.tryon_pipeline import TryOnConfig, VirtualTryOnPipeline
from src.models.model_manager import ModelManager


def create_sample_image(color1, color2, width=512, height=512):
    img = Image.new("RGB", (width, height), color=color1)
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, width - 50, height - 50], fill=color2)
    return img


def test_full_virtual_tryon_pipeline_integration(tmp_path):
    output_dir = tmp_path / "outputs" / "virtual_tryon"
    exp_dir = tmp_path / "experiments"

    # Test sequential model switching
    model_manager = ModelManager()
    model_manager.switch_to("catvton")
    assert model_manager.active_model == "catvton"

    config = TryOnConfig(
        config_dir="configs",
        output_root=str(output_dir),
        experiments_root=str(exp_dir),
        height=512,
        width=512,
        allow_fallback=True,
    )
    pipeline = VirtualTryOnPipeline(config=config)

    person_img = create_sample_image((230, 210, 190), (180, 140, 100))
    garment_img = create_sample_image((255, 255, 255), (20, 50, 180))

    person_input = PersonConditioningInput(person_image=person_img)
    garment_input = GarmentConditioningInput(garment_image=garment_img, garment_type="kurti")

    result = pipeline.run(
        person_input=person_input,
        garment_input=garment_input,
        output_filename="test_integration_tryon",
    )

    assert result.status == "completed"
    assert Path(result.image_path).exists()
    assert Path(result.metadata_path).exists()
    assert result.validation["valid"] is True
    assert result.validation["validation_status"] == "passed"
