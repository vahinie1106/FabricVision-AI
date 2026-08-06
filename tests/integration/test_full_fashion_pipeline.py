import pytest
from pathlib import Path
from PIL import Image

from src.garment_generation.pipeline.garment_generation_pipeline import GarmentGenerationConfig, GarmentGenerationPipeline
from src.virtual_tryon.models import PersonConditioningInput, GarmentConditioningInput
from src.virtual_tryon.tryon_pipeline import TryOnConfig, VirtualTryOnPipeline
from src.models.model_manager import ModelManager


def test_full_fashion_pipeline_integration(tmp_path):
    output_dir = tmp_path / "outputs"
    exp_dir = tmp_path / "experiments"

    # Initialize sequential ModelManager
    model_manager = ModelManager()

    # 1. Stage 1: Semantic Analysis Simulation
    model_manager.switch_to("qwen")
    assert model_manager.active_model == "qwen"

    simulated_fabric_metadata = {
        "material": "cotton",
        "pattern": "floral",
        "texture": "soft",
        "dominant_colors": ["royal_blue"],
        "style": "traditional",
    }
    user_customization = {
        "gender": "women",
        "garment_type": "kurti",
        "sleeve": "three_quarter_sleeve",
        "neckline": "round_neck",
        "size": "M",
    }

    # 2. Stage 2: FLUX Garment Generation
    model_manager.switch_to("flux")
    assert model_manager.active_model == "flux"

    flux_config = GarmentGenerationConfig(
        config_dir="configs",
        output_root=str(output_dir / "generated_garments"),
        experiments_root=str(exp_dir),
        height=512,
        width=512,
        allow_fallback=True,
    )
    flux_pipeline = GarmentGenerationPipeline(config=flux_config)

    flux_result = flux_pipeline.run(
        fabric_metadata=simulated_fabric_metadata,
        user_customization=user_customization,
        output_filename="full_pipeline_garment",
    )

    assert flux_result["status"] == "completed"
    assert Path(flux_result["image_path"]).exists()
    generated_garment_img = Image.open(flux_result["image_path"])

    # 3. Stage 3: CatVTON Virtual Try-On
    model_manager.switch_to("catvton")
    assert model_manager.active_model == "catvton"

    tryon_config = TryOnConfig(
        config_dir="configs",
        output_root=str(output_dir / "virtual_tryon"),
        experiments_root=str(exp_dir),
        height=512,
        width=512,
        allow_fallback=True,
    )
    tryon_pipeline = VirtualTryOnPipeline(config=tryon_config)

    dummy_person = Image.new("RGB", (512, 512), color=(230, 210, 190))
    person_in = PersonConditioningInput(person_image=dummy_person)
    garment_in = GarmentConditioningInput(
        garment_image=generated_garment_img,
        garment_type=user_customization["garment_type"],
    )

    tryon_result = tryon_pipeline.run(
        person_input=person_in,
        garment_input=garment_in,
        output_filename="full_pipeline_tryon",
    )

    assert tryon_result.status == "completed"
    assert Path(tryon_result.image_path).exists()
    assert Path(tryon_result.metadata_path).exists()
    assert tryon_result.validation["valid"] is True
