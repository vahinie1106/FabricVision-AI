import pytest
from pathlib import Path
from PIL import Image

from src.features.custom_generator.pipeline.garment_generation_pipeline import (
    GarmentGenerationConfig,
    GarmentGenerationPipeline,
)
from src.features.virtual_tryon.models import PersonConditioningInput, GarmentConditioningInput
from src.features.virtual_tryon.tryon_pipeline import TryOnConfig, VirtualTryOnPipeline
from src.common.models.model_manager import ModelManager

FLORAL_FABRIC = Path("data/uploads/30c9aacaacf444169b3959f2171b4942.jpg")


@pytest.mark.skipif(
    not FLORAL_FABRIC.exists(), reason="reference floral fabric fixture missing from data/uploads"
)
def test_full_fashion_pipeline_integration(tmp_path, monkeypatch):
    monkeypatch.setenv("CATVTON_REQUIRE_REAL", "false")
    output_dir = tmp_path / "outputs"
    exp_dir = tmp_path / "experiments"

    # Initialize sequential ModelManager
    model_manager = ModelManager()

    # 1. Stage 1: Semantic Analysis Simulation
    model_manager.switch_to("qwen")
    assert model_manager.active_model == "qwen"

    simulated_fabric_metadata = {
        "material": "cotton",
        "texture": "soft",
        "style": "traditional",
    }
    user_customization = {
        "gender": "women",
        "garment_type": "kurti",
        "sleeve": "three_quarter_sleeve",
        "neckline": "round_neck",
        "size": "M",
    }

    # 2. Stage 2: FLUX Garment Generation (FLUX.1-Kontext; pytest env uses the
    # synthetic placeholder path in FLUXInferenceEngine, no GPU/model download).
    model_manager.switch_to("flux")
    assert model_manager.active_model == "flux"

    flux_config = GarmentGenerationConfig(
        config_dir="configs",
        config_path="configs/custom_generator/flux_config.yaml",
        output_root=str(output_dir / "generated_garments"),
        experiments_root=str(exp_dir),
        height=512,
        width=512,
        allow_fallback=True,
    )
    flux_pipeline = GarmentGenerationPipeline(config=flux_config)

    fabric_image = Image.open(FLORAL_FABRIC).convert("RGB")

    flux_result = flux_pipeline.run(
        fabric_metadata=simulated_fabric_metadata,
        user_customization=user_customization,
        output_filename="full_pipeline_garment",
        reference_image=fabric_image,
    )

    assert Path(flux_result["image_path"]).exists()
    assert flux_result["metadata"]["model"] == "FLUX.1-Kontext"
    generated_garment_img = Image.open(flux_result["image_path"])

    # 3. Stage 3: CatVTON Virtual Try-On
    model_manager.switch_to("catvton")
    assert model_manager.active_model == "catvton"

    tryon_config = TryOnConfig(
        config_dir="configs",
        output_root=str(output_dir / "virtual_tryon"),
        experiments_root=str(exp_dir),
        height=512,
        width=384,
        allow_fallback=True,
    )
    tryon_pipeline = VirtualTryOnPipeline(config=tryon_config)

    dummy_person = Image.new("RGB", (512, 512), color=(230, 210, 190))
    # Non-flat person stub so GrabCut variance gate / silhouette heuristics stay realistic.
    from PIL import ImageDraw

    _d = ImageDraw.Draw(dummy_person)
    _d.ellipse([180, 40, 330, 190], fill=(210, 170, 140))
    _d.rectangle([190, 180, 320, 360], fill=(40, 120, 80))
    _d.rectangle([200, 360, 250, 500], fill=(30, 40, 60))
    _d.rectangle([260, 360, 310, 500], fill=(30, 40, 60))
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

    assert tryon_result.status in ("completed", "completed_with_fallback")
    assert Path(tryon_result.image_path).exists()
    assert Path(tryon_result.metadata_path).exists()
    assert tryon_result.validation["valid"] is True
