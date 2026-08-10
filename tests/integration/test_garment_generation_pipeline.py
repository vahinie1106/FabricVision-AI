import pytest
from pathlib import Path
from PIL import Image

from src.features.custom_generator.pipeline.garment_generation_pipeline import (
    GarmentGenerationConfig,
    GarmentGenerationPipeline,
)

# Real, known floral fabric used throughout the Day 17 FLUX.1-Kontext benchmarks:
# white/light base, red floral elements, green leaves. Deterministic pixel-derived
# appearance (verified in benchmark logs): dominant_colors=["white", "red"], pattern="floral".
FLORAL_FABRIC = Path("data/uploads/30c9aacaacf444169b3959f2171b4942.jpg")


@pytest.mark.skipif(
    not FLORAL_FABRIC.exists(), reason="reference floral fabric fixture missing from data/uploads"
)
def test_garment_generation_pipeline_dry_run(tmp_path):
    output_dir = tmp_path / "outputs" / "generated_garments"
    exp_dir = tmp_path / "experiments"

    config = GarmentGenerationConfig(
        config_dir="configs",
        config_path="configs/custom_generator/flux_config.yaml",
        output_root=str(output_dir),
        experiments_root=str(exp_dir),
        height=512,
        width=512,
        # allow_fallback=True + PYTEST_CURRENT_TEST (set by pytest) makes
        # FLUXModelLoader skip real weight loading and use the synthetic
        # placeholder path in FLUXInferenceEngine — no GPU/model download here.
        allow_fallback=True,
    )

    pipeline = GarmentGenerationPipeline(config=config)

    fabric_metadata = {
        "material": "cotton",
        "texture": "smooth",
        "style": "casual",
    }
    user_customization = {
        "gender": "women",
        "garment_type": "top",
        "sleeve": "short_sleeve",
        "neckline": "sweetheart",
        "size": "M",
    }

    fabric_image = Image.open(FLORAL_FABRIC).convert("RGB")

    result = pipeline.run(
        fabric_metadata=fabric_metadata,
        user_customization=user_customization,
        output_filename="test_top_001",
        reference_image=fabric_image,
    )

    # Current pipeline result contract: image_path/output_path/metadata/validation.
    # (No top-level "status" key — that belonged to an older API shape.)
    assert Path(result["image_path"]).exists()
    assert result["output_path"] == result["image_path"]
    assert (output_dir / "metadata" / "test_top_001.json").exists()
    assert (exp_dir / "generation_results" / "test_top_001_exp.json").exists()

    metadata = result["metadata"]

    # Architecture guarantee (Day 17): FLUX.1-Kontext only, never a Schnell fallback.
    assert metadata["model"] == "FLUX.1-Kontext"
    assert metadata["mode_key"] == "standard"
    assert metadata["height"] == 512
    assert metadata["width"] == 512

    # Fabric pixel palette must win over any UI color unless force_recolor is set
    # (Day 17 bugfix for the white/red floral -> yellow regression).
    fabric_meta_out = metadata["fabric_metadata"]
    assert fabric_meta_out["color_source"] == "fabric_pixels"
    assert fabric_meta_out["dominant_colors"] == ["white", "red"]
    assert fabric_meta_out["pattern"] == "floral"

    user_custom_out = metadata["user_customization"]
    assert user_custom_out["gender"] == "women"
    assert user_custom_out["garment_type"] == "top"
    assert user_custom_out["sleeve"] == "short_sleeve"
    assert user_custom_out["neckline"] == "sweetheart"
    assert user_custom_out["size"] == "M"

    # CLIP-safe prompt: must fit the 77-token budget with no tokenizer truncation.
    prompt_stats = metadata["prompt_stats"]
    assert prompt_stats["truncated"] is False
    assert prompt_stats["token_count"] <= prompt_stats["token_budget"]

    assert result["validation"]["valid"] is True
