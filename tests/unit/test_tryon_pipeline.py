import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from src.features.virtual_tryon.models import PersonConditioningInput, GarmentConditioningInput
from src.features.virtual_tryon.tryon_pipeline import TryOnConfig, VirtualTryOnPipeline


def create_non_flat_image(color1, color2, width=512, height=512):
    img = Image.new("RGB", (width, height), color=color1)
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, width - 50, height - 50], fill=color2)
    return img


def test_virtual_tryon_pipeline_dry_run(tmp_path, monkeypatch):
    # Unit dry-run: allow blend when weights unavailable; never default to production REQUIRE_REAL.
    monkeypatch.setenv("CATVTON_REQUIRE_REAL", "false")
    monkeypatch.delenv("CATVTON_DEBUG", raising=False)

    out_dir = tmp_path / "outputs" / "virtual_tryon"
    exp_dir = tmp_path / "experiments"

    config = TryOnConfig(
        config_dir="configs",
        output_root=str(out_dir),
        experiments_root=str(exp_dir),
        height=512,
        width=384,
        allow_fallback=True,
    )

    pipeline = VirtualTryOnPipeline(config=config)

    person_img = create_non_flat_image((220, 200, 180), (180, 140, 100))
    garment_img = create_non_flat_image((50, 100, 200), (200, 220, 250))

    person_input = PersonConditioningInput(person_image=person_img)
    garment_input = GarmentConditioningInput(garment_image=garment_img, garment_type="kurti")

    result = pipeline.run(person_input, garment_input, output_filename="test_tryon_001")

    assert result.status in ("completed", "completed_with_fallback")
    assert Path(result.image_path).exists()
    assert Path(result.metadata_path).exists()
    assert (exp_dir / "tryon_results" / "test_tryon_001_exp.json").exists()
    assert result.validation["valid"] is True

    meta = json.loads(Path(result.metadata_path).read_text(encoding="utf-8"))
    assert meta.get("mask_source") in ("grabcut", "box_fallback", "provided", "automasker")
    assert "was_fallback_used" in meta
    assert "was_real_catvton_used" in meta
    assert "inference_backend" in meta
    assert meta.get("resolution") == [512, 512] or (
        isinstance(meta.get("resolution"), list) and len(meta["resolution"]) == 2
    )
    # PNG integrity: saved file opens at expected size
    saved = Image.open(result.image_path)
    assert saved.size[0] in (384, 512) and saved.size[1] in (384, 512)
    assert Path(result.image_path).suffix.lower() == ".png"

    if meta.get("was_fallback_used"):
        assert meta.get("status") == "completed_with_fallback"
        assert meta.get("was_real_catvton_used") is False
        assert meta.get("inference_backend") == "blend_preview"
    else:
        assert meta.get("was_real_catvton_used") is True
        assert meta.get("inference_backend") in ("catvton_native", "diffusers_inpaint")


def test_require_real_rejects_blend(tmp_path, monkeypatch):
    monkeypatch.setenv("CATVTON_REQUIRE_REAL", "true")
    out_dir = tmp_path / "outputs" / "virtual_tryon"
    exp_dir = tmp_path / "experiments"
    config = TryOnConfig(
        output_root=str(out_dir),
        experiments_root=str(exp_dir),
        height=256,
        width=256,
        allow_fallback=True,  # config alone must not override REQUIRE_REAL
        model_path="models/CatVTON_DOES_NOT_EXIST_FOR_TEST",
    )
    pipeline = VirtualTryOnPipeline(config=config)
    person = PersonConditioningInput(person_image=create_non_flat_image((200, 180, 160), (100, 80, 60), 256, 256))
    garment = GarmentConditioningInput(
        garment_image=create_non_flat_image((20, 40, 200), (200, 20, 80), 256, 256),
        garment_type="shirt",
    )
    with pytest.raises(RuntimeError, match="CatVTON|box_fallback|required"):
        pipeline.run(person, garment, output_filename="require_real_test")
