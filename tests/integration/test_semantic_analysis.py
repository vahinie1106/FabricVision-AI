from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from src.features.semantic_analysis.pipeline import SemanticAnalysisPipeline, SemanticAnalysisConfig


class FakeInferenceEngine:
    """Provide deterministic semantic-analysis responses for testing."""

    def __init__(self, response_payload: dict) -> None:
        self.response_payload = response_payload

    def run(self, image_path: str | Path, prompt: str) -> str:
        return json.dumps(self.response_payload)


class LoadAwareModelLoader:
    """Minimal stub that records whether the loader was invoked."""

    def __init__(self) -> None:
        self.loaded = False
        self.model = None
        self.processor = None

    def load(self) -> tuple[object | None, object | None]:
        self.loaded = True
        return None, None


def test_pipeline_generates_metadata_and_organizes_image(tmp_path: Path) -> None:
    image_path = tmp_path / "hoodie001.jpg"
    Image.new("RGB", (160, 200), color="blue").save(image_path)

    config = SemanticAnalysisConfig(
        config_dir="configs",
        output_root=str(tmp_path / "curated_dataset"),
        model_path="models/flux_kontext/Qwen2.5-VL-3B-Instruct",
        device="cpu",
    )

    pipeline = SemanticAnalysisPipeline(
        config=config,
        inference_engine=FakeInferenceEngine(
            {
                "garment_identity": {"name": "hoodie", "gender": "men"},
                "classification": {"category": "upper_wear", "subcategory": "hoodie", "garment_type": "hoodie"},
                "physical_attributes": {"material": "cotton", "construction": "fleece"},
                "visual_attributes": {"colors": ["blue"], "patterns": ["solid"], "texture": "soft"},
                "shape_and_fit": {"silhouette": "regular", "fit": "regular", "sleeves": "long", "neckline": "hooded"},
                "style": {"occasion": "casual", "season": "all_season"},
                "fabric_behaviour": {"drape": "moderate", "flexibility": "medium", "thickness": "medium"},
                "virtual_try_on_attributes": {"ease": "regular", "stretch": "medium"},
                "ai_analysis": {"confidence": 0.92, "model_info": "qwen-test"},
            }
        ),
        model_loader=None,
    )

    result = pipeline.run(str(image_path))

    assert result["status"] == "completed"
    assert result["metadata_path"].exists()
    assert result["organized_image_path"].exists()
    assert result["metadata"]["garment_identity"]["name"] == "hoodie"
    assert result["metadata"]["classification"]["category"] == "upper_wear"


def test_pipeline_loads_model_path_from_default_config_file() -> None:
    config = SemanticAnalysisConfig(config_dir="configs", output_root="curated_dataset", device="cpu")
    pipeline = SemanticAnalysisPipeline(config=config, inference_engine=FakeInferenceEngine({}), model_loader=LoadAwareModelLoader())

    assert Path(pipeline.config.model_path).exists()
    assert Path(pipeline.config.model_path).name == "Qwen2.5-VL-3B-Instruct"


def test_pipeline_initializes_model_loader_before_inference() -> None:
    loader = LoadAwareModelLoader()
    pipeline = SemanticAnalysisPipeline(
        config=SemanticAnalysisConfig(
            config_dir="configs",
            output_root="curated_dataset",
            model_path="models/Qwen2.5-VL-3B-Instruct",
            device="cpu",
        ),
        inference_engine=FakeInferenceEngine({}),
        model_loader=loader,
    )

    assert loader.loaded is True


def test_metadata_validator_rejects_unknown_vocab_value() -> None:
    from src.features.semantic_analysis.validation import MetadataValidator

    validator = MetadataValidator(config_dir="configs")
    metadata = {
        "garment_identity": {"name": "test", "gender": "men"},
        "classification": {"category": "upper_wear", "subcategory": "hoodie", "garment_type": "hoodie"},
        "physical_attributes": {"material": "unknown_material", "construction": "woven"},
        "visual_attributes": {"colors": ["blue"], "patterns": ["solid"], "texture": "soft"},
        "shape_and_fit": {"silhouette": "regular", "fit": "regular", "sleeves": "long", "neckline": "crew"},
        "style": {"occasion": "casual", "season": "all_season"},
        "fabric_behaviour": {"drape": "moderate", "flexibility": "medium", "thickness": "medium"},
        "virtual_try_on_attributes": {"ease": "regular", "stretch": "medium"},
        "ai_analysis": {"confidence": 0.92, "model_info": "qwen-test"},
    }

    result = validator.validate(metadata)

    assert result["is_valid"] is False
    assert any(issue["field"] == "physical_attributes.material" for issue in result["issues"])
