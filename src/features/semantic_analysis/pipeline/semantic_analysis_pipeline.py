from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.common.utils.utils import load_yaml_config
from src.features.semantic_analysis.inference.qwen_inference import QwenInferenceEngine
from src.features.semantic_analysis.metadata.metadata_builder import MetadataBuilder
from src.features.semantic_analysis.model.qwen_model import QwenModelLoader
from src.features.semantic_analysis.organization.dataset_organizer import DatasetOrganizer
from src.features.semantic_analysis.parsing.response_parser import ResponseParser
from src.features.semantic_analysis.prompting.prompt_builder import PromptBuilder
from src.features.semantic_analysis.validation.metadata_normalizer import MetadataNormalizer
from src.features.semantic_analysis.validation.metadata_validator import MetadataValidator


@dataclass
class SemanticAnalysisConfig:
    """Runtime configuration for the Semantic Analysis pipeline."""

    config_dir: str = "configs"
    output_root: str = "curated_dataset"
    model_path: str = "models/Qwen2.5-VL-3B-Instruct"
    device: str = "auto"
    config_path: Optional[str] = None
    supported_formats: list[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp"])


class SemanticAnalysisPipeline:
    """Coordinate image analysis, metadata generation, validation, and dataset organization."""

    def __init__(self, config: Optional[SemanticAnalysisConfig] = None, inference_engine: Optional[object] = None, model_loader: Optional[QwenModelLoader] = None) -> None:
        self.config = config or SemanticAnalysisConfig()
        self.logger = logging.getLogger("fabricvision.semantic_analysis.pipeline")
        self._load_config_file()
        self.prompt_builder = PromptBuilder(self.config.config_dir)
        self.response_parser = ResponseParser()
        self.metadata_builder = MetadataBuilder()
        self.normalizer = MetadataNormalizer(self.config.config_dir)
        self.validator = MetadataValidator(self.config.config_dir)
        self.organizer = DatasetOrganizer(self.config.output_root)
        self.model_loader = model_loader or QwenModelLoader(self.config.model_path, self.config.device)
        if hasattr(self.model_loader, "load"):
            self.model_loader.load()
        self.inference_engine = inference_engine or QwenInferenceEngine(self.model_loader, self.config.device)

    def run(self, image_path: str | Path) -> dict[str, Any]:
        """Run the full semantic analysis workflow for a single image."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        prompt = self.prompt_builder.build(image_path)
        raw_response = self.inference_engine.run(image_path, prompt)
        parsed_response = self.response_parser.parse(raw_response)
        metadata = self.metadata_builder.build(parsed_response, image_path)
        metadata = self.normalizer.normalize(metadata)
        validation_result = self.validator.validate(metadata)
        if not validation_result["is_valid"]:
            return {
                "status": "rejected",
                "image_path": str(image_path),
                "issues": validation_result["issues"],
                "metadata": metadata,
            }

        organization_result = self.organizer.organize(image_path, metadata)
        return {
            "status": "completed",
            "image_path": str(image_path),
            "metadata": metadata,
            "metadata_path": organization_result["metadata_path"],
            "organized_image_path": organization_result["image_path"],
            "validation": validation_result,
        }

    def _load_config_file(self) -> None:
        if self.config.config_path:
            loaded_config = load_yaml_config(self.config.config_path)
            if loaded_config:
                self.config.output_root = self._resolve_path(loaded_config.get("output_root", self.config.output_root))
                self.config.model_path = self._resolve_path(loaded_config.get("model_path", self.config.model_path))
                self.config.device = loaded_config.get("device", self.config.device)

    def _resolve_path(self, path_value: str | None) -> str:
        if not path_value:
            return self.config.output_root
        path = Path(path_value)
        if not path.is_absolute():
            workspace_root = Path(__file__).resolve().parents[3]
            path = workspace_root / path
        return str(path)
