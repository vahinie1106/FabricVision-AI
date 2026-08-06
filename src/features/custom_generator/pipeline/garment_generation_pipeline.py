from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from src.common.utils.utils import load_yaml_config, serialize_json
from src.features.custom_generator.inference.flux_inference import FLUXInferenceEngine
from src.features.custom_generator.model.flux_model_loader import FLUXModelLoader
from src.features.custom_generator.prompting.garment_prompt_builder import GarmentPromptBuilder
from src.features.custom_generator.validation.garment_validator import GarmentValidator


@dataclass
class GarmentGenerationConfig:
    """Runtime configuration for FLUX Garment Generation Pipeline."""

    config_dir: str = "configs"
    model_path: str = "models/flux"
    device: str = "auto"
    precision: str = "bfloat16"
    output_root: str = "outputs/generated_garments"
    experiments_root: str = "experiments"
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 4
    guidance_scale: float = 3.5
    seed: int = 42
    prompt_version: str = "v1.0"
    generation_mode: str = "Fast Preview"
    config_path: Optional[str] = None
    allow_fallback: bool = True


class GarmentGenerationPipeline:
    """Master pipeline for conditioned FLUX garment synthesis, validation, and experiment tracking."""

    def __init__(
        self,
        config: Optional[GarmentGenerationConfig] = None,
        model_loader: Optional[FLUXModelLoader] = None,
        inference_engine: Optional[FLUXInferenceEngine] = None,
        validator: Optional[GarmentValidator] = None,
    ) -> None:
        self.config = config or GarmentGenerationConfig()
        self.logger = logging.getLogger("fabricvision.garment_generation.pipeline")
        self._load_config_files()

        self.prompt_builder = GarmentPromptBuilder(self.config.config_dir)
        self.validator = validator or GarmentValidator()
        self.model_loader = model_loader or FLUXModelLoader(
            model_path=self.config.model_path,
            device=self.config.device,
            precision=self.config.precision,
            allow_fallback=self.config.allow_fallback,
        )
        self.inference_engine = inference_engine or FLUXInferenceEngine(
            self.model_loader,
            allow_fallback=self.config.allow_fallback,
        )

    def _load_config_files(self) -> None:
        if self.config.config_path:
            loaded_flux = load_yaml_config(self.config.config_path)
            if loaded_flux:
                self.config.model_path = loaded_flux.get("model_path", self.config.model_path)
                self.config.device = loaded_flux.get("device", self.config.device)
                self.config.precision = loaded_flux.get("precision", self.config.precision)

        gen_cfg_path = Path(self.config.config_dir) / "generation_config.yaml"
        if gen_cfg_path.exists():
            loaded_gen = load_yaml_config(gen_cfg_path)
            if loaded_gen:
                self.config.num_inference_steps = loaded_gen.get("default_num_inference_steps", self.config.num_inference_steps)
                self.config.guidance_scale = loaded_gen.get("default_guidance_scale", self.config.guidance_scale)
                self.config.height = loaded_gen.get("default_height", self.config.height)
                self.config.width = loaded_gen.get("default_width", self.config.width)
                self.config.seed = loaded_gen.get("seed", self.config.seed)
                self.config.output_root = loaded_gen.get("output_root", self.config.output_root)

    def run(
        self,
        fabric_metadata: Dict[str, Any],
        user_customization: Dict[str, Any],
        output_filename: Optional[str] = None,
        reference_image: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute prompt construction, FLUX synthesis, validation, and experiment tracking."""
        # 1. Build prompts
        positive_prompt, negative_prompt = self.prompt_builder.build_prompts(
            fabric_metadata=fabric_metadata,
            user_customization=user_customization,
        )

        # 2. Run inference
        image = self.inference_engine.generate(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            reference_image=reference_image,
            height=self.config.height,
            width=self.config.width,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            seed=self.config.seed,
        )

        # 3. Validate generated image
        garment_type = user_customization.get("garment_type") or "garment"
        color_val = fabric_metadata.get("dominant_colors") or user_customization.get("color")
        val_result = self.validator.validate(
            image,
            target_garment=garment_type,
            target_color=str(color_val) if color_val else None,
        )

        # 4. Prepare output & experiment paths
        output_base = Path(self.config.output_root)
        images_dir = output_base / "images"
        metadata_dir = output_base / "metadata"
        images_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        exp_root = Path(self.config.experiments_root)
        flux_exp_dir = exp_root / "flux_experiments"
        prompt_ver_dir = exp_root / "prompt_versions"
        results_dir = exp_root / "generation_results"
        for d in [flux_exp_dir, prompt_ver_dir, results_dir]:
            d.mkdir(parents=True, exist_ok=True)

        item_id = output_filename or f"garment_{uuid.uuid4().hex[:8]}"
        image_path = images_dir / f"{item_id}.png"
        meta_path = metadata_dir / f"{item_id}.json"

        # Save image
        image.save(image_path, format="PNG")

        # Normalize schema metadata fields
        norm_gender = self.prompt_builder._normalize_token(user_customization.get("gender"), "women")
        norm_type = self.prompt_builder._normalize_token(user_customization.get("garment_type"), "kurti")
        norm_mat = self.prompt_builder.validate_and_normalize_material(
            fabric_metadata.get("material") or user_customization.get("fabric_material")
        )
        norm_color = self.prompt_builder.validate_and_normalize_color(
            fabric_metadata.get("dominant_colors") or user_customization.get("color")
        )
        norm_pattern = self.prompt_builder.validate_and_normalize_pattern(
            fabric_metadata.get("pattern") or user_customization.get("pattern")
        )

        norm_sleeve = self.prompt_builder.validate_and_normalize_sleeve(user_customization.get("sleeve"))
        norm_neck = self.prompt_builder.validate_and_normalize_neckline(user_customization.get("neckline"))
        norm_size = self.prompt_builder.validate_and_normalize_size(user_customization.get("size"))

        gen_metadata = {
            "gender": norm_gender,
            "garment_type": norm_type,
            "material": norm_mat,
            "color": norm_color,
            "pattern": norm_pattern,
            "sleeve": norm_sleeve,
            "neckline": norm_neck,
            "size": norm_size,
            "model": "FLUX.1-schnell",
            "prompt_version": self.config.prompt_version,
            "generation_mode": self.config.generation_mode,
            "seed": self.config.seed,
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "generation_parameters": {
                "height": self.config.height,
                "width": self.config.width,
                "num_inference_steps": self.config.num_inference_steps,
                "guidance_scale": self.config.guidance_scale,
                "generation_mode": self.config.generation_mode,
            },
            "validation": val_result,
            "saved_image_path": str(image_path),
        }

        # Update last_execution_stats with mode & output details
        if hasattr(self.inference_engine, "last_execution_stats") and isinstance(self.inference_engine.last_execution_stats, dict):
            self.inference_engine.last_execution_stats["generation_mode"] = self.config.generation_mode
            self.inference_engine.last_execution_stats["num_inference_steps"] = self.config.num_inference_steps
            self.inference_engine.last_execution_stats["output_path"] = str(image_path)

        # Save outputs log & experiment record
        serialize_json(gen_metadata, meta_path)
        serialize_json(gen_metadata, results_dir / f"{item_id}_exp.json")

        status_str = "completed" if val_result.get("valid", True) else "rejected"
        return {
            "status": status_str,
            "image_path": str(image_path),
            "metadata_path": str(meta_path),
            "validation": val_result,
            "metadata": gen_metadata,
        }
