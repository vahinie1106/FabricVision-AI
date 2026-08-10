from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from src.common.utils.utils import load_yaml_config, serialize_json
from src.features.custom_generator.inference.fabric_conditioning import (
    build_garment_conditioning_image,
)
from src.features.custom_generator.inference.flux_inference import FLUXInferenceEngine
from src.features.custom_generator.model.flux_model_loader import FLUXModelLoader
from src.features.custom_generator.prompting.garment_prompt_builder import GarmentPromptBuilder
from src.features.custom_generator.validation.garment_validator import GarmentValidator


ProgressCallback = Optional[Callable[[str, int], None]]


def normalize_generation_mode(mode: Optional[str]) -> str:
    """
    Map UI / API mode labels onto canonical keys: preview | standard | production.

    Backward compatible with "Fast Preview" and "High Quality".
    """
    raw = (mode or "").strip().lower().replace("-", " ").replace("_", " ")
    raw = " ".join(raw.split())
    if raw in ("preview", "fast preview", "fast", "fastpreview"):
        return "preview"
    if raw in ("production", "high quality", "hq", "high", "highquality", "prod"):
        return "production"
    if raw in ("standard", "default", "normal", ""):
        return "standard"
    # Unknown labels default to standard (quality-safe default)
    return "standard"


@dataclass
class GarmentGenerationConfig:
    """Runtime configuration for FLUX.1-Kontext garment generation."""

    config_dir: str = "configs"
    model_path: str = "models/flux-kontext"
    device: str = "auto"
    precision: str = "bfloat16"
    output_root: str = "outputs/generated_garments"
    experiments_root: str = "experiments"
    height: int = 512
    width: int = 512
    num_inference_steps: int = 4
    guidance_scale: float = 2.5
    seed: int = 42
    prompt_version: str = "v2.0"
    generation_mode: str = "standard"
    config_path: Optional[str] = None
    allow_fallback: bool = True
    png_compress_level: int = 3
    png_optimize: bool = False
    enable_torch_compile: bool = False
    attention_backend: str = "auto"
    profile: bool = True
    # Internal: resolved canonical mode key after config load
    mode_key: str = field(default="standard", repr=False)


class GarmentGenerationPipeline:
    """Fabric-conditioned FLUX.1-Kontext garment synthesis pipeline."""

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
            enable_torch_compile=self.config.enable_torch_compile,
            attention_backend=self.config.attention_backend,
        )
        self.inference_engine = inference_engine or FLUXInferenceEngine(
            self.model_loader,
            allow_fallback=self.config.allow_fallback,
        )

    def _mode_config_key(self, mode: str) -> str:
        canonical = normalize_generation_mode(mode)
        # Prefer canonical blocks; fall back to legacy aliases in YAML
        return canonical

    def _load_config_files(self) -> None:
        flux_cfg: Dict[str, Any] = {}
        if self.config.config_path:
            flux_cfg = load_yaml_config(self.config.config_path) or {}
            self.config.model_path = flux_cfg.get("model_path", self.config.model_path)
            self.config.device = flux_cfg.get("device", self.config.device)
            self.config.precision = flux_cfg.get("precision", self.config.precision)
            self.config.enable_torch_compile = bool(
                flux_cfg.get("enable_torch_compile", self.config.enable_torch_compile)
            )
            self.config.attention_backend = str(
                flux_cfg.get("attention_backend", self.config.attention_backend)
            )
            self.config.profile = bool(flux_cfg.get("profile", self.config.profile))
            # Propagate T5 sequence budget — measured 512 → ~360s encode on RTX 3050
            max_seq = flux_cfg.get("max_sequence_length")
            if max_seq is not None and not os.environ.get("FLUX_MAX_SEQUENCE_LENGTH"):
                os.environ["FLUX_MAX_SEQUENCE_LENGTH"] = str(int(max_seq))
            if flux_cfg.get("enable_vae_tiling") is False:
                os.environ.setdefault("FLUX_VAE_TILING", "false")
            # Day-17: preencode at seq=128 ≈ 48–50s and frees T5 before diffusion.
            if "FLUX_PREENCODE_PROMPT" not in os.environ:
                preencode = flux_cfg.get("preencode_prompt", True)
                os.environ["FLUX_PREENCODE_PROMPT"] = (
                    "true" if bool(preencode) else "false"
                )

        # Env overrides (Phase 19)
        env_compile = os.environ.get("FLUX_ENABLE_TORCH_COMPILE", "").strip().lower()
        if env_compile in ("1", "true", "yes", "on"):
            self.config.enable_torch_compile = True
        elif env_compile in ("0", "false", "no", "off"):
            self.config.enable_torch_compile = False

        env_attn = os.environ.get("FLUX_ATTENTION_BACKEND", "").strip()
        if env_attn:
            self.config.attention_backend = env_attn

        env_profile = os.environ.get("FLUX_PROFILE", "").strip().lower()
        if env_profile in ("1", "true", "yes", "on"):
            self.config.profile = True
            os.environ["FLUX_PROFILE"] = "true"
        elif env_profile in ("0", "false", "no", "off"):
            self.config.profile = False
            os.environ["FLUX_PROFILE"] = "false"
        elif self.config.profile:
            os.environ.setdefault("FLUX_PROFILE", "true")

        mode_key = self._mode_config_key(self.config.generation_mode)
        self.config.mode_key = mode_key
        self.config.generation_mode = {
            "preview": "Preview",
            "standard": "Standard",
            "production": "Production",
        }.get(mode_key, "Standard")

        mode_cfg = None
        if isinstance(flux_cfg, dict):
            # Try canonical key, then legacy aliases
            aliases = [mode_key]
            if mode_key == "preview":
                aliases.extend(["fast_preview"])
            elif mode_key == "production":
                aliases.extend(["high_quality"])
            for key in aliases:
                if isinstance(flux_cfg.get(key), dict):
                    mode_cfg = flux_cfg[key]
                    break

        if isinstance(mode_cfg, dict):
            self.config.height = int(mode_cfg.get("height", self.config.height))
            self.config.width = int(mode_cfg.get("width", self.config.width))
            self.config.num_inference_steps = int(
                mode_cfg.get("num_inference_steps", self.config.num_inference_steps)
            )
            self.config.guidance_scale = float(
                mode_cfg.get("guidance_scale", self.config.guidance_scale)
            )

        # Production size override (only after operator validates VRAM)
        if mode_key == "production":
            prod_size = os.environ.get("FLUX_PRODUCTION_SIZE", "").strip()
            if prod_size.isdigit():
                size = int(prod_size)
                if size in (512, 640, 768):
                    self.config.height = size
                    self.config.width = size
            prod_steps = os.environ.get("FLUX_PRODUCTION_STEPS", "").strip()
            if prod_steps.isdigit():
                self.config.num_inference_steps = max(1, int(prod_steps))

        std_steps = os.environ.get("FLUX_STANDARD_STEPS", "").strip()
        if mode_key == "standard" and std_steps.isdigit():
            self.config.num_inference_steps = max(1, int(std_steps))

        preview_steps = os.environ.get("FLUX_PREVIEW_STEPS", "").strip()
        if mode_key == "preview" and preview_steps.isdigit():
            self.config.num_inference_steps = max(1, int(preview_steps))

        gen_cfg_path = Path(self.config.config_dir) / "generation_config.yaml"
        if not gen_cfg_path.exists():
            gen_cfg_path = Path(self.config.config_dir) / "custom_generator" / "generation_config.yaml"
        if gen_cfg_path.exists():
            loaded_gen = load_yaml_config(gen_cfg_path) or {}
            if not mode_cfg:
                self.config.num_inference_steps = loaded_gen.get(
                    "default_num_inference_steps", self.config.num_inference_steps
                )
                self.config.guidance_scale = loaded_gen.get(
                    "default_guidance_scale", self.config.guidance_scale
                )
                self.config.height = loaded_gen.get("default_height", self.config.height)
                self.config.width = loaded_gen.get("default_width", self.config.width)
            # Only apply global defaults when caller left these at the dataclass
            # default. Otherwise an explicit output_root (e.g. pytest tmp_path
            # isolation, benchmark scripts) was silently overwritten — a real
            # stabilization bug, not a feature.
            _defaults = GarmentGenerationConfig.__dataclass_fields__
            if self.config.seed == _defaults["seed"].default:
                self.config.seed = loaded_gen.get("seed", self.config.seed)
            if self.config.output_root == _defaults["output_root"].default:
                self.config.output_root = loaded_gen.get("output_root", self.config.output_root)
            self.config.png_compress_level = int(
                loaded_gen.get("png_compress_level", self.config.png_compress_level)
            )
            self.config.png_optimize = bool(
                loaded_gen.get("png_optimize", self.config.png_optimize)
            )

    def run(
        self,
        fabric_metadata: Dict[str, Any],
        user_customization: Dict[str, Any],
        output_filename: Optional[str] = None,
        reference_image: Optional[Any] = None,
        progress_callback: ProgressCallback = None,
    ) -> Dict[str, Any]:
        """Execute Kontext prompt build, fabric conditioning, inference, validation."""
        t_total = time.perf_counter()
        timings: Dict[str, float] = {}

        def _progress(step: str, pct: int) -> None:
            if progress_callback is not None:
                try:
                    progress_callback(step, pct)
                except Exception:
                    pass

        if reference_image is None:
            raise RuntimeError(
                "Fabric reference image is required for FLUX.1-Kontext garment generation."
            )

        # Keep fabric in memory — avoid re-reading from disk during conditioning.
        fabric_image = reference_image
        if hasattr(fabric_image, "convert"):
            fabric_image = fabric_image.convert("RGB")

        _progress("Preparing fabric", 20)
        t0 = time.perf_counter()
        try:
            from src.features.custom_generator.inference.fabric_appearance import (
                describe_fabric_appearance,
            )

            appearance = describe_fabric_appearance(fabric_image)
            # CRITICAL: pixel palette wins over UI Color dropdown unless force_recolor.
            # Observed failure: UI color=yellow forced yellow garment despite white+red floral fabric.
            force_recolor = bool(user_customization.get("force_recolor"))
            if force_recolor and user_customization.get("color"):
                fabric_metadata = {
                    **fabric_metadata,
                    "pattern": appearance.get("pattern_hint")
                    or fabric_metadata.get("pattern"),
                    "dominant_colors": [str(user_customization.get("color"))],
                    "color_source": "ui_recolor",
                    "fabric_appearance": appearance.get("appearance_summary"),
                }
                self.logger.info(
                    "[FLUX COLOR] source=ui_recolor palette=%s",
                    fabric_metadata.get("dominant_colors"),
                )
            else:
                fabric_metadata = {
                    **fabric_metadata,
                    "pattern": appearance.get("pattern_hint")
                    or fabric_metadata.get("pattern"),
                    "dominant_colors": appearance.get("dominant_color_names")
                    or fabric_metadata.get("dominant_colors"),
                    "color_source": "fabric_pixels",
                    "fabric_appearance": appearance.get("appearance_summary"),
                }
                user_customization = {
                    k: v for k, v in user_customization.items() if k != "color"
                }
                self.logger.info(
                    "[FLUX COLOR] source=fabric_pixels palette=%s (UI color ignored)",
                    fabric_metadata.get("dominant_colors"),
                )
            self.logger.info("Fabric appearance cues: %s", appearance)
        except Exception as exc:
            self.logger.warning("Fabric appearance enrichment skipped: %s", exc)
        timings["fabric_appearance_s"] = round(time.perf_counter() - t0, 3)

        _progress("Encoding prompt", 28)
        t0 = time.perf_counter()
        positive_prompt, negative_prompt = self.prompt_builder.build_kontext_prompt(
            fabric_metadata=fabric_metadata,
            user_customization=user_customization,
        )
        timings["prompt_building_s"] = round(time.perf_counter() - t0, 3)
        prompt_stats = getattr(self.prompt_builder, "last_prompt_stats", {}) or {}
        self.logger.info("=== FINAL POSITIVE PROMPT ===\n%s", positive_prompt)
        self.logger.info("=== FINAL NEGATIVE PROMPT ===\n%s", negative_prompt)

        garment_type = str(user_customization.get("garment_type") or "shirt")
        sleeve = str(user_customization.get("sleeve") or "")

        _progress("Preparing garment conditioning", 35)
        t0 = time.perf_counter()
        conditioning_image = build_garment_conditioning_image(
            fabric_image=fabric_image,
            garment_type=garment_type,
            width=self.config.width,
            height=self.config.height,
            sleeve=sleeve,
        )
        timings["fabric_conditioning_s"] = round(time.perf_counter() - t0, 3)

        # Persist stage images for A/B audits (original already in uploads; save cond)
        debug_dir = Path(self.config.output_root) / "audit_stages"
        debug_dir.mkdir(parents=True, exist_ok=True)
        stage_id = output_filename or f"stage_{uuid.uuid4().hex[:8]}"
        try:
            fabric_image.save(debug_dir / f"{stage_id}_A_fabric.png")
            conditioning_image.save(debug_dir / f"{stage_id}_B_conditioning.png")
            self.logger.info(
                "[FLUX STAGES] fabric=%sx%s conditioning=%sx%s → %s",
                fabric_image.size[0],
                fabric_image.size[1],
                conditioning_image.size[0],
                conditioning_image.size[1],
                debug_dir,
            )
        except Exception as stage_exc:
            self.logger.warning("Stage image save skipped: %s", stage_exc)
        self.logger.info(
            "Kontext conditioning: garment silhouette filled with uploaded fabric "
            "(mode=%s %sx%s steps=%s guidance=%s)",
            self.config.generation_mode,
            self.config.width,
            self.config.height,
            self.config.num_inference_steps,
            self.config.guidance_scale,
        )

        garment_id = output_filename or f"garment_{uuid.uuid4().hex[:8]}"
        raw_dir = Path(self.config.output_root) / "raw"
        raw_path = raw_dir / f"{garment_id}_raw.png"

        image = self.inference_engine.generate(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            reference_image=conditioning_image,
            height=self.config.height,
            width=self.config.width,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            seed=self.config.seed,
            progress_callback=progress_callback,
            save_raw_path=str(raw_path),
        )

        # Apply contour-guided detail refiner (sharpens neckline, sleeves, seams; preserves fabric identity)
        try:
            from src.features.custom_generator.inference.garment_detail_refiner import (
                GarmentDetailRefiner,
            )

            refiner = GarmentDetailRefiner()
            image = refiner.refine(image, mask_fabric_interior=True, enabled=True)
        except Exception as ref_exc:
            self.logger.warning("Garment detail refiner skipped: %s", ref_exc)

        color_val = fabric_metadata.get("dominant_colors") or user_customization.get("color")
        val_result = self.validator.validate(
            image,
            target_garment=garment_type,
            target_color=str(color_val) if color_val else None,
        )

        _progress("Saving result", 92)
        t0 = time.perf_counter()
        output_base = Path(self.config.output_root)
        images_dir = output_base / "images"
        metadata_dir = output_base / "metadata"
        images_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        image_path = images_dir / f"{garment_id}.png"
        # Visually lossless PNG: moderate compress_level is faster without quality loss.
        # Do NOT resize — save exact model output resolution for frontend/download.
        if hasattr(image, "size"):
            self.logger.info(
                "[FLUX] Final save size=%sx%s (must match generation; no downscale)",
                image.size[0],
                image.size[1],
            )
        image.save(
            image_path,
            format="PNG",
            compress_level=int(self.config.png_compress_level),
            optimize=bool(self.config.png_optimize),
        )
        timings["image_saving_s"] = round(time.perf_counter() - t0, 3)
        timings["total_pipeline_s"] = round(time.perf_counter() - t_total, 3)

        if self.config.profile:
            self.logger.info(
                "\n[FLUX PIPELINE PROFILE]\n"
                "Fabric appearance: %.3f sec\n"
                "Prompt building: %.3f sec\n"
                "Fabric conditioning: %.3f sec\n"
                "Image saving: %.3f sec\n"
                "TOTAL pipeline: %.3f sec\n",
                timings["fabric_appearance_s"],
                timings["prompt_building_s"],
                timings["fabric_conditioning_s"],
                timings["image_saving_s"],
                timings["total_pipeline_s"],
            )

        metadata = {
            "garment_id": garment_id,
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "prompt_stats": prompt_stats,
            "fabric_metadata": fabric_metadata,
            "user_customization": user_customization,
            "validation": val_result,
            "generation_mode": self.config.generation_mode,
            "mode_key": self.config.mode_key,
            "model": "FLUX.1-Kontext",
            "height": self.config.height,
            "width": self.config.width,
            "num_inference_steps": self.config.num_inference_steps,
            "guidance_scale": self.config.guidance_scale,
            "image_path": str(image_path),
            "raw_image_path": str(raw_path),
            "pipeline_timings": timings,
        }
        serialize_json(metadata, metadata_dir / f"{garment_id}.json")

        exp_root = Path(self.config.experiments_root)
        flux_exp_dir = exp_root / "generation_results"
        flux_exp_dir.mkdir(parents=True, exist_ok=True)
        serialize_json(
            {
                **metadata,
                "stats": getattr(self.inference_engine, "last_execution_stats", {}),
            },
            flux_exp_dir / f"{garment_id}_exp.json",
        )

        stats = getattr(self.inference_engine, "last_execution_stats", None)
        if stats is not None:
            stats["generation_mode"] = self.config.generation_mode
            stats["mode_key"] = self.config.mode_key
            stats["pipeline_timings"] = timings
            stats["prompt_stats"] = prompt_stats

        _progress("Completed", 100)

        return {
            "image_path": str(image_path),
            "output_path": str(image_path),
            "metadata": metadata,
            "validation": val_result,
        }
