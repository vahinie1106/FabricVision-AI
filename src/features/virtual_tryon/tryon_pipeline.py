from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from PIL import Image

try:
    import torch
except ImportError:
    torch = None

from src.features.virtual_tryon.catvton_loader import CatVTONModelLoader
from src.features.virtual_tryon.garment_conditioning import GarmentConditioner
from src.features.virtual_tryon.models import GarmentConditioningInput, PersonConditioningInput, TryOnResult
from src.features.virtual_tryon.person_conditioning import PersonConditioner
from src.features.virtual_tryon.tryon_validator import TryOnValidator


@dataclass
class TryOnConfig:
    """Runtime configuration for the Virtual Try-On pipeline."""

    config_dir: str = "configs"
    output_root: str = "outputs/virtual_tryon"
    experiments_root: str = "experiments"
    model_path: str = "models/CatVTON"
    device: str = "auto"
    precision: str = "bfloat16"
    # Portrait 384x512 matches vitonhd-16k-512 training scale (NOT square 512x512).
    height: int = 512
    width: int = 384
    allow_fallback: bool = False
    num_inference_steps: int = 30
    guidance_scale: float = 2.5
    attn_ckpt_version: str = "vitonhd"
    supported_garment_types: list[str] = field(
        default_factory=lambda: ["kurti", "top", "shirt", "t_shirt", "dress", "hoodie", "jacket"]
    )


class VirtualTryOnPipeline:
    """Orchestrate person conditioning, garment conditioning, CatVTON diffusion execution, and validation."""

    def __init__(
        self,
        config: Optional[TryOnConfig] = None,
        model_loader: Optional[CatVTONModelLoader] = None,
    ) -> None:
        self.config = config or TryOnConfig()
        self.logger = logging.getLogger("fabricvision.virtual_tryon.pipeline")
        target_res = (self.config.width, self.config.height)
        self.person_conditioner = PersonConditioner(
            target_resolution=target_res, catvton_path=self.config.model_path
        )
        self.garment_conditioner = GarmentConditioner(
            target_resolution=target_res, catvton_path=self.config.model_path
        )
        from src.features.virtual_tryon.catvton_conditioning import attn_version_for_resolution

        attn_ver = self.config.attn_ckpt_version or attn_version_for_resolution(
            self.config.height, self.config.width
        )
        self.model_loader = model_loader or CatVTONModelLoader(
            model_path=self.config.model_path,
            device=self.config.device,
            precision=self.config.precision,
            allow_fallback=self.config.allow_fallback,
            attn_ckpt_version=attn_ver,
        )
        self.validator = TryOnValidator()

    def _blend_preview(self, person_img: Image.Image, garment_img: Image.Image) -> Image.Image:
        """
        Non-production composite used only when allow_fallback=True (CI / dry-run).

        Must never be presented as successful CatVTON in the production UI.
        """
        person_res = person_img.resize((self.config.width, self.config.height), Image.Resampling.LANCZOS).convert("RGBA")
        garment_res = garment_img.resize(
            (int(self.config.width * 0.6), int(self.config.height * 0.5)), Image.Resampling.LANCZOS
        ).convert("RGBA")

        offset_x = int((self.config.width - garment_res.width) / 2)
        offset_y = int(self.config.height * 0.25)

        composite = Image.new("RGBA", (self.config.width, self.config.height))
        composite.paste(person_res, (0, 0))
        composite.paste(garment_res, (offset_x, offset_y), mask=garment_res)
        return composite.convert("RGB")

    @staticmethod
    def _cloth_type_for_garment(
        garment_type: str,
        person_image: Optional[Image.Image] = None,
    ) -> str:
        """
        Map garment identity → CatVTON AutoMasker/GrabCut cloth_type.

        Unspecified 'garment' on a tall full-body person defaults to overall so a
        dress/kurti is not forced into an upper-only band (causes lower-body mess).
        Override with CATVTON_CLOTH_TYPE=upper|lower|overall.
        """
        env = os.environ.get("CATVTON_CLOTH_TYPE", "").strip().lower()
        if env in ("upper", "lower", "overall", "inner", "outer"):
            return env

        g = (garment_type or "garment").lower().replace(" ", "_")
        if g in ("pants", "trousers", "shorts", "jeans", "leggings"):
            return "lower"
        # Long / dress-like skirts use overall so GrabCut does not follow pant legs.
        if g in ("skirt", "mini_skirt", "short_skirt"):
            return "lower"
        if g in (
            "dress",
            "gown",
            "jumpsuit",
            "overall",
            "kurti",
            "kurta",
            "saree",
            "anarkali",
            "maxi",
            "maxi_dress",
            "maxi_skirt",
            "frock",
            "lehenga",
            "kaftan",
        ):
            return "overall"
        if g in (
            "top",
            "shirt",
            "t_shirt",
            "tshirt",
            "blouse",
            "hoodie",
            "jacket",
            "sweater",
            "coat",
            "blazer",
            "cardigan",
            "crop_top",
            "tank",
            "camisole",
        ):
            return "upper"
        if g in ("outer", "outerwear", "overcoat"):
            return "outer"
        # Unknown label: if person is full-body portrait, prefer overall.
        if person_image is not None:
            pw, ph = person_image.size
            if ph / max(1, pw) >= 1.35:
                return "overall"
        return "upper"

    @staticmethod
    def _require_real() -> bool:
        # Production default: fail closed unless explicitly disabled.
        raw = os.environ.get("CATVTON_REQUIRE_REAL", "true").strip().lower()
        return raw not in ("0", "false", "no", "off")

    def _validate_inputs(
        self,
        person_input: PersonConditioningInput,
        garment_input: GarmentConditioningInput,
    ) -> None:
        """Reject clearly invalid inputs before expensive inference."""
        person = person_input.person_image
        garment = garment_input.garment_image
        if not isinstance(person, Image.Image) or not isinstance(garment, Image.Image):
            raise ValueError("Person and garment must be valid images.")
        pw, ph = person.size
        gw, gh = garment.size
        if min(pw, ph) < 128:
            raise ValueError(f"Person image too small ({pw}x{ph}); minimum shorter side is 128px.")
        if min(gw, gh) < 64:
            raise ValueError(f"Garment image too small ({gw}x{gh}); minimum shorter side is 64px.")
        aspect = pw / max(1, ph)
        if aspect < 0.35 or aspect > 2.8:
            raise ValueError(
                f"Person aspect ratio {aspect:.2f} is unsupported; use a full-/half-body portrait."
            )

        # Quality-path guard: reject placeholders / fabric sheets used as "person".
        # Unit dry-runs with synthetic non-flat images still pass these heuristics.
        from src.features.virtual_tryon.person_image_validation import assess_person_image

        ok, reason = assess_person_image(person)
        if not ok:
            raise ValueError(f"Invalid person image for CatVTON try-on: {reason}")

    def run(
        self,
        person_input: PersonConditioningInput,
        garment_input: GarmentConditioningInput,
        output_filename: Optional[str] = None,
        person_filename: str = "person_01.png",
        garment_filename: str = "generated_garment.png",
    ) -> TryOnResult:
        """Execute conditioning, CatVTON diffusion try-on, validation, and file logging."""
        t0 = time.time()
        was_fallback_used = False
        inference_backend = "none"
        failed_stage: Optional[str] = None
        error_type: Optional[str] = None
        raw_output_image: Optional[Image.Image] = None
        vram_before = 0.0
        vram_after = 0.0
        guidance = float(self.config.guidance_scale)
        load_info: dict = {}
        mask_stats: dict = {}
        mask_reason = "ok"
        steps = int(self.config.num_inference_steps)
        original_person = person_input.person_image.copy() if isinstance(person_input.person_image, Image.Image) else None

        self._validate_inputs(person_input, garment_input)
        cloth_type = self._cloth_type_for_garment(
            garment_input.garment_type, person_image=person_input.person_image
        )
        self.logger.info(
            "CatVTON cloth_type=%s garment_type=%s person_orig=%s target=%sx%s",
            cloth_type,
            garment_input.garment_type,
            getattr(original_person, "size", None),
            self.config.width,
            self.config.height,
        )

        # 1. Condition inputs
        prep_person = self.person_conditioner.prepare_person_condition(
            person_input, cloth_type=cloth_type
        )
        prep_garment = self.garment_conditioner.prepare_garment_condition(garment_input)
        mask_source = getattr(self.person_conditioner, "last_mask_source", "unknown")
        mask_stats = getattr(self.person_conditioner, "last_mask_stats", {}) or {}
        mask_ok, mask_reason = getattr(
            self.person_conditioner, "last_mask_validation", (True, "ok")
        )
        from src.features.virtual_tryon.person_masker import resolve_person_mask

        mask_attempts = list(getattr(resolve_person_mask, "last_attempts", []) or [])
        mask_strategy = str(getattr(resolve_person_mask, "last_strategy", "auto") or "auto")

        if prep_person.agnostic_mask is None:
            raise ValueError("Person mask is empty after preprocessing.")
        import numpy as np

        fill = float((np.asarray(prep_person.agnostic_mask.convert("L")) > 127).mean())
        if fill < 0.02:
            raise ValueError("Person mask is effectively empty; cannot run try-on.")

        require_real = self._require_real()
        allow_fallback = self.config.allow_fallback and not require_real

        # Box masks / invalid clothing masks are not production-quality.
        if mask_source == "box_fallback" and require_real:
            raise RuntimeError(
                "CatVTON refused box_fallback mask under CATVTON_REQUIRE_REAL. "
                "GrabCut could not produce a clothing-region mask; AutoMasker "
                "(detectron2 + DensePose/SCHP) is the supported production path."
            )
        if require_real and not mask_ok:
            raise RuntimeError(
                f"CatVTON refused invalid clothing mask ({mask_reason}). "
                "GrabCut is insufficient for this input; enable AutoMasker if available."
            )

        debug = os.environ.get("CATVTON_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")
        debug_stamp = uuid.uuid4().hex[:8] if debug else None
        if debug and debug_stamp:
            self._save_debug_intermediates(
                prep_person,
                prep_garment,
                stamp=debug_stamp,
                original_person=original_person,
                original_garment=garment_input.garment_image
                if isinstance(garment_input.garment_image, Image.Image)
                else None,
                stage="conditioning",
                mask_stats=mask_stats,
            )

        # 2. Run inference
        steps = int(os.environ.get("CATVTON_STEPS", str(self.config.num_inference_steps)))
        steps = max(8, min(steps, 50))
        guidance = float(os.environ.get("CATVTON_GUIDANCE", str(self.config.guidance_scale)))

        vram_before = (
            float(torch.cuda.memory_allocated() / (1024 ** 2))
            if (torch and torch.cuda.is_available())
            else 0.0
        )
        pipeline = self.model_loader.load()
        load_info = getattr(self.model_loader, "last_load_info", {}) or {}
        self.logger.info(
            "CATVTON MODEL DEVICE=%s DTYPE=%s LOAD=%s VRAM_BEFORE=%.1fMB "
            "person=%s garment=%s mask=%s steps=%s guidance=%s",
            load_info.get("device"),
            load_info.get("dtype"),
            load_info,
            vram_before,
            prep_person.person_image.size,
            prep_garment.garment_image.size,
            prep_person.agnostic_mask.size if prep_person.agnostic_mask else None,
            steps,
            guidance,
        )

        output_image: Optional[Image.Image] = None
        raw_output_image: Optional[Image.Image] = None
        if pipeline is not None:
            # Final cloth_type + mask geometry immediately before the model call.
            _mask_arr = np.asarray(prep_person.agnostic_mask.convert("L"))
            _mb = np.where(_mask_arr > 127)
            _bbox = (
                [
                    int(_mb[1].min()),
                    int(_mb[0].min()),
                    int(_mb[1].max()),
                    int(_mb[0].max()),
                ]
                if _mb[0].size
                else None
            )
            self.logger.info(
                "CATVTON_FINAL_CLOTH_TYPE=%s mask_source=%s mask_size=%s mask_bbox=%s "
                "mask_area=%.4f steps=%s guidance=%s",
                cloth_type,
                mask_source,
                prep_person.agnostic_mask.size,
                _bbox,
                float((_mask_arr > 127).mean()),
                steps,
                guidance,
            )
            self.logger.info(
                "Executing CatVTON diffusion try-on (steps=%s, mask=%s, cloth=%s)...",
                steps,
                mask_source,
                cloth_type,
            )
            try:
                if type(pipeline).__name__ == "CatVTONPipeline" or hasattr(pipeline, "auto_attn_ckpt_load"):
                    out_res = pipeline(
                        image=prep_person.person_image,
                        condition_image=prep_garment.garment_image,
                        mask=prep_person.agnostic_mask,
                        height=self.config.height,
                        width=self.config.width,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                    )
                    if isinstance(out_res, Image.Image):
                        output_image = out_res
                    elif isinstance(out_res, (list, tuple)) and out_res:
                        output_image = out_res[0]
                    elif hasattr(out_res, "images"):
                        output_image = out_res.images[0]
                    else:
                        output_image = out_res
                    inference_backend = "catvton_native"
                else:
                    out = pipeline(
                        image=prep_person.person_image,
                        mask_image=prep_person.agnostic_mask,
                        control_image=prep_garment.garment_image,
                        height=self.config.height,
                        width=self.config.width,
                    )
                    output_image = out.images[0]
                    inference_backend = "diffusers_inpaint"
                raw_output_image = output_image.copy() if isinstance(output_image, Image.Image) else output_image
            except Exception as exc:
                self.logger.error("CatVTON diffusion error: %s", exc)
                failed_stage = "diffusion"
                error_type = type(exc).__name__
                if not allow_fallback:
                    raise RuntimeError(f"Real CatVTON try-on failed: {exc}") from exc
                output_image = self._blend_preview(prep_person.person_image, prep_garment.garment_image)
                was_fallback_used = True
                inference_backend = "blend_preview"
        else:
            self.logger.info("CatVTON pipeline not loaded; blend fallback only if allowed.")
            failed_stage = "model_load"
            error_type = "CatVTONWeightsMissing"
            if not allow_fallback:
                raise RuntimeError("CatVTON real model execution required but weights missing.")
            output_image = self._blend_preview(prep_person.person_image, prep_garment.garment_image)
            was_fallback_used = True
            inference_backend = "blend_preview"

        assert output_image is not None
        was_real_catvton_used = inference_backend == "catvton_native" and not was_fallback_used

        # No hidden post-composition: final == raw for native CatVTON.
        if was_real_catvton_used and raw_output_image is not None:
            output_image = raw_output_image

        vram_after = (
            float(torch.cuda.memory_allocated() / (1024 ** 2))
            if (torch and torch.cuda.is_available())
            else 0.0
        )

        if debug and debug_stamp:
            self._save_debug_intermediates(
                prep_person,
                prep_garment,
                stamp=debug_stamp,
                original_person=original_person,
                stage="post_inference",
                raw_output=raw_output_image or output_image,
            )

        t1 = time.time()
        inference_time = round(t1 - t0, 2)
        peak_vram_mb = (
            float(torch.cuda.max_memory_allocated() / (1024 ** 2))
            if (torch and torch.cuda.is_available())
            else 0.0
        )

        # 3. Validate result
        val_result = self.validator.validate(output_image)

        # 4. Save output & metadata
        output_base = Path(self.config.output_root)
        images_dir = output_base / "images"
        metadata_dir = output_base / "metadata"
        images_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        exp_results_dir = Path(self.config.experiments_root) / "tryon_results"
        exp_results_dir.mkdir(parents=True, exist_ok=True)

        item_id = output_filename or f"tryon_{uuid.uuid4().hex[:8]}"
        image_path = images_dir / f"{item_id}.png"
        meta_path = metadata_dir / f"{item_id}.json"
        exp_meta_path = exp_results_dir / f"{item_id}_exp.json"

        # Always PNG — never JPEG recompression on the try-on path.
        output_image.save(image_path, format="PNG")

        if debug and debug_stamp:
            debug_dir = Path(self.config.output_root) / "debug"
            try:
                output_image.save(debug_dir / f"{debug_stamp}_final_composition.png", format="PNG")
            except Exception:
                pass

        metadata_record = {
            "model": "CatVTON",
            "status": "completed_with_fallback" if was_fallback_used else "completed",
            "garment_type": garment_input.garment_type,
            "cloth_type": cloth_type,
            "person_filename": person_filename,
            "garment_filename": garment_filename,
            "saved_image_path": str(image_path),
            "inference_time_s": inference_time,
            "peak_vram_mb": peak_vram_mb,
            "vram_before_mb": round(vram_before, 2),
            "vram_after_mb": round(vram_after, 2),
            "resolution": [output_image.width, output_image.height],
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "scheduler": "DDIMScheduler",
            "attn_ckpt_version": load_info.get("attn_ckpt_version"),
            "model_device": load_info.get("device"),
            "model_dtype": load_info.get("dtype"),
            "person_input_size": list(prep_person.person_image.size),
            "garment_input_size": list(prep_garment.garment_image.size),
            "mask_input_size": list(prep_person.agnostic_mask.size)
            if prep_person.agnostic_mask
            else None,
            "validation": val_result,
            "mask_source": mask_source,
            "mask_strategy": mask_strategy,
            "mask_attempts": mask_attempts,
            "mask_stats": mask_stats,
            "mask_validation": mask_reason,
            "was_fallback_used": was_fallback_used,
            "was_real_catvton_used": was_real_catvton_used,
            "inference_backend": inference_backend,
            "failed_stage": failed_stage,
            "error_type": error_type,
            "garment_mask_computed": prep_garment.garment_mask is not None,
            "raw_equals_final": bool(was_real_catvton_used),
            "limitations": [
                "AutoMasker is preferred when DensePose/SCHP/detectron2 are available "
                "(CATVTON_USE_AUTOMASKER=auto|true); GrabCut remains the fallback.",
                "GrabCut cloth-region masks are approximate vs DensePose+SCHP agnostic masks.",
                "Garment uses resize_and_padding; person/mask use resize_and_crop (upstream).",
                "vitonhd-16k-512 + 384x512 used on RTX 3050 6GB (not mix 768x1024 / 50 steps).",
                "fit_preference / background_action are accepted by the API but unused.",
            ],
        }

        import json

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata_record, f, indent=2)

        with open(exp_meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata_record, f, indent=2)

        # Dry-run / CI may still return a completed blend; production must not treat it as success.
        result_status = "completed_with_fallback" if was_fallback_used else "completed"

        return TryOnResult(
            output_image=output_image,
            image_path=str(image_path),
            metadata_path=str(meta_path),
            status=result_status,
            validation=val_result,
        )

    def _save_debug_intermediates(
        self,
        prep_person: PersonConditioningInput,
        prep_garment: GarmentConditioningInput,
        *,
        stamp: str,
        original_person: Optional[Image.Image] = None,
        original_garment: Optional[Image.Image] = None,
        stage: str = "conditioning",
        raw_output: Optional[Image.Image] = None,
        mask_stats: Optional[dict] = None,
    ) -> None:
        """Save intermediates under outputs/virtual_tryon/debug/ when CATVTON_DEBUG=1."""
        debug_dir = Path(self.config.output_root) / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        try:
            if stage == "conditioning":
                from src.features.virtual_tryon.catvton_conditioning import (
                    prepare_catvton_person_conditioning,
                    analyze_mask,
                )
                import json

                if original_person is not None:
                    original_person.convert("RGB").save(debug_dir / f"{stamp}_original_person.png")
                if original_garment is not None:
                    original_garment.convert("RGB").save(debug_dir / f"{stamp}_garment_original.png")

                prep_person.person_image.save(debug_dir / f"{stamp}_preprocessed_person.png")
                prep_person.person_image.save(debug_dir / f"{stamp}_catvton_person_conditioning.png")
                if prep_person.agnostic_mask is not None:
                    prep_person.agnostic_mask.save(debug_dir / f"{stamp}_person_mask.png")
                    prep_person.agnostic_mask.save(debug_dir / f"{stamp}_catvton_mask_conditioning.png")
                    masked = prepare_catvton_person_conditioning(
                        prep_person.person_image, prep_person.agnostic_mask
                    )
                    masked.save(debug_dir / f"{stamp}_masked_person.png")
                    stats = mask_stats or analyze_mask(prep_person.agnostic_mask)
                    (debug_dir / f"{stamp}_mask_stats.json").write_text(
                        json.dumps(stats, indent=2), encoding="utf-8"
                    )

                prep_garment.garment_image.save(debug_dir / f"{stamp}_garment_input.png")
                prep_garment.garment_image.save(debug_dir / f"{stamp}_catvton_garment_conditioning.png")
                prep_garment.garment_image.save(debug_dir / f"{stamp}_garment_conditioning.png")
                if prep_garment.garment_mask is not None:
                    prep_garment.garment_mask.save(debug_dir / f"{stamp}_garment_mask.png")
                    prep_garment.garment_mask.save(debug_dir / f"{stamp}_garment_segmented.png")

                (debug_dir / f"{stamp}_conditioning_note.txt").write_text(
                    "CatVTON native inputs (upstream semantics):\n"
                    "  person = RGB full person (resize_and_crop)\n"
                    "  garment = RGB cloth (resize_and_padding)\n"
                    "  mask white = REPLACE clothing; black = KEEP person\n"
                    "  pipeline builds masked_image = person * (mask < 0.5)\n"
                    f"person_size={prep_person.person_image.size}\n"
                    f"garment_size={prep_garment.garment_image.size}\n"
                    f"mask_size={getattr(prep_person.agnostic_mask, 'size', None)}\n"
                    f"mask_source={getattr(self.person_conditioner, 'last_mask_source', 'unknown')}\n"
                    f"mask_stats={mask_stats}\n"
                    f"target={self.config.width}x{self.config.height}\n",
                    encoding="utf-8",
                )
            if stage == "post_inference" and raw_output is not None:
                raw_output.save(debug_dir / f"{stamp}_raw_catvton_output.png")
                raw_output.save(debug_dir / f"{stamp}_final_output.png")
            self.logger.info("CatVTON debug intermediates (%s) saved under %s", stage, debug_dir)
        except Exception as exc:
            self.logger.warning("CatVTON debug dump failed: %s", exc)
