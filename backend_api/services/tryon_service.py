import asyncio
import os
from pathlib import Path

from backend_api.services.job_manager import job_manager
from backend_api.services.generation_service import model_manager


async def process_tryon(
    job_id: str,
    garment_image_path: str,
    person_image_path: str,
    fit_preference: str,
    background_action: str,
    garment_type: str = "garment",
):
    """
    Background worker for CatVTON try-on.

    Unloads FLUX (and other residents) via ModelManager before loading CatVTON
    so the RTX 3050 6GB budget is not shared with garment generation weights.

    Production fails closed: blend_preview is never marked as a successful try-on.
    fit_preference / background_action remain API-compatible but unused by the pipeline.
    """
    try:
        job_manager.update_job(
            job_id,
            status="processing",
            progress=10,
            current_step="Loading CatVTON pipeline...",
        )

        from src.features.virtual_tryon.tryon_pipeline import (
            TryOnConfig,
            VirtualTryOnPipeline,
        )
        from src.features.virtual_tryon.models import (
            GarmentConditioningInput,
            PersonConditioningInput,
        )
        from PIL import Image
        from backend_api.config.settings import settings

        # Free FLUX / Qwen VRAM before CatVTON residency.
        model_manager.switch_to("catvton")
        loader = model_manager.catvton_manager.loader

        job_manager.update_job(job_id, progress=30, current_step="Preparing conditions...")

        # Production: no silent blend. Override with CATVTON_ALLOW_FALLBACK=1 only for labs.
        allow_fb = os.environ.get("CATVTON_ALLOW_FALLBACK", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        steps = int(os.environ.get("CATVTON_STEPS", "30"))
        # Prefer AutoMasker when DensePose/SCHP are present; GrabCut remains fallback.
        os.environ.setdefault("CATVTON_USE_AUTOMASKER", "auto")
        os.environ.setdefault("CATVTON_MASK_STRATEGY", "auto")
        tryon_root = str(settings.OUTPUT_DIR / "virtual_tryon")
        config = TryOnConfig(
            height=512,
            width=384,
            allow_fallback=allow_fb,
            num_inference_steps=steps,
            guidance_scale=2.5,
            attn_ckpt_version="vitonhd",
            output_root=tryon_root,
        )
        pipeline = VirtualTryOnPipeline(config=config, model_loader=loader)

        loop = asyncio.get_running_loop()
        resolved_garment_type = (garment_type or "garment").strip() or "garment"

        def run_sync_tryon():
            g_img = Image.open(garment_image_path).convert("RGB")
            p_img = Image.open(person_image_path).convert("RGB")

            garment_in = GarmentConditioningInput(
                garment_image=g_img,
                garment_type=resolved_garment_type,
            )
            person_in = PersonConditioningInput(person_image=p_img)

            _ = (fit_preference, background_action)
            return pipeline.run(
                person_in,
                garment_in,
                output_filename=f"tryon_{job_id}",
            )

        job_manager.update_job(
            job_id,
            progress=50,
            current_step="Mapping garment to person (CatVTON)...",
        )
        result = await loop.run_in_executor(None, run_sync_tryon)

        img_path = (
            getattr(result, "image_path", None)
            or getattr(result, "output_path", None)
            or (result.get("image_path") if isinstance(result, dict) else None)
        )
        result_url_val = None
        if img_path:
            p = Path(img_path)
            try:
                import shutil

                rel_path = p.relative_to(settings.OUTPUT_DIR)
                result_url_val = f"/outputs/{rel_path.as_posix()}"
            except ValueError:
                import shutil

                filename = p.name
                dest = settings.OUTPUT_DIR / filename
                if p.exists() and not dest.exists():
                    shutil.copy2(p, dest)
                result_url_val = f"/outputs/{filename}"

        meta_path = getattr(result, "metadata_path", None)
        extra_meta = {}
        if meta_path:
            try:
                import json

                with open(meta_path, encoding="utf-8") as f:
                    extra_meta = json.load(f)
            except Exception:
                extra_meta = {}

        was_fallback = bool(extra_meta.get("was_fallback_used")) or (
            getattr(result, "status", "") == "completed_with_fallback"
        )
        was_real = bool(extra_meta.get("was_real_catvton_used"))
        mask_source = extra_meta.get("mask_source")
        inference_backend = extra_meta.get("inference_backend")

        from src.common.models.device_manager import DeviceManager

        cat_device = (
            extra_meta.get("model_device")
            or DeviceManager.resolve_role_device("catvton", "cuda:1")
        )

        honesty = {
            "mask_source": mask_source,
            "was_fallback_used": was_fallback,
            "was_real_catvton_used": was_real,
            "inference_backend": inference_backend,
            "failed_stage": extra_meta.get("failed_stage"),
            "error_type": extra_meta.get("error_type"),
            "peak_vram_mb": extra_meta.get("peak_vram_mb"),
            "inference_time_s": extra_meta.get("inference_time_s"),
            "resolution": extra_meta.get("resolution"),
            "num_inference_steps": extra_meta.get("num_inference_steps"),
            "garment_type": resolved_garment_type,
            "cloth_type": extra_meta.get("cloth_type"),
            "fit_preference": fit_preference,
            "background_action": background_action,
            "controls_applied": False,
            "device": cat_device,
            "gpu": cat_device,
            "model_device": extra_meta.get("model_device") or cat_device,
        }

        # Never present blend / fallback as a successful production try-on.
        if was_fallback or not was_real:
            job_manager.update_job(
                job_id,
                status="failed",
                progress=100,
                current_step="Try-on fallback rejected",
                error=(
                    "Virtual try-on did not complete with real CatVTON inference "
                    f"(backend={inference_backend}, mask={mask_source}, "
                    f"was_fallback={was_fallback})."
                ),
                error_type="CATVTON_FALLBACK",
                failed_stage=extra_meta.get("failed_stage") or "fallback",
                result_url=result_url_val,
                metadata=honesty,
            )
            return

        if mask_source == "box_fallback":
            job_manager.update_job(
                job_id,
                status="failed",
                progress=100,
                current_step="Insufficient clothing mask",
                error=(
                    "Try-on refused: only a rectangular box mask was available. "
                    "This produces misleading overlays, not a valid CatVTON fit."
                ),
                error_type="CATVTON_MASK_QUALITY",
                failed_stage="masking",
                result_url=result_url_val,
                metadata=honesty,
            )
            return

        warn_mask = mask_source == "grabcut"
        job_manager.update_job(
            job_id,
            status="completed",
            progress=100,
            current_step=(
                "Try-on successful (GrabCut mask — AutoMasker unavailable)"
                if warn_mask
                else "Try-on successful (Real CatVTON)"
            ),
            result_url=result_url_val,
            metadata={
                **honesty,
                "mask_quality_warning": warn_mask,
            },
        )

    except Exception as e:
        err_type = type(e).__name__
        msg = str(e)
        if "box_fallback" in msg or "REQUIRE_REAL" in msg:
            err_type = "CATVTON_MASK_QUALITY"
            failed = "masking"
        elif "weights missing" in msg.lower() or "required but" in msg.lower():
            err_type = "CATVTON_MODEL_MISSING"
            failed = "model_load"
        else:
            failed = "tryon"
        job_manager.update_job(
            job_id,
            status="failed",
            error=msg,
            error_type=err_type,
            failed_stage=failed,
            current_step="Failed",
        )
