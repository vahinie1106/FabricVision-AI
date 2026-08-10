import asyncio
import os
import shutil
from pathlib import Path

from PIL import Image

from backend_api.config.settings import settings
from backend_api.services.job_manager import job_manager
from src.common.models.model_manager import ModelManager

# Shared VRAM-aware model orchestrator (same instance used by try-on / semantic services)
model_manager = ModelManager()


async def process_generation(
    job_id: str,
    fabric_image_path: str,
    garment_type: str,
    fit: str,
    style: str,
    gender: str = "women",
    season: str = "summer",
    occasion: str = "casual",
    fabric: str = "cotton",
    material: str = "cotton",
    texture: str = "smooth",
    color: str = "white",
    sleeve: str = "short",
    neckline: str = "round",
    generation_mode: str = "standard",
):
    """
    Background worker for FLUX.1-Kontext fabric→garment generation.

    Loads Kontext once via ModelManager, then reuses that loader for inference.
    Progress steps stay alive for long RTX 3050 jobs so frontend polling stays valid.
    """
    try:
        from src.features.custom_generator.pipeline.garment_generation_pipeline import (
            GarmentGenerationConfig,
            GarmentGenerationPipeline,
            normalize_generation_mode,
        )

        mode = generation_mode or "standard"
        mode_key = normalize_generation_mode(mode)

        already_loaded = (
            model_manager.flux_manager.loader is not None
            and getattr(model_manager.flux_manager.loader, "pipeline", None) is not None
            and model_manager.active_model == "flux"
        )

        if already_loaded:
            job_manager.update_job(
                job_id,
                status="processing",
                progress=10,
                current_step="Reusing loaded FLUX.1-Kontext pipeline",
            )
            # Previous jobs may have left transformer/VAE on GPU (~6GB).
            try:
                model_manager.flux_manager.recover_after_oom()
            except Exception:
                pass
        else:
            job_manager.update_job(
                job_id,
                status="processing",
                progress=10,
                current_step="Loading model",
            )

        exec_mode = os.environ.get("FLUX_EXECUTION_MODE", "local").strip().lower()
        if exec_mode == "remote" and os.environ.get("FLUX_REMOTE_URL"):
            job_manager.update_job(
                job_id,
                status="processing",
                progress=10,
                current_step="Connecting to Remote GPU FLUX cluster",
            )
            from src.features.custom_generator.inference.remote_flux_provider import RemoteFluxProvider

            remote_provider = RemoteFluxProvider()
            loader = None
        else:
            model_manager.flux_manager.allow_fallback = False
            model_manager.flux_manager.model_path = settings.BASE_DIR / "models" / "flux-kontext"
            model_manager.switch_to("flux")

            loader = model_manager.flux_manager.loader
            if loader is None or getattr(loader, "pipeline", None) is None:
                raise RuntimeError(
                    "FLUX.1-Kontext failed to load. Download weights into models/flux-kontext "
                    "or check GPU VRAM availability."
                )

            if already_loaded or getattr(loader, "_reuse_count", 0) > 0:
                job_manager.update_job(
                    job_id,
                    progress=15,
                    current_step="Reusing loaded FLUX.1-Kontext pipeline",
                )

        config = GarmentGenerationConfig(
            config_dir=str(settings.BASE_DIR / "configs"),
            config_path=str(
                settings.BASE_DIR / "configs" / "custom_generator" / "flux_config.yaml"
            ),
            output_root=str(settings.OUTPUT_DIR / "generated_garments"),
            experiments_root=str(settings.BASE_DIR / "experiments"),
            generation_mode=mode,
            allow_fallback=False,
        )

        pipeline = GarmentGenerationPipeline(config=config, model_loader=loader)

        def on_progress(step: str, pct: int) -> None:
            # Thread-safe enough for in-memory job store; keeps long jobs "alive"
            job_manager.update_job(
                job_id,
                status="processing",
                progress=max(10, min(99, int(pct))),
                current_step=step,
            )

        loop = asyncio.get_running_loop()

        def run_sync_generation():
            # Single disk read — keep PIL in memory for appearance + conditioning
            ref_img = Image.open(fabric_image_path).convert("RGB")
            fabric_metadata = {
                "material": material.lower(),
                "fabric": fabric.lower(),
                "texture": texture.lower(),
                "dominant_colors": (
                    []
                    if color.lower().replace(" ", "_") in ("match_fabric", "match-fabric", "")
                    else [color.lower()]
                ),
                "color": color.lower(),
                "style": style.lower(),
                "occasion": occasion.lower(),
                "season": season.lower(),
                "fit": fit.lower(),
            }
            user_customization = {
                "gender": gender.lower(),
                "garment_type": garment_type.lower().replace(" ", "_"),
                "material": material.lower(),
                "fabric": fabric.lower(),
                "texture": texture.lower(),
                "neckline": neckline.lower().replace(" ", "_"),
                "sleeve": sleeve.lower().replace(" ", "_"),
                "occasion": occasion.lower(),
                "season": season.lower(),
                "fit": fit.lower(),
                "style": style.lower(),
            }
            # Only pass UI color when user explicitly wants a recolor (not Match Fabric)
            color_key = color.lower().replace(" ", "_")
            if color_key not in ("match_fabric", "match-fabric", ""):
                user_customization["color"] = color.lower()
                user_customization["force_recolor"] = True
            else:
                user_customization["force_recolor"] = False
            return pipeline.run(
                fabric_metadata=fabric_metadata,
                user_customization=user_customization,
                reference_image=ref_img,
                progress_callback=on_progress,
            )

        job_manager.update_job(
            job_id,
            progress=20,
            current_step="Preparing fabric",
        )

        result = await loop.run_in_executor(None, run_sync_generation)

        img_path = result.get("image_path") or result.get("output_path")
        if not img_path:
            raise RuntimeError("Generation completed but no image path was returned.")

        p = Path(img_path)
        if not p.exists():
            raise RuntimeError(f"Generated image missing on disk: {img_path}")

        try:
            rel_path = p.resolve().relative_to(settings.OUTPUT_DIR.resolve())
            result_url_val = f"/outputs/{rel_path.as_posix()}"
        except ValueError:
            filename = p.name
            dest = settings.OUTPUT_DIR / filename
            if p.exists() and not dest.exists():
                shutil.copy2(p, dest)
            result_url_val = f"/outputs/{filename}"

        stats = getattr(pipeline.inference_engine, "last_execution_stats", {}) or {}
        if stats.get("was_fallback_used") or not stats.get("was_real_flux_used", True):
            raise RuntimeError(
                "FLUX.1-Kontext did not produce a real garment image (fallback detected)."
            )

        meta_result = result.get("metadata") or {}
        prompt_stats = meta_result.get("prompt_stats") or stats.get("prompt_stats") or {}

        job_manager.update_job(
            job_id,
            status="completed",
            progress=100,
            current_step="Completed",
            result_url=result_url_val,
            metadata={
                "category": garment_type,
                "fabric": fabric.capitalize(),
                "styleAffinity": style,
                "confidenceScore": 0.95,
                "model": "FLUX.1-Kontext",
                "generation_mode": pipeline.config.generation_mode,
                "mode_key": mode_key,
                "was_real_flux_used": bool(stats.get("was_real_flux_used", True)),
                "model_reused": bool(stats.get("model_reused", already_loaded)),
                "has_image": True,
                "offload_strategy": getattr(loader, "_offload_strategy", None),
                "attention_backend": getattr(loader, "_attention_backend", None),
                "torch_compile": getattr(loader, "_torch_compile_enabled", False),
                "bnb_4bit": getattr(loader, "_used_bnb_4bit", None),
                "generation_time_s": stats.get("generation_time_s"),
                "peak_vram_mb": stats.get("peak_vram_mb"),
                "height": pipeline.config.height,
                "width": pipeline.config.width,
                "num_inference_steps": pipeline.config.num_inference_steps,
                "guidance_scale": pipeline.config.guidance_scale,
                "prompt_token_count": prompt_stats.get("token_count"),
                "prompt_compacted": prompt_stats.get("prompt_compacted"),
                "prompt_truncated": prompt_stats.get("truncated"),
                "image_path": str(p),
                "positive_prompt": meta_result.get("positive_prompt"),
                "negative_prompt": meta_result.get("negative_prompt"),
            },
        )

    except Exception as e:
        # Recover GPU so a subsequent Generate is not blocked by leftover residency.
        try:
            if hasattr(model_manager, "flux_manager") and hasattr(
                model_manager.flux_manager, "recover_after_oom"
            ):
                model_manager.flux_manager.recover_after_oom()
            else:
                model_manager.clear_vram()
        except Exception:
            pass
        from backend_api.services.generation_errors import classify_generation_error

        error_type, failed_stage, log_message = classify_generation_error(e)
        import logging

        logging.getLogger("fabricvision.api.generation").exception(
            "Generation job %s failed type=%s stage=%s: %s",
            job_id,
            error_type,
            failed_stage,
            log_message,
        )
        job_manager.update_job(
            job_id,
            status="failed",
            error=log_message,
            error_type=error_type,
            failed_stage=failed_stage,
            current_step=f"Failed ({failed_stage})",
            metadata={
                "error_type": error_type,
                "failed_stage": failed_stage,
                "error_class": type(e).__name__,
            },
        )
