import asyncio
import logging
import os
import shutil
import threading
import time
from pathlib import Path

from PIL import Image

from backend_api.config.settings import settings
from backend_api.services.job_manager import job_manager
from src.common.models.model_manager import ModelManager

# Shared VRAM-aware model orchestrator (same instance used by try-on / semantic services)
model_manager = ModelManager()
logger = logging.getLogger("fabricvision.api.generation")

# Single-flight GPU generation — concurrent jobs on one T4 cause immediate OOM.
_generation_lock = threading.Lock()


def _stage_log(label: str, t0: float, *, end: bool = False, extra: str = "") -> float:
    """Print timestamped stage markers with elapsed seconds since t0 / segment start."""
    now = time.perf_counter()
    if end:
        msg = f"[GENERATION] {label} END t={now:.2f} elapsed={now - t0:.2f}s"
    else:
        msg = f"[GENERATION] {label} START t={now:.2f}"
    if extra:
        msg = f"{msg} {extra}"
    logger.info(msg)
    print(msg, flush=True)
    return now


def _process_generation_sync(
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
) -> None:
    """
    Synchronous FLUX garment worker.

    MUST run off the asyncio event loop (thread executor). Blocking model load /
    inference on the loop freezes GET /status polling and leaves the UI stuck at 5%.
    """
    t_job = time.perf_counter()
    _stage_log("GENERATION START", t_job, extra=f"job_id={job_id}")
    loader = None
    acquired = _generation_lock.acquire(blocking=False)
    if not acquired:
        job_manager.update_job(
            job_id,
            status="processing",
            progress=5,
            current_step="Waiting for GPU (another generation is running)",
        )
        _stage_log("GPU LOCK WAIT", t_job, extra=f"job_id={job_id}")
        _generation_lock.acquire(blocking=True)
        acquired = True
    try:
        from src.features.custom_generator.inference.flux_vram_policy import log_vram

        log_vram("before generation job")
    except Exception:
        pass
    try:
        from src.features.custom_generator.pipeline.garment_generation_pipeline import (
            GarmentGenerationConfig,
            GarmentGenerationPipeline,
            normalize_generation_mode,
        )

        mode = generation_mode or "standard"
        mode_key = normalize_generation_mode(mode)

        t_upload = _stage_log("UPLOAD/PARSE INPUT", t_job)
        already_loaded = (
            model_manager.flux_manager.loader is not None
            and getattr(model_manager.flux_manager.loader, "pipeline", None) is not None
            and model_manager.active_model == "flux"
        )
        _stage_log(
            "UPLOAD/PARSE INPUT",
            t_upload,
            end=True,
            extra=f"fabric={fabric_image_path!s} already_loaded={already_loaded}",
        )

        if already_loaded:
            print(
                f"[FLUX GENERATE] pid={os.getpid()} waiting_for_warmup=false "
                f"warmup_state=ready model_reused=true",
                flush=True,
            )
            job_manager.update_job(
                job_id,
                status="processing",
                progress=12,
                current_step="FLUX READY (in-memory hit) — skipping reload",
            )
            try:
                model_manager.flux_manager.recover_after_oom()
            except Exception:
                pass
        else:
            # Parent-process prefetch does NOT populate this process. Wait for
            # API-process warmup (or let Generate become the single loader).
            job_manager.update_job(
                job_id,
                status="processing",
                progress=8,
                current_step="Checking API-process FLUX residency",
            )
            try:
                from backend_api.services.flux_warmup import (
                    get_warmup_status,
                    wait_until_flux_ready,
                    warm_flux_in_api_process,
                )

                warm_status = get_warmup_status()
                print(
                    f"[FLUX GENERATE] pid={os.getpid()} waiting_for_warmup=true "
                    f"warmup_state={warm_status.get('state')} model_reused=false",
                    flush=True,
                )
                print(
                    f"[FLUX LOAD TRACE] process_id={os.getpid()} request_id={job_id} "
                    f"loader_instance={id(getattr(model_manager.flux_manager, 'loader', None))} "
                    f"cache_state=pending_api_load pipeline_exists=false "
                    f"pipeline_load_start model_reused=false",
                    flush=True,
                )
                # Kick warmup only when idle (not when already failed — surface that).
                _ws = str(warm_status.get("state") or warm_status.get("state_raw") or "").upper()
                if _ws in ("IDLE",) and os.environ.get(
                    "FLUX_WARMUP_ON_STARTUP", "true"
                ).strip().lower() not in ("0", "false", "no", "off"):
                    threading.Thread(
                        target=lambda: warm_flux_in_api_process(),
                        name="flux-warmup-kick",
                        daemon=True,
                    ).start()

                def _wait_progress(step: str, pct: int) -> None:
                    job_manager.update_job(
                        job_id,
                        status="processing",
                        progress=max(5, min(17, int(pct))),
                        current_step=step,
                    )

                warm_status = wait_until_flux_ready(
                    timeout_s=float(os.environ.get("FLUX_WARMUP_WAIT_S", "900")),
                    progress_callback=_wait_progress,
                )
                already_loaded = (
                    model_manager.flux_manager.loader is not None
                    and getattr(model_manager.flux_manager.loader, "pipeline", None)
                    is not None
                    and model_manager.active_model == "flux"
                )
                print(
                    f"[FLUX GENERATE] pid={os.getpid()} waiting_for_warmup=false "
                    f"warmup_state={warm_status.get('state')} "
                    f"model_reused={already_loaded}",
                    flush=True,
                )
                _ws2 = str(warm_status.get("state") or warm_status.get("state_raw") or "").upper()
                if _ws2 == "FAILED" and not already_loaded:
                    err = warm_status.get("error") or "unknown warmup failure"
                    # Surface the failure in the live step, clear sticky FAILED, then
                    # allow Generate to become the sole on-demand loader (lock-serialized).
                    # This avoids a forever-stuck 14% wait on a dead warmup, and avoids
                    # leaving the job failed without a retry path.
                    job_manager.update_job(
                        job_id,
                        status="processing",
                        progress=8,
                        current_step=(
                            f"API warmup failed ({err}); loading FLUX on demand"
                        ),
                    )
                    print(
                        f"[FLUX GENERATE] pid={os.getpid()} waiting_for_warmup=false "
                        f"warmup_state=failed model_reused=false "
                        f"action=on_demand_load error={err}",
                        flush=True,
                    )
                    from backend_api.services.flux_warmup import reset_warmup_state

                    reset_warmup_state()
                elif already_loaded:
                    job_manager.update_job(
                        job_id,
                        status="processing",
                        progress=12,
                        current_step="FLUX READY (API warmup) — skipping reload",
                    )
                else:
                    # Warmup skipped/disabled — Generate becomes the sole loader.
                    job_manager.update_job(
                        job_id,
                        status="processing",
                        progress=8,
                        current_step="Initializing FLUX (cache check / dependencies)",
                    )
            except Exception as wait_exc:
                print(
                    f"[FLUX GENERATE] pid={os.getpid()} waiting_for_warmup=error "
                    f"warmup_state=unknown model_reused=false "
                    f"error={type(wait_exc).__name__}: {wait_exc}",
                    flush=True,
                )
                job_manager.update_job(
                    job_id,
                    status="processing",
                    progress=8,
                    current_step="Initializing FLUX (cache check / dependencies)",
                )

        exec_mode = os.environ.get("FLUX_EXECUTION_MODE", "local").strip().lower()
        t_model = _stage_log("MODEL LOAD", t_job)

        def on_progress(step: str, pct: int) -> None:
            job_manager.update_job(
                job_id,
                status="processing",
                progress=max(5, min(99, int(pct))),
                current_step=step,
            )

        if exec_mode == "remote" and os.environ.get("FLUX_REMOTE_URL"):
            job_manager.update_job(
                job_id,
                status="processing",
                progress=10,
                current_step="Connecting to Remote GPU FLUX cluster",
            )
            from src.features.custom_generator.inference.remote_flux_provider import (
                RemoteFluxProvider,
            )

            RemoteFluxProvider()
            loader = None
            _stage_log("MODEL LOAD", t_model, end=True, extra="remote=True")
        else:
            hf_id = os.environ.get("FLUX_KONTEXT_MODEL_ID", "").strip() or None
            if not hf_id:
                try:
                    from src.common.utils.utils import load_yaml_config

                    flux_yaml = (
                        load_yaml_config(
                            settings.BASE_DIR
                            / "configs"
                            / "custom_generator"
                            / "flux_config.yaml"
                        )
                        or {}
                    )
                    hf_id = (flux_yaml.get("hf_model_id") or "").strip() or None
                except Exception:
                    hf_id = None

            model_manager.flux_manager.allow_fallback = False
            model_manager.flux_manager.model_path = (
                settings.BASE_DIR / "models" / "flux-kontext"
            )
            model_manager.flux_manager.hf_model_id = hf_id
            model_manager.flux_manager.progress_callback = on_progress
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

            print(
                f"[FLUX LOAD TRACE] process_id={os.getpid()} request_id={job_id} "
                f"loader_instance={id(getattr(model_manager.flux_manager, 'loader', None))} "
                f"cache_state=switch_to_flux "
                f"pipeline_exists={bool(getattr(getattr(model_manager.flux_manager, 'loader', None), 'pipeline', None))} "
                f"pipeline_load_start model_reused={already_loaded}",
                flush=True,
            )
            try:
                model_manager.switch_to("flux")
            except Exception as load_exc:
                raise RuntimeError(
                    str(load_exc) or "FLUX.1-Kontext failed to load"
                ) from load_exc

            loader = model_manager.flux_manager.loader
            reused = already_loaded or bool(getattr(loader, "_reuse_count", 0))
            print(
                f"[FLUX LOAD TRACE] process_id={os.getpid()} request_id={job_id} "
                f"loader_instance={id(loader) if loader else None} "
                f"cache_state={getattr(loader, '_cache_status', None)} "
                f"pipeline_exists={bool(getattr(loader, 'pipeline', None))} "
                f"pipeline_load_end model_reused={reused}",
                flush=True,
            )
            print(
                f"[FLUX GENERATE] pid={os.getpid()} waiting_for_warmup=false "
                f"warmup_state=done model_reused={reused}",
                flush=True,
            )
            if loader is None or getattr(loader, "pipeline", None) is None:
                raise RuntimeError(
                    "MODEL_LOAD_ERROR: FLUX.1-Kontext pipeline is not initialized after load. "
                    "Weights may be missing from models/flux-kontext (gitignored) — "
                    "run python scripts/download_flux_kontext.py or ensure Kaggle can reach "
                    "Hugging Face for eramth/flux-kontext-4bit."
                )

            runtime = {}
            if hasattr(loader, "get_runtime_info"):
                runtime = loader.get_runtime_info() or {}
            cache = runtime.get("cache_status", "unknown")
            init_s = runtime.get("init_time_s")
            dl_s = runtime.get("download_time_s")
            _stage_log(
                "MODEL LOAD",
                t_model,
                end=True,
                extra=(
                    f"reused={already_loaded} cache={cache} "
                    f"init_s={init_s} download_s={dl_s} "
                    f"offload={runtime.get('offload_strategy')} "
                    f"vram_mb={runtime.get('gpu_vram_mb')}"
                ),
            )

            if already_loaded or getattr(loader, "_reuse_count", 0) > 0:
                job_manager.update_job(
                    job_id,
                    progress=15,
                    current_step="FLUX READY (in-memory hit) — generating",
                )
            else:
                job_manager.update_job(
                    job_id,
                    progress=18,
                    current_step="FLUX READY (in-memory) — starting fabric prep",
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

        job_manager.update_job(
            job_id,
            progress=20,
            current_step="Preparing fabric",
        )

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
        color_key = color.lower().replace(" ", "_")
        if color_key not in ("match_fabric", "match-fabric", ""):
            user_customization["color"] = color.lower()
            user_customization["force_recolor"] = True
        else:
            user_customization["force_recolor"] = False

        result = pipeline.run(
            fabric_metadata=fabric_metadata,
            user_customization=user_customization,
            reference_image=ref_img,
            progress_callback=on_progress,
        )

        t_save = _stage_log("IMAGE SAVE", t_job)
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
        _stage_log("IMAGE SAVE", t_save, end=True, extra=f"url={result_url_val}")

        stats = getattr(pipeline.inference_engine, "last_execution_stats", {}) or {}
        if stats.get("was_fallback_used") or not stats.get("was_real_flux_used", True):
            raise RuntimeError(
                "FLUX.1-Kontext did not produce a real garment image (fallback detected)."
            )

        t_meta = _stage_log("METADATA", t_job)
        meta_result = result.get("metadata") or {}
        prompt_stats = meta_result.get("prompt_stats") or stats.get("prompt_stats") or {}

        metadata = {
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
                "cache_status": getattr(loader, "_cache_status", None),
                "model_init_time_s": getattr(loader, "_init_time_s", None),
                "model_download_time_s": getattr(loader, "_download_time_s", None),
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
            "result_url": result_url_val,
        }
        _stage_log(
            "METADATA",
            t_meta,
            end=True,
            extra=f"keys={sorted(metadata.keys())}",
        )

        job_manager.update_job(
            job_id,
            status="completed",
            progress=100,
            current_step="Completed",
            result_url=result_url_val,
            metadata=metadata,
        )
        _stage_log("JOB COMPLETE", t_job, end=True, extra=f"job_id={job_id}")

    except Exception as e:
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
        logger.exception(
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
    finally:
        try:
            from src.features.custom_generator.inference.flux_vram_policy import log_vram

            log_vram("after generation job cleanup")
        except Exception:
            pass
        if acquired:
            _generation_lock.release()


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

    Offloads all blocking work (model load + inference) to a thread so
    GET /api/v1/status/{job_id} stays responsive during multi-minute jobs.
    """
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        lambda: _process_generation_sync(
            job_id,
            fabric_image_path,
            garment_type,
            fit,
            style,
            gender,
            season,
            occasion,
            fabric,
            material,
            texture,
            color,
            sleeve,
            neckline,
            generation_mode,
        ),
    )
