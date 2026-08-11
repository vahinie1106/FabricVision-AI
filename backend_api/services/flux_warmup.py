"""Warm FLUX inside the FastAPI process (not the run_kaggle parent).

Parent-process prefetch only writes weights to disk. The API child still paid a
multi-minute cold ``from_pretrained`` on the first Generate while the UI sat at
8% Loading model. Warming here makes the resident ModelManager ready before
user traffic.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("fabricvision.api.flux_warmup")


def warm_flux_in_api_process() -> Dict[str, Any]:
    """
    Load FLUX into the shared generation ``model_manager`` singleton.

    Returns a small status dict for logs / health metadata.
    Skipped under pytest unless FLUX_WARMUP_ON_STARTUP=force.
    """
    force = os.environ.get("FLUX_WARMUP_ON_STARTUP", "true").strip().lower()
    if force in ("0", "false", "no", "off"):
        logger.info("[FLUX WARMUP] skipped (FLUX_WARMUP_ON_STARTUP=%s)", force)
        return {"skipped": True, "reason": "disabled"}

    if os.environ.get("PYTEST_CURRENT_TEST") and force != "force":
        logger.info("[FLUX WARMUP] skipped under pytest")
        return {"skipped": True, "reason": "pytest"}

    t0 = time.perf_counter()
    print("[FLUX TIMING] WARMUP_START t=%.2f" % t0, flush=True)

    from backend_api.config.settings import settings
    from backend_api.services.generation_service import model_manager
    from src.common.utils.utils import load_yaml_config

    already = (
        model_manager.flux_manager.loader is not None
        and getattr(model_manager.flux_manager.loader, "pipeline", None) is not None
        and model_manager.active_model == "flux"
    )
    if already:
        print("[FLUX TIMING] WARMUP_END already_in_memory=True", flush=True)
        return {"skipped": True, "reason": "already_loaded", "in_memory": True}

    hf_id = os.environ.get("FLUX_KONTEXT_MODEL_ID", "").strip() or None
    if not hf_id:
        try:
            flux_yaml = (
                load_yaml_config(
                    settings.BASE_DIR / "configs" / "custom_generator" / "flux_config.yaml"
                )
                or {}
            )
            hf_id = (flux_yaml.get("hf_model_id") or "").strip() or None
        except Exception:
            hf_id = None

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    model_manager.flux_manager.allow_fallback = False
    model_manager.flux_manager.model_path = settings.BASE_DIR / "models" / "flux-kontext"
    model_manager.flux_manager.hf_model_id = hf_id

    def _cb(step: str, pct: int) -> None:
        print(f"[FLUX WARMUP] {pct}% {step}", flush=True)

    model_manager.flux_manager.progress_callback = _cb
    model_manager.switch_to("flux")
    loader = model_manager.flux_manager.loader
    if loader is None or getattr(loader, "pipeline", None) is None:
        raise RuntimeError("FLUX warmup failed: pipeline not initialized")

    info = loader.get_runtime_info() if hasattr(loader, "get_runtime_info") else {}
    elapsed = round(time.perf_counter() - t0, 2)
    print(
        f"[FLUX TIMING] WARMUP_END t={time.perf_counter():.2f} duration={elapsed}s "
        f"cache={info.get('cache_status')} init_s={info.get('init_time_s')} "
        f"download_s={info.get('download_time_s')} offload={info.get('offload_strategy')} "
        f"in_memory=True",
        flush=True,
    )
    logger.info("[FLUX WARMUP] complete in %.2fs cache=%s", elapsed, info.get("cache_status"))
    return {
        "skipped": False,
        "elapsed_s": elapsed,
        "cache_status": info.get("cache_status"),
        "init_time_s": info.get("init_time_s"),
        "download_time_s": info.get("download_time_s"),
        "offload_strategy": info.get("offload_strategy"),
        "in_memory": True,
        "gpu_name": info.get("gpu_name"),
        "gpu_vram_mb": info.get("gpu_vram_mb"),
    }


def flux_in_memory_ready() -> bool:
    from backend_api.services.generation_service import model_manager

    return (
        model_manager.active_model == "flux"
        and model_manager.flux_manager.loader is not None
        and getattr(model_manager.flux_manager.loader, "pipeline", None) is not None
    )
