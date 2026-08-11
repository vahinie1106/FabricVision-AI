"""Warm FLUX inside the FastAPI process (not the run_kaggle parent).

CRITICAL ARCHITECTURE:
  scripts/run_kaggle.py runs in a PARENT process. Prefetch there only fills the
  on-disk Hugging Face / models/ cache, then unloads. FastAPI is a CHILD process
  (uvicorn Popen) with a separate address space — it does NOT inherit the
  parent's in-memory FluxKontextPipeline.

  Therefore the ONLY in-memory FLUX that Generate can reuse is the one loaded
  by this module inside the API process (shared ``generation_service.model_manager``).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("fabricvision.api.flux_warmup")

ProgressCb = Optional[Callable[[str, int], None]]

_lock = threading.Lock()
_ready = threading.Event()
_state: Dict[str, Any] = {
    "state": "idle",  # idle | loading | ready | failed | skipped
    "pid": None,
    "error": None,
    "result": None,
    "started_at": None,
    "finished_at": None,
}


def reset_warmup_state() -> None:
    """Reset module status (tests / explicit retry after FAILED)."""
    with _lock:
        _state.update(
            {
                "state": "idle",
                "pid": None,
                "error": None,
                "result": None,
                "started_at": None,
                "finished_at": None,
            }
        )
        _ready.clear()


def get_warmup_status() -> Dict[str, Any]:
    """Public status for /api/v1/flux-status and generation waits."""
    with _lock:
        out = dict(_state)
    out["ready"] = _ready.is_set() and out.get("state") == "ready"
    out["in_memory"] = flux_in_memory_ready()
    return out


def flux_in_memory_ready() -> bool:
    try:
        from backend_api.services.generation_service import model_manager

        return (
            model_manager.active_model == "flux"
            and model_manager.flux_manager.loader is not None
            and getattr(model_manager.flux_manager.loader, "pipeline", None) is not None
        )
    except Exception:
        return False


def wait_until_flux_ready(
    *,
    timeout_s: float = 900.0,
    progress_callback: ProgressCb = None,
    poll_s: float = 1.0,
) -> Dict[str, Any]:
    """
    Block until API-process FLUX is in memory (or failed/skipped/timeout).

    Used by Generate so it does not start a second from_pretrained while warmup
    is already loading in this process.
    """
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if flux_in_memory_ready():
            _ready.set()
            with _lock:
                if _state["state"] not in ("ready", "failed", "skipped"):
                    _state["state"] = "ready"
            return get_warmup_status()

        status = get_warmup_status()
        state = status.get("state")
        if state == "failed":
            # Allow Generate to observe failure; caller decides whether to raise or reload.
            return status
        if state == "skipped":
            # Warmup disabled / pytest — caller loads on demand via switch_to.
            return status
        if state == "idle":
            # No warmup claimed yet — do not spin; caller may load (lock-serialized).
            return status

        if progress_callback is not None and state == "loading":
            started = status.get("started_at")
            elapsed = int(time.perf_counter() - started) if started else 0
            try:
                progress_callback(
                    f"Waiting for API-process FLUX warmup ({elapsed}s, pid={status.get('pid')})",
                    9,
                )
            except Exception:
                pass

        if _ready.wait(timeout=poll_s):
            if flux_in_memory_ready():
                return get_warmup_status()
        # continue polling while state == loading

    raise TimeoutError(
        f"FLUX API-process warmup not ready within {timeout_s:.0f}s "
        f"(status={get_warmup_status()})"
    )


def warm_flux_in_api_process() -> Dict[str, Any]:
    """
    Load FLUX into the shared generation ``model_manager`` singleton.

    Returns a small status dict for logs / health metadata.
    Skipped under pytest unless FLUX_WARMUP_ON_STARTUP=force.
    """
    force = os.environ.get("FLUX_WARMUP_ON_STARTUP", "true").strip().lower()
    if force in ("0", "false", "no", "off"):
        with _lock:
            _state.update({"state": "skipped", "result": {"skipped": True, "reason": "disabled"}})
        _ready.set()
        print(f"[FLUX WARMUP] pid={os.getpid()} state=SKIPPED reason=disabled", flush=True)
        logger.info("[FLUX WARMUP] skipped (FLUX_WARMUP_ON_STARTUP=%s)", force)
        return {"skipped": True, "reason": "disabled"}

    if os.environ.get("PYTEST_CURRENT_TEST") and force != "force":
        with _lock:
            _state.update({"state": "skipped", "result": {"skipped": True, "reason": "pytest"}})
        _ready.set()
        print(f"[FLUX WARMUP] pid={os.getpid()} state=SKIPPED reason=pytest", flush=True)
        logger.info("[FLUX WARMUP] skipped under pytest")
        return {"skipped": True, "reason": "pytest"}

    # Single-flight: if another thread is already warming, wait for it.
    with _lock:
        if _state["state"] == "ready" and flux_in_memory_ready():
            print(
                f"[FLUX WARMUP] pid={os.getpid()} state=READY reason=already_loaded",
                flush=True,
            )
            return {"skipped": True, "reason": "already_loaded", "in_memory": True}
        if _state["state"] == "loading":
            already_loading = True
        else:
            already_loading = False
            _state.update(
                {
                    "state": "loading",
                    "pid": os.getpid(),
                    "error": None,
                    "started_at": time.perf_counter(),
                    "finished_at": None,
                }
            )
            _ready.clear()

    if already_loading:
        print(
            f"[FLUX WARMUP] pid={os.getpid()} state=STARTING "
            "(waiting on in-flight warmup)",
            flush=True,
        )
        return wait_until_flux_ready(timeout_s=900.0)

    t0 = time.perf_counter()
    print(f"[FLUX WARMUP] pid={os.getpid()} state=STARTING", flush=True)
    print(
        f"[FLUX LOAD TRACE] process_id={os.getpid()} request_id=warmup "
        f"loader_instance=pending cache_state=pending pipeline_exists=false "
        f"pipeline_load_start model_reused=false",
        flush=True,
    )
    print(
        f"[FLUX TIMING] WARMUP_START t={t0:.2f} pid={os.getpid()} "
        f"(API process — not run_kaggle parent)",
        flush=True,
    )

    try:
        from backend_api.config.settings import settings
        from backend_api.services.generation_service import model_manager
        from src.common.utils.utils import load_yaml_config

        already = flux_in_memory_ready()
        if already:
            loader = model_manager.flux_manager.loader
            print(f"[FLUX WARMUP] pid={os.getpid()} state=READY", flush=True)
            print(
                f"[FLUX LOAD TRACE] process_id={os.getpid()} request_id=warmup "
                f"loader_instance={id(loader)} cache_state=in_memory "
                f"pipeline_exists=true pipeline_load_end model_reused=true",
                flush=True,
            )
            result = {"skipped": True, "reason": "already_loaded", "in_memory": True}
            with _lock:
                _state.update(
                    {
                        "state": "ready",
                        "result": result,
                        "finished_at": time.perf_counter(),
                    }
                )
            _ready.set()
            return result

        hf_id = os.environ.get("FLUX_KONTEXT_MODEL_ID", "").strip() or None
        if not hf_id:
            try:
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

        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        model_manager.flux_manager.allow_fallback = False
        model_manager.flux_manager.model_path = (
            settings.BASE_DIR / "models" / "flux-kontext"
        )
        model_manager.flux_manager.hf_model_id = hf_id

        def _cb(step: str, pct: int) -> None:
            print(f"[FLUX WARMUP] pid={os.getpid()} {pct}% {step}", flush=True)

        model_manager.flux_manager.progress_callback = _cb
        model_manager.switch_to("flux")
        loader = model_manager.flux_manager.loader
        if loader is None or getattr(loader, "pipeline", None) is None:
            raise RuntimeError("FLUX warmup failed: pipeline not initialized")

        info = loader.get_runtime_info() if hasattr(loader, "get_runtime_info") else {}
        elapsed = round(time.perf_counter() - t0, 2)
        print(f"[FLUX WARMUP] pid={os.getpid()} state=READY", flush=True)
        print(
            f"[FLUX TIMING] WARMUP_END t={time.perf_counter():.2f} duration={elapsed}s "
            f"pid={os.getpid()} cache={info.get('cache_status')} "
            f"init_s={info.get('init_time_s')} download_s={info.get('download_time_s')} "
            f"offload={info.get('offload_strategy')} in_memory=True",
            flush=True,
        )
        print(
            f"[FLUX LOAD TRACE] process_id={os.getpid()} request_id=warmup "
            f"loader_instance={id(loader)} cache_state={info.get('cache_status')} "
            f"pipeline_exists=true pipeline_load_end model_reused=false",
            flush=True,
        )
        logger.info(
            "[FLUX WARMUP] complete in %.2fs cache=%s pid=%s",
            elapsed,
            info.get("cache_status"),
            os.getpid(),
        )
        result = {
            "skipped": False,
            "elapsed_s": elapsed,
            "cache_status": info.get("cache_status"),
            "init_time_s": info.get("init_time_s"),
            "download_time_s": info.get("download_time_s"),
            "offload_strategy": info.get("offload_strategy"),
            "in_memory": True,
            "gpu_name": info.get("gpu_name"),
            "gpu_vram_mb": info.get("gpu_vram_mb"),
            "pid": os.getpid(),
        }
        with _lock:
            _state.update(
                {
                    "state": "ready",
                    "result": result,
                    "finished_at": time.perf_counter(),
                    "error": None,
                }
            )
        _ready.set()
        return result
    except Exception as exc:
        with _lock:
            _state.update(
                {
                    "state": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "finished_at": time.perf_counter(),
                }
            )
        _ready.set()
        print(f"[FLUX WARMUP] pid={os.getpid()} state=FAILED", flush=True)
        print(
            f"[FLUX LOAD TRACE] process_id={os.getpid()} request_id=warmup "
            f"loader_instance=none cache_state=failed pipeline_exists=false "
            f"pipeline_load_end model_reused=false error={type(exc).__name__}: {exc}",
            flush=True,
        )
        raise
