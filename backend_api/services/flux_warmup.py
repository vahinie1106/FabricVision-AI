"""Warm FLUX inside the FastAPI process (not the run_kaggle parent).

CRITICAL ARCHITECTURE:
  scripts/run_kaggle.py runs in a PARENT process. Prefetch there only fills the
  on-disk Hugging Face / models/ cache, then unloads. FastAPI is a CHILD process
  (uvicorn Popen) with a separate address space — it does NOT inherit the
  parent's in-memory FluxKontextPipeline.

  Therefore the ONLY in-memory FLUX that Generate can reuse is the one loaded
  by this module inside the API process (shared ``generation_service.model_manager``).

Readiness layers (do not conflate):
  - /api/v1/health  → API liveness (never waits for FLUX)
  - /health         → Next.js frontend reachable
  - /api/v1/flux-status → FLUX warmup state in THIS process
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("fabricvision.api.flux_warmup")

ProgressCb = Optional[Callable[[str, int], None]]

# Public state names exposed by /api/v1/flux-status.
_PUBLIC_STATE = {
    "idle": "IDLE",
    "loading": "STARTING",
    "ready": "READY",
    "failed": "FAILED",
    "skipped": "SKIPPED",
}

_lock = threading.Lock()
_ready = threading.Event()
_state: Dict[str, Any] = {
    "state": "idle",  # idle | loading | ready | failed | skipped
    "pid": None,
    "error": None,
    "result": None,
    "started_at": None,
    "finished_at": None,
    "progress": 0,
    "current_step": None,
    "stage": None,
    "cache_status": None,
    "pipeline_exists": False,
    "model_reused": False,
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
                "progress": 0,
                "current_step": None,
                "stage": None,
                "cache_status": None,
                "pipeline_exists": False,
                "model_reused": False,
            }
        )
        _ready.clear()


def update_warmup_progress(
    step: str,
    pct: int,
    *,
    stage: Optional[str] = None,
    cache_status: Optional[str] = None,
) -> None:
    """Update live progress visible via /api/v1/flux-status."""
    with _lock:
        if _state["state"] not in ("loading", "ready"):
            return
        _state["progress"] = max(0, min(100, int(pct)))
        _state["current_step"] = step
        if stage:
            _state["stage"] = stage
        if cache_status is not None:
            _state["cache_status"] = cache_status
        _state["pipeline_exists"] = flux_in_memory_ready()


def get_warmup_status() -> Dict[str, Any]:
    """Public status for /api/v1/flux-status and generation waits."""
    with _lock:
        raw = dict(_state)
    started = raw.get("started_at")
    finished = raw.get("finished_at")
    now = time.perf_counter()
    if started is not None:
        end = finished if finished is not None else now
        load_duration_s = round(end - float(started), 2)
    else:
        load_duration_s = None

    in_mem = flux_in_memory_ready()
    public_state = _PUBLIC_STATE.get(str(raw.get("state")), str(raw.get("state")).upper())
    out: Dict[str, Any] = {
        "state": public_state,
        "state_raw": raw.get("state"),
        "pid": raw.get("pid"),
        "error": raw.get("error"),
        "result": raw.get("result"),
        "started_at": started,
        "finished_at": finished,
        "load_started_at": started,
        "load_duration_s": load_duration_s,
        "progress": raw.get("progress") or 0,
        "current_step": raw.get("current_step"),
        "stage": raw.get("stage") or raw.get("current_step"),
        "cache_status": raw.get("cache_status"),
        "pipeline_exists": bool(raw.get("pipeline_exists") or in_mem),
        "model_reused": bool(raw.get("model_reused")),
        "ready": _ready.is_set() and raw.get("state") == "ready",
        "in_memory": in_mem,
    }
    return out


def flux_in_memory_ready() -> bool:
    try:
        from backend_api.services.generation_service import model_manager

        return (
            model_manager.active_model == "flux"
            and model_manager.flux_manager.loader is not None
            and getattr(model_manager.flux_manager.loader, "pipeline", None)
            is not None
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
                    _state["pipeline_exists"] = True
                    _state["progress"] = max(int(_state.get("progress") or 0), 100)
            return get_warmup_status()

        status = get_warmup_status()
        state_raw = status.get("state_raw")
        if state_raw == "failed":
            return status
        if state_raw == "skipped":
            return status
        if state_raw == "idle":
            return status

        if progress_callback is not None and state_raw == "loading":
            elapsed = int(status.get("load_duration_s") or 0)
            step = status.get("current_step") or "Waiting for API-process FLUX warmup"
            try:
                progress_callback(
                    f"{step} ({elapsed}s, pid={status.get('pid')})",
                    max(9, min(17, int(status.get("progress") or 9))),
                )
            except Exception:
                pass

        if _ready.wait(timeout=poll_s):
            if flux_in_memory_ready():
                return get_warmup_status()

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
            _state.update(
                {
                    "state": "skipped",
                    "result": {"skipped": True, "reason": "disabled"},
                    "progress": 0,
                    "current_step": "FLUX warmup disabled",
                    "stage": "SKIPPED",
                }
            )
        _ready.set()
        print(f"[FLUX WARMUP] pid={os.getpid()} state=SKIPPED reason=disabled", flush=True)
        logger.info("[FLUX WARMUP] skipped (FLUX_WARMUP_ON_STARTUP=%s)", force)
        return {"skipped": True, "reason": "disabled"}

    if os.environ.get("PYTEST_CURRENT_TEST") and force != "force":
        with _lock:
            _state.update(
                {
                    "state": "skipped",
                    "result": {"skipped": True, "reason": "pytest"},
                    "progress": 0,
                    "current_step": "FLUX warmup skipped under pytest",
                    "stage": "SKIPPED",
                }
            )
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
                    "progress": 1,
                    "current_step": "FLUX warmup starting",
                    "stage": "STARTING",
                    "cache_status": None,
                    "pipeline_exists": False,
                    "model_reused": False,
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
        f"pipeline_load_start model_reused=false "
        f"(note: cache_state here is in-memory residency, not HF disk cache)",
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
                        "progress": 100,
                        "current_step": "FLUX READY (already in memory)",
                        "stage": "READY",
                        "pipeline_exists": True,
                        "model_reused": True,
                        "cache_status": "in_memory",
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
            update_warmup_progress(step, pct, stage="LOADING")
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
            f"[FLUX TIMING] WARMUP_READY t={time.perf_counter():.2f} duration={elapsed}s "
            f"pid={os.getpid()} disk_cache={info.get('cache_status')} "
            f"init_s={info.get('init_time_s')} download_s={info.get('download_time_s')} "
            f"offload={info.get('offload_strategy')} pipeline_exists=true",
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
                    "progress": 100,
                    "current_step": "FLUX READY",
                    "stage": "READY",
                    "cache_status": info.get("cache_status"),
                    "pipeline_exists": True,
                    "model_reused": False,
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
                    "progress": int(_state.get("progress") or 0),
                    "current_step": f"FLUX warmup failed: {type(exc).__name__}",
                    "stage": "FAILED",
                    "pipeline_exists": False,
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
