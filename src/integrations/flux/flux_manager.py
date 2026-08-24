from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from src.common.models.device_manager import DeviceManager

ProgressCallback = Optional[Callable[[str, int], None]]


class FluxManager:
    """Specialized model manager for FLUX.1-Kontext diffusion model lifecycle."""

    def __init__(
        self,
        model_path: str | Path = "models/flux-kontext",
        device: str = "auto",
        precision: str = "bfloat16",
        allow_fallback: bool = True,
        hf_model_id: Optional[str] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.precision = precision
        self.allow_fallback = allow_fallback
        self.hf_model_id = hf_model_id
        self.logger = logging.getLogger("fabricvision.models.flux_manager")
        self.device_manager = DeviceManager()
        self.loader: Optional[Any] = None
        self.progress_callback: ProgressCallback = None
        # Prevent warmup + Generate (or duplicate requests) from racing into
        # multiple concurrent from_pretrained() calls.
        self._load_lock = threading.Lock()

    def load(self, progress_callback: ProgressCallback = None) -> Any | None:
        """Load FLUX.1-Kontext weights into memory (reuses resident pipeline)."""
        import os

        cb = progress_callback if progress_callback is not None else self.progress_callback
        with self._load_lock:
            pipe_exists = (
                self.loader is not None
                and getattr(self.loader, "pipeline", None) is not None
            )
            print(
                f"[FLUX LOAD TRACE] process_id={os.getpid()} request_id=flux_manager "
                f"loader_instance={id(self.loader) if self.loader else None} "
                f"in_memory={'hit' if pipe_exists else 'miss'} "
                f"disk_cache=n/a "
                f"pipeline_exists={pipe_exists} "
                f"{'pipeline_load_end model_reused=true' if pipe_exists else 'pipeline_load_start model_reused=false'} "
                f"(in_memory miss means from_pretrained will run in THIS process; "
                f"not the same as HF disk CACHE MISS)",
                flush=True,
            )
            if pipe_exists:
                self.logger.info("[FLUX] Reusing loaded Kontext pipeline via FluxManager")
                return self.loader.load(progress_callback=cb)

            self.logger.info(
                "[FLUX] Model initialization started via FluxManager from %s",
                self.model_path,
            )
            print("[FLUX] model loading started", flush=True)
            from src.features.custom_generator.model.flux_model_loader import FLUXModelLoader

            self.loader = FLUXModelLoader(
                model_path=self.model_path,
                device=self.device,
                precision=self.precision,
                allow_fallback=self.allow_fallback,
                hf_model_id=self.hf_model_id,
            )
            self.logger.info("FLUX model ID: %s", self.loader.hf_model_id)
            pipeline = self.loader.load(progress_callback=cb)
            print(
                f"[FLUX LOAD TRACE] process_id={os.getpid()} request_id=flux_manager "
                f"loader_instance={id(self.loader)} "
                f"cache_state={getattr(self.loader, '_cache_status', None)} "
                f"pipeline_exists={pipeline is not None} "
                f"pipeline_load_end model_reused=false",
                flush=True,
            )
            if pipeline is not None:
                self.logger.info("[FLUX] Model initialization completed via FluxManager")
                print("[FLUX] model loading completed", flush=True)
                try:
                    dev = getattr(pipeline, "_execution_device", None) or self.device
                    print(f"[FLUX] pipeline device = {dev}", flush=True)
                except Exception:
                    pass
            return pipeline

    def unload(self) -> None:
        """Unload FLUX pipeline and free GPU memory."""
        with self._load_lock:
            if self.loader is not None:
                self.logger.info("Unloading FLUX.1-Kontext pipeline...")
                if hasattr(self.loader, "park_on_cpu"):
                    try:
                        self.loader.park_on_cpu()
                    except Exception as exc:
                        self.logger.warning("park_on_cpu during unload failed: %s", exc)
                self.loader._pipeline = None
                self.loader = None
                self.device_manager.clear_vram()

    def recover_after_oom(self) -> None:
        """Park resident FLUX modules and reset the CUDA allocator for Retry."""
        import gc

        if self.loader is not None and hasattr(self.loader, "park_on_cpu"):
            self.loader.park_on_cpu()
        else:
            self.device_manager.clear_vram()
        try:
            from src.features.custom_generator.inference.flux_vram_policy import (
                cleanup_cuda_after_failure,
            )

            cleanup_cuda_after_failure()
        except Exception:
            gc.collect()
            self.device_manager.clear_vram()
