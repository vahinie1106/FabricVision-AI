from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from src.common.models.device_manager import DeviceManager


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

    def load(self) -> Any | None:
        """Load FLUX.1-Kontext weights into memory (reuses resident pipeline)."""
        if self.loader is not None and getattr(self.loader, "pipeline", None) is not None:
            self.logger.info("[FLUX] Reusing loaded Kontext pipeline via FluxManager")
            # Touch loader.load() so reuse counters / logs stay consistent
            return self.loader.load()

        self.logger.info(
            "[FLUX] Model initialization started via FluxManager from %s",
            self.model_path,
        )
        from src.features.custom_generator.model.flux_model_loader import FLUXModelLoader

        self.loader = FLUXModelLoader(
            model_path=self.model_path,
            device=self.device,
            precision=self.precision,
            allow_fallback=self.allow_fallback,
            hf_model_id=self.hf_model_id,
        )
        self.logger.info("FLUX model ID: %s", self.loader.hf_model_id)
        pipeline = self.loader.load()
        if pipeline is not None:
            self.logger.info("[FLUX] Model initialization completed via FluxManager")
        return pipeline

    def unload(self) -> None:
        """Unload FLUX pipeline and free GPU memory."""
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
        """Park resident FLUX modules on CPU so the next job can start cleanly."""
        if self.loader is not None and hasattr(self.loader, "park_on_cpu"):
            self.loader.park_on_cpu()
        else:
            self.device_manager.clear_vram()
