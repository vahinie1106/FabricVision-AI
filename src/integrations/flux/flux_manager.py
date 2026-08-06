from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from src.common.models.device_manager import DeviceManager


class FluxManager:
    """Specialized model manager for FLUX.1-schnell diffusion model lifecycle."""

    def __init__(
        self,
        model_path: str | Path = "models/flux",
        device: str = "auto",
        precision: str = "bfloat16",
        allow_fallback: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.precision = precision
        self.allow_fallback = allow_fallback
        self.logger = logging.getLogger("fabricvision.models.flux_manager")
        self.device_manager = DeviceManager()
        self.loader: Optional[Any] = None

    def load(self) -> Any | None:
        """Load FLUX model weights into memory dynamically."""
        if self.loader is not None and getattr(self.loader, "pipeline", None) is not None:
            return self.loader.pipeline

        self.logger.info("Loading FLUX model via FluxManager from %s", self.model_path)
        from src.features.custom_generator.model.flux_model_loader import FLUXModelLoader
        self.loader = FLUXModelLoader(
            model_path=self.model_path,
            device=self.device,
            precision=self.precision,
            allow_fallback=self.allow_fallback,
        )
        return self.loader.load()

    def unload(self) -> None:
        """Unload FLUX pipeline and free GPU memory."""
        if self.loader is not None:
            self.logger.info("Unloading FLUX model pipeline...")
            self.loader._pipeline = None
            self.loader = None
            self.device_manager.clear_vram()
