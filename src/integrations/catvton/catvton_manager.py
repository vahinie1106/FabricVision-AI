from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from src.common.models.device_manager import DeviceManager


class CatVTONManager:
    """Specialized model manager for CatVTON virtual try-on model lifecycle."""

    def __init__(self, model_path: str | Path = "models/CatVTON", device: str = "auto") -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.logger = logging.getLogger("fabricvision.models.catvton_manager")
        self.device_manager = DeviceManager()
        self._loader = None

    @property
    def loader(self) -> Any | None:
        """Return the managed CatVTON loader, or None if it has not been created yet."""
        return self._loader

    def load(self) -> Any | None:
        """Load CatVTON model weights into GPU memory with offloading."""
        self.logger.info("Initializing CatVTON Manager from %s", self.model_path)
        if self._loader is None:
            from src.features.virtual_tryon.catvton_loader import CatVTONModelLoader
            self._loader = CatVTONModelLoader(model_path=self.model_path, device=self.device)
        return self._loader.load()

    def unload(self) -> None:
        """Unload CatVTON model weights and free GPU VRAM."""
        self.logger.info("Unloading CatVTON model weights...")
        if self._loader is not None:
            self._loader.unload()
            self._loader = None
        self.device_manager.clear_vram()
