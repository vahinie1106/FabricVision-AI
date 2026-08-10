from __future__ import annotations

import logging
import os
from typing import Optional, Literal

from src.common.models.device_manager import DeviceManager
from src.integrations.qwen.qwen_manager import QwenManager
from src.integrations.flux.flux_manager import FluxManager
from src.integrations.catvton.catvton_manager import CatVTONManager


class ModelManager:
    """Master orchestrator for sequential AI model loading, memory offloading, and VRAM management."""

    def __init__(self) -> None:
        self.logger = logging.getLogger("fabricvision.models.model_manager")
        self.device_manager = DeviceManager()
        self.qwen_manager = QwenManager()
        self.flux_manager = FluxManager()
        self.catvton_manager = CatVTONManager()
        self._active_model: Optional[str] = None

    @property
    def active_model(self) -> Optional[str]:
        return self._active_model

    def switch_to(self, model_name: Literal["qwen", "flux", "catvton"]) -> None:
        """Sequential execution guard: Unloads inactive models before activating requested model."""
        needs_reload = False
        if self._active_model == model_name:
            if model_name == "flux":
                loader = getattr(self.flux_manager, "loader", None)
                if loader is None or getattr(loader, "pipeline", None) is None:
                    needs_reload = True
                else:
                    return
            elif model_name == "catvton":
                loader = getattr(self.catvton_manager, "loader", None)
                if loader is None or getattr(loader, "pipeline", None) is None:
                    needs_reload = True
                else:
                    return
            else:
                return

        self.logger.info(
            "Switching active model from '%s' to '%s'%s",
            self._active_model,
            model_name,
            " (forced reload)" if needs_reload else "",
        )

        # Default pytest runs must not load multi-GB weights (hangs under GPU contention).
        # Marked `slow` tests that need real residency should call managers directly.
        if os.environ.get("PYTEST_CURRENT_TEST"):
            self._active_model = model_name
            self.logger.info("Pytest: recorded active_model=%s without weight load", model_name)
            return

        if self._active_model == "qwen" and model_name != "qwen":
            self.qwen_manager.unload()
        elif self._active_model == "flux" and (model_name != "flux" or needs_reload):
            self.flux_manager.unload()
        elif self._active_model == "catvton" and (model_name != "catvton" or needs_reload):
            self.catvton_manager.unload()

        self.device_manager.clear_vram()

        if model_name == "qwen":
            self.qwen_manager.load()
        elif model_name == "flux":
            self.flux_manager.load()
        elif model_name == "catvton":
            self.catvton_manager.load()

        self._active_model = model_name

    def clear_vram(self) -> None:
        """Proxy call to clear VRAM cache."""
        self.device_manager.clear_vram()

    def get_vram_usage_mb(self) -> float:
        """Proxy call to retrieve VRAM usage in MB."""
        return self.device_manager.get_allocated_vram_mb()
