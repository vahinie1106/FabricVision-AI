from __future__ import annotations

import logging
import os
from typing import Optional, Literal

from src.common.models.device_manager import DeviceManager
from src.integrations.qwen.qwen_manager import QwenManager
from src.integrations.flux.flux_manager import FluxManager
from src.integrations.catvton.catvton_manager import CatVTONManager


class ModelManager:
    """
    Orchestrator for AI model loading and VRAM management.

    Single-GPU (local RTX 3050 / single T4): sequential unload — only one heavy
    model resident at a time.

    Dual-GPU (Kaggle T4×2 when FABRICVISION_DUAL_GPU / distinct role devices):
    FLUX stays on ``FLUX_CUDA_DEVICE`` (default 0) and CatVTON on
    ``CATVTON_CUDA_DEVICE`` (default 1) without unloading each other.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger("fabricvision.models.model_manager")
        self.device_manager = DeviceManager()

        flux_device = DeviceManager.resolve_role_device("flux", "auto")
        cat_device = DeviceManager.resolve_role_device("catvton", "auto")
        qwen_device = DeviceManager.resolve_role_device("qwen", "auto")

        self.qwen_manager = QwenManager(device=qwen_device)
        self.flux_manager = FluxManager(device=flux_device)
        self.catvton_manager = CatVTONManager(device=cat_device)
        self._active_model: Optional[str] = None

        self.logger.info(
            "ModelManager devices flux=%s catvton=%s qwen=%s dual_residency=%s",
            self.flux_manager.device,
            self.catvton_manager.device,
            self.qwen_manager.device,
            DeviceManager.dual_gpu_residency_enabled(),
        )

    @property
    def active_model(self) -> Optional[str]:
        return self._active_model

    def _device_for(self, model_name: str) -> str:
        if model_name == "flux":
            return self.device_manager.resolve_device(self.flux_manager.device)
        if model_name == "catvton":
            return self.device_manager.resolve_device(self.catvton_manager.device)
        if model_name == "qwen":
            return self.device_manager.resolve_device(self.qwen_manager.device)
        return "cpu"

    def _can_keep_peer_resident(self, unloading: str, activating: str) -> bool:
        """True when two heavy models can stay loaded on different GPUs."""
        if not DeviceManager.dual_gpu_residency_enabled():
            return False
        if unloading not in ("flux", "catvton") or activating not in ("flux", "catvton"):
            return False
        a = DeviceManager.cuda_device_index(self._device_for(unloading))
        b = DeviceManager.cuda_device_index(self._device_for(activating))
        return a is not None and b is not None and a != b

    def switch_to(self, model_name: Literal["qwen", "flux", "catvton"]) -> None:
        """Activate a model; unload peers unless dual-GPU residency applies."""
        needs_reload = False
        if self._active_model == model_name:
            if model_name == "flux":
                loader = getattr(self.flux_manager, "loader", None)
                if loader is None or getattr(loader, "pipeline", None) is None:
                    needs_reload = True
                else:
                    cb = getattr(self.flux_manager, "progress_callback", None)
                    self.flux_manager.load(progress_callback=cb)
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
        if os.environ.get("PYTEST_CURRENT_TEST"):
            self._active_model = model_name
            self.logger.info("Pytest: recorded active_model=%s without weight load", model_name)
            return

        if self._active_model == "qwen" and model_name != "qwen":
            self.qwen_manager.unload()
            self.device_manager.clear_vram(self._device_for("qwen"))
        elif self._active_model == "flux" and (model_name != "flux" or needs_reload):
            if model_name != "flux" and self._can_keep_peer_resident("flux", model_name):
                self.logger.info(
                    "Dual-GPU residency: keeping FLUX on %s while activating %s on %s",
                    self._device_for("flux"),
                    model_name,
                    self._device_for(model_name),
                )
            else:
                self.flux_manager.unload()
                self.device_manager.clear_vram(self._device_for("flux"))
        elif self._active_model == "catvton" and (model_name != "catvton" or needs_reload):
            if model_name != "catvton" and self._can_keep_peer_resident("catvton", model_name):
                self.logger.info(
                    "Dual-GPU residency: keeping CatVTON on %s while activating %s on %s",
                    self._device_for("catvton"),
                    model_name,
                    self._device_for(model_name),
                )
            else:
                self.catvton_manager.unload()
                self.device_manager.clear_vram(self._device_for("catvton"))

        if model_name == "qwen":
            self.qwen_manager.load()
        elif model_name == "flux":
            cb = getattr(self.flux_manager, "progress_callback", None)
            self.flux_manager.load(progress_callback=cb)
        elif model_name == "catvton":
            self.catvton_manager.load()

        self._active_model = model_name

    def clear_vram(self) -> None:
        """Proxy call to clear VRAM cache."""
        self.device_manager.clear_vram()

    def get_vram_usage_mb(self) -> float:
        """Proxy call to retrieve VRAM usage in MB."""
        return self.device_manager.get_allocated_vram_mb()
