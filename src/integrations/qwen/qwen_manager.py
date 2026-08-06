from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from src.common.models.device_manager import DeviceManager
from src.features.semantic_analysis.model.qwen_model import QwenModelLoader


class QwenManager:
    """Specialized model manager for Qwen2.5-VL vision-language model lifecycle."""

    def __init__(self, model_path: str | Path = "models/Qwen2.5-VL-3B-Instruct", device: str = "auto") -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.logger = logging.getLogger("fabricvision.models.qwen_manager")
        self.device_manager = DeviceManager()
        self.loader: Optional[QwenModelLoader] = None

    def load(self) -> tuple[Any | None, Any | None]:
        """Load Qwen2.5-VL model weights into memory."""
        if self.loader is not None and self.loader.model is not None:
            return self.loader.model, self.loader.processor

        self.logger.info("Loading Qwen model via QwenManager from %s", self.model_path)
        self.loader = QwenModelLoader(self.model_path, self.device)
        return self.loader.load()

    def unload(self) -> None:
        """Unload Qwen model weights and free GPU memory."""
        if self.loader is not None:
            self.logger.info("Unloading Qwen2.5-VL model weights...")
            self.loader._model = None
            self.loader._processor = None
            self.loader = None
            self.device_manager.clear_vram()
