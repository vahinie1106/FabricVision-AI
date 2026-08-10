from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

try:
    import torch
except Exception:  # pragma: no cover - optional dependency guard
    torch = None


class QwenModelLoader:
    """Load the local Qwen2.5-VL model and processor with graceful fallbacks."""

    def __init__(self, model_path: str | Path = "models/Qwen2.5-VL-3B-Instruct", device: str = "auto") -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.logger = logging.getLogger("fabricvision.semantic_analysis.model")
        self._model = None
        self._processor = None

    def _is_complete_model_dir(self) -> bool:
        if not self.model_path.exists() or not (self.model_path / "config.json").exists():
            return False
        index_file = self.model_path / "model.safetensors.index.json"
        single_file = self.model_path / "model.safetensors"
        if not index_file.exists() and not single_file.exists():
            return False
        weight_files = list(self.model_path.glob("*.safetensors")) + list(self.model_path.glob("*.bin"))
        return len(weight_files) >= 1

    def load(self) -> tuple[object | None, object | None]:
        """Load the model and processor if available, otherwise return None values."""
        if self._model is not None and self._processor is not None:
            return self._model, self._processor

        import os

        if os.environ.get("PYTEST_CURRENT_TEST"):
            # Mirrors FLUXModelLoader's pytest guard: real weights exist locally
            # for Qwen2.5-VL (~7.5GB) and would otherwise load onto the GPU during
            # a default `pytest -q` run. Tests that need the real model opt in
            # explicitly via the `slow` marker.
            self.logger.info("Pytest environment detected; skipping Qwen weight load.")
            return None, None

        if not self._is_complete_model_dir():
            self.logger.warning("Qwen model path incomplete or missing: %s", self.model_path)
            return None, None

        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except Exception as exc:  # pragma: no cover - dependency availability guard
            self.logger.warning("Transformers could not be imported: %s", exc)
            return None, None

        selected_device = self._resolve_device()
        self.logger.info("Loading Qwen model from %s on %s", self.model_path, selected_device)
        try:
            processor = AutoProcessor.from_pretrained(str(self.model_path), use_fast=True)
            dtype = self._resolve_dtype()
            
            import os
            use_low_cpu_mem = False if (os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("FAST_TEST")) else True
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(self.model_path),
                torch_dtype=dtype,
                device_map="cuda" if selected_device == "cuda" else None,
                low_cpu_mem_usage=use_low_cpu_mem,
            )
            if selected_device != "cuda" and hasattr(model, "to"):
                model.to(selected_device)

            model.eval()
            self._model = model
            self._processor = processor
            return model, processor
        except (Exception, OSError, RuntimeError) as exc:  # pragma: no cover - runtime guard
            self.logger.warning("Failed to load Qwen model: %s", exc)
            return None, None

    def _resolve_device(self) -> str:
        if self.device == "auto":
            if torch is not None and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return self.device

    def _resolve_dtype(self) -> object:
        if torch is None:
            return None
        if self.device == "auto" and torch.cuda.is_available():
            return torch.bfloat16
        return torch.float32

    @property
    def model(self) -> object | None:
        return self._model

    @property
    def processor(self) -> object | None:
        return self._processor
