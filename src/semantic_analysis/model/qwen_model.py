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

    def __init__(self, model_path: str | Path, device: str = "auto") -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.logger = logging.getLogger("fabricvision.semantic_analysis.model")
        self._model = None
        self._processor = None

    def load(self) -> tuple[object | None, object | None]:
        """Load the model and processor if available, otherwise return None values."""
        if not self.model_path.exists():
            self.logger.warning("Qwen model path does not exist: %s", self.model_path)
            return None, None

        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
        except Exception as exc:  # pragma: no cover - dependency availability guard
            self.logger.warning("Transformers could not be imported: %s", exc)
            return None, None

        selected_device = self._resolve_device()
        self.logger.info("Loading Qwen model from %s on %s", self.model_path, selected_device)
        try:
            processor = AutoProcessor.from_pretrained(str(self.model_path), use_fast=True)
            dtype = self._resolve_dtype()
            
            if self.device == "auto" and selected_device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    str(self.model_path), 
                    quantization_config=quantization_config,
                    device_map="auto"
                )
            else:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(self.model_path), torch_dtype=dtype)
                model.to(selected_device)
                
            model.eval()
            self._model = model
            self._processor = processor
            return model, processor
        except Exception as exc:  # pragma: no cover - runtime guard
            self.logger.exception("Failed to load Qwen model: %s", exc)
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
