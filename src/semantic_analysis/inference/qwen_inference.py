from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


class QwenInferenceEngine:
    """Run inference with the local Qwen vision-language model."""

    def __init__(self, model_loader: object, device: str = "cpu") -> None:
        self.model_loader = model_loader
        self.device = device
        self.logger = logging.getLogger("fabricvision.semantic_analysis.inference")

    def run(self, image_path: str | Path, prompt: str) -> str:
        """Run the model and return a raw response string."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if self.model_loader is None:
            self.logger.warning("No model loader configured; returning empty response")
            return "{}"

        model = getattr(self.model_loader, "model", None)
        processor = getattr(self.model_loader, "processor", None)
        if model is None or processor is None:
            self.logger.warning("Qwen model or processor not available; returning empty response")
            return "{}"

        self.logger.info("Running Qwen inference for %s", image_path)
        try:
            from PIL import Image as PilImage

            image = PilImage.open(image_path).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(images=image, text=text, return_tensors="pt")
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
            response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            if not response or not response.strip():
                return "{}"
            return response
        except Exception as exc:  # pragma: no cover - runtime guard
            self.logger.exception("Inference failed: %s", exc)
            return "{}"
