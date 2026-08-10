"""Remote / Hybrid GPU execution provider for FLUX.1-Kontext garment generation.

Allows seamless switching between local RTX 3050 inference and remote GPU cluster
inference (e.g. Replicate, RunPod, Modal, or custom FastAPI GPU endpoint) without
changing frontend or pipeline interfaces.
"""

from __future__ import annotations

import abc
import io
import json
import logging
import os
from typing import Any, Dict, Optional

from PIL import Image

try:
    import urllib.request
    import urllib.error
except ImportError:
    urllib = None  # type: ignore

logger = logging.getLogger("fabricvision.garment_generation.remote_provider")


class BaseFluxProvider(abc.ABC):
    """Abstract interface for FLUX.1-Kontext generation execution."""

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        reference_image: Optional[Image.Image] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 16,
        guidance_scale: float = 3.5,
        seed: Optional[int] = 42,
        progress_callback: Optional[Any] = None,
        save_raw_path: Optional[str] = None,
    ) -> Image.Image:
        """Generate garment image."""
        pass


class LocalFluxProvider(BaseFluxProvider):
    """Wraps local FLUXInferenceEngine."""

    def __init__(self, inference_engine: Any) -> None:
        self.inference_engine = inference_engine

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        reference_image: Optional[Image.Image] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 16,
        guidance_scale: float = 3.5,
        seed: Optional[int] = 42,
        progress_callback: Optional[Any] = None,
        save_raw_path: Optional[str] = None,
    ) -> Image.Image:
        return self.inference_engine.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            reference_image=reference_image,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            progress_callback=progress_callback,
            save_raw_path=save_raw_path,
        )


class RemoteFluxProvider(BaseFluxProvider):
    """Client for remote high-performance GPU FLUX endpoint."""

    def __init__(self, remote_url: Optional[str] = None, api_key: Optional[str] = None) -> None:
        self.remote_url = remote_url or os.environ.get("FLUX_REMOTE_URL")
        self.api_key = api_key or os.environ.get("FLUX_REMOTE_API_KEY")

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        reference_image: Optional[Image.Image] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 16,
        guidance_scale: float = 3.5,
        seed: Optional[int] = 42,
        progress_callback: Optional[Any] = None,
        save_raw_path: Optional[str] = None,
    ) -> Image.Image:
        if not self.remote_url:
            raise RuntimeError(
                "FLUX_REMOTE_URL environment variable is not configured for RemoteFluxProvider."
            )

        logger.info("Dispatching FLUX generation to remote GPU endpoint: %s", self.remote_url)

        # Convert reference_image to PNG bytes
        ref_bytes = io.BytesIO()
        if reference_image is not None:
            reference_image.save(ref_bytes, format="PNG")
        import base64

        encoded_ref = base64.b64encode(ref_bytes.getvalue()).decode("utf-8")

        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "reference_image_base64": encoded_ref,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
        }

        req = urllib.request.Request(
            self.remote_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                img_b64 = res_data.get("image_base64")
                if not img_b64:
                    raise RuntimeError("Remote response missing 'image_base64'")
                img_bytes = base64.b64decode(img_b64)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                return img
        except Exception as exc:
            logger.error("Remote GPU inference request failed: %s", exc)
            raise RuntimeError(f"Remote FLUX inference failed: {exc}") from exc
