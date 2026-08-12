from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Any

from src.common.models.device_manager import DeviceManager

try:
    import torch
except ImportError:
    torch = None


class CatVTONModelLoader:
    """Load and manage CatVTON virtual try-on model weights with production 6GB VRAM safety."""

    def __init__(
        self,
        model_path: str | Path = "models/CatVTON",
        device: str = "auto",
        precision: str = "bfloat16",
        allow_fallback: bool = True,
        base_ckpt: str = "runwayml/stable-diffusion-inpainting",
        attn_ckpt_version: str = "vitonhd",
    ) -> None:
        self.model_path = Path(model_path)
        self.device_setting = device
        self.precision = precision
        self.allow_fallback = allow_fallback
        self.base_ckpt = base_ckpt
        self.attn_ckpt_version = attn_ckpt_version
        self.logger = logging.getLogger("fabricvision.virtual_tryon.model_loader")
        self.device_manager = DeviceManager()
        self._pipeline = None
        self.last_load_info: dict = {}

    def _is_complete_local_dir(self) -> bool:
        if not self.model_path.exists():
            return False
        weight_files = list(self.model_path.rglob("*.safetensors")) + list(self.model_path.rglob("*.bin")) + list(self.model_path.rglob("*.pth")) + list(self.model_path.rglob("*.pkl"))
        return len(weight_files) >= 1

    def load(self) -> Any | None:
        """Load CatVTON diffusion pipeline with bfloat16, CPU offload, and attention slicing."""
        if self._pipeline is not None:
            return self._pipeline

        import os

        if os.environ.get("PYTEST_CURRENT_TEST") and self.allow_fallback:
            # Mirrors FLUXModelLoader's pytest guard: real CatVTON checkpoints
            # exist locally and would otherwise load a real diffusion pipeline
            # during a default `pytest -q` run. Tests that need the real model
            # opt in explicitly via the `slow` marker.
            self.logger.info("Pytest environment detected; skipping CatVTON weight load.")
            return None

        target_device = self.device_manager.resolve_device(self.device_setting)
        self.logger.info("Initializing CatVTON")

        if not self._is_complete_local_dir():
            msg = f"CatVTON model weights not found at {self.model_path}."
            self.logger.warning(msg)
            if not self.allow_fallback:
                raise RuntimeError(msg)
            return None

        dtype = None
        if torch is not None:
            dtype = torch.bfloat16 if (self.precision == "bfloat16" and torch.cuda.is_available()) else torch.float32

        try:
            pipeline = None
            self.logger.info("Loading model weights")
            self.logger.info("Loading VAE")

            # Check if custom CatVTON repo pipeline is present
            catvton_pipeline_script = self.model_path / "model" / "pipeline.py"
            if catvton_pipeline_script.exists():
                catvton_dir_str = str(self.model_path.resolve())
                if catvton_dir_str not in sys.path:
                    sys.path.insert(0, catvton_dir_str)
                try:
                    from model.pipeline import CatVTONPipeline  # type: ignore

                    version = self.attn_ckpt_version
                    # Prefer available local attention folders.
                    if version == "vitonhd" and not (self.model_path / "vitonhd-16k-512").exists():
                        version = "mix" if (self.model_path / "mix-48k-1024").exists() else version
                    self.logger.info(
                        "Loading CatVTONPipeline attn_ckpt_version=%s dtype=%s device=%s",
                        version,
                        dtype,
                        target_device,
                    )
                    pipeline = CatVTONPipeline(
                        base_ckpt=self.base_ckpt,
                        attn_ckpt=str(self.model_path),
                        attn_ckpt_version=version,
                        weight_dtype=dtype,
                        device=target_device,
                        skip_safety_check=True,
                    )
                    self.last_load_info = {
                        "attn_ckpt_version": version,
                        "device": str(target_device),
                        "dtype": str(dtype),
                        "base_ckpt": self.base_ckpt,
                    }
                except Exception as cat_exc:
                    self.logger.warning("Could not instantiate native CatVTONPipeline: %s; trying AutoPipeline", cat_exc)

            if pipeline is None:
                from diffusers import AutoPipelineForInpainting
                pipeline = AutoPipelineForInpainting.from_pretrained(
                    str(self.model_path),
                    torch_dtype=dtype,
                )

            if DeviceManager.is_cuda_device(target_device):
                self.logger.info("Moving CatVTON to %s", target_device)
                dual = DeviceManager.dual_gpu_residency_enabled()
                gpu_idx = DeviceManager.cuda_device_index(target_device) or 0
                # On a dedicated second T4, keep the pipeline GPU-resident.
                # Sequential offload is for single-GPU / 6GB sharing with FLUX.
                if dual or gpu_idx > 0:
                    if hasattr(pipeline, "to"):
                        pipeline.to(target_device)
                    self.logger.info(
                        "CatVTON GPU-resident on %s (dual_gpu=%s)",
                        target_device,
                        dual,
                    )
                    if hasattr(pipeline, "enable_attention_slicing"):
                        pipeline.enable_attention_slicing()
                else:
                    if hasattr(pipeline, "enable_sequential_cpu_offload"):
                        try:
                            pipeline.enable_sequential_cpu_offload(gpu_id=gpu_idx)
                        except TypeError:
                            pipeline.enable_sequential_cpu_offload()
                        self.logger.info("CPU offloading enabled")
                    if hasattr(pipeline, "enable_attention_slicing"):
                        pipeline.enable_attention_slicing()
                        self.logger.info("Attention slicing enabled")
            else:
                if hasattr(pipeline, "to"):
                    pipeline.to(target_device)

            self.logger.info("CatVTON ready")
            self._pipeline = pipeline
            return pipeline
        except Exception as exc:
            msg = f"Failed to load CatVTON model from '{self.model_path}': {exc}"
            self.logger.warning(msg)
            if not self.allow_fallback:
                raise RuntimeError(msg) from exc
            return None

    def unload(self) -> None:
        """Unload pipeline and clear GPU VRAM."""
        if self._pipeline is not None:
            self.logger.info("Unloading CatVTON model weights...")
            self._pipeline = None
            self.device_manager.clear_vram()

    @property
    def pipeline(self) -> Any | None:
        return self._pipeline
