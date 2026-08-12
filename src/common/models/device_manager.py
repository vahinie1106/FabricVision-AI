"""Shared device resolution and VRAM cleanup utilities."""

from __future__ import annotations

import gc
import os
from typing import Any, Dict, List, Optional


class DeviceManager:
    """Resolve compute devices and perform safe GPU memory cleanup."""

    @staticmethod
    def is_cuda_device(device: str | None) -> bool:
        """True for ``cuda``, ``cuda:0``, ``cuda:1``, …"""
        if not device:
            return False
        d = str(device).strip().lower()
        return d == "cuda" or d.startswith("cuda:")

    @staticmethod
    def cuda_device_index(device: str | None) -> Optional[int]:
        """Return GPU index for a CUDA device string, or None."""
        if not device:
            return None
        d = str(device).strip().lower()
        if d == "cuda":
            return 0
        if d.startswith("cuda:"):
            try:
                return int(d.split(":", 1)[1])
            except ValueError:
                return None
        return None

    @staticmethod
    def dual_gpu_residency_enabled() -> bool:
        """Keep FLUX + CatVTON loaded on separate GPUs when configured."""
        raw = os.environ.get("FABRICVISION_DUAL_GPU", "").strip().lower()
        if raw in ("0", "false", "no", "off"):
            return False
        if raw in ("1", "true", "yes", "on"):
            return True
        # Auto: enabled when two role devices resolve to different CUDA indices.
        flux = DeviceManager.resolve_role_device("flux")
        cat = DeviceManager.resolve_role_device("catvton")
        fi = DeviceManager.cuda_device_index(flux)
        ci = DeviceManager.cuda_device_index(cat)
        return fi is not None and ci is not None and fi != ci

    @staticmethod
    def inventory_gpus() -> List[Dict[str, Any]]:
        """List CUDA devices (empty when CUDA unavailable). Never raises."""
        try:
            import torch
        except ImportError:
            return []
        try:
            if not torch.cuda.is_available():
                return []
            out: List[Dict[str, Any]] = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                out.append(
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "total_memory_mb": round(float(props.total_memory) / (1024**2), 1),
                        "major": int(props.major),
                        "minor": int(props.minor),
                    }
                )
            return out
        except Exception:
            return []

    @staticmethod
    def resolve_role_device(role: str, fallback: str = "auto") -> str:
        """
        Resolve device for a model role.

        Env (highest priority):
          FLUX_CUDA_DEVICE / CATVTON_CUDA_DEVICE / QWEN_CUDA_DEVICE
            — integer index (``0``) or full string (``cuda:1``)
          FABRICVISION_FLUX_DEVICE / FABRICVISION_CATVTON_DEVICE
        """
        role = (role or "").strip().lower()
        env_keys = {
            "flux": ("FLUX_CUDA_DEVICE", "FABRICVISION_FLUX_DEVICE"),
            "catvton": ("CATVTON_CUDA_DEVICE", "FABRICVISION_CATVTON_DEVICE"),
            "qwen": ("QWEN_CUDA_DEVICE", "FABRICVISION_QWEN_DEVICE"),
        }.get(role, ())
        for key in env_keys:
            raw = os.environ.get(key, "").strip()
            if not raw:
                continue
            if raw.isdigit():
                return f"cuda:{raw}"
            return raw
        return fallback

    def resolve_device(self, device: str = "auto") -> str:
        """Resolve a device string.

        ``"auto"`` selects CUDA when available, otherwise CPU.
        Explicit values such as ``"cuda"``, ``"cuda:1"``, or ``"cpu"`` are kept.
        If PyTorch is unavailable, ``"auto"`` falls back to ``"cpu"``.
        """
        requested = (device or "auto").strip()
        lowered = requested.lower()
        if lowered != "auto":
            if lowered.isdigit():
                return f"cuda:{lowered}"
            return requested

        try:
            import torch
        except ImportError:
            return "cpu"

        try:
            if torch.cuda.is_available():
                return "cuda:0" if torch.cuda.device_count() > 1 else "cuda"
        except Exception:
            pass
        return "cpu"

    def clear_vram(self, device: str | None = None) -> None:
        """Run GC and empty the CUDA cache when available. Never raises."""
        try:
            gc.collect()
        except Exception:
            pass

        try:
            import torch
        except ImportError:
            return

        try:
            if not torch.cuda.is_available():
                return
        except Exception:
            return

        idx = self.cuda_device_index(device) if device else None
        try:
            if idx is not None:
                with torch.cuda.device(idx):
                    torch.cuda.empty_cache()
            else:
                torch.cuda.empty_cache()
        except Exception:
            pass

        ipc_collect: Optional[object] = getattr(torch.cuda, "ipc_collect", None)
        if callable(ipc_collect):
            try:
                ipc_collect()
            except Exception:
                pass

    def get_allocated_vram_mb(self, device: str | None = None) -> float:
        """Return currently allocated CUDA VRAM in megabytes."""
        try:
            import torch
        except ImportError:
            return 0.0

        try:
            if not torch.cuda.is_available():
                return 0.0
            idx = self.cuda_device_index(device)
            if idx is None:
                return float(torch.cuda.memory_allocated() / (1024**2))
            return float(torch.cuda.memory_allocated(idx) / (1024**2))
        except Exception:
            return 0.0
