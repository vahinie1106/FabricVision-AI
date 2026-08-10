"""Shared device resolution and VRAM cleanup utilities."""

from __future__ import annotations

import gc
from typing import Optional


class DeviceManager:
    """Resolve compute devices and perform safe GPU memory cleanup."""

    def resolve_device(self, device: str = "auto") -> str:
        """Resolve a device string.

        ``"auto"`` selects CUDA when available, otherwise CPU.
        Explicit values such as ``"cuda"`` or ``"cpu"`` are returned unchanged.
        If PyTorch is unavailable, ``"auto"`` falls back to ``"cpu"``.
        """
        requested = (device or "auto").strip().lower()
        if requested != "auto":
            return device if device is not None else requested

        try:
            import torch
        except ImportError:
            return "cpu"

        try:
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def clear_vram(self) -> None:
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

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        ipc_collect: Optional[object] = getattr(torch.cuda, "ipc_collect", None)
        if callable(ipc_collect):
            try:
                ipc_collect()
            except Exception:
                pass

    def get_allocated_vram_mb(self) -> float:
        """Return currently allocated CUDA VRAM in megabytes."""
        try:
            import torch
        except ImportError:
            return 0.0

        try:
            if not torch.cuda.is_available():
                return 0.0

            return float(torch.cuda.memory_allocated() / (1024 ** 2))
        except Exception:
            return 0.0
