"""Runtime VRAM diagnostics and completion-first FLUX generation policy.

Do NOT assume Tesla T4 ≡ 768×768 GPU-resident is safe. Decide from measured
physical capacity + allocator headroom after the model is resident.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger("fabricvision.flux.vram_policy")


@dataclass(frozen=True)
class VramDiagnostics:
    gpu_name: str
    physical_total_mb: float
    allocated_mb: float
    reserved_mb: float
    free_mb: float
    max_allocated_mb: float
    max_reserved_mb: float
    cuda_available: bool

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StandardGenPolicy:
    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    prefer_model_cpu_offload: bool
    enable_vae_tiling: bool
    reason: str
    profile: str


def collect_vram_diagnostics() -> VramDiagnostics:
    """Read live GPU / CUDA allocator state. Never raises."""
    try:
        import torch
    except Exception:
        return VramDiagnostics(
            gpu_name="unavailable",
            physical_total_mb=0.0,
            allocated_mb=0.0,
            reserved_mb=0.0,
            free_mb=0.0,
            max_allocated_mb=0.0,
            max_reserved_mb=0.0,
            cuda_available=False,
        )

    if not torch.cuda.is_available():
        return VramDiagnostics(
            gpu_name="cpu",
            physical_total_mb=0.0,
            allocated_mb=0.0,
            reserved_mb=0.0,
            free_mb=0.0,
            max_allocated_mb=0.0,
            max_reserved_mb=0.0,
            cuda_available=False,
        )

    try:
        props = torch.cuda.get_device_properties(0)
        total = float(props.total_memory) / (1024**2)
        name = torch.cuda.get_device_name(0)
    except Exception:
        total = 0.0
        name = "unknown"

    try:
        allocated = float(torch.cuda.memory_allocated() / (1024**2))
        reserved = float(torch.cuda.memory_reserved() / (1024**2))
        max_alloc = float(torch.cuda.max_memory_allocated() / (1024**2))
        try:
            max_res = float(torch.cuda.max_memory_reserved() / (1024**2))
        except Exception:
            max_res = reserved
    except Exception:
        allocated = reserved = max_alloc = max_res = 0.0

    free = max(0.0, total - reserved)
    return VramDiagnostics(
        gpu_name=name,
        physical_total_mb=round(total, 1),
        allocated_mb=round(allocated, 1),
        reserved_mb=round(reserved, 1),
        free_mb=round(free, 1),
        max_allocated_mb=round(max_alloc, 1),
        max_reserved_mb=round(max_res, 1),
        cuda_available=True,
    )


def log_vram(tag: str, diag: Optional[VramDiagnostics] = None) -> VramDiagnostics:
    """Emit a single-line VRAM snapshot for Kaggle log scraping."""
    d = diag or collect_vram_diagnostics()
    line = (
        f"[VRAM] {tag} gpu={d.gpu_name} total_mb={d.physical_total_mb:.0f} "
        f"alloc_mb={d.allocated_mb:.0f} reserved_mb={d.reserved_mb:.0f} "
        f"free_mb={d.free_mb:.0f} peak_alloc_mb={d.max_allocated_mb:.0f} "
        f"peak_reserved_mb={d.max_reserved_mb:.0f}"
    )
    logger.info(line)
    print(line, flush=True)
    return d


def _env_int(name: str) -> Optional[int]:
    raw = os.environ.get(name, "").strip()
    if raw.isdigit():
        return int(raw)
    return None


def _env_truthy(name: str) -> Optional[bool]:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return None
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return None


def select_standard_generation_policy(
    *,
    physical_mb: Optional[float] = None,
    free_mb: Optional[float] = None,
    offload_strategy: Optional[str] = None,
) -> StandardGenPolicy:
    """
    Completion-first Standard policy.

    Priority: finish without OOM > quality > speed.
    Env overrides always win for resolution/steps when set.
    """
    diag = collect_vram_diagnostics()
    phys = float(physical_mb if physical_mb is not None else diag.physical_total_mb)
    free = float(free_mb if free_mb is not None else diag.free_mb)
    offload = (offload_strategy or "").strip().lower()
    gpu_resident = offload in ("gpu_resident", "none", "")

    # Explicit env wins (operators / smoke tests).
    forced_res = _env_int("FLUX_GENERATION_RESOLUTION") or _env_int("FLUX_PRODUCTION_SIZE")
    forced_steps = _env_int("FLUX_STANDARD_STEPS")
    allow_high_res = _env_truthy("FLUX_ALLOW_HIGH_RES")
    offload_env = _env_truthy("FLUX_MODEL_CPU_OFFLOAD")

    # Low-VRAM path (RTX 3050 class): keep conservative YAML-like defaults.
    if phys > 0 and phys < 14000:
        return StandardGenPolicy(
            height=forced_res or 512,
            width=forced_res or 512,
            num_inference_steps=forced_steps or 3,
            guidance_scale=3.0,
            prefer_model_cpu_offload=True if offload_env is None else offload_env,
            enable_vae_tiling=True,
            reason=f"low_vram phys={phys:.0f}MB",
            profile="standard_low_vram",
        )

    # ~14–20 GB class (typical Kaggle T4 ~15109 MiB):
    # GPU-resident NF4 already occupies most of the card. 768² activations have
    # repeatedly OOM'd here — do NOT auto-upgrade to 768 unless headroom is real
    # or the operator opts in.
    prefer_offload = True if offload_env is None else offload_env
    # Prefer residency only when explicitly disabled offload AND enough free headroom.
    if offload_env is False and free >= 7000:
        prefer_offload = False
    elif offload_env is False:
        # Keep GPU-resident (warmup already paid for it) but stay at safe res.
        prefer_offload = False

    # High-res only when free headroom is large OR explicit allow flag / forced 768+.
    high_res_ok = False
    if forced_res and forced_res >= 768:
        high_res_ok = True
    elif allow_high_res is True and free >= 6500:
        high_res_ok = True
    elif allow_high_res is True and not gpu_resident:
        # Offload path can sometimes survive 768; still gated by explicit allow.
        high_res_ok = True
    elif forced_res is None and allow_high_res is not False and free >= 8000:
        # Measured headroom: only then auto 768.
        high_res_ok = True

    if high_res_ok:
        res = forced_res or 768
        steps = forced_steps or 12
        profile = "standard_high_res"
        reason = (
            f"high_res_ok free={free:.0f}MB phys={phys:.0f}MB "
            f"offload={prefer_offload} allow={allow_high_res}"
        )
    else:
        res = forced_res or 512
        # 8 steps: better than blurry 3-step 3050 preset; safer peak than 12@768.
        steps = forced_steps or 8
        profile = "standard_t4_safe"
        reason = (
            f"completion_first free={free:.0f}MB phys={phys:.0f}MB "
            f"gpu_resident_hint={gpu_resident} (768 gated)"
        )

    return StandardGenPolicy(
        height=int(res),
        width=int(res),
        num_inference_steps=int(steps),
        guidance_scale=3.0,
        prefer_model_cpu_offload=bool(prefer_offload),
        enable_vae_tiling=True,
        reason=reason,
        profile=profile,
    )


def recommend_oom_fallback(
    *,
    height: int,
    width: int,
    num_inference_steps: int,
) -> Optional[Dict[str, int]]:
    """Return a smaller config to retry after diffusion OOM, or None if already minimal."""
    area = int(height) * int(width)
    if area >= 768 * 768:
        return {
            "height": 512,
            "width": 512,
            "num_inference_steps": max(4, min(int(num_inference_steps), 8)),
        }
    if area >= 512 * 512 and num_inference_steps > 4:
        return {
            "height": 512,
            "width": 512,
            "num_inference_steps": 4,
        }
    if area > 384 * 384:
        return {
            "height": 384,
            "width": 384,
            "num_inference_steps": max(3, min(int(num_inference_steps), 4)),
        }
    return None
