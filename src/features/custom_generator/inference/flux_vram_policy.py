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
    device_free_mb: float = 0.0

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


@dataclass(frozen=True)
class HybridVramPlan:
    """T4 GPU-resident lifecycle — never per-step model_cpu_offload."""

    evict_t5_before_diffusion: bool
    park_vae_during_denoise: bool
    decode_latents_separately: bool
    park_transformer_before_vae: bool
    enable_vae_tiling: bool
    enable_vae_slicing: bool
    per_step_cpu_offload: bool
    reason: str
    profile: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Production uses the same shape; kept as an alias for call-site clarity.
ProductionGenPolicy = StandardGenPolicy


def collect_vram_diagnostics(device: str | None = None) -> VramDiagnostics:
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
            device_free_mb=0.0,
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
            device_free_mb=0.0,
        )

    from src.common.models.device_manager import DeviceManager

    if device is None:
        device = DeviceManager.resolve_role_device("flux", "cuda:0")
    idx = DeviceManager.cuda_device_index(device)
    if idx is None:
        idx = 0
    idx = max(0, min(idx, torch.cuda.device_count() - 1))

    try:
        props = torch.cuda.get_device_properties(idx)
        total = float(props.total_memory) / (1024**2)
        name = torch.cuda.get_device_name(idx)
    except Exception:
        total = 0.0
        name = "unknown"

    try:
        allocated = float(torch.cuda.memory_allocated(idx) / (1024**2))
        reserved = float(torch.cuda.memory_reserved(idx) / (1024**2))
        max_alloc = float(torch.cuda.max_memory_allocated(idx) / (1024**2))
        try:
            max_res = float(torch.cuda.max_memory_reserved(idx) / (1024**2))
        except Exception:
            max_res = reserved
    except Exception:
        allocated = reserved = max_alloc = max_res = 0.0

    free = max(0.0, total - reserved)
    device_free = 0.0
    try:
        free_b, _total_b = torch.cuda.mem_get_info(idx)
        device_free = float(free_b) / (1024**2)
    except Exception:
        device_free = free
    return VramDiagnostics(
        gpu_name=f"{name}[cuda:{idx}]",
        physical_total_mb=round(total, 1),
        allocated_mb=round(allocated, 1),
        reserved_mb=round(reserved, 1),
        free_mb=round(free, 1),
        max_allocated_mb=round(max_alloc, 1),
        max_reserved_mb=round(max_res, 1),
        cuda_available=True,
        device_free_mb=round(device_free, 1),
    )


def log_vram(tag: str, diag: Optional[VramDiagnostics] = None) -> VramDiagnostics:
    """Emit a single-line VRAM snapshot for Kaggle log scraping."""
    d = diag or collect_vram_diagnostics()
    line = (
        f"[VRAM] {tag} gpu={d.gpu_name} total_mb={d.physical_total_mb:.0f} "
        f"alloc_mb={d.allocated_mb:.0f} reserved_mb={d.reserved_mb:.0f} "
        f"free_mb={d.free_mb:.0f} device_free_mb={d.device_free_mb:.0f} "
        f"peak_alloc_mb={d.max_allocated_mb:.0f} "
        f"peak_reserved_mb={d.max_reserved_mb:.0f}"
    )
    logger.info(line)
    print(line, flush=True)
    return d


def log_nvidia_smi(tag: str = "snapshot") -> str:
    """nvidia-smi utilization / VRAM for GPU 0 and GPU 1 (Kaggle T4×2)."""
    try:
        import subprocess

        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=8,
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        line = f"[GPU SMI] {tag} unavailable: {exc}"
        logger.info(line)
        print(line, flush=True)
        return ""
    for raw in out.strip().splitlines():
        line = (
            f"[GPU SMI] {tag} index,name,util.gpu,util.mem,mem.used,mem.total = "
            f"{raw.strip()}"
        )
        logger.info(line)
        print(line, flush=True)
    return out


def log_runtime_device_report(tag: str = "stack") -> None:
    """PyTorch / CUDA / GPU inventory + nvidia-smi. Never raises."""
    log_vram(tag)
    log_nvidia_smi(tag)
    try:
        import torch
        import diffusers
        import transformers

        bits = [
            f"[FLUX STACK] {tag}",
            f"torch={torch.__version__}",
            f"cuda={torch.version.cuda}",
            f"cuda_available={torch.cuda.is_available()}",
            f"transformers={getattr(transformers, '__version__', '?')}",
            f"diffusers={getattr(diffusers, '__version__', '?')}",
        ]
        if torch.cuda.is_available():
            bits.append(f"device_count={torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                major, minor = torch.cuda.get_device_capability(i)
                total_mb = torch.cuda.get_device_properties(i).total_memory / (1024**2)
                alloc_mb = torch.cuda.memory_allocated(i) / (1024**2)
                bits.append(
                    f"gpu{i}={torch.cuda.get_device_name(i)} "
                    f"sm={major}.{minor} total_mb={total_mb:.0f} alloc_mb={alloc_mb:.0f}"
                )
        line = " ".join(bits)
        logger.info(line)
        print(line, flush=True)
    except Exception as exc:
        line = f"[FLUX STACK] {tag} inspect failed: {exc}"
        logger.info(line)
        print(line, flush=True)


def inspect_pipeline_quantization(pipeline: Any) -> Dict[str, Any]:
    """Report transformer quantization — NF4 is not implied by the model id."""
    info: Dict[str, Any] = {
        "transformer_class": None,
        "transformer_device": None,
        "transformer_dtype": None,
        "linear4bit_modules": 0,
        "linear_modules": 0,
        "transformer_param_mb": 0.0,
        "vae_param_mb": 0.0,
        "t5_param_mb": 0.0,
        "clip_param_mb": 0.0,
        "t5_device": None,
        "vae_device": None,
        "bnb_nf4": False,
    }
    if pipeline is None:
        return info

    def _param_mb(mod: Any) -> float:
        if mod is None:
            return 0.0
        total = 0
        try:
            for p in mod.parameters():
                try:
                    total += int(p.numel()) * int(p.element_size())
                except Exception:
                    continue
        except Exception:
            return 0.0
        return round(total / (1024**2), 1)

    def _dev_dtype(mod: Any) -> tuple[Any, Any]:
        if mod is None:
            return None, None
        try:
            p = next(mod.parameters())
            return str(p.device), str(p.dtype)
        except Exception:
            return None, None

    tr = getattr(pipeline, "transformer", None)
    if tr is not None:
        info["transformer_class"] = type(tr).__name__
        info["transformer_device"], info["transformer_dtype"] = _dev_dtype(tr)
        info["transformer_param_mb"] = _param_mb(tr)
        n4 = 0
        nlin = 0
        try:
            for m in tr.modules():
                name = type(m).__name__
                if "Linear4bit" in name or "Params4bit" in name:
                    n4 += 1
                elif name == "Linear":
                    nlin += 1
        except Exception:
            pass
        info["linear4bit_modules"] = n4
        info["linear_modules"] = nlin
        info["bnb_nf4"] = n4 > 0
    vae = getattr(pipeline, "vae", None)
    info["vae_param_mb"] = _param_mb(vae)
    info["vae_device"], _ = _dev_dtype(vae)
    te2 = getattr(pipeline, "text_encoder_2", None)
    info["t5_param_mb"] = _param_mb(te2)
    info["t5_device"], _ = _dev_dtype(te2)
    te = getattr(pipeline, "text_encoder", None)
    info["clip_param_mb"] = _param_mb(te)
    line = (
        f"[FLUX QUANT] class={info['transformer_class']} "
        f"device={info['transformer_device']} dtype={info['transformer_dtype']} "
        f"Linear4bit={info['linear4bit_modules']} Linear={info['linear_modules']} "
        f"nf4={info['bnb_nf4']} transformer_mb={info['transformer_param_mb']} "
        f"vae_mb={info['vae_param_mb']} t5_mb={info['t5_param_mb']} "
        f"t5_device={info['t5_device']} vae_device={info['vae_device']}"
    )
    logger.info(line)
    print(line, flush=True)
    return info


def select_hybrid_vram_plan(
    *,
    physical_mb: float,
    offload_strategy: str = "",
    height: int = 512,
    width: int = 512,
) -> HybridVramPlan:
    """T4: GPU-resident transformer, T5/VAE not co-resident during denoise.

    Do NOT enable per-step ``enable_model_cpu_offload`` here — that was the
    5-minute stall. Separate VAE decode from the denoise call so fp32 VAE
    activations do not overlap transformer workspace at ~85% (step 8/8).
    """
    _ = (height, width)
    offload = (offload_strategy or "").strip().lower()
    per_step = offload == "model_cpu_offload"
    high_vram = float(physical_mb or 0.0) >= 14000
    if high_vram and not per_step:
        return HybridVramPlan(
            evict_t5_before_diffusion=True,
            park_vae_during_denoise=True,
            decode_latents_separately=True,
            park_transformer_before_vae=True,
            enable_vae_tiling=True,
            enable_vae_slicing=True,
            per_step_cpu_offload=False,
            reason=(
                "t4_hybrid: transformer GPU-resident during denoise; "
                "park transformer before fp32 VAE decode; T5 evict; VAE CPU in denoise"
            ),
            profile="t4_hybrid_gpu_resident",
        )
    return HybridVramPlan(
        evict_t5_before_diffusion=True,
        park_vae_during_denoise=False,
        decode_latents_separately=False,
        park_transformer_before_vae=False,
        enable_vae_tiling=True,
        enable_vae_slicing=True,
        per_step_cpu_offload=offload == "model_cpu_offload",
        reason="low_vram_or_offload: keep existing pipeline() image decode",
        profile="low_vram_offload",
    )


def cleanup_cuda_after_failure() -> None:
    """Recover allocator after OOM. Not a per-step hook."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
    except Exception:
        pass


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
    # Standard must NOT inherit FLUX_PRODUCTION_SIZE (Production-only 700+ target).
    forced_res = _env_int("FLUX_GENERATION_RESOLUTION")
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
    # Standard target: 712×712 / 8 steps / guidance 3.0, GPU-resident NF4.
    # Do NOT auto-upgrade Standard to 768×12 (that is Production).
    # Offload only when the operator explicitly sets FLUX_MODEL_CPU_OFFLOAD=true.
    prefer_offload = False if offload_env is None else bool(offload_env)

    # Opt-in 768 Standard only via FLUX_ALLOW_HIGH_RES or forced 768+.
    high_res_ok = False
    if forced_res and forced_res >= 768:
        high_res_ok = True
    elif allow_high_res is True:
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
        res = forced_res or 712
        steps = forced_steps or 8
        profile = "standard_t4_safe"
        reason = (
            f"t4_standard_712x8 free={free:.0f}MB phys={phys:.0f}MB "
            f"gpu_resident_hint={gpu_resident}"
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


def select_production_generation_policy(
    *,
    physical_mb: Optional[float] = None,
    free_mb: Optional[float] = None,
    offload_strategy: Optional[str] = None,
    yaml_height: int = 768,
    yaml_steps: int = 12,
    yaml_guidance: float = 3.0,
) -> ProductionGenPolicy:
    """
    Production / High-Quality policy.

    Locked target: 768×768 / 12 steps / guidance 3.0.

    Production must NOT inherit Standard knobs:
    FLUX_GENERATION_RESOLUTION, FLUX_GENERATION_STEPS, FLUX_STANDARD_STEPS.
    VRAM size must NOT silently clamp Production to 512×3.
    Override only via FLUX_PRODUCTION_RESOLUTION / FLUX_PRODUCTION_SIZE /
    FLUX_PRODUCTION_STEPS / FLUX_PRODUCTION_GUIDANCE.
    """
    from src.features.custom_generator.inference.flux_inference import (
        ALLOWED_FLUX_GENERATION_RESOLUTIONS,
        DEFAULT_KAGGLE_PRODUCTION_GUIDANCE,
        DEFAULT_KAGGLE_PRODUCTION_RESOLUTION,
        DEFAULT_KAGGLE_PRODUCTION_STEPS,
        resolve_flux_production_guidance,
        resolve_flux_production_resolution,
        resolve_flux_production_steps,
    )

    diag = collect_vram_diagnostics()
    phys = float(physical_mb if physical_mb is not None else diag.physical_total_mb)
    free = float(free_mb if free_mb is not None else diag.free_mb)
    offload = (offload_strategy or "").strip().lower()
    _ = offload  # reserved for diagnostics / prefer_offload below

    offload_env = _env_truthy("FLUX_MODEL_CPU_OFFLOAD")
    if offload_env is None:
        prefer_offload = phys > 0 and phys < 14000
    else:
        prefer_offload = bool(offload_env)

    # yaml_* are hints only when they already match the locked Production target.
    height_default = (
        int(yaml_height)
        if int(yaml_height) >= DEFAULT_KAGGLE_PRODUCTION_RESOLUTION
        else DEFAULT_KAGGLE_PRODUCTION_RESOLUTION
    )
    steps_default = (
        int(yaml_steps)
        if int(yaml_steps) >= DEFAULT_KAGGLE_PRODUCTION_STEPS
        else DEFAULT_KAGGLE_PRODUCTION_STEPS
    )
    guidance_default = (
        float(yaml_guidance)
        if abs(float(yaml_guidance) - DEFAULT_KAGGLE_PRODUCTION_GUIDANCE) <= 0.05
        else DEFAULT_KAGGLE_PRODUCTION_GUIDANCE
    )

    res = resolve_flux_production_resolution(default=height_default)
    if res not in ALLOWED_FLUX_GENERATION_RESOLUTIONS:
        res = DEFAULT_KAGGLE_PRODUCTION_RESOLUTION
    base_steps = resolve_flux_production_steps(default=steps_default)
    guidance = resolve_flux_production_guidance(default=guidance_default)

    profile = "production_locked_768"
    reason = (
        f"production_locked_768 free={free:.0f}MB phys={phys:.0f}MB "
        f"res={res} steps={base_steps} guidance={guidance} "
        "(isolated from FLUX_GENERATION_RESOLUTION / FLUX_STANDARD_STEPS)"
    )

    return ProductionGenPolicy(
        height=int(res),
        width=int(res),
        num_inference_steps=int(base_steps),
        guidance_scale=float(guidance),
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
    if area >= 712 * 712 and int(height) == 712 and int(width) == 712:
        # Standard 712 target: never silently demote to 512 or cut steps.
        return None
    if area >= 768 * 768:
        return {
            "height": 720,
            "width": 720,
            "num_inference_steps": max(8, min(int(num_inference_steps), 10)),
        }
    if area >= 720 * 720:
        return {
            "height": 704,
            "width": 704,
            "num_inference_steps": max(8, min(int(num_inference_steps), 10)),
        }
    if area >= 704 * 704:
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
