"""Memory-efficient SDPA helpers for FLUX Kontext on 16GB-class GPUs.

Goal: keep NF4 + FluxKontext unchanged while making transformer attention work on
Tesla T4 (sm_75) where BF16 + efficient SDPA is unavailable and Flash rejects
large head dims.

Architecture:
  - VAE encode/decode: default PyTorch SDPA (MATH allowed)
  - Flux transformer.forward only: efficient SDPA preferred, MATH disabled
  - On GPUs where BF16 efficient SDPA fails: cast Q/K/V to FP16 for the SDPA
    call only, restore output dtype (do NOT convert the whole model)

Compatible with Diffusers 0.37.x (direct F.sdpa) and 0.39.x (dispatch /
``_native_efficient`` which still calls ``F.scaled_dot_product_attention``).
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore

logger = logging.getLogger("fabricvision.garment_generation.attention")


class NoSupportedEfficientAttention(RuntimeError):
    """Raised when no safe transformer attention path is available."""


# ---------------------------------------------------------------------------
# Hardware / capability probes
# ---------------------------------------------------------------------------


def probe_cuda_capability() -> Optional[Tuple[int, int]]:
    if torch is None or not torch.cuda.is_available():
        return None
    try:
        return tuple(torch.cuda.get_device_capability(0))  # type: ignore[return-value]
    except Exception:
        return None


def probe_hardware() -> Dict[str, Any]:
    """Collect honest hardware facts (no fabricated values)."""
    out: Dict[str, Any] = {
        "torch_available": torch is not None,
        "cuda_available": False,
        "gpu_name": None,
        "cuda_compute_capability": None,
        "cuda_compute_capability_str": None,
        "vram_total_mb": None,
        "torch_version": None,
        "cuda_version": None,
    }
    if torch is None:
        return out
    out["torch_version"] = getattr(torch, "__version__", None)
    out["cuda_available"] = bool(torch.cuda.is_available())
    out["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
    if not out["cuda_available"]:
        return out
    try:
        out["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        out["gpu_name"] = None
    cap = probe_cuda_capability()
    if cap is not None:
        out["cuda_compute_capability"] = cap
        out["cuda_compute_capability_str"] = f"{cap[0]}.{cap[1]}"
    try:
        out["vram_total_mb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**2), 1
        )
    except Exception:
        out["vram_total_mb"] = None
    return out


def probe_sdpa_flags() -> Dict[str, Any]:
    """Read current CUDA SDPA enable flags. Does not claim which kernel will run."""
    out: Dict[str, Any] = {
        "torch_available": torch is not None,
        "cuda_available": False,
        "has_sdpa": False,
        "has_nn_attention_sdpa_kernel": False,
        "has_backends_sdp_kernel": False,
        "flash_sdp_enabled": None,
        "mem_efficient_sdp_enabled": None,
        "math_sdp_enabled": None,
        "cudnn_sdp_enabled": None,
    }
    if torch is None:
        return out
    out["has_sdpa"] = hasattr(torch.nn.functional, "scaled_dot_product_attention")
    out["has_nn_attention_sdpa_kernel"] = hasattr(torch.nn, "attention") and hasattr(
        torch.nn.attention, "sdpa_kernel"
    )
    out["has_backends_sdp_kernel"] = hasattr(torch.backends, "cuda") and hasattr(
        torch.backends.cuda, "sdp_kernel"
    )
    out["cuda_available"] = bool(torch.cuda.is_available())
    if not out["cuda_available"]:
        return out
    try:
        out["flash_sdp_enabled"] = bool(torch.backends.cuda.flash_sdp_enabled())
    except Exception:
        out["flash_sdp_enabled"] = None
    try:
        out["mem_efficient_sdp_enabled"] = bool(
            torch.backends.cuda.mem_efficient_sdp_enabled()
        )
    except Exception:
        out["mem_efficient_sdp_enabled"] = None
    try:
        out["math_sdp_enabled"] = bool(torch.backends.cuda.math_sdp_enabled())
    except Exception:
        out["math_sdp_enabled"] = None
    try:
        if hasattr(torch.backends.cuda, "cudnn_sdp_enabled"):
            out["cudnn_sdp_enabled"] = bool(torch.backends.cuda.cudnn_sdp_enabled())
    except Exception:
        out["cudnn_sdp_enabled"] = None
    return out


def _try_efficient_sdpa(dtype: "torch.dtype", head_dim: int = 64) -> bool:
    """Runtime probe: does EFFICIENT SDPA accept this dtype on the current GPU?"""
    if torch is None or not torch.cuda.is_available() or F is None:
        return False
    try:
        q = torch.randn(1, 1, 8, head_dim, device="cuda", dtype=dtype)
        k = q.clone()
        v = q.clone()
        if hasattr(torch.nn, "attention") and hasattr(torch.nn.attention, "sdpa_kernel"):
            from torch.nn.attention import SDPBackend

            with torch.nn.attention.sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                _ = F.scaled_dot_product_attention(q, k, v)
        elif hasattr(torch.backends.cuda, "sdp_kernel"):
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=False, enable_mem_efficient=True
            ):
                _ = F.scaled_dot_product_attention(q, k, v)
        else:
            return False
        return True
    except Exception:
        return False


def needs_fp16_attention_operands(
    *,
    force: Optional[bool] = None,
    capability: Optional[Tuple[int, int]] = None,
) -> bool:
    """
    True when BF16 Q/K/V should be cast to FP16 for efficient SDPA.

    Capability heuristic: pre-Ampere (major < 8) commonly lacks BF16 efficient
    kernels (e.g. Tesla T4 sm_75). Ampere+ keeps BF16 when the runtime probe
    succeeds. ``force`` overrides for unit tests.
    """
    if force is not None:
        return bool(force)
    if torch is None or not torch.cuda.is_available():
        return False
    cap = capability if capability is not None else probe_cuda_capability()
    if cap is not None and cap[0] < 8:
        return True
    # Ampere+: keep BF16 if efficient accepts it; otherwise cast.
    if _try_efficient_sdpa(torch.bfloat16):
        return False
    return True


def probe_efficient_kernel_available() -> bool:
    """True if EFFICIENT SDPA works for fp16 and/or bf16 on this GPU."""
    if torch is None or not torch.cuda.is_available():
        return False
    return _try_efficient_sdpa(torch.float16) or _try_efficient_sdpa(torch.bfloat16)


# ---------------------------------------------------------------------------
# SDPA contexts (transformer-only; never wrap whole pipeline)
# ---------------------------------------------------------------------------


def _build_nn_attention_backends() -> List[Any]:
    """
    Prefer memory-efficient; optionally allow Flash as secondary.

    Flash is NOT relied on for FLUX (head dims often > 256). Efficient is primary.
    MATH is never included here.
    """
    from torch.nn.attention import SDPBackend

    backends: List[Any] = []
    if hasattr(SDPBackend, "EFFICIENT_ATTENTION"):
        backends.append(SDPBackend.EFFICIENT_ATTENTION)
    # Flash as optional secondary — PyTorch may skip it when head_dim is too large.
    if hasattr(SDPBackend, "FLASH_ATTENTION"):
        backends.append(SDPBackend.FLASH_ATTENTION)
    return backends


def memory_efficient_attention_context() -> contextlib.AbstractContextManager:
    """
    Context manager that disables MATH SDPA for the wrapped region.

    WARNING: Do not wrap the entire FluxKontext ``pipeline(...)`` call with this.
    Use ``transformer_only_memory_efficient_attention``.
    """
    if torch is None or not torch.cuda.is_available():
        return contextlib.nullcontext()

    if hasattr(torch.nn, "attention") and hasattr(torch.nn.attention, "sdpa_kernel"):
        backends = _build_nn_attention_backends()
        if not backends:
            raise NoSupportedEfficientAttention(
                "torch.nn.attention.SDPBackend has neither EFFICIENT_ATTENTION nor "
                "FLASH_ATTENTION; cannot disable MATH safely."
            )
        return torch.nn.attention.sdpa_kernel(backends)

    if hasattr(torch.backends.cuda, "sdp_kernel"):
        return torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_mem_efficient=True,
            enable_math=False,
        )

    raise NoSupportedEfficientAttention(
        "No supported PyTorch SDPA kernel context API "
        "(need torch.nn.attention.sdpa_kernel or torch.backends.cuda.sdp_kernel). "
        "NO_SUPPORTED_EFFICIENT_ATTENTION_BACKEND"
    )


def _make_compat_sdpa(
    original: Callable[..., Any],
    *,
    cast_bf16_to_fp16: bool,
    state: Dict[str, Any],
) -> Callable[..., Any]:
    """Wrap F.scaled_dot_product_attention for dtype-compatible efficient SDPA."""

    def _compat_sdpa(query, key, value, *args, **kwargs):  # noqa: ANN001
        state["attention_device"] = str(getattr(query, "device", None))
        state["attention_dtype_requested"] = str(getattr(query, "dtype", None))
        q, k, v = query, key, value
        cast_used = False
        if (
            cast_bf16_to_fp16
            and torch is not None
            and getattr(query, "dtype", None) == torch.bfloat16
        ):
            q = query.to(dtype=torch.float16)
            k = key.to(dtype=torch.float16)
            v = value.to(dtype=torch.float16)
            cast_used = True
        state["attention_qkv_dtype"] = str(getattr(q, "dtype", None))
        state["attention_dtype_effective"] = state["attention_qkv_dtype"]
        state["attention_fallback_used"] = cast_used
        if cast_used:
            state["attention_fallback_reason"] = (
                "bf16_efficient_sdpa_unavailable_cast_qkv_to_fp16"
            )
        else:
            state["attention_fallback_reason"] = None

        out = original(q, k, v, *args, **kwargs)
        if cast_used and out is not None and hasattr(out, "to"):
            out = out.to(dtype=query.dtype)
        return out

    return _compat_sdpa


@contextlib.contextmanager
def transformer_only_memory_efficient_attention(
    pipeline: Any,
    *,
    force_fp16_cast: Optional[bool] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Restrict efficient / MATH-disabled SDPA to ``pipeline.transformer.forward``.

    Temporarily wraps ``F.scaled_dot_product_attention`` ONLY while inside
    transformer.forward so Diffusers ``_native_efficient`` / Flux processors get
    FP16 Q/K/V on T4-class GPUs. Restores the original function in ``finally``.
    Yields a mutable runtime state dict for diagnostics.
    """
    state: Dict[str, Any] = {
        "attention_scope": "transformer_forward_only",
        "vae_attention": "normal/default",
        "attention_dtype_requested": None,
        "attention_dtype_effective": None,
        "attention_qkv_dtype": None,
        "attention_device": None,
        "attention_fallback_used": False,
        "attention_fallback_reason": None,
        "cast_bf16_to_fp16": False,
        "sdpa_wrapped": False,
        "forward_restored": False,
        "sdpa_restored": False,
    }

    transformer = getattr(pipeline, "transformer", None)
    if transformer is None or not hasattr(transformer, "forward"):
        yield state
        return

    cast = needs_fp16_attention_operands(force=force_fp16_cast)
    state["cast_bf16_to_fp16"] = cast

    original_forward = transformer.forward
    original_sdpa = F.scaled_dot_product_attention if F is not None else None

    def _forward_with_efficient_sdpa(*args: Any, **kwargs: Any) -> Any:
        if F is None or original_sdpa is None:
            return original_forward(*args, **kwargs)
        wrapped = _make_compat_sdpa(
            original_sdpa, cast_bf16_to_fp16=cast, state=state
        )
        F.scaled_dot_product_attention = wrapped  # type: ignore[assignment]
        state["sdpa_wrapped"] = True
        try:
            with memory_efficient_attention_context():
                return original_forward(*args, **kwargs)
        finally:
            F.scaled_dot_product_attention = original_sdpa  # type: ignore[assignment]
            state["sdpa_restored"] = True

    transformer.forward = _forward_with_efficient_sdpa  # type: ignore[method-assign]
    try:
        yield state
    finally:
        # Remove instance override so the class ``forward`` is used again.
        try:
            delattr(transformer, "forward")
        except Exception:
            transformer.forward = original_forward  # type: ignore[method-assign]
        state["forward_restored"] = True
        if F is not None and original_sdpa is not None:
            F.scaled_dot_product_attention = original_sdpa  # type: ignore[assignment]
            state["sdpa_restored"] = True


def _try_diffusers_native_efficient(pipeline: Any) -> Optional[str]:
    """
    Option A: Diffusers transformer.set_attention_backend when present.

    Does not import ``set_attention_backend`` from attention_dispatch (absent on 0.37).
    """
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None or not hasattr(transformer, "set_attention_backend"):
        return None
    # Prefer memory-efficient native SDPA. Flash is not suitable for large head dims.
    candidates = ("_native_efficient", "native")
    last_err: Optional[BaseException] = None
    for name in candidates:
        try:
            transformer.set_attention_backend(name)
            logger.info("[FLUX] Diffusers transformer attention backend: %s", name)
            return name
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    if last_err is not None:
        logger.warning(
            "[FLUX] Diffusers set_attention_backend unavailable/failed (%s); "
            "using PyTorch SDPA context instead",
            last_err,
        )
    return None


def configure_memory_efficient_attention(
    pipeline: Any,
    *,
    force_fp16_cast: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Configure Flux *transformer* attention for T4-compatible efficient SDPA.

    MATH is disabled only inside ``transformer.forward``. VAE keeps default SDPA.
    """
    hw = probe_hardware()
    flags_before = probe_sdpa_flags()
    cast = needs_fp16_attention_operands(force=force_fp16_cast)
    efficient_ok = probe_efficient_kernel_available()

    diag: Dict[str, Any] = {
        "attention_backend_requested": "memory_efficient",
        "diffusers_attention_backend": None,
        "pytorch_context_api": None,
        "attention_math_enabled": False,
        "attention_flash_enabled": flags_before.get("flash_sdp_enabled"),
        "attention_mem_efficient_enabled": flags_before.get("mem_efficient_sdp_enabled"),
        "attention_backend_effective": None,
        "flux_transformer_attention": "memory_efficient_requested",
        "flux_transformer_math": False,
        "vae_attention": "normal/default",
        "attention_scope": "transformer_forward_only",
        "attention_dtype_requested": "bfloat16",
        "attention_dtype_effective": "float16" if cast else "bfloat16",
        "attention_qkv_dtype": "float16" if cast else "bfloat16",
        "attention_device": "cuda" if hw.get("cuda_available") else "cpu",
        "cuda_compute_capability": hw.get("cuda_compute_capability_str"),
        "gpu_name": hw.get("gpu_name"),
        "attention_efficient_kernel_available": efficient_ok,
        "attention_fallback_used": cast,
        "attention_fallback_reason": (
            "bf16_efficient_sdpa_unavailable_cast_qkv_to_fp16" if cast else None
        ),
        "attention_config_ok": False,
        "error": None,
    }

    if torch is None:
        diag["error"] = "torch_unavailable"
        return diag
    if not flags_before.get("has_sdpa"):
        diag["error"] = "NO_SUPPORTED_EFFICIENT_ATTENTION_BACKEND"
        return diag

    diffusers_backend = _try_diffusers_native_efficient(pipeline)
    diag["diffusers_attention_backend"] = diffusers_backend

    try:
        with memory_efficient_attention_context():
            flags_inside = probe_sdpa_flags()
            diag["attention_math_enabled"] = bool(flags_inside.get("math_sdp_enabled"))
            diag["attention_flash_enabled"] = flags_inside.get("flash_sdp_enabled")
            diag["attention_mem_efficient_enabled"] = flags_inside.get(
                "mem_efficient_sdp_enabled"
            )
        if diag["attention_math_enabled"] is True:
            diag["error"] = "MATH_STILL_ENABLED_INSIDE_CONTEXT"
            diag["attention_backend_effective"] = (
                "configured_flags_only;math_not_disabled"
            )
            return diag
    except NoSupportedEfficientAttention as exc:
        diag["error"] = "NO_SUPPORTED_EFFICIENT_ATTENTION_BACKEND"
        diag["attention_backend_effective"] = str(exc)
        return diag
    except Exception as exc:  # noqa: BLE001
        diag["error"] = f"{type(exc).__name__}:{exc}"
        return diag

    if flags_before.get("has_nn_attention_sdpa_kernel"):
        diag["pytorch_context_api"] = "torch.nn.attention.sdpa_kernel"
        backends = []
        try:
            from torch.nn.attention import SDPBackend

            if hasattr(SDPBackend, "EFFICIENT_ATTENTION"):
                backends.append("EFFICIENT_ATTENTION")
            if hasattr(SDPBackend, "FLASH_ATTENTION"):
                backends.append("FLASH_ATTENTION")
        except Exception:
            backends = ["EFFICIENT_ATTENTION"]
        diag["attention_backend_effective"] = (
            "sdpa_kernel(" + ",".join(backends) + ";math=disabled)"
        )
    elif flags_before.get("has_backends_sdp_kernel"):
        diag["pytorch_context_api"] = "torch.backends.cuda.sdp_kernel"
        diag["attention_backend_effective"] = (
            "backends.cuda.sdp_kernel(flash=True,mem_efficient=True,math=False)"
        )
    else:
        diag["error"] = "NO_SUPPORTED_EFFICIENT_ATTENTION_BACKEND"
        return diag

    if cast:
        diag["attention_backend_effective"] = (
            f"{diag['attention_backend_effective']}+qkv_fp16_cast"
        )

    if diffusers_backend:
        diag["attention_backend_effective"] = (
            f"diffusers:{diffusers_backend}+{diag['attention_backend_effective']}"
        )

    mem_ok = bool(diag.get("attention_mem_efficient_enabled"))
    flash_ok = bool(diag.get("attention_flash_enabled"))
    if not mem_ok and not flash_ok and not efficient_ok:
        diag["error"] = "NO_SUPPORTED_EFFICIENT_ATTENTION_BACKEND"
        diag["attention_config_ok"] = False
        return diag

    # Require efficient path to be usable for at least fp16 (T4 target).
    if torch.cuda.is_available() and not _try_efficient_sdpa(torch.float16):
        diag["error"] = "NO_SUPPORTED_EFFICIENT_ATTENTION_BACKEND"
        diag["attention_efficient_kernel_available"] = False
        return diag

    diag["attention_config_ok"] = True
    logger.info(
        "[FLUX] Attention config: scope=transformer_forward_only math=%s "
        "cast_bf16_to_fp16=%s effective=%s capability=%s",
        diag["attention_math_enabled"],
        cast,
        diag["attention_backend_effective"],
        diag["cuda_compute_capability"],
    )
    return diag


def format_attention_diag_lines(diag: Dict[str, Any]) -> List[str]:
    """Smoke-friendly KEY=value lines (honest transformer vs VAE scope)."""
    return [
        f"ATTENTION_BACKEND_REQUESTED={diag.get('attention_backend_requested')}",
        f"ATTENTION_SCOPE={diag.get('attention_scope', 'transformer_forward_only')}",
        f"FLUX_TRANSFORMER_ATTENTION={diag.get('flux_transformer_attention', 'memory_efficient_requested')}",
        f"FLUX_TRANSFORMER_MATH={diag.get('flux_transformer_math', False)}",
        f"ATTENTION_MATH_ENABLED={diag.get('attention_math_enabled')}",
        f"ATTENTION_FLASH_ENABLED={diag.get('attention_flash_enabled')}",
        f"ATTENTION_MEM_EFFICIENT_ENABLED={diag.get('attention_mem_efficient_enabled')}",
        f"ATTENTION_BACKEND_EFFECTIVE={diag.get('attention_backend_effective')}",
        f"VAE_ATTENTION={diag.get('vae_attention', 'normal/default')}",
        f"ATTENTION_DTYPE_REQUESTED={diag.get('attention_dtype_requested')}",
        f"ATTENTION_DTYPE_EFFECTIVE={diag.get('attention_dtype_effective')}",
        f"ATTENTION_QKV_DTYPE={diag.get('attention_qkv_dtype')}",
        f"ATTENTION_DEVICE={diag.get('attention_device')}",
        f"CUDA_COMPUTE_CAPABILITY={diag.get('cuda_compute_capability')}",
        f"ATTENTION_EFFICIENT_KERNEL_AVAILABLE={diag.get('attention_efficient_kernel_available')}",
        f"ATTENTION_FALLBACK_USED={diag.get('attention_fallback_used')}",
        f"ATTENTION_FALLBACK_REASON={diag.get('attention_fallback_reason')}",
        f"ATTENTION_DIFFUSERS_BACKEND={diag.get('diffusers_attention_backend')}",
        f"ATTENTION_PYTORCH_CONTEXT_API={diag.get('pytorch_context_api')}",
        f"ATTENTION_CONFIG_OK={diag.get('attention_config_ok')}",
    ]


def merge_runtime_attention_state(
    diag: Dict[str, Any], runtime: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Overlay runtime SDPA wrap observations onto configure() diagnostics."""
    if not runtime:
        return diag
    out = dict(diag)
    for key in (
        "attention_dtype_requested",
        "attention_dtype_effective",
        "attention_qkv_dtype",
        "attention_device",
        "attention_fallback_used",
        "attention_fallback_reason",
    ):
        if runtime.get(key) is not None:
            out[key] = runtime[key]
    return out
