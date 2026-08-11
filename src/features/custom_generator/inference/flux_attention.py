"""Memory-efficient SDPA helpers for FLUX Kontext on 16GB-class GPUs.

Goal: prevent PyTorch SDPA MATH backend from materializing an O(N^2) attention
workspace (~6.19 GiB at Kontext 1024 / N~8320) while keeping NF4 + FluxKontext
unchanged.

Compatible with:
  - PyTorch 2.x ``torch.nn.attention.sdpa_kernel`` (preferred)
  - Older ``torch.backends.cuda.sdp_kernel`` context
  - Optional Diffusers ``transformer.set_attention_backend("_native_efficient")``
    when present (0.39+); skipped cleanly on 0.37.x
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Dict, Iterator, List, Optional

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

logger = logging.getLogger("fabricvision.garment_generation.attention")


class NoSupportedEfficientAttention(RuntimeError):
    """Raised when MATH is the only available SDPA path (unsafe for 1024 Kontext)."""


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


def _build_nn_attention_backends() -> List[Any]:
    """Prefer flash then mem-efficient; never include MATH."""
    from torch.nn.attention import SDPBackend

    backends: List[Any] = []
    # Flash first when the global toggle is on; T4 may still reject at runtime,
    # in which case PyTorch tries the next listed backend (efficient).
    if hasattr(SDPBackend, "FLASH_ATTENTION"):
        backends.append(SDPBackend.FLASH_ATTENTION)
    if hasattr(SDPBackend, "EFFICIENT_ATTENTION"):
        backends.append(SDPBackend.EFFICIENT_ATTENTION)
    return backends


def memory_efficient_attention_context() -> contextlib.AbstractContextManager:
    """
    Context manager that disables MATH SDPA for the wrapped region.

    Preferred API: ``torch.nn.attention.sdpa_kernel``.
    Fallback: ``torch.backends.cuda.sdp_kernel(enable_math=False, ...)``.

    WARNING: Do not wrap the entire FluxKontext ``pipeline(...)`` call with this.
    VAE encode/decode on T4 often requires MATH (head dims incompatible with
    flash/mem-efficient). Use ``transformer_only_memory_efficient_attention``.
    """
    if torch is None or not torch.cuda.is_available():
        return contextlib.nullcontext()

    # Option B1: modern API (PyTorch 2.3+)
    if hasattr(torch.nn, "attention") and hasattr(torch.nn.attention, "sdpa_kernel"):
        backends = _build_nn_attention_backends()
        if not backends:
            raise NoSupportedEfficientAttention(
                "torch.nn.attention.SDPBackend has neither FLASH_ATTENTION nor "
                "EFFICIENT_ATTENTION; cannot disable MATH safely."
            )
        return torch.nn.attention.sdpa_kernel(backends)

    # Option B2: older context manager
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


@contextlib.contextmanager
def transformer_only_memory_efficient_attention(pipeline: Any) -> Iterator[None]:
    """
    Restrict MATH-disabled SDPA to ``pipeline.transformer.forward`` only.

    VAE encode (prepare_latents) and VAE decode stay on default PyTorch SDPA
    (MATH allowed). Flux transformer denoise prefers flash/mem-efficient.
    """
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None or not hasattr(transformer, "forward"):
        # No transformer to wrap — do not apply a global MATH=False context.
        yield
        return

    original_forward = transformer.forward

    def _forward_with_efficient_sdpa(*args: Any, **kwargs: Any) -> Any:
        with memory_efficient_attention_context():
            return original_forward(*args, **kwargs)

    transformer.forward = _forward_with_efficient_sdpa  # type: ignore[method-assign]
    try:
        yield
    finally:
        transformer.forward = original_forward  # type: ignore[method-assign]


def _try_diffusers_native_efficient(pipeline: Any) -> Optional[str]:
    """
    Option A: Diffusers transformer.set_attention_backend when present.

    Returns backend name string on success, None if API missing / fails.
    Does not import ``set_attention_backend`` from attention_dispatch (absent on 0.37).
    """
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None or not hasattr(transformer, "set_attention_backend"):
        return None
    # Prefer memory-efficient native SDPA; skip flash-only names that often
    # require Ampere+ and external packages.
    candidates = ("_native_efficient", "native")
    last_err: Optional[BaseException] = None
    for name in candidates:
        try:
            transformer.set_attention_backend(name)
            logger.info("[FLUX] Diffusers transformer attention backend: %s", name)
            return name
        except Exception as exc:  # noqa: BLE001 — probe installed Diffusers only
            last_err = exc
            continue
    if last_err is not None:
        logger.warning(
            "[FLUX] Diffusers set_attention_backend unavailable/failed (%s); "
            "using PyTorch SDPA context instead",
            last_err,
        )
    return None


def configure_memory_efficient_attention(pipeline: Any) -> Dict[str, Any]:
    """
    Configure Flux *transformer* attention for memory-efficient SDPA.

    MATH is disabled only inside ``transformer.forward`` (via
    ``transformer_only_memory_efficient_attention``). VAE keeps default SDPA
    (MATH allowed). Does not install xFormers or FlashAttention packages.
    """
    flags_before = probe_sdpa_flags()
    diag: Dict[str, Any] = {
        "attention_backend_requested": "memory_efficient",
        "diffusers_attention_backend": None,
        "pytorch_context_api": None,
        # Scoped: transformer denoise only (not global / not VAE).
        "attention_math_enabled": False,
        "attention_flash_enabled": flags_before.get("flash_sdp_enabled"),
        "attention_mem_efficient_enabled": flags_before.get("mem_efficient_sdp_enabled"),
        "attention_backend_effective": None,
        "flux_transformer_attention": "memory_efficient_requested",
        "flux_transformer_math": False,
        "vae_attention": "normal/default",
        "attention_scope": "transformer_forward_only",
        "attention_config_ok": False,
        "error": None,
    }

    if torch is None:
        diag["error"] = "torch_unavailable"
        return diag
    if not flags_before.get("has_sdpa"):
        diag["error"] = "NO_SUPPORTED_EFFICIENT_ATTENTION_BACKEND"
        return diag

    # Option A (optional): Diffusers Flux transformer backend when API exists.
    diffusers_backend = _try_diffusers_native_efficient(pipeline)
    diag["diffusers_attention_backend"] = diffusers_backend

    # Option B (required for MATH suppression on T4): PyTorch SDPA context API.
    try:
        # Validate we can enter/exit the context (does not run a forward).
        with memory_efficient_attention_context():
            flags_inside = probe_sdpa_flags()
            # Inside sdpa_kernel, PyTorch reports only allowed backends as enabled.
            diag["attention_math_enabled"] = bool(flags_inside.get("math_sdp_enabled"))
            diag["attention_flash_enabled"] = flags_inside.get("flash_sdp_enabled")
            diag["attention_mem_efficient_enabled"] = flags_inside.get(
                "mem_efficient_sdp_enabled"
            )
        if diag["attention_math_enabled"] is True:
            # Context failed to suppress MATH — treat as unsupported.
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

            if hasattr(SDPBackend, "FLASH_ATTENTION"):
                backends.append("FLASH_ATTENTION")
            if hasattr(SDPBackend, "EFFICIENT_ATTENTION"):
                backends.append("EFFICIENT_ATTENTION")
        except Exception:
            backends = ["EFFICIENT_ATTENTION"]
        # Honest effective string: requested context backends, not a claimed kernel.
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

    if diffusers_backend:
        diag["attention_backend_effective"] = (
            f"diffusers:{diffusers_backend}+{diag['attention_backend_effective']}"
        )

    # Require that mem-efficient and/or flash can be enabled on this build.
    mem_ok = bool(diag.get("attention_mem_efficient_enabled"))
    flash_ok = bool(diag.get("attention_flash_enabled"))
    if not mem_ok and not flash_ok:
        # Inside context both might briefly report True for allowed ones; if both
        # False, the GPU/build cannot serve an efficient path.
        diag["error"] = "NO_SUPPORTED_EFFICIENT_ATTENTION_BACKEND"
        diag["attention_config_ok"] = False
        return diag

    diag["attention_config_ok"] = True
    logger.info(
        "[FLUX] Attention config: requested=memory_efficient math=%s flash=%s "
        "mem_efficient=%s effective=%s",
        diag["attention_math_enabled"],
        diag["attention_flash_enabled"],
        diag["attention_mem_efficient_enabled"],
        diag["attention_backend_effective"],
    )
    return diag


def format_attention_diag_lines(diag: Dict[str, Any]) -> List[str]:
    """Smoke-friendly KEY=value lines (honest transformer vs VAE scope)."""
    return [
        f"ATTENTION_BACKEND_REQUESTED={diag.get('attention_backend_requested')}",
        f"ATTENTION_SCOPE={diag.get('attention_scope', 'transformer_forward_only')}",
        f"FLUX_TRANSFORMER_ATTENTION={diag.get('flux_transformer_attention', 'memory_efficient_requested')}",
        f"FLUX_TRANSFORMER_MATH={diag.get('flux_transformer_math', False)}",
        # Legacy key: MATH disabled for transformer only, not globally.
        f"ATTENTION_MATH_ENABLED={diag.get('attention_math_enabled')}",
        f"ATTENTION_FLASH_ENABLED={diag.get('attention_flash_enabled')}",
        f"ATTENTION_MEM_EFFICIENT_ENABLED={diag.get('attention_mem_efficient_enabled')}",
        f"ATTENTION_BACKEND_EFFECTIVE={diag.get('attention_backend_effective')}",
        f"VAE_ATTENTION={diag.get('vae_attention', 'normal/default')}",
        f"ATTENTION_DIFFUSERS_BACKEND={diag.get('diffusers_attention_backend')}",
        f"ATTENTION_PYTORCH_CONTEXT_API={diag.get('pytorch_context_api')}",
        f"ATTENTION_CONFIG_OK={diag.get('attention_config_ok')}",
    ]
