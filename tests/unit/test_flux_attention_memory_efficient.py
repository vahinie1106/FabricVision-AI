"""Unit tests for FLUX memory-efficient / T4-compatible SDPA helpers."""

from __future__ import annotations

from src.features.custom_generator.inference.flux_attention import (
    configure_memory_efficient_attention,
    format_attention_diag_lines,
    merge_runtime_attention_state,
    needs_fp16_attention_operands,
    probe_hardware,
    probe_sdpa_flags,
    transformer_only_memory_efficient_attention,
)


def flags_has_sdpa_context() -> bool:
    flags = probe_sdpa_flags()
    return bool(
        flags.get("has_nn_attention_sdpa_kernel") or flags.get("has_backends_sdp_kernel")
    )


def test_probe_sdpa_flags_returns_expected_keys():
    flags = probe_sdpa_flags()
    for key in (
        "torch_available",
        "has_sdpa",
        "has_nn_attention_sdpa_kernel",
        "has_backends_sdp_kernel",
        "flash_sdp_enabled",
        "mem_efficient_sdp_enabled",
        "math_sdp_enabled",
    ):
        assert key in flags


def test_probe_hardware_keys():
    hw = probe_hardware()
    for key in (
        "torch_available",
        "cuda_available",
        "gpu_name",
        "cuda_compute_capability",
        "cuda_compute_capability_str",
        "torch_version",
    ):
        assert key in hw


def test_needs_fp16_cast_capability_injection():
    """Pre-Ampere (e.g. T4 7.5) must request FP16 Q/K/V cast; Ampere+ may not."""
    assert needs_fp16_attention_operands(force=True) is True
    assert needs_fp16_attention_operands(force=False) is False
    assert needs_fp16_attention_operands(capability=(7, 5), force=None) is True
    # Ampere capability alone does not force cast (runtime probe may still decide).
    # force=False proves override works without requiring a T4.


def test_configure_memory_efficient_attention_disables_math_when_cuda():
    """On CUDA builds, configuration must report MATH disabled inside the context."""
    import torch

    class _DummyPipe:
        transformer = None

    diag = configure_memory_efficient_attention(_DummyPipe())
    lines = format_attention_diag_lines(diag)
    assert any(line.startswith("ATTENTION_BACKEND_REQUESTED=memory_efficient") for line in lines)
    assert any(line.startswith("ATTENTION_SCOPE=transformer_forward_only") for line in lines)
    assert any(line.startswith("VAE_ATTENTION=normal/default") for line in lines)
    assert any(
        line.startswith("FLUX_TRANSFORMER_ATTENTION=memory_efficient_requested")
        for line in lines
    )
    assert any(line.startswith("ATTENTION_DTYPE_REQUESTED=") for line in lines)
    assert any(line.startswith("ATTENTION_DTYPE_EFFECTIVE=") for line in lines)

    if not torch.cuda.is_available() or not flags_has_sdpa_context():
        assert diag.get("attention_config_ok") is False or diag.get("error")
        return

    assert diag.get("attention_config_ok") is True
    assert diag.get("attention_math_enabled") is False
    assert diag.get("flux_transformer_math") is False
    assert diag.get("vae_attention") == "normal/default"
    assert diag.get("attention_mem_efficient_enabled") or diag.get(
        "attention_flash_enabled"
    ) or diag.get("attention_efficient_kernel_available")
    assert diag.get("attention_backend_effective")
    assert "math=disabled" in str(diag.get("attention_backend_effective")) or (
        "math=False" in str(diag.get("attention_backend_effective"))
    )


def test_forced_fp16_cast_updates_diagnostics():
    import torch

    if not torch.cuda.is_available() or not flags_has_sdpa_context():
        return

    class _DummyPipe:
        transformer = None

    diag = configure_memory_efficient_attention(_DummyPipe(), force_fp16_cast=True)
    assert diag.get("attention_config_ok") is True
    assert diag.get("attention_fallback_used") is True
    assert diag.get("attention_dtype_effective") == "float16"
    assert diag.get("attention_qkv_dtype") == "float16"
    assert "fp16" in str(diag.get("attention_fallback_reason") or "").lower() or (
        "cast" in str(diag.get("attention_fallback_reason") or "").lower()
    )


def test_transformer_only_context_does_not_wrap_vae_execution():
    """
    Regression: MATH-disabled SDPA must apply only inside transformer.forward.

    VAE encode/decode paths must keep default SDPA (MATH allowed).
    """
    import torch
    import torch.nn as nn

    if not torch.cuda.is_available() or not flags_has_sdpa_context():
        return

    class _Transformer(nn.Module):
        def forward(self, x):  # noqa: ANN001
            return probe_sdpa_flags().get("math_sdp_enabled"), x

    class _VAE(nn.Module):
        def encode(self, x):  # noqa: ANN001
            return probe_sdpa_flags().get("math_sdp_enabled")

    class _Pipe:
        def __init__(self) -> None:
            self.transformer = _Transformer()
            self.vae = _VAE()

    pipe = _Pipe()
    assert probe_sdpa_flags().get("math_sdp_enabled") is True

    with transformer_only_memory_efficient_attention(pipe) as state:
        vae_math = pipe.vae.encode(torch.zeros(1, device="cpu"))
        xf_math, _ = pipe.transformer(torch.zeros(1, device="cpu"))

    assert vae_math is True, "VAE path must keep MATH SDPA available"
    assert xf_math is False, "transformer.forward must disable MATH SDPA"
    assert probe_sdpa_flags().get("math_sdp_enabled") is True
    assert state.get("forward_restored") is True
    assert state.get("sdpa_restored") is True


def test_bf16_qkv_cast_to_fp16_inside_transformer_sdpa():
    """BF16 Q/K/V are cast to FP16 for SDPA when force_fp16_cast=True; output restored."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if not torch.cuda.is_available() or not flags_has_sdpa_context():
        return

    seen = {}

    class _Transformer(nn.Module):
        def forward(self, x):  # noqa: ANN001
            # Mimic Diffusers calling F.scaled_dot_product_attention with BF16.
            q = torch.randn(1, 2, 8, 64, device="cuda", dtype=torch.bfloat16)
            k = q.clone()
            v = q.clone()
            out = F.scaled_dot_product_attention(q, k, v)
            return out.dtype, out

    class _Pipe:
        def __init__(self) -> None:
            self.transformer = _Transformer().cuda()

    pipe = _Pipe()
    with transformer_only_memory_efficient_attention(pipe, force_fp16_cast=True) as state:
        out_dtype, _ = pipe.transformer(torch.zeros(1, device="cuda"))

    assert state.get("attention_fallback_used") is True
    assert "float16" in str(state.get("attention_qkv_dtype"))
    assert "bfloat16" in str(state.get("attention_dtype_requested"))
    assert out_dtype == torch.bfloat16, "output dtype must be restored to BF16"
    assert state.get("sdpa_restored") is True
    assert state.get("forward_restored") is True
    # Global SDPA must remain callable after restore (no leftover broken wrap).
    q = torch.randn(1, 1, 4, 32, device="cuda", dtype=torch.float16)
    _ = F.scaled_dot_product_attention(q, q, q)


def test_original_forward_restored_after_context():
    import torch.nn as nn

    class _Transformer(nn.Module):
        def forward(self, x):  # noqa: ANN001
            return x

    class _Pipe:
        def __init__(self) -> None:
            self.transformer = _Transformer()

    pipe = _Pipe()
    original_func = _Transformer.forward
    with transformer_only_memory_efficient_attention(pipe):
        # During the context, Module.forward attribute is replaced.
        assert pipe.transformer.__dict__.get("forward") is not None
    # After exit, instance override is gone / original restored.
    restored = pipe.transformer.__dict__.get("forward")
    assert restored is None or getattr(restored, "__func__", restored) is original_func
    assert pipe.transformer.forward.__func__ is original_func
    assert pipe.transformer(1) == 1


def test_merge_runtime_attention_state():
    base = {
        "attention_dtype_requested": "bfloat16",
        "attention_dtype_effective": "float16",
        "attention_qkv_dtype": "float16",
    }
    runtime = {
        "attention_dtype_requested": "torch.bfloat16",
        "attention_dtype_effective": "torch.float16",
        "attention_qkv_dtype": "torch.float16",
        "attention_device": "cuda:0",
        "attention_fallback_used": True,
        "attention_fallback_reason": "bf16_efficient_sdpa_unavailable_cast_qkv_to_fp16",
    }
    merged = merge_runtime_attention_state(base, runtime)
    assert merged["attention_device"] == "cuda:0"
    assert merged["attention_fallback_used"] is True
    assert "torch.float16" in str(merged["attention_qkv_dtype"])


def test_sdpa_and_forward_restored_after_transformer_exception():
    """Temporary SDPA wrap must restore even when transformer.forward raises."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if not torch.cuda.is_available() or not flags_has_sdpa_context():
        return

    original_sdpa = F.scaled_dot_product_attention

    class _Transformer(nn.Module):
        def forward(self, x):  # noqa: ANN001
            raise RuntimeError("synthetic transformer failure")

    class _Pipe:
        def __init__(self) -> None:
            self.transformer = _Transformer()

    pipe = _Pipe()
    with transformer_only_memory_efficient_attention(pipe, force_fp16_cast=True) as state:
        try:
            pipe.transformer(torch.zeros(1))
            assert False, "expected synthetic failure"
        except RuntimeError as exc:
            assert "synthetic transformer failure" in str(exc)

    assert state.get("forward_restored") is True
    assert state.get("sdpa_restored") is True
    assert pipe.transformer.forward.__func__ is _Transformer.forward
    # SDPA still usable globally
    q = torch.randn(1, 1, 4, 32, device="cuda", dtype=torch.float16)
    _ = F.scaled_dot_product_attention(q, q, q)
    assert F.scaled_dot_product_attention is original_sdpa or callable(
        F.scaled_dot_product_attention
    )


def test_flux_inference_error_classification_helpers():
    from src.features.custom_generator.inference.flux_inference import FLUXInferenceEngine

    assert FLUXInferenceEngine._is_cuda_oom(RuntimeError("CUDA out of memory"))
    assert not FLUXInferenceEngine._is_cuda_oom(
        RuntimeError("No available kernel. Aborting execution.")
    )
    assert FLUXInferenceEngine._is_attention_kernel_error(
        RuntimeError("No available kernel. Aborting execution.")
    )
    assert FLUXInferenceEngine._is_attention_kernel_error(
        RuntimeError("ATTENTION_KERNEL_UNAVAILABLE: test")
    )
    assert not FLUXInferenceEngine._is_attention_kernel_error(
        RuntimeError("CUDA out of memory")
    )


def test_classify_attention_kernel_unavailable():
    from backend_api.services.generation_errors import classify_generation_error

    err_type, stage, _ = classify_generation_error(
        RuntimeError("No available kernel. Aborting execution.")
    )
    assert err_type == "ATTENTION_KERNEL_UNAVAILABLE"
    assert stage == "diffusion"

    err_type2, _, _ = classify_generation_error(
        RuntimeError(
            "CUDA out of memory. Tried to allocate 80.00 MiB. GPU 0 has a total capacity of 14.00 GiB"
        )
    )
    assert err_type2 in ("OUT_OF_MEMORY", "CUDA_OOM")


def test_default_generation_resolution_remains_768():
    from src.features.custom_generator.inference.flux_inference import (
        DEFAULT_FLUX_GENERATION_RESOLUTION,
        resolve_flux_generation_resolution,
    )
    import os

    os.environ.pop("FLUX_GENERATION_RESOLUTION", None)
    os.environ.pop("FLUX_PRODUCTION_SIZE", None)
    assert DEFAULT_FLUX_GENERATION_RESOLUTION == 768
    assert resolve_flux_generation_resolution() == 768


def test_production_pipeline_wires_flux_inference_engine():
    from src.features.custom_generator.pipeline.garment_generation_pipeline import (
        GarmentGenerationPipeline,
    )
    from src.features.custom_generator.inference.flux_inference import FLUXInferenceEngine
    import inspect

    assert "FLUXInferenceEngine" in inspect.getsource(GarmentGenerationPipeline)
    src = inspect.getsource(FLUXInferenceEngine.generate)
    assert "transformer_only_memory_efficient_attention" in src
    # Must not wrap the whole pipeline with the global MATH=False context.
    assert "with memory_efficient_attention_context():" not in src
