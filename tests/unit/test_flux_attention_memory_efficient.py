"""Unit tests for FLUX memory-efficient SDPA configuration helpers."""

from __future__ import annotations

from src.features.custom_generator.inference.flux_attention import (
    configure_memory_efficient_attention,
    format_attention_diag_lines,
    probe_sdpa_flags,
    transformer_only_memory_efficient_attention,
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

    if not torch.cuda.is_available() or not flags_has_sdpa_context():
        # CPU-only CI: helper should fail clearly rather than claim success.
        assert diag.get("attention_config_ok") is False or diag.get("error")
        return

    assert diag.get("attention_config_ok") is True
    assert diag.get("attention_math_enabled") is False
    assert diag.get("flux_transformer_math") is False
    assert diag.get("vae_attention") == "normal/default"
    assert diag.get("attention_mem_efficient_enabled") or diag.get(
        "attention_flash_enabled"
    )
    assert diag.get("attention_backend_effective")
    assert "math=disabled" in str(diag.get("attention_backend_effective")) or (
        "math=False" in str(diag.get("attention_backend_effective"))
    )


def test_transformer_only_context_does_not_wrap_vae_execution():
    """
    Regression: MATH-disabled SDPA must apply only inside transformer.forward.

    VAE encode/decode paths must keep default SDPA (MATH allowed). Wrapping the
    entire pipeline(...) previously caused T4 'No available kernel' in VAE attn.
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
    # Baseline outside any wrap: MATH should be enabled by default on CUDA.
    assert probe_sdpa_flags().get("math_sdp_enabled") is True

    with transformer_only_memory_efficient_attention(pipe):
        # Simulated VAE encode (prepare_latents) — must NOT see MATH=False.
        vae_math = pipe.vae.encode(torch.zeros(1, device="cpu"))
        # Transformer denoise — MATH disabled.
        xf_math, _ = pipe.transformer(torch.zeros(1, device="cpu"))

    assert vae_math is True, "VAE path must keep MATH SDPA available"
    assert xf_math is False, "transformer.forward must disable MATH SDPA"
    # After exit, global flags restored.
    assert probe_sdpa_flags().get("math_sdp_enabled") is True


def flags_has_sdpa_context() -> bool:
    flags = probe_sdpa_flags()
    return bool(
        flags.get("has_nn_attention_sdpa_kernel") or flags.get("has_backends_sdp_kernel")
    )
