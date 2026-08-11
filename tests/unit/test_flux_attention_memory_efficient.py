"""Unit tests for FLUX memory-efficient SDPA configuration helpers."""

from __future__ import annotations

from src.features.custom_generator.inference.flux_attention import (
    configure_memory_efficient_attention,
    format_attention_diag_lines,
    probe_sdpa_flags,
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

    if not torch.cuda.is_available() or not flags_has_sdpa_context():
        # CPU-only CI: helper should fail clearly rather than claim success.
        assert diag.get("attention_config_ok") is False or diag.get("error")
        return

    assert diag.get("attention_config_ok") is True
    assert diag.get("attention_math_enabled") is False
    assert diag.get("attention_mem_efficient_enabled") or diag.get(
        "attention_flash_enabled"
    )
    assert diag.get("attention_backend_effective")
    assert "math=disabled" in str(diag.get("attention_backend_effective")) or (
        "math=False" in str(diag.get("attention_backend_effective"))
    )


def flags_has_sdpa_context() -> bool:
    flags = probe_sdpa_flags()
    return bool(
        flags.get("has_nn_attention_sdpa_kernel") or flags.get("has_backends_sdp_kernel")
    )
