"""T4 must stay GPU-resident; pre-Ampere is not an offload trigger."""

from __future__ import annotations

import pytest

from src.features.custom_generator.model.flux_model_loader import (
    should_prefer_model_cpu_offload,
)


def test_auto_offload_only_below_14gb():
    assert should_prefer_model_cpu_offload(physical_mb=6144.0, offload_env="") is True
    assert should_prefer_model_cpu_offload(physical_mb=13999.0, offload_env="") is True
    assert should_prefer_model_cpu_offload(physical_mb=14000.0, offload_env="") is False
    assert should_prefer_model_cpu_offload(physical_mb=15109.0, offload_env="") is False


def test_t4_class_auto_is_gpu_resident_even_if_pre_ampere():
    # sm_75 is a dtype concern (fp16), not CPU offload. 15GB T4 stays resident.
    assert should_prefer_model_cpu_offload(physical_mb=15109.0) is False


def test_explicit_env_wins():
    assert (
        should_prefer_model_cpu_offload(physical_mb=15109.0, offload_env="true") is True
    )
    assert (
        should_prefer_model_cpu_offload(physical_mb=6144.0, offload_env="false") is False
    )


def test_t4_standard_policy_defaults_no_offload(monkeypatch):
    monkeypatch.delenv("FLUX_MODEL_CPU_OFFLOAD", raising=False)
    monkeypatch.delenv("FLUX_GENERATION_RESOLUTION", raising=False)
    monkeypatch.delenv("FLUX_STANDARD_STEPS", raising=False)
    monkeypatch.delenv("FLUX_ALLOW_HIGH_RES", raising=False)
    from src.features.custom_generator.inference.flux_vram_policy import (
        select_standard_generation_policy,
    )

    policy = select_standard_generation_policy(
        physical_mb=15109.0,
        free_mb=2500.0,
        offload_strategy="gpu_resident",
    )
    assert policy.height == 512
    assert policy.width == 512
    assert policy.num_inference_steps == 8
    assert policy.prefer_model_cpu_offload is False
    assert policy.profile == "standard_t4_safe"


def test_loader_auto_t4_does_not_enable_offload(monkeypatch):
    """15109 MB + pre-Ampere capability must not select model_cpu_offload."""
    torch = pytest.importorskip("torch")

    from src.features.custom_generator.model.flux_model_loader import FLUXModelLoader

    monkeypatch.delenv("FLUX_MODEL_CPU_OFFLOAD", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda idx=0: (7, 5))

    loader = FLUXModelLoader(allow_fallback=True)
    monkeypatch.setattr(loader, "_gpu_vram_mb", lambda device_index=0: 15109.0)
    assert (
        should_prefer_model_cpu_offload(
            physical_mb=loader._gpu_vram_mb(),
            offload_env="",
        )
        is False
    )
    # Pre-Ampere still selects fp16 compute, not offload.
    assert loader._is_pre_ampere_gpu(0) is True
    assert loader._resolve_torch_dtype() == torch.float16


def test_resolve_flux_target_honors_role_env(monkeypatch):
    monkeypatch.setenv("FLUX_CUDA_DEVICE", "0")
    from src.features.custom_generator.model.flux_model_loader import FLUXModelLoader

    loader = FLUXModelLoader(allow_fallback=True, device="auto")
    # Without CUDA this still returns cpu; with CUDA it must be cuda:0.
    resolved = loader._resolve_flux_target_device()
    assert resolved in ("cuda:0", "cuda", "cpu")
    if resolved.startswith("cuda"):
        assert resolved in ("cuda:0", "cuda")


def test_loader_offload_auto_does_not_use_pre_ampere():
    from pathlib import Path

    src = Path("src/features/custom_generator/model/flux_model_loader.py").read_text(
        encoding="utf-8"
    )
    assert "physical_mb < 14000 or self._is_pre_ampere_gpu" not in src
    assert "should_prefer_model_cpu_offload" in src
    inf = Path("src/features/custom_generator/inference/flux_inference.py").read_text(
        encoding="utf-8"
    )
    assert "Skipping park_on_cpu before generate" in inf
