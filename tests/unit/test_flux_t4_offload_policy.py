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
    assert policy.height == 712
    assert policy.width == 712
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
    assert "output_type=latent" in inf
    assert "_park_vae_for_denoise" in inf
    assert "_park_transformer_for_vae" in inf
    assert "park_transformer_before_vae" in inf
    assert "_evict_text_encoders" in inf.split("Prompt embed cache HIT")[1].split(
        "return embeds"
    )[0]


def test_hybrid_t4_plan_evicts_t5_parks_vae_no_step_offload():
    from src.features.custom_generator.inference.flux_vram_policy import (
        select_hybrid_vram_plan,
    )

    plan = select_hybrid_vram_plan(
        physical_mb=15109.0,
        offload_strategy="gpu_resident",
        height=512,
        width=512,
    )
    assert plan.per_step_cpu_offload is False
    assert plan.evict_t5_before_diffusion is True
    assert plan.park_vae_during_denoise is True
    assert plan.decode_latents_separately is True
    assert plan.enable_vae_tiling is True
    assert plan.park_transformer_before_vae is True
    assert plan.enable_vae_slicing is True
    assert plan.profile == "t4_hybrid_gpu_resident"


def test_hybrid_6gb_does_not_use_t4_latent_decode():
    from src.features.custom_generator.inference.flux_vram_policy import (
        select_hybrid_vram_plan,
    )

    plan = select_hybrid_vram_plan(
        physical_mb=6144.0,
        offload_strategy="model_cpu_offload",
        height=512,
        width=512,
    )
    assert plan.decode_latents_separately is False
    assert plan.park_vae_during_denoise is False
    assert plan.park_transformer_before_vae is False
    assert plan.per_step_cpu_offload is True


def test_kaggle_gpu_roles_flux0_catvton1():
    from pathlib import Path

    src = Path("scripts/run_kaggle.py").read_text(encoding="utf-8")
    assert 'setdefault("FLUX_CUDA_DEVICE", "0")' in src
    assert 'setdefault("CATVTON_CUDA_DEVICE", "1")' in src


def test_loader_reuses_resident_pipeline():
    from pathlib import Path

    src = Path("src/features/custom_generator/model/flux_model_loader.py").read_text(
        encoding="utf-8"
    )
    assert "if self._pipeline is not None:" in src
    assert "self._reuse_count += 1" in src


def test_oom_recovery_clears_cuda_allocator():
    from pathlib import Path

    mgr = Path("src/integrations/flux/flux_manager.py").read_text(encoding="utf-8")
    pol = Path(
        "src/features/custom_generator/inference/flux_vram_policy.py"
    ).read_text(encoding="utf-8")
    assert "cleanup_cuda_after_failure" in mgr
    assert "reset_peak_memory_stats" in pol
    inf = Path("src/features/custom_generator/inference/flux_inference.py").read_text(
        encoding="utf-8"
    )
    assert "cleanup_cuda_after_failure" in inf


def test_park_vae_does_not_move_transformer():
    from src.features.custom_generator.model.flux_model_loader import FLUXModelLoader

    loader = FLUXModelLoader(allow_fallback=True)
    loader._offload_strategy = "gpu_resident"
    loader._used_bnb_4bit = True

    class _Mod:
        def __init__(self):
            self.moves = []
            self._device = "cuda"

        def to(self, device=None, **kwargs):
            if device is not None:
                self.moves.append(str(device))
                self._device = str(device)
            return self

        def parameters(self):
            class _P:
                def __init__(self, device):
                    self.device = device
                    self.dtype = "float32"

            yield _P(self._device)

    class _Pipe:
        def __init__(self):
            self.transformer = _Mod()
            self.vae = _Mod()

    pipe = _Pipe()
    loader._pipeline = pipe
    info = loader.park_vae_to_cpu()
    assert info["parked"] is True
    assert "cpu" in pipe.vae.moves
    assert pipe.transformer.moves == []


def test_park_transformer_does_not_move_vae():
    pytest.importorskip("torch")
    from src.features.custom_generator.model.flux_model_loader import FLUXModelLoader

    loader = FLUXModelLoader(allow_fallback=True)
    loader._offload_strategy = "gpu_resident"
    loader._used_bnb_4bit = True

    class _Mod:
        def __init__(self):
            self.moves = []
            self._device = "cuda:0"

        def to(self, device=None, **kwargs):
            if device is not None:
                self.moves.append(str(device))
                self._device = str(device)
            return self

        def modules(self):
            return iter(())

        def parameters(self):
            class _P:
                def __init__(self, device):
                    self.device = device
                    self.dtype = "float16"

            yield _P(self._device)

    class _Pipe:
        def __init__(self):
            self.transformer = _Mod()
            self.vae = _Mod()

    pipe = _Pipe()
    loader._pipeline = pipe
    info = loader.park_transformer_to_cpu()
    assert info["parked"] is True
    assert "cpu" in pipe.transformer.moves
    assert pipe.vae.moves == []
    restored = loader.ensure_transformer_on_device()
    if restored.get("skipped"):
        return
    assert restored["restored"] is True
    assert str(pipe.transformer._device).startswith("cuda")


def test_hybrid_plan_parks_transformer_before_vae_not_during_denoise():
    from pathlib import Path

    inf = Path("src/features/custom_generator/inference/flux_inference.py").read_text(
        encoding="utf-8"
    )
    denoise_idx = inf.find('log_vram("after_denoise_before_vae")')
    park_idx = inf.find("self._park_transformer_for_vae(pipeline)", denoise_idx)
    vae_idx = inf.find("_decode_latents_with_oom_retry", denoise_idx)
    assert denoise_idx != -1 and park_idx != -1 and vae_idx != -1
    assert denoise_idx < park_idx < vae_idx
    assert "enable_model_cpu_offload()" not in inf.split("def generate")[1].split("def _generate_synthetic")[0]
    assert 'log_vram("transformer parked before VAE")' in inf
    assert "delivered_resolution" in inf


def test_t4_vae_stays_fp32_with_tiling_and_latent_output():
    from pathlib import Path

    loader = Path("src/features/custom_generator/model/flux_model_loader.py").read_text(
        encoding="utf-8"
    )
    inf = Path("src/features/custom_generator/inference/flux_inference.py").read_text(
        encoding="utf-8"
    )
    assert "vae.to(dtype=torch.float32)" in loader
    assert "enable_tiling" in inf
    assert "enable_slicing" in inf
    assert 'kwargs["output_type"] = "latent"' in inf
    assert "FLUX_STANDARD_NO_OOM_FALLBACK" in inf
    kaggle = Path("scripts/run_kaggle.py").read_text(encoding="utf-8")
    assert 'setdefault("FLUX_GENERATION_RESOLUTION", "712")' in kaggle
    assert 'setdefault("FLUX_STANDARD_STEPS", "8")' in kaggle
    assert 'setdefault("FLUX_CUDA_DEVICE", "0")' in kaggle
    assert 'setdefault("CATVTON_CUDA_DEVICE", "1")' in kaggle

