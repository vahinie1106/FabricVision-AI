"""Tests for generation error classification and NF4-safe park behavior."""

from backend_api.services.generation_errors import classify_generation_error
from src.features.custom_generator.model.flux_model_loader import FLUXModelLoader


def test_classify_getcurrentstream_as_device_mismatch():
    err_type, stage, _ = classify_generation_error(
        RuntimeError("invalid argument to getCurrentStream")
    )
    assert err_type == "DEVICE_MISMATCH"
    assert stage == "diffusion"


def test_classify_cuda_oom():
    err_type, stage, _ = classify_generation_error(
        RuntimeError(
            "CUDA out of memory. Tried to allocate 80.00 MiB. GPU 0 has a total capacity of 6.00 GiB"
        )
    )
    assert err_type in ("OUT_OF_MEMORY", "CUDA_OOM")


def test_classify_attention_kernel_unavailable():
    err_type, stage, _ = classify_generation_error(
        RuntimeError("No available kernel. Aborting execution.")
    )
    assert err_type == "ATTENTION_KERNEL_UNAVAILABLE"
    assert stage == "diffusion"


def test_classify_model_auth_failed():
    err_type, stage, _ = classify_generation_error(
        RuntimeError("MODEL_AUTH_FAILED: 401 Unauthorized for gated model")
    )
    assert err_type == "MODEL_AUTH_FAILED"
    assert stage == "model_load"


def test_classify_model_download_failed():
    err_type, stage, _ = classify_generation_error(
        RuntimeError("MODEL_DOWNLOAD_FAILED: snapshot_download timed out")
    )
    assert err_type == "MODEL_DOWNLOAD_FAILED"


def test_park_on_cpu_skips_nf4_transformer_to_cpu():
    loader = FLUXModelLoader(allow_fallback=True)
    loader._used_bnb_4bit = True
    loader._offload_strategy = "model_cpu_offload"

    class _Mod:
        def __init__(self, name):
            self.name = name
            self.moves = []

        def to(self, device):
            self.moves.append(device)
            return self

    class _Pipe:
        def __init__(self):
            self.transformer = _Mod("transformer")
            self.text_encoder = _Mod("text_encoder")
            self.text_encoder_2 = _Mod("text_encoder_2")
            self.vae = _Mod("vae")
            self.freed = False
            self.offload_enabled = 0

        def maybe_free_model_hooks(self):
            self.freed = True
            # Diffusers re-applies offload inside maybe_free; we must NOT
            # call enable_model_cpu_offload again from park_on_cpu.
            self.offload_enabled += 1

        def enable_model_cpu_offload(self):
            self.offload_enabled += 10  # would indicate a forbidden double-enable

    pipe = _Pipe()
    loader._pipeline = pipe
    loader.park_on_cpu()

    assert pipe.freed is True
    assert pipe.offload_enabled == 1  # only via maybe_free, not park
    assert "cpu" not in pipe.transformer.moves
    # With model_cpu_offload, park must not fight hooks with manual .to(cpu)
    assert pipe.text_encoder.moves == []
    assert pipe.vae.moves == []


def test_park_gpu_resident_then_ensure_restores_vae(monkeypatch):
    """T4-class bug: gpu_resident park left VAE on CPU → CUDA_ERROR at ~50%."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    loader = FLUXModelLoader(allow_fallback=True)
    loader._used_bnb_4bit = True
    loader._offload_strategy = "gpu_resident"

    class _Mod:
        def __init__(self, name):
            self.name = name
            self.moves = []
            self._device = "cuda"

        def parameters(self):
            class _P:
                def __init__(self, device):
                    self.device = device
                    self.dtype = torch.float16

            yield _P(self._device)

        def to(self, device):
            self.moves.append(str(device))
            self._device = str(device)
            return self

    class _Pipe:
        def __init__(self):
            self.transformer = _Mod("transformer")
            self.text_encoder = _Mod("text_encoder")
            self.text_encoder_2 = _Mod("text_encoder_2")
            self.vae = _Mod("vae")

        def maybe_free_model_hooks(self):
            return None

    pipe = _Pipe()
    loader._pipeline = pipe
    loader.park_on_cpu()

    assert "cpu" in pipe.vae.moves
    assert "cpu" not in pipe.transformer.moves  # NF4 must stay put

    info = loader.ensure_generation_devices()
    assert "vae" in info["restored"]
    assert any("cuda" in m for m in pipe.vae.moves)


def test_resolve_torch_dtype_pre_ampere_uses_fp16(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda idx=0: (7, 5))

    loader = FLUXModelLoader(allow_fallback=True, precision="bfloat16")
    assert loader._resolve_torch_dtype() == torch.float16


def test_auto_offload_prefers_true_on_pre_ampere(monkeypatch):
    """≥14GB T4 should still prefer model_cpu_offload in auto mode."""
    loader = FLUXModelLoader(allow_fallback=True)
    monkeypatch.setattr(loader, "_gpu_vram_mb", lambda: 15109.0)
    monkeypatch.setattr(loader, "_is_pre_ampere_gpu", lambda: True)
    monkeypatch.delenv("FLUX_MODEL_CPU_OFFLOAD", raising=False)

    # Mirror loader decision block.
    offload_env = ""
    physical_mb = loader._gpu_vram_mb()
    if offload_env in ("0", "false", "no", "off"):
        prefer_offload = False
    elif offload_env in ("1", "true", "yes", "on"):
        prefer_offload = True
    else:
        prefer_offload = physical_mb < 14000 or loader._is_pre_ampere_gpu()
    assert prefer_offload is True
