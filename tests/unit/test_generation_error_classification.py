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
    assert err_type == "CUDA_OOM"


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
