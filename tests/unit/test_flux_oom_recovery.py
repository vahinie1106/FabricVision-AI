"""Tests for CUDA OOM detection and FLUX park-on-CPU recovery helpers."""

from src.features.custom_generator.inference.flux_inference import FLUXInferenceEngine


def test_is_cuda_oom_detects_torch_message():
    class FakeOOM(RuntimeError):
        pass

    assert FLUXInferenceEngine._is_cuda_oom(
        FakeOOM(
            "CUDA out of memory. Tried to allocate 20.00 MiB. "
            "GPU 0 has a total capacity of 6.00 GiB of which 0 bytes is free."
        )
    )
    assert FLUXInferenceEngine._is_cuda_oom(type("OutOfMemoryError", (Exception,), {})("oom"))
    assert not FLUXInferenceEngine._is_cuda_oom(RuntimeError("weights missing for flux"))


def test_park_on_cpu_without_pipeline_is_safe():
    class DummyLoader:
        _pipeline = None

        def park_on_cpu(self):
            return {
                "allocated_before_mb": 0.0,
                "allocated_after_mb": 0.0,
                "reserved_before_mb": 0.0,
                "reserved_after_mb": 0.0,
            }

    engine = FLUXInferenceEngine(model_loader=DummyLoader(), allow_fallback=True)
    engine._park_pipeline(None)
