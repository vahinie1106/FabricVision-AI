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


def test_vae_decode_oom_retries_same_latent_size():
    from PIL import Image

    class DummyLoader:
        pipeline = None

        def park_on_cpu(self):
            return {}

    engine = FLUXInferenceEngine(model_loader=DummyLoader(), allow_fallback=True)
    calls = {"n": 0}

    def fake_decode(_pipeline, _latents, height, width):
        calls["n"] += 1
        calls["size"] = (height, width)
        if calls["n"] == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB during vae decode")
        return Image.new("RGB", (int(width), int(height)))

    engine._decode_flux_latents = fake_decode  # type: ignore[method-assign]
    engine._park_vae_for_denoise = lambda _p: None  # type: ignore[method-assign]
    engine._ensure_generation_devices = lambda _p: None  # type: ignore[method-assign]
    engine._enable_vae_tile_slice = lambda _p: None  # type: ignore[method-assign]

    class _Pipe:
        vae = None

    image = engine._decode_latents_with_oom_retry(_Pipe(), object(), 704, 704)
    assert calls["n"] == 2
    assert calls["size"] == (704, 704)
    assert image.size == (704, 704)


def test_vae_decode_oom_retry_does_not_fallback_to_512():
    class DummyLoader:
        pipeline = None

        def park_on_cpu(self):
            return {}

    engine = FLUXInferenceEngine(model_loader=DummyLoader(), allow_fallback=True)

    def always_oom(_pipeline, _latents, height, width):
        raise RuntimeError(f"CUDA out of memory during VAE decode at {width}x{height}")

    engine._decode_flux_latents = always_oom  # type: ignore[method-assign]
    engine._park_vae_for_denoise = lambda _p: None  # type: ignore[method-assign]
    engine._ensure_generation_devices = lambda _p: None  # type: ignore[method-assign]
    engine._enable_vae_tile_slice = lambda _p: None  # type: ignore[method-assign]

    class _Pipe:
        vae = None

    try:
        engine._decode_latents_with_oom_retry(_Pipe(), object(), 704, 704)
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        msg = str(exc).lower()
        assert "512" not in msg or "no silent 512" in msg
        assert "vae decode" in msg

