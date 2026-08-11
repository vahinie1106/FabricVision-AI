#!/usr/bin/env python3
"""Diagnose FLUX.1-Kontext model loading + optional real 1024 smoke inference.

Run inside Kaggle (or locally) from the repo root:

  python scripts/debug_flux_load.py
  python scripts/debug_flux_load.py --smoke
  python scripts/debug_flux_load.py --smoke --smoke-steps 1

``--smoke`` always requests an explicit 1024x1024 Kontext generation (never 256
with silent Diffusers upscaling). It uses the same FLUXInferenceEngine path as
the Custom Garment Generator (pre-encode, CPU offload, VAE tiling for 1024).

Never prints secrets.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Large-res smoke / garment gens need VAE tiling to avoid decode OOM on 16GB.
os.environ.setdefault("FLUX_VAE_TILING", "true")
os.environ.setdefault("FLUX_PREENCODE_PROMPT", "true")


def _vram_mb() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {
            "total_mb": 0.0,
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "max_allocated_mb": 0.0,
            "free_mb": 0.0,
        }
    total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    alloc = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    max_alloc = torch.cuda.max_memory_allocated() / (1024**2)
    free = max(0.0, total - reserved)
    return {
        "total_mb": round(total, 1),
        "allocated_mb": round(alloc, 1),
        "reserved_mb": round(reserved, 1),
        "max_allocated_mb": round(max_alloc, 1),
        "free_mb": round(free, 1),
    }


def _module_dtype(mod) -> str:
    if mod is None:
        return "none"
    try:
        for p in mod.parameters():
            return str(p.dtype)
    except Exception:
        pass
    return "unknown"


def _quant_state(mod) -> str:
    if mod is None:
        return "none"
    try:
        names = []
        for m in list(mod.modules())[:80]:
            n = type(m).__name__
            if "4bit" in n.lower() or "Linear4bit" in n:
                names.append(n)
        if names:
            return f"bnb_nf4_modules={len(names)} sample={names[0]}"
    except Exception as exc:
        return f"inspect_failed:{type(exc).__name__}"
    return "no_Linear4bit_detected"


def _apply_1024_memory_opts(pipe, loader) -> list[str]:
    """Legitimate Diffusers/accelerate opts for 1024 Kontext on ~16GB."""
    applied: list[str] = []
    # Ensure model CPU offload remains active (already set by loader for NF4).
    offload = getattr(loader, "_offload_strategy", None)
    if offload:
        applied.append(f"offload={offload}")

    vae = getattr(pipe, "vae", None)
    if vae is not None:
        if hasattr(vae, "enable_slicing"):
            vae.enable_slicing()
            applied.append("vae_slicing")
        if hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
            applied.append("vae_tiling")

    # Prefer SDPA / already configured attention from loader.
    attn = getattr(loader, "_attention_backend", None)
    if attn:
        applied.append(f"attention={attn}")

    if hasattr(loader, "park_on_cpu"):
        loader.park_on_cpu()
        applied.append("park_on_cpu_before_smoke")

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            applied.append("cuda_empty_cache+reset_peak")
    except Exception:
        pass
    return applied


def _run_smoke(loader, pipe, steps: int = 1) -> int:
    """
    Real 1024x1024 Kontext generation via production FLUXInferenceEngine.

    Returns 0 on success, 1 on failure (including honest OOM).
    """
    from PIL import Image

    import torch

    from src.features.custom_generator.inference.flux_inference import FLUXInferenceEngine

    height = width = 1024
    print(f"SMOKE_RESOLUTION={width}x{height}", flush=True)
    print(f"SMOKE_STEPS={int(steps)}", flush=True)
    print("SMOKE_INFERENCE=START", flush=True)

    applied = _apply_1024_memory_opts(pipe, loader)
    print(f"SMOKE_MEMORY_OPTS={','.join(applied)}", flush=True)

    vram_total = _vram_mb()
    print(f"GPU_VRAM_TOTAL_MB={vram_total['total_mb']}", flush=True)
    print(f"VRAM_AFTER_LOAD_ALLOC_MB={vram_total['allocated_mb']}", flush=True)
    print(f"VRAM_AFTER_LOAD_RESERVED_MB={vram_total['reserved_mb']}", flush=True)
    print(f"VRAM_AFTER_LOAD_FREE_MB={vram_total['free_mb']}", flush=True)

    transformer = getattr(pipe, "transformer", None)
    print(f"MODEL_DTYPE={getattr(loader, 'precision', 'unknown')}", flush=True)
    print(f"TRANSFORMER_DTYPE={_module_dtype(transformer)}", flush=True)
    print(f"QUANTIZATION_STATE={_quant_state(transformer)}", flush=True)
    print(
        f"OFFLOAD_STRATEGY={getattr(loader, '_offload_strategy', None)}",
        flush=True,
    )
    print(
        f"ATTENTION_BACKEND={getattr(loader, '_attention_backend', None)}",
        flush=True,
    )
    # Detailed SDPA configuration (MATH must be disabled for T4 1024).
    try:
        from src.features.custom_generator.inference.flux_attention import (
            configure_memory_efficient_attention,
            format_attention_diag_lines,
        )

        diag = getattr(loader, "_attention_diag", None) or {}
        if not diag.get("attention_config_ok"):
            diag = configure_memory_efficient_attention(pipe)
            try:
                loader._attention_diag = diag
                if diag.get("attention_config_ok"):
                    loader._attention_backend = "memory_efficient_sdpa"
            except Exception:
                pass
        for line in format_attention_diag_lines(diag):
            print(line, flush=True)
        if not diag.get("attention_config_ok"):
            print(
                f"ATTENTION_ERROR={diag.get('error') or 'NO_SUPPORTED_EFFICIENT_ATTENTION_BACKEND'}",
                flush=True,
            )
            print("SMOKE_INFERENCE=FAILED", flush=True)
            print("ERROR=NO_SUPPORTED_EFFICIENT_ATTENTION_BACKEND", flush=True)
            return 1
    except Exception as attn_exc:
        print(f"ATTENTION_DIAG_ERROR={type(attn_exc).__name__}: {attn_exc}", flush=True)
        print("SMOKE_INFERENCE=FAILED", flush=True)
        print("ERROR=NO_SUPPORTED_EFFICIENT_ATTENTION_BACKEND", flush=True)
        return 1
    print(f"BNB_4BIT={getattr(loader, '_used_bnb_4bit', None)}", flush=True)

    # Conditioning image must already be 1024 so Diffusers does not resize silently.
    img = Image.new("RGB", (width, height), color=(180, 40, 40))
    engine = FLUXInferenceEngine(loader, allow_fallback=False)

    t0 = time.perf_counter()
    before = _vram_mb()
    print(f"VRAM_BEFORE_INFERENCE_ALLOC_MB={before['allocated_mb']}", flush=True)
    print(f"VRAM_BEFORE_INFERENCE_RESERVED_MB={before['reserved_mb']}", flush=True)
    print(f"VRAM_BEFORE_INFERENCE_FREE_MB={before['free_mb']}", flush=True)

    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        out = engine.generate(
            prompt="a simple red cotton kurti garment product photo, studio lighting",
            negative_prompt="",
            reference_image=img,
            height=height,
            width=width,
            num_inference_steps=int(steps),
            guidance_scale=2.5,
            seed=42,
        )
        elapsed = round(time.perf_counter() - t0, 3)
        after = _vram_mb()
        peak = after["max_allocated_mb"]
        out_w, out_h = out.size
        print("SMOKE_INFERENCE=SUCCESS", flush=True)
        print(f"OUTPUT_RESOLUTION={out_w}x{out_h}", flush=True)
        print(f"ELAPSED_INFERENCE_S={elapsed}", flush=True)
        print(f"VRAM_PEAK_ALLOC_MB={peak}", flush=True)
        print(f"VRAM_AFTER_INFERENCE_ALLOC_MB={after['allocated_mb']}", flush=True)
        print(f"VRAM_AFTER_INFERENCE_RESERVED_MB={after['reserved_mb']}", flush=True)
        stats = getattr(engine, "last_execution_stats", {}) or {}
        for key in (
            "peak_vram_mb",
            "generation_time_s",
            "inference_time_s",
            "prompt_encoding_time_s",
            "offload_strategy",
            "was_real_flux_used",
            "height",
            "width",
            "output_size",
        ):
            if key in stats:
                print(f"STATS_{key.upper()}={stats[key]}", flush=True)
        if out_w != 1024 or out_h != 1024:
            print(
                "WARNING: output resolution is not 1024x1024 — Diffusers may have resized",
                flush=True,
            )
            return 1
        return 0
    except Exception as exc:
        elapsed = round(time.perf_counter() - t0, 3)
        after = _vram_mb()
        msg = str(exc)
        lower = msg.lower()
        is_oom = "out of memory" in lower or "outofmemory" in type(exc).__name__.lower()
        print("SMOKE_INFERENCE=FAILED", flush=True)
        if is_oom:
            print("ERROR=OUT_OF_MEMORY", flush=True)
        else:
            print(f"ERROR={type(exc).__name__}: {exc}", flush=True)
        print(f"RESOLUTION={width}x{height}", flush=True)
        print(f"ELAPSED_INFERENCE_S={elapsed}", flush=True)
        print(f"VRAM_PEAK_ALLOC_MB={after['max_allocated_mb']}", flush=True)
        print(f"VRAM_AFTER_FAIL_ALLOC_MB={after['allocated_mb']}", flush=True)
        print(f"VRAM_AFTER_FAIL_RESERVED_MB={after['reserved_mb']}", flush=True)
        print(f"VRAM_AFTER_FAIL_FREE_MB={after['free_mb']}", flush=True)
        print(
            "OOM_NOTE=At 1024x1024 FluxKontext must hold NF4 transformer activations, "
            "latents, and VAE working memory. If PyTorch already holds ~14GiB before the "
            "next large allocation, the denoise/VAE step fails even with model_cpu_offload. "
            "Production path uses pre-encode + encoder eviction + VAE tiling; sequential "
            "CPU offload is NOT used with bitsandbytes NF4 (meta-tensor corruption risk).",
            flush=True,
        )
        traceback.print_exc()
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose FLUX.1-Kontext model loading")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="After load, run a REAL explicit 1024x1024 Kontext inference smoke test",
    )
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=1,
        help="Denoise steps for --smoke (default 1)",
    )
    args = parser.parse_args()

    print("=== FabricVision FLUX load diagnostics ===", flush=True)
    print(f"ROOT={ROOT}", flush=True)

    try:
        from src.common.utils.ensure_bitsandbytes import ensure_bitsandbytes

        bnb_ver = ensure_bitsandbytes(auto_install=True)
        print(f"BITSANDBYTES_VERSION={bnb_ver}", flush=True)
        import importlib.metadata as md

        print(f"BITSANDBYTES_METADATA={md.version('bitsandbytes')}", flush=True)
    except Exception as exc:
        print(f"BITSANDBYTES_ERROR={type(exc).__name__}: {exc}", flush=True)
        print("LOAD_RESULT=FAILED", flush=True)
        print(f"ERROR={type(exc).__name__}: {exc}", flush=True)
        return 1

    hf_present = bool(
        (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    )
    print(f"HF_TOKEN_PRESENT={hf_present}", flush=True)

    try:
        import torch

        print(f"TORCH={torch.__version__}", flush=True)
        print(f"CUDA_AVAILABLE={torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available():
            print(f"GPU={torch.cuda.get_device_name(0)}", flush=True)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            print(f"GPU_VRAM_MB={vram:.0f}", flush=True)
        else:
            print("GPU=NO GPU", flush=True)
    except Exception as exc:
        print(f"TORCH_IMPORT_ERROR={type(exc).__name__}: {exc}", flush=True)

    try:
        import diffusers

        print(f"DIFFUSERS_VERSION={diffusers.__version__}", flush=True)
        from diffusers import FluxKontextPipeline

        print("PIPELINE_CLASS=FluxKontextPipeline", flush=True)
        _ = FluxKontextPipeline
    except Exception as exc:
        print(f"DIFFUSERS_ERROR={type(exc).__name__}: {exc}", flush=True)
        print("PIPELINE_CLASS=UNAVAILABLE", flush=True)

    try:
        import transformers

        print(f"TRANSFORMERS_VERSION={transformers.__version__}", flush=True)
    except Exception as exc:
        print(f"TRANSFORMERS_ERROR={type(exc).__name__}: {exc}", flush=True)

    from src.features.custom_generator.model.flux_model_loader import FLUXModelLoader

    model_path = ROOT / "models" / "flux-kontext"
    loader = FLUXModelLoader(
        model_path=model_path,
        allow_fallback=False,
    )
    print(f"MODEL_ID={loader.hf_model_id}", flush=True)
    print(f"MODEL_PATH={model_path}", flush=True)
    print(f"MODEL_EXISTS={model_path.exists()}", flush=True)
    print(
        f"MODEL_INDEX_EXISTS={(model_path / 'model_index.json').exists()}",
        flush=True,
    )

    try:
        pipe = loader.load()
        info = loader.get_runtime_info()
        print("LOAD_RESULT=SUCCESS", flush=True)
        print("ERROR=", flush=True)
        for key in (
            "model_kind",
            "bnb_4bit",
            "offload_strategy",
            "attention_backend",
            "gpu_name",
            "gpu_vram_mb",
            "cuda_allocated_mb",
            "diffusers_version",
            "bitsandbytes_version",
            "hf_model_id",
            "model_path",
        ):
            if key in info:
                print(f"{key.upper()}={info[key]}", flush=True)
        print(f"PIPELINE_LOADED={pipe is not None}", flush=True)

        if args.smoke:
            return _run_smoke(loader, pipe, steps=max(1, int(args.smoke_steps)))
        return 0
    except Exception as exc:
        print("LOAD_RESULT=FAILED", flush=True)
        print(f"ERROR={type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
