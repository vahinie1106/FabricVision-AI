#!/usr/bin/env python3
"""Diagnose FLUX.1-Kontext model loading (no image generation).

Run inside Kaggle (or locally) from the repo root:

  python scripts/debug_flux_load.py

Never prints secrets. Reports MODEL_ID / path / CUDA / versions / LOAD_RESULT.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main() -> int:
    print("=== FabricVision FLUX load diagnostics ===", flush=True)
    print(f"ROOT={ROOT}", flush=True)

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

    try:
        import bitsandbytes as bnb

        print(f"BITSANDBYTES_VERSION={getattr(bnb, '__version__', 'unknown')}", flush=True)
    except Exception as exc:
        print(f"BITSANDBYTES_ERROR={type(exc).__name__}: {exc}", flush=True)

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
        print(f"ERROR=", flush=True)
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
        return 0
    except Exception as exc:
        print("LOAD_RESULT=FAILED", flush=True)
        print(f"ERROR={type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
