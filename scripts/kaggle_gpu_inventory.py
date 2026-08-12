#!/usr/bin/env python3
"""Detect Kaggle / local CUDA devices and print dual-T4 readiness.

Run inside the Kaggle notebook (T4×2 accelerator) or locally:

    python scripts/kaggle_gpu_inventory.py

Also configures dual-GPU role env when 2+ devices are present (same as
``configure_kaggle_flux_runtime``) so a follow-up ``run_kaggle.py`` inherits them.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    try:
        import torch
    except ImportError:
        print("torch not installed", file=sys.stderr)
        return 2

    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    print("device_count", torch.cuda.device_count() if torch.cuda.is_available() else 0)

    from src.common.models.device_manager import DeviceManager

    gpus = DeviceManager.inventory_gpus()
    for g in gpus:
        print(
            f"GPU[{g['index']}] {g['name']} "
            f"total_memory_mb={g['total_memory_mb']} "
            f"capability={g['major']}.{g['minor']}"
        )
        if torch.cuda.is_available():
            print(
                f"  allocated_mb={torch.cuda.memory_allocated(g['index']) / (1024**2):.1f} "
                f"reserved_mb={torch.cuda.memory_reserved(g['index']) / (1024**2):.1f}"
            )

    # Mirror Kaggle launcher dual-GPU pin when 2+ cards exist.
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        os.environ.setdefault("FABRICVISION_PROFILE", "kaggle_t4x2")
        os.environ.setdefault("FABRICVISION_DUAL_GPU", "1")
        os.environ.setdefault("FLUX_CUDA_DEVICE", "0")
        os.environ.setdefault("CATVTON_CUDA_DEVICE", "1")
        print("dual_gpu_profile=ENABLED")
    else:
        print("dual_gpu_profile=single_or_cpu")

    report = {
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": gpus,
        "flux_device": DeviceManager.resolve_role_device("flux", "auto"),
        "catvton_device": DeviceManager.resolve_role_device("catvton", "auto"),
        "dual_residency": DeviceManager.dual_gpu_residency_enabled(),
        "profile": os.environ.get("FABRICVISION_PROFILE"),
        "resolved_flux": DeviceManager().resolve_device(
            DeviceManager.resolve_role_device("flux", "auto")
        ),
        "resolved_catvton": DeviceManager().resolve_device(
            DeviceManager.resolve_role_device("catvton", "auto")
        ),
    }
    print(json.dumps(report, indent=2))

    out = ROOT / "experiments" / "kaggle" / "gpu_inventory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", out)

    if report["device_count"] < 1:
        print("FAIL: no CUDA devices — cannot run FLUX/CatVTON on GPU")
        return 1
    if report["device_count"] >= 2 and not report["dual_residency"]:
        print("WARN: 2+ GPUs visible but dual residency not active (check env)")
    if report["device_count"] >= 2:
        print("OK: T4×2-class dual GPU inventory ready (FLUX cuda:0 / CatVTON cuda:1)")
    else:
        print("OK: single GPU inventory (local-dev or single T4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
