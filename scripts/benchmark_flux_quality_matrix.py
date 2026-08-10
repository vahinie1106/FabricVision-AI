"""Controlled FLUX quality-first matrix (steps / resolution / precision).

Preserves LOW_VRAM NF4+offload defaults. Quality cells opt into higher steps,
higher resolution, and optional full precision via env overrides.

Usage (venv):
  python scripts/benchmark_flux_quality_matrix.py --fabric path/to/fabric.jpg
  python scripts/benchmark_flux_quality_matrix.py --fabric path/to/fabric.jpg --only baseline_512_3_nf4,q15_512_nf4
  python scripts/benchmark_flux_quality_matrix.py --fabric path/to/fabric.jpg --plan-only

Does NOT invent human quality scores. Objective image stats are diagnostic only.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("fabricvision.flux_quality_matrix")

# Controlled matrix — change one axis at a time across the ladder.
MATRIX: list[dict[str, Any]] = [
    {
        "id": "baseline_512_3_nf4",
        "mode": "standard",
        "steps": 3,
        "height": 512,
        "width": 512,
        "guidance": 3.0,
        "quant": "nf4",
        "offload": "true",
        "min_physical_vram_mb": 0,
    },
    {
        "id": "q15_512_nf4",
        "mode": "quality_15",
        "steps": 15,
        "height": 512,
        "width": 512,
        "guidance": 3.0,
        "quant": "nf4",
        "offload": "true",
        "min_physical_vram_mb": 0,
    },
    {
        "id": "q20_512_nf4",
        "mode": "quality_20",
        "steps": 20,
        "height": 512,
        "width": 512,
        "guidance": 3.0,
        "quant": "nf4",
        "offload": "true",
        "min_physical_vram_mb": 0,
    },
    {
        "id": "q30_512_nf4",
        "mode": "quality_30",
        "steps": 30,
        "height": 512,
        "width": 512,
        "guidance": 3.0,
        "quant": "nf4",
        "offload": "true",
        "min_physical_vram_mb": 0,
    },
    {
        "id": "q20_768_nf4",
        "mode": "quality_768",
        "steps": 20,
        "height": 768,
        "width": 768,
        "guidance": 3.0,
        "quant": "nf4",
        "offload": "true",
        "min_physical_vram_mb": 10000,
        "note": "Prefer square 768; skip on <10GB physical unless --force",
    },
    {
        "id": "q20_768x1024_nf4",
        "mode": "quality_20",
        "steps": 20,
        "height": 1024,
        "width": 768,
        "guidance": 3.0,
        "quant": "nf4",
        "offload": "true",
        "min_physical_vram_mb": 14000,
        "optional": True,
        "note": "Portrait cell; only when operator confirms non-square Kontext framing",
    },
    {
        "id": "q20_512_full",
        "mode": "quality_20",
        "steps": 20,
        "height": 512,
        "width": 512,
        "guidance": 3.0,
        "quant": "full",
        "offload": "auto",
        "min_physical_vram_mb": 14000,
        "note": "Full precision; expect OOM under 14GB dedicated VRAM",
    },
]


def _physical_vram_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 1)
    except Exception:
        pass
    return 0.0


def _device_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu"


def objective_image_stats(img: Image.Image) -> dict[str, Any]:
    """Diagnostic pixel stats — not human quality scores."""
    rgb = img.convert("RGB")
    gray = np.asarray(rgb.convert("L"), dtype=np.float32)
    arr = np.asarray(rgb, dtype=np.float32)
    # Simple Laplacian variance without scipy dependency.
    lap = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    gx = np.abs(gray[:, 1:] - gray[:, :-1]).mean()
    gy = np.abs(gray[1:, :] - gray[:-1, :]).mean()
    # Non-near-white pixels (garment body proxy on white studio backgrounds).
    near_white = (arr[:, :, 0] > 245) & (arr[:, :, 1] > 245) & (arr[:, :, 2] > 245)
    non_empty_pct = float((~near_white).mean() * 100.0)
    return {
        "resolution": [int(rgb.size[0]), int(rgb.size[1])],
        "laplacian_variance": round(float(np.var(lap)), 3),
        "edge_energy": round(float((gx + gy) / 2.0), 3),
        "image_variance": round(float(np.var(gray)), 3),
        "non_empty_pixel_pct": round(non_empty_pct, 2),
        "note": "Objective diagnostics only; not perceived quality scores.",
    }


def _apply_cell_env(cell: dict[str, Any]) -> dict[str, str | None]:
    """Set FLUX env for one cell; return previous values for restore."""
    keys = (
        "FLUX_QUANTIZATION",
        "FLUX_DISABLE_NF4",
        "FLUX_MODEL_CPU_OFFLOAD",
        "FLUX_PROFILE",
        "FLUX_VAE_TILING",
        "FLUX_MAX_SEQUENCE_LENGTH",
    )
    previous = {k: os.environ.get(k) for k in keys}
    os.environ["FLUX_PROFILE"] = "true"
    os.environ.setdefault("FLUX_MAX_SEQUENCE_LENGTH", "128")
    os.environ.setdefault("FLUX_VAE_TILING", "false")

    quant = cell["quant"]
    if quant == "nf4":
        os.environ["FLUX_QUANTIZATION"] = "nf4"
        os.environ.pop("FLUX_DISABLE_NF4", None)
    else:
        os.environ["FLUX_QUANTIZATION"] = "full"
        os.environ["FLUX_DISABLE_NF4"] = "true"

    offload = cell.get("offload", "true")
    if offload == "auto":
        # Loader auto-decides from physical VRAM when full precision.
        os.environ.pop("FLUX_MODEL_CPU_OFFLOAD", None)
    else:
        os.environ["FLUX_MODEL_CPU_OFFLOAD"] = str(offload)
    return previous


def _restore_env(previous: dict[str, str | None]) -> None:
    for k, v in previous.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def run_cell(
    cell: dict[str, Any],
    fabric: Image.Image,
    fabric_path: Path,
    out_dir: Path,
    garment_type: str,
    sleeve: str,
    neckline: str,
    seed: int,
) -> dict[str, Any]:
    from src.common.models.model_manager import ModelManager
    from src.features.custom_generator.pipeline.garment_generation_pipeline import (
        GarmentGenerationConfig,
        GarmentGenerationPipeline,
    )

    previous = _apply_cell_env(cell)
    record: dict[str, Any] = {
        "id": cell["id"],
        "configuration": deepcopy(cell),
        "status": "pending",
        "device": _device_name(),
        "physical_gpu_vram_mb": _physical_vram_mb(),
    }
    try:
        # Force reload so quantization/offload switches take effect.
        mgr = ModelManager()
        if getattr(mgr, "flux_manager", None) is not None:
            try:
                mgr.flux_manager.unload()
            except Exception:
                pass
            # Ensure switch_to does not early-return on stale active_model.
            mgr._active_model = None
        mgr.flux_manager.allow_fallback = False
        mgr.flux_manager.model_path = ROOT / "models" / "flux-kontext"

        t_load = time.perf_counter()
        mgr.switch_to("flux")
        load_s = round(time.perf_counter() - t_load, 3)
        loader = mgr.flux_manager.loader
        if loader is None or loader.pipeline is None:
            raise RuntimeError("Failed to load FluxKontextPipeline")

        runtime = loader.get_runtime_info() if hasattr(loader, "get_runtime_info") else {}
        cfg = GarmentGenerationConfig(
            config_dir=str(ROOT / "configs"),
            config_path=str(ROOT / "configs" / "custom_generator" / "flux_config.yaml"),
            output_root=str(ROOT / "outputs" / "generated_garments"),
            experiments_root=str(ROOT / "experiments"),
            generation_mode=cell["mode"],
            allow_fallback=False,
            seed=seed,
            height=int(cell["height"]),
            width=int(cell["width"]),
            num_inference_steps=int(cell["steps"]),
            guidance_scale=float(cell["guidance"]),
        )
        pipe = GarmentGenerationPipeline(config=cfg, model_loader=loader)
        pipe.config.seed = seed
        pipe.config.height = int(cell["height"])
        pipe.config.width = int(cell["width"])
        pipe.config.num_inference_steps = int(cell["steps"])
        pipe.config.guidance_scale = float(cell["guidance"])

        fabric_metadata = {
            "material": "cotton",
            "fabric": "cotton",
            "texture": "smooth",
            "style": "casual",
            "fit": "regular",
            "dominant_colors": [],
            "color": "match_fabric",
        }
        user = {
            "gender": "women",
            "garment_type": garment_type,
            "material": "cotton",
            "neckline": neckline,
            "sleeve": sleeve,
            "fit": "regular",
            "style": "casual",
            "occasion": "casual",
            "season": "summer",
            "force_recolor": False,
        }

        t0 = time.perf_counter()
        result = pipe.run(
            fabric_metadata=fabric_metadata,
            user_customization=user,
            output_filename=f"flux_quality_{cell['id']}",
            reference_image=fabric,
        )
        wall_s = round(time.perf_counter() - t0, 3)

        image_path = result.get("image_path") or result.get("output_path")
        meta_path = result.get("metadata_path")
        meta: dict[str, Any] = {}
        # Pipeline writes metadata next to images and an experiments *_exp.json.
        candidates = []
        if meta_path:
            candidates.append(Path(meta_path))
        if image_path:
            stem = Path(image_path).stem
            candidates.extend(
                [
                    ROOT / "outputs" / "generated_garments" / "metadata" / f"{stem}.json",
                    ROOT / "experiments" / "generation_results" / f"{stem}_exp.json",
                ]
            )
        for cand in candidates:
            if cand.exists():
                meta = json.loads(cand.read_text(encoding="utf-8"))
                meta_path = str(cand)
                break

        prompt_stats = (
            meta.get("prompt_stats")
            or result.get("prompt_stats")
            or (meta.get("stats") or {}).get("prompt_stats")
            or {}
        )
        stats = meta.get("stats") or {}
        img_stats = {}
        if image_path and Path(image_path).exists():
            img_stats = objective_image_stats(Image.open(image_path))

        record.update(
            {
                "status": result.get("status") or "completed",
                "load_time_s": load_s,
                "generation_time_s": wall_s,
                "diffusion_time_s": stats.get("diffusion_time_s")
                or stats.get("diffusion_s")
                or meta.get("diffusion_time_s")
                or meta.get("timings", {}).get("diffusion_s"),
                "vae_decode_time_s": stats.get("vae_decode_time_s")
                or stats.get("vae_decode_s")
                or meta.get("vae_decode_time_s")
                or meta.get("timings", {}).get("vae_decode_s"),
                "resolution": [cell["width"], cell["height"]],
                "steps": cell["steps"],
                "guidance": cell["guidance"],
                "precision_quantization": runtime.get("quantization_profile")
                or ("nf4" if runtime.get("bnb_4bit") else "full_precision"),
                "bnb_4bit": runtime.get("bnb_4bit"),
                "offload_strategy": runtime.get("offload_strategy"),
                "peak_cuda_allocated_mb": stats.get("peak_cuda_allocated_mb")
                or meta.get("peak_cuda_allocated_mb")
                or stats.get("peak_vram_mb")
                or meta.get("peak_vram_mb"),
                "peak_cuda_reserved_mb": stats.get("peak_cuda_reserved_mb")
                or meta.get("peak_cuda_reserved_mb"),
                "physical_gpu_vram_mb": stats.get("physical_gpu_vram_mb")
                or meta.get("physical_gpu_vram_mb")
                or record["physical_gpu_vram_mb"],
                "output_path": image_path,
                "metadata_path": meta_path,
                "prompt_stats": {
                    "token_count": prompt_stats.get("token_count"),
                    "token_budget": prompt_stats.get("token_budget"),
                    "truncated": prompt_stats.get("truncated"),
                    "prompt_compacted": prompt_stats.get("prompt_compacted"),
                },
                "objective_image_stats": img_stats,
                "fabric_path": str(fabric_path),
                "loader_runtime": runtime,
                "was_fallback_used": stats.get("was_fallback_used", meta.get("was_fallback_used")),
                "was_real_flux_used": stats.get("was_real_flux_used"),
            }
        )
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("Cell %s failed", cell["id"])
    finally:
        _restore_env(previous)
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="FLUX quality-first benchmark matrix")
    parser.add_argument("--fabric", required=False, help="Path to fabric image")
    parser.add_argument("--garment-type", default="kurti")
    parser.add_argument("--sleeve", default="three_quarter_sleeve")
    parser.add_argument("--neckline", default="v_neck")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated cell ids to run (default: all non-optional that fit VRAM)",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Include optional cells (e.. 768x1024)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run cells even when physical VRAM is below min_physical_vram_mb",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write planned cells JSON without running inference",
    )
    parser.add_argument(
        "--output",
        default=str(
            ROOT / "experiments" / "generation_results" / "flux_quality_matrix.json"
        ),
    )
    args = parser.parse_args()

    physical = _physical_vram_mb()
    device = _device_name()
    only = {x.strip() for x in args.only.split(",") if x.strip()}

    planned: list[dict[str, Any]] = []
    for cell in MATRIX:
        if only and cell["id"] not in only:
            continue
        if cell.get("optional") and not args.include_optional and not (
            only and cell["id"] in only
        ):
            planned.append(
                {
                    **cell,
                    "decision": "skipped_optional",
                    "reason": "optional cell; pass --include-optional or --only",
                }
            )
            continue
        min_v = float(cell.get("min_physical_vram_mb") or 0)
        if not args.force and physical > 0 and physical < min_v:
            planned.append(
                {
                    **cell,
                    "decision": "skipped_insufficient_vram",
                    "reason": (
                        f"physical_vram_mb={physical} < required {min_v}; "
                        "re-run on ~16GB GPU or pass --force"
                    ),
                    "device": device,
                    "physical_gpu_vram_mb": physical,
                }
            )
            continue
        planned.append({**cell, "decision": "run", "device": device, "physical_gpu_vram_mb": physical})

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.plan_only:
        payload = {
            "status": "plan_only",
            "device": device,
            "physical_gpu_vram_mb": physical,
            "cells": planned,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 0

    if not args.fabric:
        print("ERROR: --fabric is required unless --plan-only", file=sys.stderr)
        return 2
    fabric_path = Path(args.fabric)
    if not fabric_path.is_file():
        print(f"ERROR: fabric not found: {fabric_path}", file=sys.stderr)
        return 2

    fabric = Image.open(fabric_path).convert("RGB")
    results: list[dict[str, Any]] = []
    for cell in planned:
        if cell.get("decision") != "run":
            results.append(
                {
                    "id": cell["id"],
                    "status": cell["decision"],
                    "reason": cell.get("reason"),
                    "configuration": cell,
                    "device": device,
                    "physical_gpu_vram_mb": physical,
                }
            )
            continue
        logger.info("=== Running FLUX cell %s ===", cell["id"])
        results.append(
            run_cell(
                cell=cell,
                fabric=fabric,
                fabric_path=fabric_path,
                out_dir=out_path.parent,
                garment_type=args.garment_type,
                sleeve=args.sleeve,
                neckline=args.neckline,
                seed=args.seed,
            )
        )
        out_path.write_text(
            json.dumps(
                {
                    "status": "partial",
                    "device": device,
                    "physical_gpu_vram_mb": physical,
                    "results": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    payload = {
        "status": "completed",
        "device": device,
        "physical_gpu_vram_mb": physical,
        "fabric_path": str(fabric_path),
        "results": results,
        "recommended_note": (
            "Pick the best cell by inspecting outputs + objective stats. "
            "Do not treat laplacian_variance as perceptual quality."
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
