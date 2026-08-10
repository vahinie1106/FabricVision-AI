"""Day 17 fix — run a real Standard FLUX.1-Kontext generation with full profiling.

Usage (venv):
  python scripts/run_day17_fix_generation.py --image path/to/fabric.png
  python scripts/run_day17_fix_generation.py --image path/to/fabric.png --mode Production
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Fabric image path")
    parser.add_argument("--mode", default="Standard", choices=["Preview", "Standard", "Production"])
    parser.add_argument("--garment-type", default="kurti")
    parser.add_argument("--sleeve", default="three_quarter_sleeve")
    parser.add_argument("--neckline", default="v_neck")
    parser.add_argument("--color", default="royal blue")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", default="day17_fix")
    args = parser.parse_args()

    os.environ.setdefault("FLUX_PROFILE", "true")
    os.environ.setdefault("FLUX_MAX_SEQUENCE_LENGTH", "128")
    os.environ.setdefault("FLUX_VAE_TILING", "false")

    from PIL import Image

    from src.common.models.model_manager import ModelManager
    from src.features.custom_generator.pipeline.garment_generation_pipeline import (
        GarmentGenerationConfig,
        GarmentGenerationPipeline,
    )

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Image not found: {img_path}", file=sys.stderr)
        return 2

    fabric = Image.open(img_path).convert("RGB")
    mgr = ModelManager()
    mgr.flux_manager.allow_fallback = False
    mgr.flux_manager.model_path = ROOT / "models" / "flux-kontext"

    t_load = time.perf_counter()
    print("[BENCH] Loading / reusing FLUX.1-Kontext...", flush=True)
    mgr.switch_to("flux")
    load_s = round(time.perf_counter() - t_load, 3)
    loader = mgr.flux_manager.loader
    if loader is None or loader.pipeline is None:
        print("Failed to load Kontext pipeline", file=sys.stderr)
        return 3
    print(f"[BENCH] Model ready in {load_s}s reused={getattr(loader, '_reuse_count', 0)}", flush=True)

    cfg = GarmentGenerationConfig(
        config_dir=str(ROOT / "configs"),
        config_path=str(ROOT / "configs" / "custom_generator" / "flux_config.yaml"),
        output_root=str(ROOT / "outputs" / "generated_garments"),
        experiments_root=str(ROOT / "experiments"),
        generation_mode=args.mode,
        allow_fallback=False,
        seed=args.seed,
    )
    # Force seed onto config after YAML load
    pipe = GarmentGenerationPipeline(config=cfg, model_loader=loader)
    pipe.config.seed = args.seed

    fabric_metadata = {
        "material": "cotton",
        "fabric": "cotton",
        "texture": "smooth",
        "style": "casual",
        "fit": "slim",
    }
    user = {
        "gender": "women",
        "garment_type": args.garment_type,
        "material": "cotton",
        "neckline": args.neckline,
        "sleeve": args.sleeve,
        "fit": "slim",
        "style": "casual",
        "occasion": "casual",
        "season": "summer",
        "force_recolor": False,
    }
    color_key = args.color.lower().replace(" ", "_")
    if color_key not in ("match_fabric", "match-fabric", ""):
        user["color"] = args.color
        user["force_recolor"] = True
        fabric_metadata["dominant_colors"] = [args.color]
        fabric_metadata["color"] = args.color
    else:
        fabric_metadata["dominant_colors"] = []
        fabric_metadata["color"] = "match_fabric"

    print(
        f"[BENCH] Generating mode={pipe.config.generation_mode} "
        f"{pipe.config.width}x{pipe.config.height} steps={pipe.config.num_inference_steps} "
        f"guidance={pipe.config.guidance_scale}",
        flush=True,
    )
    t0 = time.perf_counter()
    result = pipe.run(
        fabric_metadata=fabric_metadata,
        user_customization=user,
        reference_image=fabric,
        output_filename=args.label,
    )
    wall = round(time.perf_counter() - t0, 3)
    stats = getattr(pipe.inference_engine, "last_execution_stats", {}) or {}

    report = {
        "label": args.label,
        "mode": pipe.config.generation_mode,
        "wall_clock_s": wall,
        "model_switch_s": load_s,
        "image_path": result.get("image_path"),
        "raw_image_path": (result.get("metadata") or {}).get("raw_image_path"),
        "config": {
            "height": pipe.config.height,
            "width": pipe.config.width,
            "steps": pipe.config.num_inference_steps,
            "guidance": pipe.config.guidance_scale,
            "seed": pipe.config.seed,
            "max_sequence_length": os.environ.get("FLUX_MAX_SEQUENCE_LENGTH"),
        },
        "stats": stats,
        "prompt_stats": (result.get("metadata") or {}).get("prompt_stats"),
    }
    out = ROOT / "experiments" / "generation_results" / f"{args.label}_bench.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"[BENCH] Wrote {out}", flush=True)
    print(f"[BENCH] TOTAL wall={wall}s generation_time_s={stats.get('generation_time_s')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
