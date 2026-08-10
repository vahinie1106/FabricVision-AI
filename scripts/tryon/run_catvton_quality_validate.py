"""CatVTON quality validation — requires a REAL person photo + garment image.

Tiers (explicit):
  1. unit/dry-run — pytest with synthetic images (not this script)
  2. infrastructure — real CatVTON weights load + denoise
  3. quality — real person photo + real garment (this script)

Usage:
  python scripts/tryon/run_catvton_quality_validate.py \\
    --person path/to/real_person.jpg \\
    --garment path/to/garment.png \\
    --garment-type dress
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("CATVTON_REQUIRE_REAL", "1")
os.environ.setdefault("CATVTON_ALLOW_FALLBACK", "0")

from src.features.virtual_tryon.models import (  # noqa: E402
    GarmentConditioningInput,
    PersonConditioningInput,
)
from src.features.virtual_tryon.person_image_validation import assess_person_image  # noqa: E402
from src.features.virtual_tryon.tryon_pipeline import TryOnConfig, VirtualTryOnPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Real CatVTON quality validation (requires a real person photo)."
    )
    parser.add_argument("--person", required=True, help="Path to a real person photograph")
    parser.add_argument("--garment", required=True, help="Path to garment / clothing image")
    parser.add_argument("--garment-type", default="dress")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--label", default="catvton_quality")
    parser.add_argument(
        "--mask-strategy",
        default="auto",
        choices=["auto", "automasker", "grabcut", "provided_only"],
        help="Mask resolution policy (AutoMasker needs detectron2+fvcore).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    person_path = Path(args.person)
    garment_path = Path(args.garment)
    if not person_path.is_file():
        print(f"ERROR: person image not found: {person_path}", file=sys.stderr)
        print(
            "Provide a real full-/half-body person photograph. "
            "Do not use fabric swatches or examples/person/person_01.png placeholder.",
            file=sys.stderr,
        )
        return 2
    if not garment_path.is_file():
        print(f"ERROR: garment image not found: {garment_path}", file=sys.stderr)
        return 2

    person = Image.open(person_path).convert("RGB")
    garment = Image.open(garment_path).convert("RGB")
    ok, reason = assess_person_image(person)
    if not ok:
        print(f"ERROR: invalid person image for quality validation: {reason}", file=sys.stderr)
        return 3

    os.environ["CATVTON_MASK_STRATEGY"] = args.mask_strategy
    if args.mask_strategy == "automasker":
        os.environ["CATVTON_USE_AUTOMASKER"] = "true"

    config = TryOnConfig(
        height=args.height,
        width=args.width,
        allow_fallback=False,
        num_inference_steps=args.steps,
        guidance_scale=2.5,
        attn_ckpt_version="vitonhd",
        output_root=str(ROOT / "outputs" / "virtual_tryon"),
        experiments_root=str(ROOT / "experiments"),
    )
    pipeline = VirtualTryOnPipeline(config=config)
    result = pipeline.run(
        person_input=PersonConditioningInput(person_image=person),
        garment_input=GarmentConditioningInput(
            garment_image=garment,
            garment_type=args.garment_type,
        ),
        output_filename=args.label,
        person_filename=person_path.name,
        garment_filename=garment_path.name,
    )

    meta = {}
    if result.metadata_path and Path(result.metadata_path).exists():
        meta = json.loads(Path(result.metadata_path).read_text(encoding="utf-8"))

    summary = {
        "validation_tier": "quality",
        "status": result.status,
        "image_path": result.image_path,
        "metadata_path": result.metadata_path,
        "person_path": str(person_path),
        "garment_path": str(garment_path),
        "person_assessment": reason,
        "cloth_type": meta.get("cloth_type"),
        "mask_source": meta.get("mask_source"),
        "mask_strategy": meta.get("mask_strategy"),
        "mask_attempts": meta.get("mask_attempts"),
        "was_real_catvton_used": meta.get("was_real_catvton_used"),
        "was_fallback_used": meta.get("was_fallback_used"),
        "inference_backend": meta.get("inference_backend"),
        "resolution": meta.get("resolution"),
        "steps": meta.get("num_inference_steps"),
        "guidance": meta.get("guidance_scale"),
        "inference_time_s": meta.get("inference_time_s"),
        "peak_vram_mb": meta.get("peak_vram_mb"),
        "model_device": meta.get("model_device"),
    }
    out = ROOT / "experiments" / "tryon_results" / f"{args.label}_quality.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if not meta.get("was_real_catvton_used") or meta.get("was_fallback_used"):
        print("ERROR: quality run did not use real CatVTON.", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
