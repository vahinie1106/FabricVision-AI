"""One-shot Day-18 full-dress CatVTON validation.

Requires explicit --person and --garment paths. Will refuse fabric sheets or
geometric placeholders used as the person image.
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
os.environ.setdefault("CATVTON_DEBUG", "1")
os.environ.setdefault("CATVTON_STEPS", "30")
os.environ.setdefault("CATVTON_GUIDANCE", "2.5")
os.environ["CATVTON_ALLOW_FALLBACK"] = "0"

from src.features.virtual_tryon.models import (  # noqa: E402
    GarmentConditioningInput,
    PersonConditioningInput,
)
from src.features.virtual_tryon.person_image_validation import assess_person_image  # noqa: E402
from src.features.virtual_tryon.tryon_pipeline import TryOnConfig, VirtualTryOnPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Day-18 CatVTON dress validation")
    parser.add_argument(
        "--person",
        required=True,
        help="Path to a REAL person photograph (not a fabric swatch / placeholder)",
    )
    parser.add_argument(
        "--garment",
        required=True,
        help="Path to garment image (e.g. outputs/generated_garments/images/...)",
    )
    parser.add_argument("--garment-type", default="dress")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    person_path = Path(args.person)
    garment_path = Path(args.garment)
    if not person_path.exists() or not garment_path.exists():
        raise SystemExit(
            f"Missing inputs person={person_path.exists()} ({person_path}) "
            f"garment={garment_path.exists()} ({garment_path})"
        )

    person = Image.open(person_path).convert("RGB")
    ok, reason = assess_person_image(person)
    if not ok:
        raise SystemExit(
            f"Invalid person image for Day-18 quality validation: {reason}\n"
            f"Do not use fabric uploads or examples/person/person_01.png."
        )

    config = TryOnConfig(
        height=512,
        width=384,
        allow_fallback=False,
        num_inference_steps=30,
        guidance_scale=2.5,
        attn_ckpt_version="vitonhd",
        output_root=str(ROOT / "outputs" / "virtual_tryon"),
        experiments_root=str(ROOT / "experiments"),
    )
    pipeline = VirtualTryOnPipeline(config=config)
    result = pipeline.run(
        person_input=PersonConditioningInput(person_image=person),
        garment_input=GarmentConditioningInput(
            garment_image=Image.open(garment_path).convert("RGB"),
            garment_type=args.garment_type,
        ),
        output_filename="tryon_day18_full_dress",
        person_filename=person_path.name,
        garment_filename=garment_path.name,
    )
    meta = {}
    if result.metadata_path and Path(result.metadata_path).exists():
        meta = json.loads(Path(result.metadata_path).read_text(encoding="utf-8"))
    summary = {
        "validation_tier": "quality" if ok else "infrastructure",
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
        "raw_equals_final": meta.get("raw_equals_final"),
        "resolution": meta.get("resolution") or meta.get("output_dimensions"),
        "steps": meta.get("num_inference_steps") or meta.get("steps"),
        "guidance": meta.get("guidance_scale") or meta.get("guidance"),
        "runtime": meta.get("runtime_sec") or meta.get("runtime"),
        "checkpoint": meta.get("attn_ckpt_version") or meta.get("checkpoint"),
        "inference_backend": meta.get("inference_backend"),
        "mask_stats": meta.get("mask_stats"),
    }
    out = ROOT / "experiments" / "tryon_results" / "day18_full_dress_validate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
