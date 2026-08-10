"""One-shot Day-18 full-dress CatVTON validation (same person + pink dress)."""
from __future__ import annotations

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
from src.features.virtual_tryon.tryon_pipeline import TryOnConfig, VirtualTryOnPipeline  # noqa: E402


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    person_path = ROOT / "data" / "uploads" / "8dd96489cf1a4262a9eeccfac617fd90.jpg"
    garment_path = ROOT / "outputs" / "generated_garments" / "images" / "garment_aca36127.png"
    if not person_path.exists() or not garment_path.exists():
        raise SystemExit(f"Missing inputs person={person_path.exists()} garment={garment_path.exists()}")

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
        person_input=PersonConditioningInput(person_image=Image.open(person_path).convert("RGB")),
        garment_input=GarmentConditioningInput(
            garment_image=Image.open(garment_path).convert("RGB"),
            garment_type="dress",
        ),
        output_filename="tryon_day18_full_dress",
        person_filename=person_path.name,
        garment_filename=garment_path.name,
    )
    meta = {}
    if result.metadata_path and Path(result.metadata_path).exists():
        meta = json.loads(Path(result.metadata_path).read_text(encoding="utf-8"))
    summary = {
        "status": result.status,
        "image_path": result.image_path,
        "metadata_path": result.metadata_path,
        "cloth_type": meta.get("cloth_type"),
        "mask_source": meta.get("mask_source"),
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
