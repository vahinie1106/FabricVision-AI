from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from PIL import Image, ImageDraw

workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from src.virtual_tryon.models import PersonConditioningInput, GarmentConditioningInput
from src.virtual_tryon.tryon_pipeline import TryOnConfig, VirtualTryOnPipeline
from src.models.model_manager import ModelManager


def main():
    parser = argparse.ArgumentParser(description="FabricVision-AI Automated CatVTON Virtual Try-On Pipeline")
    parser.add_argument("--person_image", type=str, default=None, help="Path to target person image")
    parser.add_argument("--garment_image", type=str, default=None, help="Path to FLUX generated garment image")
    parser.add_argument("--allow_fallback", action="store_true", default=False, help="Allow fallback generation if model weights missing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # 1. Resolve Person Image
    if args.person_image:
        person_path = Path(args.person_image)
    else:
        person_path = workspace_root / "examples" / "person" / "person_01.png"
        if not person_path.exists():
            person_path.parent.mkdir(parents=True, exist_ok=True)
            img = Image.new("RGB", (1024, 1024), color=(240, 240, 245))
            draw = ImageDraw.Draw(img)
            draw.ellipse([412, 120, 612, 320], fill=(235, 205, 185))
            draw.polygon([(360, 320), (664, 320), (740, 850), (284, 850)], fill=(220, 220, 230))
            img.save(person_path, "PNG")

    person_img = Image.open(person_path)

    # 2. Resolve FLUX Generated Garment Image
    if args.garment_image:
        garment_path = Path(args.garment_image)
    else:
        garment_dir = workspace_root / "outputs" / "garment_generation" / "images"
        if not garment_dir.exists():
            garment_dir = workspace_root / "outputs" / "generated_garments" / "images"
        garment_files = list(garment_dir.glob("*.png")) if garment_dir.exists() else []
        if garment_files:
            garment_path = garment_files[0]
        else:
            garment_path = workspace_root / "outputs" / "garment_generation" / "images" / "flux_garment.png"
            garment_path.parent.mkdir(parents=True, exist_ok=True)
            g_img = Image.new("RGB", (512, 512), color=(255, 255, 255))
            draw = ImageDraw.Draw(g_img)
            draw.rectangle([128, 100, 384, 450], fill=(20, 50, 180), outline=(200, 200, 210), width=3)
            g_img.save(garment_path, "PNG")

    garment_img = Image.open(garment_path)

    # 3. Model lifecycle management: Switch to CatVTON sequentially
    model_manager = ModelManager()
    print("=== Step 1: Sequential Model Switch to CatVTON ===")
    model_manager.switch_to("catvton")

    # 4. Initialize CatVTON Pipeline
    config = TryOnConfig(
        config_dir=str(workspace_root / "configs"),
        output_root=str(workspace_root / "outputs" / "virtual_tryon"),
        experiments_root=str(workspace_root / "experiments"),
        height=1024,
        width=1024,
        allow_fallback=args.allow_fallback,
    )
    pipeline = VirtualTryOnPipeline(config=config)

    # 5. Run Virtual Try-On
    person_input = PersonConditioningInput(person_image=person_img)
    garment_input = GarmentConditioningInput(garment_image=garment_img, garment_type="kurti")

    print("\n=== Step 2: Running Real CatVTON Virtual Try-On Pipeline ===")
    result = pipeline.run(
        person_input=person_input,
        garment_input=garment_input,
        output_filename=f"tryon_{garment_path.stem}",
        person_filename=person_path.name,
        garment_filename=garment_path.name,
    )

    print(f"\nTry-On Status: {result.status}")
    print(f"Final Try-On Image Path: {result.image_path}")
    print(f"Metadata Summary Path: {result.metadata_path}")

    # Clean up GPU memory
    model_manager.clear_vram()


if __name__ == "__main__":
    main()
