from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from PIL import Image

workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from src.dataset_management.utils import serialize_json
from src.garment_generation.pipeline.garment_generation_pipeline import GarmentGenerationConfig, GarmentGenerationPipeline
from src.models.model_manager import ModelManager
from src.semantic_analysis.pipeline.semantic_analysis_pipeline import SemanticAnalysisConfig, SemanticAnalysisPipeline
from src.virtual_tryon.models import GarmentConditioningInput, PersonConditioningInput
from src.virtual_tryon.tryon_pipeline import TryOnConfig, VirtualTryOnPipeline


def main():
    parser = argparse.ArgumentParser(description="FabricVision-AI Production Dataset Processing Pipeline")
    parser.add_argument("--gender", type=str, default="women", choices=["women", "men", "all"], help="Gender category to process")
    parser.add_argument("--limit", type=int, default=5, help="Number of garments per category to process (default: 5)")
    parser.add_argument("--allow_fallback", action="store_true", default=False, help="Allow fallback generation if model weights missing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("fabricvision.dataset_pipeline")

    print(f"=== Starting Production Dataset Pipeline (Gender: {args.gender}, Limit: {args.limit}) ===")

    # 1. Gather dataset input garment images
    dataset_root = workspace_root / "datasets" / "fashion_garments" / "img"
    image_paths = []
    
    if args.gender in ["women", "all"]:
        women_dir = dataset_root / "WOMEN"
        if women_dir.exists():
            image_paths.extend(list(women_dir.rglob("*.jpg"))[: args.limit])
            
    if args.gender in ["men", "all"]:
        men_dir = dataset_root / "MEN"
        if men_dir.exists():
            image_paths.extend(list(men_dir.rglob("*.jpg"))[: args.limit])

    if not image_paths:
        sample_dir = workspace_root / "datasets" / "sample_image"
        if sample_dir.exists():
            image_paths = list(sample_dir.glob("*.jpg"))[: args.limit]

    logger.info("Found %d dataset garment images for processing", len(image_paths))

    person_path = workspace_root / "examples" / "person" / "person_01.png"
    person_img = Image.open(person_path) if person_path.exists() else Image.new("RGB", (1024, 1024), color=(240, 240, 245))

    output_root = workspace_root / "outputs"
    model_manager = ModelManager()

    # Initialize pipeline configs
    semantic_pipeline = SemanticAnalysisPipeline(config=SemanticAnalysisConfig(
        config_dir=str(workspace_root / "configs"),
        output_root=str(output_root / "semantic_analysis"),
    ))

    flux_pipeline = GarmentGenerationPipeline(config=GarmentGenerationConfig(
        config_dir=str(workspace_root / "configs"),
        output_root=str(output_root / "garment_generation"),
        experiments_root=str(workspace_root / "experiments"),
        height=512,
        width=512,
        num_inference_steps=4,
        guidance_scale=3.5,
        allow_fallback=args.allow_fallback,
    ))

    tryon_pipeline = VirtualTryOnPipeline(config=TryOnConfig(
        config_dir=str(workspace_root / "configs"),
        output_root=str(output_root / "virtual_tryon"),
        experiments_root=str(workspace_root / "experiments"),
        height=512,
        width=512,
        allow_fallback=args.allow_fallback,
    ))

    processed_records = []

    for idx, img_p in enumerate(image_paths, 1):
        garment_id = f"dataset_{idx:03d}_{img_p.stem[:15]}"
        logger.info("\n--- Processing Garment [%d/%d]: %s ---", idx, len(image_paths), garment_id)

        # 1. Semantic Analysis (Qwen2.5-VL)
        model_manager.switch_to("qwen")
        sem_res = semantic_pipeline.run(img_p)
        extracted_meta = sem_res.get("metadata", {})

        user_customization = {
            "gender": extracted_meta.get("gender") or "women",
            "garment_type": extracted_meta.get("garment_type") or "kurti",
            "sleeve": extracted_meta.get("sleeve") or "three_quarter_sleeve",
            "neckline": extracted_meta.get("neckline") or "round_neck",
            "size": extracted_meta.get("size") or "M",
        }

        # 2. FLUX Garment Generation
        model_manager.switch_to("flux")
        flux_res = flux_pipeline.run(
            fabric_metadata=extracted_meta,
            user_customization=user_customization,
            output_filename=garment_id,
        )

        gen_img_path = flux_res.get("image_path")
        garment_img = Image.open(gen_img_path) if gen_img_path and Path(gen_img_path).exists() else Image.new("RGB", (512, 512), (20, 50, 180))

        # 3. CatVTON Virtual Try-On
        model_manager.switch_to("catvton")
        person_in = PersonConditioningInput(person_image=person_img)
        garment_in = GarmentConditioningInput(garment_image=garment_img, garment_type=user_customization["garment_type"])

        tryon_res = tryon_pipeline.run(
            person_input=person_in,
            garment_input=garment_in,
            output_filename=f"tryon_{garment_id}",
            person_filename=person_path.name,
            garment_filename=Path(gen_img_path).name if gen_img_path else "garment.png",
        )

        processed_records.append({
            "garment_id": garment_id,
            "input_garment_path": str(img_p),
            "semantic_metadata": extracted_meta,
            "flux_output_path": gen_img_path,
            "tryon_output_path": tryon_res.image_path,
            "status": "completed",
        })

    model_manager.clear_vram()

    summary_file = output_root / "reports" / "dataset_processing_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    serialize_json({"total_processed": len(processed_records), "records": processed_records}, summary_file)

    print(f"\n=== Dataset Pipeline Completed ({len(processed_records)} items processed) ===")
    print(f"Summary report saved to: {summary_file}")


if __name__ == "__main__":
    main()
