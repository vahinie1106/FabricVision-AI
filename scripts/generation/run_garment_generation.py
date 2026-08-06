from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from src.garment_generation.pipeline.garment_generation_pipeline import GarmentGenerationConfig, GarmentGenerationPipeline
from src.semantic_analysis.pipeline.semantic_analysis_pipeline import SemanticAnalysisConfig, SemanticAnalysisPipeline


def main():
    parser = argparse.ArgumentParser(description="FabricVision-AI Automated Garment Generation from Semantic Analysis")
    parser.add_argument("--image_path", type=str, default=None, help="Path to input fabric/garment image")
    parser.add_argument("--allow_fallback", action="store_true", default=False, help="Allow fallback generation if model weights missing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # 1. Resolve input garment image
    if args.image_path:
        input_image_path = Path(args.image_path)
    else:
        sample_dir = workspace_root / "datasets" / "sample_image"
        sample_images = list(sample_dir.glob("*.jpg")) if sample_dir.exists() else []
        if sample_images:
            input_image_path = sample_images[0]
        else:
            input_image_path = workspace_root / "examples" / "person" / "person_01.png"

    print(f"=== Step 1: Running Semantic Analysis on Input Image: {input_image_path.name} ===")
    
    # 2. Run Semantic Analysis Pipeline (Qwen2.5-VL / Metadata Builder)
    semantic_config = SemanticAnalysisConfig(
        config_dir=str(workspace_root / "configs"),
        output_root=str(workspace_root / "outputs" / "semantic_analysis"),
    )
    semantic_pipeline = SemanticAnalysisPipeline(config=semantic_config)
    semantic_result = semantic_pipeline.run(input_image_path)

    extracted_metadata = semantic_result.get("metadata", {})
    print("\nExtracted Semantic Analysis Metadata:")
    print(json.dumps(extracted_metadata, indent=2))

    # 3. Formulate Customization Parameters from Metadata
    user_customization = {
        "gender": extracted_metadata.get("gender") or "women",
        "garment_type": extracted_metadata.get("garment_type") or "kurti",
        "sleeve": extracted_metadata.get("sleeve") or "three_quarter_sleeve",
        "neckline": extracted_metadata.get("neckline") or "round_neck",
        "size": extracted_metadata.get("size") or "M",
    }

    # 4. Run FLUX Garment Generation Pipeline
    print("\n=== Step 2: Running Real FLUX Garment Generation Pipeline ===")
    flux_config = GarmentGenerationConfig(
        config_dir=str(workspace_root / "configs"),
        output_root=str(workspace_root / "outputs" / "garment_generation"),
        experiments_root=str(workspace_root / "experiments"),
        height=512,
        width=512,
        num_inference_steps=4,
        guidance_scale=3.5,
        allow_fallback=args.allow_fallback,
    )
    
    flux_pipeline = GarmentGenerationPipeline(config=flux_config)
    result = flux_pipeline.run(
        fabric_metadata=extracted_metadata,
        user_customization=user_customization,
        output_filename=f"flux_gen_{input_image_path.stem}",
    )

    print(f"\nGeneration Status: {result['status']}")
    print(f"Generated Garment Image Path: {result['image_path']}")
    print(f"Output Metadata Summary Path: {result['metadata_path']}")

    stats = getattr(flux_pipeline.inference_engine, "last_execution_stats", {})
    print("\n=== Real Inference Proof & Hardware Metrics ===")
    print(f"- Was Fallback Used?: {stats.get('was_fallback_used', False)}")
    print(f"- Was Real FLUX Weights Used?: {stats.get('was_real_flux_used', True)}")
    print(f"- Before Inference VRAM: {stats.get('vram_before_mb', 0.0):.2f} MB")
    print(f"- After Inference VRAM: {stats.get('vram_after_mb', 0.0):.2f} MB")
    print(f"- Peak VRAM Memory: {stats.get('peak_vram_mb', 0.0):.2f} MB")
    print(f"- Total Generation Time: {stats.get('generation_time_s', 0.0):.2f} seconds")


if __name__ == "__main__":
    main()
