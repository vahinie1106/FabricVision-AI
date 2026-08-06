import json
import logging
import sys
from pathlib import Path

workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from src.garment_generation.pipeline.garment_generation_pipeline import GarmentGenerationConfig, GarmentGenerationPipeline

BENCHMARK_GARMENTS = [
    # Women (5 garments)
    {"gender": "women", "garment_type": "kurti", "material": "cotton", "color": "royal_blue", "pattern": "floral", "sleeve": "three_quarter_sleeve", "neckline": "round_neck", "size": "M"},
    {"gender": "women", "garment_type": "saree", "material": "silk", "color": "crimson", "pattern": "paisley", "sleeve": "short_sleeve", "neckline": "sweetheart_neck", "size": "M"},
    {"gender": "women", "garment_type": "lehenga", "material": "velvet", "color": "emerald_green", "pattern": "embroidered", "sleeve": "sleeveless", "neckline": "v_neck", "size": "S"},
    {"gender": "women", "garment_type": "dress", "material": "chiffon", "color": "pastel_pink", "pattern": "solid", "sleeve": "long_sleeve", "neckline": "square_neck", "size": "M"},
    {"gender": "women", "garment_type": "jacket", "material": "denim", "color": "black", "pattern": "solid", "sleeve": "long_sleeve", "neckline": "collar", "size": "L"},
    # Men (5 garments)
    {"gender": "men", "garment_type": "shirt", "material": "linen", "color": "white", "pattern": "striped", "sleeve": "long_sleeve", "neckline": "collar", "size": "L"},
    {"gender": "men", "garment_type": "t_shirt", "material": "cotton", "color": "navy_blue", "pattern": "solid", "sleeve": "short_sleeve", "neckline": "crew_neck", "size": "M"},
    {"gender": "men", "garment_type": "hoodie", "material": "fleece", "color": "charcoal_grey", "pattern": "solid", "sleeve": "long_sleeve", "neckline": "hooded", "size": "XL"},
    {"gender": "men", "garment_type": "jacket", "material": "leather", "color": "brown", "pattern": "solid", "sleeve": "long_sleeve", "neckline": "collar", "size": "L"},
    {"gender": "men", "garment_type": "kurta", "material": "cotton", "color": "mustard_yellow", "pattern": "printed", "sleeve": "long_sleeve", "neckline": "mandarin_collar", "size": "L"},
]

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    config = GarmentGenerationConfig(
        config_dir=str(workspace_root / "configs"),
        output_root=str(workspace_root / "outputs" / "generated_garments"),
        experiments_root=str(workspace_root / "experiments"),
        height=1024,
        width=1024,
        num_inference_steps=4,
        guidance_scale=3.5,
    )
    
    pipeline = GarmentGenerationPipeline(config=config)
    results = []

    print(f"=== Starting FLUX 10-Garment Benchmark Run ({len(BENCHMARK_GARMENTS)} items) ===")
    
    for idx, item in enumerate(BENCHMARK_GARMENTS, 1):
        fabric_metadata = {
            "material": item["material"],
            "pattern": item["pattern"],
            "texture": "smooth",
            "dominant_colors": [item["color"]],
            "style": "classic"
        }
        user_customization = {
            "gender": item["gender"],
            "garment_type": item["garment_type"],
            "sleeve": item["sleeve"],
            "neckline": item["neckline"],
            "size": item["size"]
        }
        
        filename = f"benchmark_{idx:02d}_{item['gender']}_{item['garment_type']}"
        print(f"\n[{idx}/10] Generating {item['gender']} {item['garment_type']} ({item['color']}, {item['material']}, {item['pattern']})...")
        
        out = pipeline.run(
            fabric_metadata=fabric_metadata,
            user_customization=user_customization,
            output_filename=filename
        )
        
        meta = out["metadata"]
        print(f"  - Status: {out['status']}")
        print(f"  - Output Image: {out['image_path']}")
        print(f"  - Resolution: {meta['generation_parameters']['width']}x{meta['generation_parameters']['height']}")
        print(f"  - Validation Valid: {meta['validation']['valid']}")
        
        results.append(out)

    summary_file = workspace_root / "experiments" / "generation_results" / "benchmark_10_garments_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump([r["metadata"] for r in results], f, indent=2)

    print(f"\n=== Benchmark Complete! Saved summary to {summary_file} ===")

if __name__ == "__main__":
    main()
