from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from PIL import Image, ImageDraw

workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

try:
    import torch
except ImportError:
    torch = None

from src.dataset_management.utils import serialize_json
from src.garment_generation.pipeline.garment_generation_pipeline import GarmentGenerationConfig, GarmentGenerationPipeline
from src.models.model_manager import ModelManager
from src.virtual_tryon.models import GarmentConditioningInput, PersonConditioningInput
from src.virtual_tryon.tryon_pipeline import TryOnConfig, VirtualTryOnPipeline


def main():
    parser = argparse.ArgumentParser(description="FabricVision-AI Automated Benchmark Pipeline")
    parser.add_argument("--limit", type=int, default=10, help="Number of garments to process in benchmark (default: 10)")
    parser.add_argument("--allow_fallback", action="store_true", default=False, help="Allow fallback execution during benchmark")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("fabricvision.benchmark")

    print(f"=== Starting FabricVision-AI Automated Benchmark ({args.limit} Garments) ===")

    # 1. Collect sample garment dataset images
    dataset_dir = workspace_root / "datasets" / "sample_image"
    garment_images = list(dataset_dir.glob("*.jpg")) if dataset_dir.exists() else []
    
    if len(garment_images) < args.limit:
        # Fallback to recursively gathering images under datasets/
        all_dataset_imgs = list((workspace_root / "datasets").rglob("*.jpg")) + list((workspace_root / "datasets").rglob("*.png"))
        garment_images.extend([img for img in all_dataset_imgs if img not in garment_images])

    selected_images = garment_images[: args.limit]
    
    # If no dataset images are found, generate dummy input files
    if not selected_images:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for i in range(args.limit):
            dummy_file = dataset_dir / f"benchmark_sample_{i+1:02d}.jpg"
            img = Image.new("RGB", (512, 512), color=(200 + i * 5, 210, 220))
            draw = ImageDraw.Draw(img)
            draw.rectangle([100, 100, 412, 412], fill=(20, 50, 180))
            img.save(dummy_file, "JPEG")
            selected_images.append(dummy_file)

    # 2. Setup person image for virtual try-on
    person_path = workspace_root / "examples" / "person" / "person_01.png"
    if not person_path.exists():
        person_path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (1024, 1024), color=(240, 240, 245))
        draw = ImageDraw.Draw(img)
        draw.ellipse([412, 120, 612, 320], fill=(235, 205, 185))
        draw.polygon([(360, 320), (664, 320), (740, 850), (284, 850)], fill=(220, 220, 230))
        img.save(person_path, "PNG")
    person_img = Image.open(person_path)

    model_manager = ModelManager()

    # 3. Benchmark tracking statistics
    results = []
    success_count = 0
    validation_pass_count = 0
    failure_count = 0
    total_generation_time = 0.0
    total_tryon_time = 0.0
    peak_gpu_vram_mb = 0.0

    output_root = workspace_root / "outputs"
    reports_dir = output_root / "reports" / "benchmark"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Initialize pipelines
    flux_config = GarmentGenerationConfig(
        config_dir=str(workspace_root / "configs"),
        output_root=str(output_root / "garment_generation"),
        experiments_root=str(workspace_root / "experiments"),
        height=512,
        width=512,
        num_inference_steps=4,
        guidance_scale=3.5,
        allow_fallback=args.allow_fallback,
    )
    flux_pipeline = GarmentGenerationPipeline(config=flux_config)

    tryon_config = TryOnConfig(
        config_dir=str(workspace_root / "configs"),
        output_root=str(output_root / "virtual_tryon"),
        experiments_root=str(workspace_root / "experiments"),
        height=512,
        width=512,
        allow_fallback=args.allow_fallback,
    )
    tryon_pipeline = VirtualTryOnPipeline(config=tryon_config)

    t_start_all = time.time()

    # 4. Iterate over garments
    for idx, img_path in enumerate(selected_images, 1):
        garment_id = f"benchmark_{idx:03d}_{img_path.stem[:20]}"
        logger.info("Processing garment %d/%d: %s", idx, len(selected_images), garment_id)

        sample_metadata = {
            "gender": "women" if idx % 2 != 0 else "men",
            "garment_type": "kurti" if idx % 2 != 0 else "shirt",
            "material": "cotton",
            "pattern": "floral" if idx % 2 != 0 else "solid",
            "dominant_colors": ["royal_blue"] if idx % 2 != 0 else ["navy"],
            "sleeve": "three_quarter_sleeve" if idx % 2 != 0 else "long_sleeve",
            "neckline": "round_neck" if idx % 2 != 0 else "collar",
            "size": "M",
        }

        # Stage 1: FLUX Garment Generation
        t0_gen = time.time()
        model_manager.switch_to("flux")
        gen_res = flux_pipeline.run(
            fabric_metadata=sample_metadata,
            user_customization=sample_metadata,
            output_filename=garment_id,
        )
        t1_gen = time.time()
        gen_duration = round(t1_gen - t0_gen, 2)
        total_generation_time += gen_duration

        gen_status = gen_res.get("status") == "completed"
        gen_img_path = gen_res.get("image_path")

        # Stage 2: CatVTON Virtual Try-On
        t0_try = time.time()
        model_manager.switch_to("catvton")
        if gen_img_path and Path(gen_img_path).exists():
            generated_garment_img = Image.open(gen_img_path)
        else:
            generated_garment_img = Image.new("RGB", (512, 512), color=(20, 50, 180))

        person_input = PersonConditioningInput(person_image=person_img)
        garment_input = GarmentConditioningInput(garment_image=generated_garment_img, garment_type=sample_metadata["garment_type"])

        tryon_res = tryon_pipeline.run(
            person_input=person_input,
            garment_input=garment_input,
            output_filename=f"tryon_{garment_id}",
            person_filename=person_path.name,
            garment_filename=Path(gen_img_path).name if gen_img_path else "garment.png",
        )
        t1_try = time.time()
        tryon_duration = round(t1_try - t0_try, 2)
        total_tryon_time += tryon_duration

        tryon_status = tryon_res.status == "completed"
        val_valid = gen_res.get("validation", {}).get("valid", True) and tryon_res.validation.get("valid", True)

        if gen_status and tryon_status:
            success_count += 1
        else:
            failure_count += 1

        if val_valid:
            validation_pass_count += 1

        vram_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if (torch and torch.cuda.is_available()) else 0.0
        peak_gpu_vram_mb = max(peak_gpu_vram_mb, vram_mb)

        results.append({
            "garment_index": idx,
            "garment_id": garment_id,
            "gender": sample_metadata["gender"],
            "garment_type": sample_metadata["garment_type"],
            "generation_status": "completed" if gen_status else "failed",
            "tryon_status": "completed" if tryon_status else "failed",
            "validation_passed": val_valid,
            "generation_time_s": gen_duration,
            "tryon_time_s": tryon_duration,
            "total_time_s": round(gen_duration + tryon_duration, 2),
            "peak_vram_mb": vram_mb,
        })

    model_manager.clear_vram()
    t_end_all = time.time()
    total_benchmark_time = round(t_end_all - t_start_all, 2)

    # 5. Compile Summary Report JSON
    summary_report = {
        "benchmark_id": f"benchmark_{int(time.time())}",
        "total_garments_processed": len(selected_images),
        "generation_success_count": success_count,
        "generation_success_rate": round(success_count / len(selected_images), 4),
        "validation_pass_count": validation_pass_count,
        "validation_pass_rate": round(validation_pass_count / len(selected_images), 4),
        "failure_count": failure_count,
        "total_benchmark_time_s": total_benchmark_time,
        "avg_generation_time_s": round(total_generation_time / max(len(selected_images), 1), 2),
        "avg_tryon_time_s": round(total_tryon_time / max(len(selected_images), 1), 2),
        "avg_pipeline_time_s": round((total_generation_time + total_tryon_time) / max(len(selected_images), 1), 2),
        "peak_gpu_vram_mb": round(peak_gpu_vram_mb, 2),
        "detailed_results": results,
    }

    summary_json_path = reports_dir / "benchmark_summary.json"
    serialize_json(summary_report, summary_json_path)

    # 6. Generate Markdown Report
    md_content = f"""# FabricVision-AI — Automated Benchmark Execution Report

* **Garments Processed**: {len(selected_images)}
* **Overall Success Rate**: {summary_report['generation_success_rate'] * 100:.1f}% ({success_count}/{len(selected_images)})
* **Validation Pass Rate**: {summary_report['validation_pass_rate'] * 100:.1f}% ({validation_pass_count}/{len(selected_images)})
* **Total Execution Time**: {total_benchmark_time}s
* **Average Pipeline Latency**: {summary_report['avg_pipeline_time_s']}s / garment
* **Peak GPU VRAM Allocated**: {summary_report['peak_gpu_vram_mb']} MB

---

## Detailed Benchmark Breakdown

| Garment ID | Gender | Type | Generation | Try-On | Validation | Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for r in results:
        md_content += f"| `{r['garment_id']}` | {r['gender']} | {r['garment_type']} | {r['generation_status']} | {r['tryon_status']} | {'Passed' if r['validation_passed'] else 'Failed'} | {r['total_time_s']}s |\n"

    summary_md_path = reports_dir / "benchmark_report.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n=== Benchmark Complete ===")
    print(f"Summary JSON saved to: {summary_json_path}")
    print(f"Markdown Report saved to: {summary_md_path}")
    print(json.dumps(summary_report, indent=2))


if __name__ == "__main__":
    main()
