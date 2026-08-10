"""Benchmark FLUX.1-Kontext generation candidates for FabricVision-AI.

Compares:
- Baseline (512x512, 3-4 steps)
- Candidate A (512x512, 12 steps, pre-encode, GPU resident)
- Candidate B (512x512, 16 steps, pre-encode, GPU resident)
- Candidate C (512x512, 16 steps + contour detail refiner)

Saves results and stage images to experiments/generation_results/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fabricvision.benchmark")


def compute_image_metrics(img: Image.Image) -> dict[str, float]:
    """Compute visual sharpness and edge metrics."""
    rgb = img.convert("RGB")
    gray = rgb.convert("L")
    arr = np.asarray(gray, dtype=np.float32)

    # Laplacian variance (standard focus / blur metric)
    # Discrete Laplacian kernel
    kernel_laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    from scipy.signal import convolve2d

    lap = convolve2d(arr, kernel_laplacian, mode="valid")
    lap_var = float(np.var(lap))

    # Gradient magnitude (edge energy)
    gx = np.abs(arr[:, 1:] - arr[:, :-1]).mean()
    gy = np.abs(arr[1:, :] - arr[:-1, :]).mean()
    edge_energy = float((gx + gy) / 2.0)

    # High frequency energy ratio
    fft = np.fft.fft2(arr)
    fft_shift = np.fft.fftshift(fft)
    h, w = arr.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    y, x = np.ogrid[:h, :w]
    mask_low = (y - cy) ** 2 + (x - cx) ** 2 <= r**2
    total_energy = np.sum(np.abs(fft_shift) ** 2)
    low_energy = np.sum(np.abs(fft_shift[mask_low]) ** 2)
    high_freq_ratio = float((total_energy - low_energy) / (total_energy + 1e-8))

    return {
        "laplacian_variance": round(lap_var, 2),
        "edge_energy": round(edge_energy, 3),
        "high_freq_ratio": round(high_freq_ratio, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="FLUX.1-Kontext Quality & Performance Benchmark")
    parser.add_argument("--fabric", default="assets/sample_fabric.jpg", help="Path to fabric image")
    parser.add_argument("--garment", default="dress", help="Garment type")
    parser.add_argument("--sleeve", default="short_sleeve", help="Sleeve style")
    parser.add_argument("--neckline", default="round_neck", help="Neckline style")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="experiments/generation_results", help="Output directory")
    args = parser.parse_args()

    fabric_path = Path(args.fabric)
    if not fabric_path.exists():
        # Fallback to test image if sample_fabric doesn't exist
        test_imgs = list(Path("tests/test_images").glob("*.jpg")) + list(Path("tests/test_images").glob("*.png"))
        if test_imgs:
            fabric_path = test_imgs[0]
            logger.info("Using test image: %s", fabric_path)
        else:
            # Create a synthetic test fabric
            fabric_path = Path("experiments/test_fabric.png")
            fabric_path.parent.mkdir(parents=True, exist_ok=True)
            synth = Image.new("RGB", (512, 512), (240, 240, 245))
            from PIL import ImageDraw
            draw = ImageDraw.Draw(synth)
            for i in range(0, 512, 40):
                draw.line([(i, 0), (i, 512)], fill=(220, 60, 60), width=8)
                draw.line([(0, i), (512, i)], fill=(40, 160, 80), width=8)
            synth.save(fabric_path)
            logger.info("Created synthetic fabric image at %s", fabric_path)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from src.features.custom_generator.pipeline.garment_generation_pipeline import (
        GarmentGenerationConfig,
        GarmentGenerationPipeline,
    )

    configs_to_test = [
        {
            "name": "Baseline_512x512_3step",
            "mode": "standard",
            "steps": 3,
            "guidance": 2.5,
            "preencode": "false",
            "size": 512,
        },
        {
            "name": "Candidate_A_512x512_12step_preencode",
            "mode": "standard",
            "steps": 12,
            "guidance": 3.0,
            "preencode": "true",
            "size": 512,
        },
        {
            "name": "Candidate_B_512x512_16step_preencode",
            "mode": "production",
            "steps": 16,
            "guidance": 3.5,
            "preencode": "true",
            "size": 512,
        },
    ]

    fabric_image = Image.open(fabric_path).convert("RGB")
    results = []

    for cfg in configs_to_test:
        logger.info("==================================================")
        logger.info("Running Candidate: %s", cfg["name"])
        logger.info("Steps: %s, Guidance: %s, Pre-encode: %s", cfg["steps"], cfg["guidance"], cfg["preencode"])
        logger.info("==================================================")

        os.environ["FLUX_PREENCODE_PROMPT"] = cfg["preencode"]

        pipeline_config = GarmentGenerationConfig(
            config_dir="configs",
            config_path="configs/custom_generator/flux_config.yaml",
            output_root=str(out_dir / cfg["name"]),
            height=cfg["size"],
            width=cfg["size"],
            num_inference_steps=cfg["steps"],
            guidance_scale=cfg["guidance"],
            seed=args.seed,
            allow_fallback=False,
        )

        pipeline = GarmentGenerationPipeline(config=pipeline_config)

        fabric_metadata = {
            "material": "cotton",
            "fabric": "cotton",
            "texture": "smooth",
            "dominant_colors": [],
            "color": "match_fabric",
            "style": "casual",
            "occasion": "casual",
            "season": "summer",
            "fit": "regular",
        }

        user_customization = {
            "gender": "women",
            "garment_type": args.garment,
            "material": "cotton",
            "sleeve": args.sleeve,
            "neckline": args.neckline,
            "fit": "regular",
            "style": "casual",
            "force_recolor": False,
        }

        t_start = time.perf_counter()
        try:
            res = pipeline.run(
                fabric_metadata=fabric_metadata,
                user_customization=user_customization,
                reference_image=fabric_image,
            )
            dur = round(time.perf_counter() - t_start, 2)

            img_path = res["image_path"]
            gen_img = Image.open(img_path)
            metrics = compute_image_metrics(gen_img)
            stats = getattr(pipeline.inference_engine, "last_execution_stats", {})

            candidate_result = {
                "name": cfg["name"],
                "total_time_s": dur,
                "generation_time_s": stats.get("generation_time_s", dur),
                "model_load_time_s": stats.get("model_load_time_s", 0),
                "prompt_encoding_time_s": stats.get("prompt_encoding_time_s", 0),
                "diffusion_s": stats.get("diffusion_pipeline_s", 0),
                "vae_decode_time_s": stats.get("vae_decode_time_s", 0),
                "peak_vram_mb": stats.get("peak_vram_mb", 0),
                "cpu_ram_after_mb": stats.get("cpu_ram_after_mb", 0),
                "num_inference_steps": cfg["steps"],
                "guidance_scale": cfg["guidance"],
                "was_real_flux_used": stats.get("was_real_flux_used", False),
                "metrics": metrics,
                "image_path": img_path,
            }
            results.append(candidate_result)
            logger.info("Completed %s in %.2fs (Peak VRAM: %.1f MB, Laplacian: %.2f)",
                        cfg["name"], dur, candidate_result["peak_vram_mb"], metrics["laplacian_variance"])
        except Exception as exc:
            logger.error("Candidate %s failed: %s", cfg["name"], exc, exc_info=True)

    summary_file = out_dir / "benchmark_summary.json"
    summary_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n================ BENCHMARK SUMMARY ================")
    print(json.dumps(results, indent=2))
    print(f"\nSaved summary to {summary_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
