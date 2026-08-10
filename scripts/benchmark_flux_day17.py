"""Day 17 FLUX.1-Kontext performance / quality micro-benchmark helpers.

Measures prompt build + mode config resolution without requiring a full GPU run.
Full GPU timing belongs in live generation (see [FLUX PROFILE] logs).

Usage:
  python scripts/benchmark_flux_day17.py
  python scripts/benchmark_flux_day17.py --live-smoke  # optional short pipeline probe
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def benchmark_prompt() -> dict:
    from src.features.custom_generator.prompting.garment_prompt_builder import (
        CLIP_MAX_TOKENS,
        GarmentPromptBuilder,
    )

    builder = GarmentPromptBuilder(ROOT / "configs")
    fabric_metadata = {
        "material": "cotton",
        "pattern": "floral",
        "texture": "smooth",
        "dominant_colors": ["navy blue"],
    }
    user = {
        "gender": "women",
        "garment_type": "dress",
        "sleeve": "short_sleeve",
        "neckline": "round_neck",
        "fit": "slim",
        "style": "casual",
        "occasion": "casual",
        "season": "summer",
    }
    t0 = time.perf_counter()
    prompt, _ = builder.build_kontext_prompt(fabric_metadata, user)
    elapsed = time.perf_counter() - t0
    stats = dict(builder.last_prompt_stats)
    stats["prompt"] = prompt
    stats["prompt_build_s"] = round(elapsed, 4)
    stats["clip_budget"] = CLIP_MAX_TOKENS
    return stats


def benchmark_modes() -> list[dict]:
    from src.features.custom_generator.pipeline.garment_generation_pipeline import (
        GarmentGenerationConfig,
        GarmentGenerationPipeline,
    )

    rows = []
    for mode in ("Preview", "Standard", "Production", "Fast Preview", "High Quality"):
        cfg = GarmentGenerationConfig(
            config_dir=str(ROOT / "configs"),
            config_path=str(ROOT / "configs" / "custom_generator" / "flux_config.yaml"),
            generation_mode=mode,
            allow_fallback=True,
        )
        # Construct pipeline only to apply YAML mode presets (no weight load in pytest-like path)
        pipe = GarmentGenerationPipeline.__new__(GarmentGenerationPipeline)
        pipe.config = cfg
        pipe.logger = __import__("logging").getLogger("benchmark")
        pipe._load_config_files()
        rows.append(
            {
                "requested_mode": mode,
                "resolved_mode": pipe.config.generation_mode,
                "mode_key": pipe.config.mode_key,
                "height": pipe.config.height,
                "width": pipe.config.width,
                "steps": pipe.config.num_inference_steps,
                "guidance": pipe.config.guidance_scale,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(ROOT / "experiments" / "generation_results" / "day17_benchmark.json"),
    )
    args = parser.parse_args()

    report = {
        "note": (
            "Prompt/mode micro-benchmark only. Wall-clock generation times must come from "
            "actual RTX 3050 runs via [FLUX PROFILE] logs — do not invent timings."
        ),
        "prompt": benchmark_prompt(),
        "modes": benchmark_modes(),
        "recommended_defaults": {
            "preview": "384x384 / 3 steps / guidance 2.5",
            "standard": "512x512 / 4 steps / guidance 2.5",
            "production": "512x512 / 8 steps / guidance 2.5 (768 optional via FLUX_PRODUCTION_SIZE)",
            "attention": "sdpa (auto)",
            "torch_compile": False,
            "offload": "model_cpu_offload",
            "quantization": "nf4",
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
