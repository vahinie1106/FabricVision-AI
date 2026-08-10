"""Day 17 FINAL — resolution + step benchmarks for FLUX.1-Kontext on RTX 3050 6GB.

Loads the model ONCE, then runs controlled A/B tests.
Catches OOM without crashing. Never invents timings.

Usage:
  python scripts/benchmark_day17_resolution_steps.py --phase resolution
  python scripts/benchmark_day17_resolution_steps.py --phase steps --best-resolution 512
  python scripts/benchmark_day17_resolution_steps.py --phase all
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("day17.benchmark")

DEFAULT_FABRIC = ROOT / "data" / "uploads" / "30c9aacaacf444169b3959f2171b4942.jpg"
OUT_IMG = ROOT / "outputs" / "generated_garments" / "day17_resolution_tests"
EXP = ROOT / "experiments" / "generation_results"


def _edge_energy(path: Path) -> float | None:
    try:
        import numpy as np
        from PIL import Image

        a = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
        gx = float(np.abs(a[:, 1:] - a[:, :-1]).mean())
        gy = float(np.abs(a[1:, :] - a[:-1, :]).mean())
        return round((gx + gy) / 2.0, 3)
    except Exception:
        return None


def _mean_rgb(path: Path) -> list[float] | None:
    try:
        import numpy as np
        from PIL import Image

        a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        return [round(float(x), 1) for x in a.mean(axis=(0, 1))]
    except Exception:
        return None


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_md_table(path: Path, title: str, rows: list[dict], columns: list[str]) -> None:
    lines = [f"# {title}", "", "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    lines.append("")
    lines.append("_All timings measured with `time.perf_counter()` on local RTX 3050 6GB. No fabricated values._")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_pipeline(loader: Any, seed: int) -> Any:
    from src.features.custom_generator.pipeline.garment_generation_pipeline import (
        GarmentGenerationConfig,
        GarmentGenerationPipeline,
    )

    cfg = GarmentGenerationConfig(
        config_dir=str(ROOT / "configs"),
        config_path=str(ROOT / "configs" / "custom_generator" / "flux_config.yaml"),
        output_root=str(ROOT / "outputs" / "generated_garments"),
        experiments_root=str(ROOT / "experiments"),
        generation_mode="Standard",
        allow_fallback=False,
        seed=seed,
    )
    pipe = GarmentGenerationPipeline(config=cfg, model_loader=loader)
    pipe.config.seed = seed
    return pipe


def _run_one(
    *,
    pipe: Any,
    fabric: Any,
    resolution: int,
    steps: int,
    guidance: float,
    seed: int,
    label: str,
    out_name: str,
) -> dict[str, Any]:
    """Run one generation; catch OOM and return structured result."""
    import torch

    pipe.config.height = resolution
    pipe.config.width = resolution
    pipe.config.num_inference_steps = steps
    pipe.config.guidance_scale = guidance
    pipe.config.seed = seed
    pipe.config.mode_key = "standard"
    pipe.config.generation_mode = "Standard"

    fabric_metadata = {
        "material": "cotton",
        "fabric": "cotton",
        "texture": "smooth",
        "style": "casual",
        "fit": "slim",
        "dominant_colors": [],
        "color": "match_fabric",
    }
    user = {
        "gender": "women",
        "garment_type": "top",
        "material": "cotton",
        "neckline": "sweetheart_neck",
        "sleeve": "short_sleeve",
        "fit": "slim",
        "style": "casual",
        "occasion": "casual",
        "season": "summer",
        "force_recolor": False,
    }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

    row: dict[str, Any] = {
        "label": label,
        "resolution": resolution,
        "steps": steps,
        "guidance": guidance,
        "seed": seed,
        "status": "ok",
        "error": None,
        "image_path": None,
        "wall_clock_s": None,
        "generation_time_s": None,
        "peak_vram_mb": None,
        "cpu_ram_after_mb": None,
        "prompt": None,
        "prompt_stats": None,
        "per_step_durations_s": None,
        "edge_energy": None,
        "mean_rgb": None,
        "output_size": None,
    }

    print(
        f"\n[BENCH] START {label} res={resolution} steps={steps} guidance={guidance}",
        flush=True,
    )
    t0 = time.perf_counter()
    try:
        result = pipe.run(
            fabric_metadata=fabric_metadata,
            user_customization=user,
            reference_image=fabric,
            output_filename=label,
        )
        wall = round(time.perf_counter() - t0, 3)
        stats = getattr(pipe.inference_engine, "last_execution_stats", {}) or {}
        meta = result.get("metadata") or {}

        src = Path(result.get("image_path") or "")
        dest = OUT_IMG / out_name
        OUT_IMG.mkdir(parents=True, exist_ok=True)
        if src.exists():
            dest.write_bytes(src.read_bytes())

        row.update(
            {
                "status": "ok",
                "wall_clock_s": wall,
                "generation_time_s": stats.get("generation_time_s"),
                "peak_vram_mb": stats.get("peak_vram_mb"),
                "cpu_ram_after_mb": stats.get("cpu_ram_after_mb"),
                "model_load_time_s": stats.get("model_load_time_s"),
                "prompt_encoding_time_s": stats.get("prompt_encoding_time_s"),
                "inference_time_s": stats.get("inference_time_s"),
                "vae_decode_time_s": stats.get("vae_decode_time_s"),
                "per_step_durations_s": stats.get("per_step_durations_s"),
                "pipeline_timings": meta.get("pipeline_timings"),
                "prompt": meta.get("positive_prompt"),
                "prompt_stats": meta.get("prompt_stats"),
                "image_path": str(dest),
                "raw_image_path": meta.get("raw_image_path"),
                "output_size": stats.get("output_size"),
                "edge_energy": _edge_energy(dest) if dest.exists() else None,
                "mean_rgb": _mean_rgb(dest) if dest.exists() else None,
                "model_reused": stats.get("model_reused"),
                "attention_backend": stats.get("attention_backend"),
                "offload_strategy": stats.get("offload_strategy"),
                "bnb_4bit": stats.get("bnb_4bit"),
            }
        )
        print(
            f"[BENCH] OK {label} wall={wall}s peak_vram={row['peak_vram_mb']} "
            f"edge={row['edge_energy']} mean_rgb={row['mean_rgb']}",
            flush=True,
        )
    except Exception as exc:
        wall = round(time.perf_counter() - t0, 3)
        err = f"{type(exc).__name__}: {exc}"
        is_oom = "out of memory" in str(exc).lower() or "cuda" in str(exc).lower() and "memory" in str(exc).lower()
        row.update(
            {
                "status": "RESOLUTION_UNAVAILABLE" if is_oom else "FAILED",
                "error": err,
                "wall_clock_s": wall,
                "traceback": traceback.format_exc()[-2000:],
            }
        )
        print(f"[BENCH] FAIL {label} status={row['status']} err={err}", flush=True)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return row


def run_resolution_phase(pipe: Any, fabric: Any, seed: int, guidance: float, steps: int) -> list[dict]:
    rows = []
    for res in (384, 512, 640, 768):
        row = _run_one(
            pipe=pipe,
            fabric=fabric,
            resolution=res,
            steps=steps,
            guidance=guidance,
            seed=seed,
            label=f"day17_res_{res}_s{steps}",
            out_name=f"resolution_{res}_steps{steps}.png",
        )
        rows.append(row)
        _write_json(EXP / "day17_resolution_benchmark.json", {"phase": "resolution", "rows": rows})
        # Stop climbing if OOM — higher res will also fail
        if row["status"] == "RESOLUTION_UNAVAILABLE":
            print(f"[BENCH] Stopping resolution climb after OOM at {res}", flush=True)
            break
    return rows


def run_step_phase(
    pipe: Any, fabric: Any, seed: int, guidance: float, resolution: int
) -> list[dict]:
    rows = []
    for steps in (2, 3, 4):
        row = _run_one(
            pipe=pipe,
            fabric=fabric,
            resolution=resolution,
            steps=steps,
            guidance=guidance,
            seed=seed,
            label=f"day17_steps_{steps}_r{resolution}",
            out_name=f"best_resolution_steps{steps}.png",
        )
        rows.append(row)
        _write_json(
            EXP / "day17_step_benchmark.json",
            {"phase": "steps", "best_resolution": resolution, "rows": rows},
        )
    return rows


def _recommend(res_rows: list[dict], step_rows: list[dict] | None) -> dict[str, Any]:
    """Pick best practical config from measured rows (quality proxies + stability)."""
    ok = [r for r in res_rows if r.get("status") == "ok"]
    if not ok:
        return {"error": "no successful resolution runs"}

    # Prefer higher edge_energy and non-yellow mean (B channel not collapsed), then lower runtime
    def score(r: dict) -> tuple:
        rgb = r.get("mean_rgb") or [0, 0, 0]
        # Penalize strong yellow cast (high R/G, low B) like the old bug
        yellow_penalty = max(0.0, (rgb[0] + rgb[1]) / 2 - rgb[2] - 40)
        edge = float(r.get("edge_energy") or 0)
        # Prefer larger resolution lightly (detail capacity) but not if much slower for same edge
        res = int(r.get("resolution") or 0)
        runtime = float(r.get("generation_time_s") or r.get("wall_clock_s") or 1e9)
        return (edge - yellow_penalty * 0.05, res, -runtime)

    best_res = max(ok, key=score)
    best_steps = None
    if step_rows:
        ok_s = [r for r in step_rows if r.get("status") == "ok"]
        if ok_s:
            best_steps = max(ok_s, key=score)

    return {
        "best_resolution_row": best_res,
        "best_step_row": best_steps,
        "recommendation": {
            "preview": {"height": 384, "width": 384, "steps": 2, "guidance": 2.5},
            "standard": {
                "height": best_res["resolution"],
                "width": best_res["resolution"],
                "steps": (best_steps or best_res)["steps"],
                "guidance": (best_steps or best_res)["guidance"],
            },
            "production": {
                "height": best_res["resolution"],
                "width": best_res["resolution"],
                "steps": max(4, int((best_steps or best_res)["steps"])),
                "guidance": 2.5,
            },
            "notes": (
                "Recommendation derived from measured edge_energy, color cast, resolution, "
                "and runtime. Visual inspection of PNGs remains required before shipping."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=str(DEFAULT_FABRIC))
    parser.add_argument("--phase", choices=["resolution", "steps", "all"], default="all")
    parser.add_argument("--best-resolution", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance", type=float, default=2.5)
    parser.add_argument("--resolution-steps", type=int, default=3)
    args = parser.parse_args()

    os.environ.setdefault("FLUX_PROFILE", "true")
    os.environ.setdefault("FLUX_MAX_SEQUENCE_LENGTH", "128")
    os.environ.setdefault("FLUX_VAE_TILING", "false")
    os.environ.setdefault("FLUX_ENABLE_TORCH_COMPILE", "false")
    os.environ.setdefault("FLUX_PREENCODE_PROMPT", "false")

    from PIL import Image

    from src.common.models.model_manager import ModelManager

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Fabric not found: {img_path}", file=sys.stderr)
        return 2

    fabric = Image.open(img_path).convert("RGB")
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)

    mgr = ModelManager()
    mgr.flux_manager.allow_fallback = False
    mgr.flux_manager.model_path = ROOT / "models" / "flux-kontext"

    t_load = time.perf_counter()
    print("[FLUX] Model initialization started", flush=True)
    mgr.switch_to("flux")
    load_s = round(time.perf_counter() - t_load, 3)
    loader = mgr.flux_manager.loader
    if loader is None or getattr(loader, "pipeline", None) is None:
        print("Failed to load Kontext", file=sys.stderr)
        return 3
    print(f"[FLUX] Model initialization completed ({load_s}s)", flush=True)

    pipe = _build_pipeline(loader, args.seed)

    res_rows: list[dict] = []
    step_rows: list[dict] = []

    if args.phase in ("resolution", "all"):
        res_rows = run_resolution_phase(
            pipe, fabric, args.seed, args.guidance, args.resolution_steps
        )
        _write_json(
            EXP / "day17_resolution_benchmark.json",
            {
                "hardware": "NVIDIA RTX 3050 Laptop GPU 6GB",
                "model": "FLUX.1-Kontext",
                "quantization": "nf4",
                "offload": "model_cpu_offload",
                "attention": "sdpa",
                "fabric": str(img_path),
                "seed": args.seed,
                "guidance": args.guidance,
                "steps_fixed": args.resolution_steps,
                "model_init_s": load_s,
                "rows": res_rows,
            },
        )
        _write_md_table(
            EXP / "day17_resolution_benchmark.md",
            "Day 17 Resolution Benchmark (measured)",
            [
                {
                    "Resolution": r["resolution"],
                    "Steps": r["steps"],
                    "Guidance": r["guidance"],
                    "Runtime": r.get("generation_time_s") or r.get("wall_clock_s"),
                    "Peak VRAM": r.get("peak_vram_mb"),
                    "Edge": r.get("edge_energy"),
                    "Mean RGB": r.get("mean_rgb"),
                    "Status": r.get("status"),
                }
                for r in res_rows
            ],
            [
                "Resolution",
                "Steps",
                "Guidance",
                "Runtime",
                "Peak VRAM",
                "Edge",
                "Mean RGB",
                "Status",
            ],
        )

    best_res = args.best_resolution
    if not best_res and res_rows:
        ok = [r for r in res_rows if r.get("status") == "ok"]
        if ok:
            # Visual QA (Day 17): edge_energy inflates 384 via pixelation.
            # Prefer 512 when present; else closest successful <=640 with best
            # quality/runtime balance (do NOT auto-pick max edge_energy).
            preferred = {r["resolution"] for r in ok}
            if 512 in preferred:
                best_res = 512
            else:
                best_res = min(ok, key=lambda r: float(r.get("wall_clock_s") or 1e9))[
                    "resolution"
                ]
            print(
                f"[BENCH] Auto best_resolution={best_res} "
                f"(visual-preferred; edge_energy alone is unreliable)",
                flush=True,
            )

    if args.phase in ("steps", "all"):
        if not best_res:
            print("No best resolution available for step phase", file=sys.stderr)
            return 4
        print(f"[BENCH] Step phase at resolution={best_res}", flush=True)
        step_rows = run_step_phase(pipe, fabric, args.seed, args.guidance, best_res)
        _write_json(
            EXP / "day17_step_benchmark.json",
            {
                "hardware": "NVIDIA RTX 3050 Laptop GPU 6GB",
                "best_resolution": best_res,
                "seed": args.seed,
                "guidance": args.guidance,
                "rows": step_rows,
            },
        )
        _write_md_table(
            EXP / "day17_step_benchmark.md",
            f"Day 17 Step Benchmark @ {best_res}² (measured)",
            [
                {
                    "Resolution": r["resolution"],
                    "Steps": r["steps"],
                    "Guidance": r["guidance"],
                    "Runtime": r.get("generation_time_s") or r.get("wall_clock_s"),
                    "Peak VRAM": r.get("peak_vram_mb"),
                    "Edge": r.get("edge_energy"),
                    "Mean RGB": r.get("mean_rgb"),
                    "Status": r.get("status"),
                }
                for r in step_rows
            ],
            [
                "Resolution",
                "Steps",
                "Guidance",
                "Runtime",
                "Peak VRAM",
                "Edge",
                "Mean RGB",
                "Status",
            ],
        )

    rec = _recommend(res_rows or [], step_rows or None)
    _write_json(EXP / "day17_final_recommendation.json", rec)
    print(json.dumps(rec, indent=2), flush=True)
    print("[BENCH] Complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
