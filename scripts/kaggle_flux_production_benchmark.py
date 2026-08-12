#!/usr/bin/env python3
"""Kaggle T4×2 FLUX Production resolution ladder benchmark.

Runs the 700+ Production matrix on the live FastAPI backend:

  resolutions: 704, 720, 768
  steps:       8, 10, 12
  guidance:    3.0

Records timing, VRAM, OOM, and output paths. Does NOT pick a winner
automatically — inspect quality artifacts under experiments/kaggle/.

Usage (inside Kaggle after API is up on :8000):

    python scripts/kaggle_flux_production_benchmark.py
    python scripts/kaggle_flux_production_benchmark.py --api http://127.0.0.1:8000/api/v1
    python scripts/kaggle_flux_production_benchmark.py --only 704:8,720:10,768:12

Requires dual-GPU inventory first:

    python scripts/kaggle_gpu_inventory.py
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "kaggle" / "production_ladder"
DEFAULT_API = "http://127.0.0.1:8000/api/v1"

LADDER = [
    (704, 8),
    (704, 10),
    (704, 12),
    (720, 8),
    (720, 10),
    (720, 12),
    (768, 8),
    (768, 10),
    (768, 12),
]


def multipart(fields: dict[str, str], files: dict[str, Path]) -> tuple[bytes, str]:
    boundary = "----prodladder"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        parts.append(str(value).encode())
        parts.append(b"\r\n")
    for name, path in files.items():
        data = path.read_bytes()
        ctype = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{path.name}"\r\n'.encode()
        )
        parts.append(f"Content-Type: {ctype}\r\n\r\n".encode())
        parts.append(data)
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), boundary


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode())


def post_generate(api: str, fabric: Path, resolution: int, steps: int) -> dict:
    fields = {
        "garment_type": "dress",
        "fit": "Slim Fit",
        "style": "Casual",
        "gender": "Women",
        "season": "Summer",
        "occasion": "Casual",
        "fabric": "Cotton",
        "material": "Cotton",
        "texture": "Smooth",
        "color": "match_fabric",
        "sleeve": "Short",
        "neckline": "Round",
        "generation_mode": "Production",
    }
    # Backend prefers FLUX_PRODUCTION_* env; Form cannot set env, so we rely on
    # process env set by the caller / run_kaggle. Document that operators should
    # export FLUX_PRODUCTION_RESOLUTION / FLUX_PRODUCTION_STEPS per cell, OR we
    # hit a small control endpoint if present. For this ladder we set env in-process
    # of the *client* only when using in-process pipeline; for API mode, restart
    # is required OR we pass generation_mode=Production and pre-set server env.
    #
    # Practical approach: write a sidecar env request file the server does not
    # read — instead document exporting before each run. To make API-mode work
    # without restart, include resolution/steps in Form if backend accepts them.
    # Current API does not — so this script uses os.environ on the *server*
    # via a best-effort / not available. For Kaggle, run each cell by setting
    # env before uvicorn OR use direct pipeline mode (--in-process).
    body, boundary = multipart(fields, {"fabric_image": fabric})
    req = urllib.request.Request(f"{api}/generate", data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode())


def poll(api: str, job_id: str, timeout_s: int = 3600) -> dict:
    t0 = time.time()
    last: dict = {}
    while True:
        last = get_json(f"{api}/status/{job_id}")
        elapsed = int(time.time() - t0)
        print(
            f"  [{elapsed}s] {last.get('status')} {last.get('stage')} "
            f"{last.get('progress')} {last.get('current_step')}",
            flush=True,
        )
        if last.get("status") in ("completed", "failed"):
            return last
        if elapsed > timeout_s:
            last["status"] = "failed"
            last["error"] = f"timeout after {timeout_s}s"
            return last
        time.sleep(15)


def run_in_process(fabric: Path, resolution: int, steps: int, guidance: float) -> dict:
    """Direct pipeline run — preferred for controlled Kaggle ladder cells."""
    import os

    os.environ["FLUX_PRODUCTION_RESOLUTION"] = str(resolution)
    os.environ["FLUX_PRODUCTION_SIZE"] = str(resolution)
    os.environ["FLUX_PRODUCTION_STEPS"] = str(steps)
    os.environ.setdefault("FLUX_VAE_TILING", "true")

    from PIL import Image

    from src.features.custom_generator.pipeline.garment_generation_pipeline import (
        GarmentGenerationConfig,
        GarmentGenerationPipeline,
    )
    from src.common.models.device_manager import DeviceManager
    from src.integrations.flux.flux_manager import FluxManager

    t_load0 = time.perf_counter()
    flux_dev = DeviceManager.resolve_role_device("flux", "cuda:0")
    mgr = FluxManager(device=flux_dev, allow_fallback=False)
    loader = mgr.load()
    load_s = round(time.perf_counter() - t_load0, 2)
    if loader is None or getattr(loader, "pipeline", None) is None:
        return {
            "ok": False,
            "oom": False,
            "error": "FLUX failed to load",
            "model_load_time_s": load_s,
        }

    cfg = GarmentGenerationConfig(
        generation_mode="Production",
        height=resolution,
        width=resolution,
        num_inference_steps=steps,
        guidance_scale=guidance,
        allow_fallback=False,
        config_path=str(ROOT / "configs" / "custom_generator" / "flux_config.yaml"),
        output_root=str(OUT / "images"),
    )
    pipe = GarmentGenerationPipeline(config=cfg, model_loader=loader)
    # Ladder cell wins over YAML / policy defaults for this measured run.
    pipe.config.height = resolution
    pipe.config.width = resolution
    pipe.config.num_inference_steps = steps
    pipe.config.guidance_scale = guidance
    pipe.config.generation_mode = "Production"
    pipe.config.mode_key = "production"

    fabric_img = Image.open(fabric).convert("RGB")
    t_inf0 = time.perf_counter()
    oom = False
    err = None
    result: dict = {}
    try:
        result = pipe.run(
            fabric_metadata={
                "fabric": "cotton",
                "color": "match_fabric",
                "texture": "smooth",
                "material": "cotton",
            },
            user_customization={
                "garment_type": "dress",
                "fit": "slim",
                "style": "casual",
                "gender": "women",
                "sleeve": "short",
                "neckline": "round",
                "color": "match_fabric",
            },
            reference_image=fabric_img,
            output_filename=f"ladder_{resolution}_{steps}",
        )
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        oom = "out of memory" in err.lower() or "cuda" in err.lower() and "memory" in err.lower()
    inf_s = round(time.perf_counter() - t_inf0, 2)

    import torch

    idx = DeviceManager.cuda_device_index(flux_dev) or 0
    vram_alloc = (
        round(torch.cuda.memory_allocated(idx) / (1024**2), 1)
        if torch.cuda.is_available()
        else 0.0
    )
    vram_peak = (
        round(torch.cuda.max_memory_allocated(idx) / (1024**2), 1)
        if torch.cuda.is_available()
        else 0.0
    )

    return {
        "ok": err is None and bool(result.get("was_real_flux_used", result.get("image_path"))),
        "oom": oom,
        "error": err,
        "model_load_time_s": load_s,
        "inference_time_s": inf_s,
        "total_time_s": round(load_s + inf_s, 2),
        "resolution": f"{pipe.config.width}x{pipe.config.height}",
        "steps": pipe.config.num_inference_steps,
        "guidance": pipe.config.guidance_scale,
        "gpu": DeviceManager.inventory_gpus(),
        "vram_alloc_mb": vram_alloc,
        "vram_peak_mb": vram_peak,
        "image_path": result.get("image_path") or result.get("result_url"),
        "was_real_flux_used": result.get("was_real_flux_used"),
        "result_meta": {
            k: result[k]
            for k in (
                "was_real_flux_used",
                "num_inference_steps",
                "height",
                "width",
                "guidance_scale",
                "generation_time_s",
                "peak_vram_mb",
            )
            if k in result
        },
    }


def pick_fabric() -> Path:
    for p in [
        ROOT / "data" / "uploads" / "val_cotton.png",
        ROOT / "frontend" / "public" / "studio" / "fabric.png",
    ]:
        if p.exists():
            return p
    pngs = list((ROOT / "data" / "uploads").glob("*.png"))
    if not pngs:
        raise FileNotFoundError("No fabric image found")
    return pngs[0]


def parse_only(raw: str | None) -> list[tuple[int, int]]:
    if not raw:
        return list(LADDER)
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        res_s, steps_s = part.split(":")
        out.append((int(res_s), int(steps_s)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Kaggle FLUX Production 700+ ladder")
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--in-process", action="store_true", default=True)
    parser.add_argument("--api-mode", action="store_true", help="Use FastAPI instead of in-process")
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--only", default="", help="e.g. 704:8,720:10,768:12")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    in_process = not args.api_mode

    OUT.mkdir(parents=True, exist_ok=True)
    fabric = pick_fabric()
    print(f"fabric={fabric}", flush=True)
    print(f"mode={'in-process' if in_process else 'api'}", flush=True)

    rows = []
    for resolution, steps in parse_only(args.only or None):
        label = f"{resolution}x{resolution}_s{steps}"
        print(f"\n=== LADDER {label} guidance={args.guidance} ===", flush=True)
        row: dict = {
            "label": label,
            "resolution": resolution,
            "steps": steps,
            "guidance": args.guidance,
        }
        try:
            if in_process:
                row.update(
                    run_in_process(fabric, resolution, steps, args.guidance)
                )
            else:
                # API mode: operator must set server env for each cell.
                print(
                    "API mode requires server env "
                    f"FLUX_PRODUCTION_RESOLUTION={resolution} "
                    f"FLUX_PRODUCTION_STEPS={steps} before job",
                    flush=True,
                )
                t0 = time.time()
                init = post_generate(args.api, fabric, resolution, steps)
                st = poll(args.api, init["job_id"], timeout_s=args.timeout)
                meta = st.get("metadata") or {}
                err = st.get("error")
                row.update(
                    {
                        "ok": st.get("status") == "completed"
                        and bool(meta.get("was_real_flux_used")),
                        "oom": bool(err and "memory" in str(err).lower()),
                        "error": err,
                        "model_load_time_s": meta.get("model_init_time_s"),
                        "inference_time_s": meta.get("generation_time_s"),
                        "total_time_s": round(time.time() - t0, 2),
                        "resolution": f"{meta.get('width')}x{meta.get('height')}",
                        "steps": meta.get("num_inference_steps"),
                        "guidance": meta.get("guidance_scale"),
                        "vram_peak_mb": meta.get("peak_vram_mb"),
                        "image_path": st.get("result_url"),
                        "was_real_flux_used": meta.get("was_real_flux_used"),
                        "job_id": init.get("job_id"),
                    }
                )
        except Exception as exc:
            row.update(
                {
                    "ok": False,
                    "oom": "out of memory" in str(exc).lower(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        rows.append(row)
        (OUT / f"{label}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(json.dumps(row, indent=2)[:2000], flush=True)

    report = {
        "ladder": rows,
        "note": (
            "Select best Production config by quality + stability. "
            "Minimum production resolution is 700+. Do not claim success "
            "without inspecting output images."
        ),
    }
    (OUT / "ladder_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nWROTE", OUT / "ladder_report.json", flush=True)
    ok_n = sum(1 for r in rows if r.get("ok"))
    print(f"completed_ok={ok_n}/{len(rows)}", flush=True)
    return 0 if ok_n else 1


if __name__ == "__main__":
    raise SystemExit(main())
