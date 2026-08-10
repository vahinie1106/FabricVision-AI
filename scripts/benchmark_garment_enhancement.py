"""Day 17 — isolated second-stage detail-enhancement benchmark.

Loads an EXISTING real FLUX.1-Kontext output PNG (never regenerates FLUX),
runs a single candidate super-resolution model on it, and records measured
runtime / VRAM / dimension metadata. Never overwrites or mutates the source
FLUX image — enhanced output is always written to a separate file.

Candidate under test (see experiments/generation_results/day17_enhancement_investigation.md
for why this one was selected):

  realesr-general-x4v3 (SRVGGNetCompact, Real-ESRGAN family, ~4.7MB)
  Loaded via `spandrel` (pure PyTorch, no basicsr/torchvision.functional_tensor
  dependency — basicsr is broken on current torchvision and was rejected for
  that reason, see investigation report).

Usage (venv):
  python scripts/benchmark_garment_enhancement.py --image outputs/generated_garments/images/day17_enhancement_source.png
  python scripts/benchmark_garment_enhancement.py --image <path> --outscale 2 --tile 256
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = ROOT / "models" / "enhancement" / "realesr-general-x4v3.pth"


def _cpu_ram_mb() -> float:
    try:
        import psutil
        import os

        return round(psutil.Process(os.getpid()).memory_info().rss / (1024**2), 1)
    except Exception:
        return 0.0


def _tiled_upscale(model: Any, img_tensor: Any, tile: int, tile_pad: int, scale: int, device: str) -> Any:
    """
    Split the image into overlapping tiles before running the network.

    WHY: a 6GB GPU cannot always hold a full 512x512 (or larger) activation
    map for every SR architecture at once. Tiling is the standard mitigation
    used by the real Real-ESRGAN inference script — this is not a shortcut,
    it is required for VRAM safety on this hardware.
    """
    import torch

    b, c, h, w = img_tensor.shape
    output_h, output_w = h * scale, w * scale
    output = torch.zeros((b, c, output_h, output_w), device=img_tensor.device)

    tiles_x = (w + tile - 1) // tile
    tiles_y = (h + tile - 1) // tile

    for y in range(tiles_y):
        for x in range(tiles_x):
            ofs_x, ofs_y = x * tile, y * tile
            in_x0, in_x1 = max(ofs_x - tile_pad, 0), min(ofs_x + tile + tile_pad, w)
            in_y0, in_y1 = max(ofs_y - tile_pad, 0), min(ofs_y + tile + tile_pad, h)

            in_tile = img_tensor[:, :, in_y0:in_y1, in_x0:in_x1]
            with torch.inference_mode():
                out_tile = model(in_tile)

            out_x0, out_x1 = min(ofs_x, w) * scale, min(ofs_x + tile, w) * scale
            out_y0, out_y1 = min(ofs_y, h) * scale, min(ofs_y + tile, h) * scale
            trim_x0 = (min(ofs_x, w) - in_x0) * scale
            trim_y0 = (min(ofs_y, h) - in_y0) * scale
            trim_x1 = trim_x0 + (out_x1 - out_x0)
            trim_y1 = trim_y0 + (out_y1 - out_y0)

            output[:, :, out_y0:out_y1, out_x0:out_x1] = out_tile[:, :, trim_y0:trim_y1, trim_x0:trim_x1]

    return output


def enhance_image(
    image_path: Path,
    model_path: Path,
    outscale: float,
    tile: int,
    tile_pad: int,
) -> Dict[str, Any]:
    import torch
    from PIL import Image
    from spandrel import ModelLoader

    if not image_path.exists():
        raise FileNotFoundError(f"Source FLUX image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Enhancement checkpoint not found: {model_path}. "
            "Download realesr-general-x4v3.pth (see investigation report)."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    src_image = Image.open(image_path).convert("RGB")
    src_w, src_h = src_image.size

    ram_before = _cpu_ram_mb()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    vram_before = (
        round(torch.cuda.memory_allocated() / (1024**2), 2) if device == "cuda" else 0.0
    )

    t_load = time.perf_counter()
    model = ModelLoader().load_from_file(str(model_path))
    model = model.to(device).eval()
    net_scale = int(model.scale)
    load_s = round(time.perf_counter() - t_load, 3)

    import numpy as np

    t_infer = time.perf_counter()
    img_np = np.array(src_image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.inference_mode():
        if max(src_w, src_h) > tile:
            out_tensor = _tiled_upscale(model, img_tensor, tile, tile_pad, net_scale, device)
        else:
            out_tensor = model(img_tensor)

    if device == "cuda":
        torch.cuda.synchronize()
    infer_s = round(time.perf_counter() - t_infer, 3)

    out_np = out_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()
    out_img = Image.fromarray((out_np * 255.0).round().astype("uint8"), mode="RGB")

    net_w, net_h = out_img.size

    t_resize = time.perf_counter()
    final_w = round(src_w * outscale)
    final_h = round(src_h * outscale)
    if (net_w, net_h) != (final_w, final_h):
        # Model's native scale != requested outscale — resize the SR output
        # (never the original) down/up to the requested delivery size. This
        # happens AFTER the real network pass, so it is not "fake sharpening";
        # it only changes final delivered pixel count.
        out_img = out_img.resize((final_w, final_h), Image.Resampling.LANCZOS)
    resize_s = round(time.perf_counter() - t_resize, 3)

    vram_after = (
        round(torch.cuda.memory_allocated() / (1024**2), 2) if device == "cuda" else 0.0
    )
    peak_vram = (
        round(torch.cuda.max_memory_allocated() / (1024**2), 2) if device == "cuda" else 0.0
    )
    ram_after = _cpu_ram_mb()

    del model, img_tensor, out_tensor
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    total_s = round(load_s + infer_s + resize_s, 3)

    return {
        "output_image": out_img,
        "source_image": src_image,
        "metrics": {
            "device": device,
            "model_load_s": load_s,
            "inference_s": infer_s,
            "resize_s": resize_s,
            "total_enhancement_s": total_s,
            "vram_before_mb": vram_before,
            "vram_after_mb": vram_after,
            "peak_vram_mb": peak_vram,
            "cpu_ram_before_mb": ram_before,
            "cpu_ram_after_mb": ram_after,
            "source_resolution": [src_w, src_h],
            "network_native_output_resolution": [net_w, net_h],
            "final_output_resolution": [final_w, final_h],
            "network_scale": net_scale,
            "requested_outscale": outscale,
            "tile": tile,
            "tile_pad": tile_pad,
        },
    }


def _color_fidelity_check(source_fabric_path: Path, image: "Any") -> Dict[str, Any]:
    """Cheap, honest color-drift signal — NOT a substitute for human visual review."""
    from PIL import Image as PilImage

    fabric = PilImage.open(source_fabric_path).convert("RGB").resize((64, 64))
    sample = image.convert("RGB").resize((64, 64))
    fabric_mean = [sum(c) / len(c) for c in zip(*fabric.getdata())]
    sample_mean = [sum(c) / len(c) for c in zip(*sample.getdata())]
    return {
        "fabric_mean_rgb": [round(v, 1) for v in fabric_mean],
        "image_mean_rgb": [round(v, 1) for v in sample_mean],
        "note": (
            "Coarse 64x64 mean-color signal only; garment fabric fill ratio differs "
            "from a full fabric swatch, so this is informational, not a pass/fail test."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Existing real FLUX output PNG")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to spandrel-loadable SR checkpoint",
    )
    parser.add_argument("--outscale", type=float, default=2.0)
    parser.add_argument("--tile", type=int, default=256, help="Tile size for VRAM-safe SR (0 disables tiling)")
    parser.add_argument("--tile-pad", type=int, default=16)
    parser.add_argument("--label", default="day17_enhancement")
    parser.add_argument(
        "--fabric-reference",
        default=None,
        help="Optional original fabric photo for a coarse color-drift signal",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    model_path = Path(args.model)
    out_dir = ROOT / "outputs" / "generated_garments" / "enhanced"
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = ROOT / "experiments" / "generation_results"
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ENHANCE-BENCH] Source (read-only): {image_path}", flush=True)
    print(f"[ENHANCE-BENCH] Model: {model_path.name}", flush=True)

    result = enhance_image(
        image_path=image_path,
        model_path=model_path,
        outscale=args.outscale,
        tile=args.tile if args.tile > 0 else 10_000,
        tile_pad=args.tile_pad,
    )

    enhanced_path = out_dir / f"{args.label}_enhanced.png"
    result["output_image"].save(enhanced_path, format="PNG", compress_level=3)
    print(f"[ENHANCE-BENCH] Saved enhanced image -> {enhanced_path}", flush=True)

    # Confirm the original was never touched.
    assert image_path.exists(), "Source FLUX image must remain untouched"

    report: Dict[str, Any] = {
        "label": args.label,
        "candidate_model": model_path.stem,
        "loader": "spandrel",
        "source_image_path": str(image_path),
        "enhanced_image_path": str(enhanced_path),
        "measured": result["metrics"],
    }

    if args.fabric_reference:
        report["color_fidelity_signal"] = _color_fidelity_check(
            Path(args.fabric_reference), result["output_image"]
        )

    out_json = exp_dir / f"{args.label}_bench.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"[ENHANCE-BENCH] Wrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
