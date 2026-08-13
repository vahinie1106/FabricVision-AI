"""Persist and verify final garment PNG outputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from PIL import Image
import numpy as np

logger = logging.getLogger("fabricvision.garment_generation.output")


def persist_and_verify_garment_png(
    image: Image.Image,
    destination: str | Path,
    *,
    expected_size: tuple[int, int] | None = None,
    compress_level: int = 3,
) -> Dict[str, Any]:
    """
    Save ``image`` to ``destination`` and verify it is a valid non-black PNG.

    Raises RuntimeError if save or verification fails — callers must treat this
    as a hard generation failure (do not mark the job completed).
    """
    if image is None or not isinstance(image, Image.Image):
        raise RuntimeError("Final garment image is missing or not a PIL Image.")

    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)

    rgb = image.convert("RGB") if image.mode not in ("RGB", "RGBA") else image
    if rgb.mode == "RGBA":
        # Flatten onto white for consistent product delivery.
        bg = Image.new("RGB", rgb.size, (255, 255, 255))
        bg.paste(rgb, mask=rgb.split()[-1])
        rgb = bg
    elif rgb.mode != "RGB":
        rgb = rgb.convert("RGB")

    rgb.save(dest, format="PNG", compress_level=int(compress_level), optimize=False)

    if not dest.exists():
        raise RuntimeError(f"Final garment PNG was not written: {dest}")

    file_size = int(dest.stat().st_size)
    if file_size <= 0:
        raise RuntimeError(f"Final garment PNG is empty (0 bytes): {dest}")

    try:
        with Image.open(dest) as loaded:
            loaded.load()
            mode = loaded.mode
            width, height = loaded.size
            arr = np.asarray(loaded.convert("RGB"))
    except Exception as exc:
        raise RuntimeError(
            f"Final garment PNG cannot be reopened as a valid image: {dest} ({exc})"
        ) from exc

    if mode not in ("RGB", "RGBA"):
        raise RuntimeError(f"Final garment PNG has unexpected mode={mode}: {dest}")
    if width < 8 or height < 8:
        raise RuntimeError(f"Final garment PNG is too small ({width}x{height}): {dest}")
    if expected_size is not None:
        ew, eh = int(expected_size[0]), int(expected_size[1])
        if (width, height) != (ew, eh):
            logger.warning(
                "Final PNG size %sx%s differs from generation %sx%s (continuing)",
                width,
                height,
                ew,
                eh,
            )

    pix_min = float(arr.min())
    pix_max = float(arr.max())
    pix_mean = float(arr.mean())
    pix_std = float(arr.std())
    zero_pct = float(np.mean(arr == 0) * 100.0)

    if pix_max <= 0 or (pix_std == 0.0 and pix_mean == 0.0):
        raise RuntimeError(
            f"Final garment PNG is completely black/empty "
            f"(min={pix_min}, max={pix_max}, mean={pix_mean}, std={pix_std}): {dest}"
        )

    stats = {
        "path": str(dest.resolve()),
        "file_size": file_size,
        "width": width,
        "height": height,
        "mode": mode,
        "min": pix_min,
        "max": pix_max,
        "mean": pix_mean,
        "std": pix_std,
        "zero_pct": zero_pct,
    }
    logger.info(
        "[GARMENT OUTPUT] saved=%s size=%s bytes %sx%s mode=%s "
        "min=%.1f max=%.1f mean=%.2f std=%.2f zero%%=%.2f",
        stats["path"],
        file_size,
        width,
        height,
        mode,
        pix_min,
        pix_max,
        pix_mean,
        pix_std,
        zero_pct,
    )
    print(f"[FLUX] output size = {width}x{height}", flush=True)
    print(f"[FLUX] output mode = {mode}", flush=True)
    print(f"[FLUX] output saved = {stats['path']}", flush=True)
    print(f"[FLUX] file size = {file_size}", flush=True)
    print("[FLUX] real FLUX output = true", flush=True)
    print(
        f"[GARMENT OUTPUT] FINAL PATH={stats['path']} "
        f"FILE SIZE={file_size} IMAGE SIZE={width}x{height} MODE={mode} "
        f"MIN={pix_min} MAX={pix_max} MEAN={pix_mean:.2f} STD={pix_std:.2f} "
        f"ZERO%={zero_pct:.2f}",
        flush=True,
    )
    return stats
