"""CatVTON input conditioning matching upstream Zheng-Chong/CatVTON semantics.

Upstream CatVTONPipeline.__call__ (models/CatVTON/model/pipeline.py):
  - person + mask: resize_and_crop(width, height)
  - garment (condition_image): resize_and_padding(width, height)  # preserve silhouette
  - mask polarity: white (1) = clothing region to REPLACE
  - masked person inside the model: image * (mask < 0.5)  # black = keep person

We must NOT invent a custom person×mask paste as the model input; the native
pipeline builds the agnostic latent itself. Our job is to supply the same
PIL person / garment / mask the official app.py would supply.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps

logger = logging.getLogger("fabricvision.virtual_tryon.catvton_conditioning")

# Cache keyed by resolved CatVTON root path (or the requested path string).
_UTILS_CACHE: Dict[str, Tuple[Callable[..., Image.Image], Callable[..., Image.Image]]] = {}


def _resize_and_crop(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Project-owned mirror of CatVTON ``utils.resize_and_crop`` (center-crop + LANCZOS)."""
    w, h = image.size
    target_w, target_h = size
    if w / h < target_w / target_h:
        new_w = w
        new_h = w * target_h // target_w
    else:
        new_h = h
        new_w = h * target_w // target_h
    image = image.crop(
        ((w - new_w) // 2, (h - new_h) // 2, (w + new_w) // 2, (h + new_h) // 2)
    )
    return image.resize(size, Image.LANCZOS)


def _resize_and_padding(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Project-owned mirror of CatVTON ``utils.resize_and_padding`` (fit + white letterbox)."""
    w, h = image.size
    target_w, target_h = size
    if w / h < target_w / target_h:
        new_h = target_h
        new_w = w * target_h // h
    else:
        new_w = target_w
        new_h = h * target_w // w
    image = image.resize((new_w, new_h), Image.LANCZOS)
    padding = Image.new("RGB", size, (255, 255, 255))
    padding.paste(image, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return padding


def _import_catvton_utils(catvton_root: Path):
    """Load CatVTON resize helpers from an explicit utils.py path, with local fallback.

    Prefers the real ``{catvton_root}/utils.py`` when present (loaded via importlib
    from that filesystem path — no ``sys.path`` mutation / bare ``import utils``).
    If the CatVTON checkout is missing or its utils module cannot be imported,
    returns project-owned implementations that match upstream resize semantics.
    This does not imply that real CatVTON inference is available.
    """
    root = Path(catvton_root)
    try:
        cache_key = str(root.resolve())
    except OSError:
        cache_key = str(root)

    cached = _UTILS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    utils_path = root / "utils.py"
    if utils_path.is_file():
        try:
            module_name = f"fabricvision_catvton_utils_{abs(hash(cache_key))}"
            spec = importlib.util.spec_from_file_location(module_name, utils_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot create import spec for {utils_path}")
            module = importlib.util.module_from_spec(spec)
            # Register before exec so any late self-references resolve cleanly.
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            resize_and_crop = getattr(module, "resize_and_crop")
            resize_and_padding = getattr(module, "resize_and_padding")
            if not callable(resize_and_crop) or not callable(resize_and_padding):
                raise AttributeError("CatVTON utils.py missing resize helpers")
            pair = (resize_and_crop, resize_and_padding)
            _UTILS_CACHE[cache_key] = pair
            logger.debug("Loaded CatVTON resize utils from %s", utils_path)
            return pair
        except Exception as exc:
            logger.warning(
                "Failed to load CatVTON utils from %s (%s); "
                "using project-owned resize fallbacks for conditioning.",
                utils_path,
                exc,
            )

    pair = (_resize_and_crop, _resize_and_padding)
    _UTILS_CACHE[cache_key] = pair
    logger.debug(
        "CatVTON utils unavailable at %s; using project-owned resize fallbacks.",
        utils_path,
    )
    return pair


def resize_person_and_mask(
    person: Image.Image,
    mask: Image.Image,
    size: Tuple[int, int],
    catvton_root: Path = Path("models/CatVTON"),
) -> Tuple[Image.Image, Image.Image]:
    """Official person/mask resize: center crop to aspect then LANCZOS resize."""
    resize_and_crop, _ = _import_catvton_utils(catvton_root)
    person_rgb = person.convert("RGB")
    mask_l = mask.convert("L")
    return resize_and_crop(person_rgb, size), resize_and_crop(mask_l, size)


def tight_crop_garment_rgb(
    garment: Image.Image,
    white_threshold: int = 245,
    margin: int = 4,
) -> Image.Image:
    """
    Crop near-white padding so letterboxing keeps more useful garment pixels.

    Does not stretch or distort — only removes empty canvas before
    resize_and_padding. Preserves neckline through hem of the foreground.
    """
    rgb = garment.convert("RGB")
    arr = np.asarray(rgb)
    fg = ~(
        (arr[:, :, 0] >= white_threshold)
        & (arr[:, :, 1] >= white_threshold)
        & (arr[:, :, 2] >= white_threshold)
    )
    ys, xs = np.where(fg)
    if ys.size < 16:
        return rgb
    h, w = arr.shape[:2]
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(h, int(ys.max()) + 1 + margin)
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(w, int(xs.max()) + 1 + margin)
    if (y1 - y0) < 16 or (x1 - x0) < 16:
        return rgb
    return Image.fromarray(arr[y0:y1, x0:x1], mode="RGB")


def resize_garment_condition(
    garment: Image.Image,
    size: Tuple[int, int],
    catvton_root: Path = Path("models/CatVTON"),
) -> Image.Image:
    """Official garment resize: tight-crop then letterbox pad (no stretch/crop of silhouette)."""
    _, resize_and_padding = _import_catvton_utils(catvton_root)
    cropped = tight_crop_garment_rgb(garment.convert("RGB"))
    return resize_and_padding(cropped, size)


def prepare_catvton_person_conditioning(
    person: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    """
    Visualization / debug helper mirroring pipeline: masked_image = image * (mask < 0.5).

    White mask pixels are zeroed (clothing removed). This image is NOT passed as a
    separate tensor when using native CatVTONPipeline — the pipeline recomputes it —
    but we save it so we can verify polarity and coverage.
    """
    person_rgb = person.convert("RGB")
    mask_l = mask.convert("L")
    if mask_l.size != person_rgb.size:
        mask_l = mask_l.resize(person_rgb.size, Image.Resampling.NEAREST)
    p = np.asarray(person_rgb, dtype=np.float32)
    m = np.asarray(mask_l, dtype=np.float32) / 255.0
    # Upstream: keep where mask < 0.5
    keep = (m < 0.5).astype(np.float32)[..., None]
    out = (p * keep).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def soft_blur_mask(mask: Image.Image, blur_radius: int = 9) -> Image.Image:
    """Match official app.py mask_processor.blur(..., blur_factor=9)."""
    if blur_radius <= 0:
        return mask.convert("L")
    return mask.convert("L").filter(ImageFilter.GaussianBlur(radius=blur_radius))


def analyze_mask(mask: Image.Image) -> Dict[str, Any]:
    """Programmatic mask QA used in debug dumps and production validation."""
    arr = np.asarray(mask.convert("L"), dtype=np.uint8)
    h, w = arr.shape
    binary = arr > 127
    area = float(binary.mean())
    ys, xs = np.where(binary)
    if ys.size == 0:
        bbox = None
    else:
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    upper = binary[: int(h * 0.55), :]
    lower = binary[int(h * 0.55) :, :]
    legs = binary[int(h * 0.72) :, :]
    thighs = binary[int(h * 0.62) : int(h * 0.82), :]
    head = binary[: int(h * 0.12), :]
    # Rough background: columns near left/right edges
    edge = np.concatenate([binary[:, : max(1, w // 20)], binary[:, -max(1, w // 20) :]], axis=1)

    # Connected components + lower-body split detection (pants vs dress)
    try:
        import cv2

        num, _, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8) * 255, 8)
        n_fg = max(0, num - 1)
    except Exception:
        n_fg = -1

    def _row_segments(y: int) -> int:
        row = binary[min(max(0, y), h - 1)].astype(np.uint8)
        d = np.diff(np.concatenate([[0], row, [0]]))
        return int((d == 1).sum())

    mid_thigh_y = int(h * 0.78)
    calf_y = int(h * 0.88)
    lower_split_segments = max(_row_segments(mid_thigh_y), _row_segments(calf_y))
    bottom_frac = round(float(bbox[3]) / max(1, h), 4) if bbox else 0.0

    return {
        "mask_area_ratio": round(area, 4),
        "mask_bbox": bbox,
        "mask_bottom_frac": bottom_frac,
        "upper_body_area_ratio": round(float(upper.mean()), 4) if upper.size else 0.0,
        "lower_body_area_ratio": round(float(lower.mean()), 4) if lower.size else 0.0,
        "thigh_fill_ratio": round(float(thighs.mean()), 4) if thighs.size else 0.0,
        "leg_fill_ratio": round(float(legs.mean()), 4) if legs.size else 0.0,
        "head_fill_ratio": round(float(head.mean()), 4) if head.size else 0.0,
        "background_edge_fill_ratio": round(float(edge.mean()), 4) if edge.size else 0.0,
        "lower_split_segments": lower_split_segments,
        "connected_components": n_fg,
        "resolution": [w, h],
    }


def validate_clothing_mask(
    mask: Image.Image,
    cloth_type: str = "upper",
) -> Tuple[bool, str]:
    """
    Reject masks that cannot drive a legitimate try-on.

    Returns (ok, reason).
    """
    stats = analyze_mask(mask)
    area = stats["mask_area_ratio"]
    if area < 0.04:
        return False, f"mask too small (area_ratio={area})"
    if area > 0.85 and cloth_type not in ("overall", "dress", "outer"):
        return False, f"mask too large (area_ratio={area})"
    if area > 0.92:
        return False, f"mask too large (area_ratio={area})"
    if stats["background_edge_fill_ratio"] > 0.35:
        return False, "mask spills into image edges/background"
    if stats["head_fill_ratio"] > 0.45:
        return False, "mask covers face/head excessively"
    if cloth_type == "upper" and stats["leg_fill_ratio"] > 0.25:
        return False, f"upper mask covers legs (leg_fill={stats['leg_fill_ratio']})"
    if cloth_type == "upper" and stats["upper_body_area_ratio"] < 0.03:
        return False, "upper mask has negligible torso coverage"

    # Dress / overall: require skirt-length coverage. Do NOT require leg_fill==0 —
    # a full dress legitimately occupies pixels in front of the legs.
    if cloth_type in ("overall", "dress", "outer"):
        bottom = float(stats.get("mask_bottom_frac") or 0.0)
        if bottom < 0.72:
            return False, (
                f"overall mask stops prematurely above dress region "
                f"(bottom_frac={bottom})"
            )
        if float(stats.get("lower_body_area_ratio") or 0.0) < 0.06:
            return False, "overall mask has insufficient skirt/lower coverage"
        if float(stats.get("thigh_fill_ratio") or 0.0) < 0.04:
            return False, "overall mask missing thigh/skirt band (looks like upper-only)"
        # Split pant-legs at mid-thigh/calf → truncated dress transfers
        if int(stats.get("lower_split_segments") or 0) >= 2:
            return False, (
                f"overall mask is split into pant legs "
                f"(segments={stats['lower_split_segments']}); need continuous dress region"
            )

    if stats["connected_components"] == 0:
        return False, "mask has no foreground"
    if stats["connected_components"] > 6:
        return False, f"mask too fragmented ({stats['connected_components']} components)"
    return True, "ok"


def attn_version_for_resolution(height: int, width: int) -> str:
    """
    Select attention checkpoint matching training resolution.

    mix-48k-1024 is trained near 768x1024; using it at 512x512 warps spatial priors.
    vitonhd-16k-512 / dresscode-16k-512 match 512-scale inference on 6GB GPUs.
    """
    if max(height, width) <= 512:
        return "vitonhd"
    return "mix"
