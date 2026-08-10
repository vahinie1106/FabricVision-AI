"""Build FLUX Kontext conditioning images for fabric→garment synthesis.

FLUX.1 Kontext preserves input composition. A raw fabric swatch therefore
reproduces a textile fill. We build a garment-shaped mockup filled with the
uploaded fabric so Kontext edits clothing structure while keeping material identity.
"""

from __future__ import annotations

import logging
from typing import Sequence, Tuple

from PIL import Image, ImageDraw

logger = logging.getLogger("fabricvision.garment_generation.fabric_conditioning")


def _cover_fabric(fabric: Image.Image, width: int, height: int) -> Image.Image:
    """
    Fill the garment canvas with fabric while preserving print scale when possible.

    Softness note: shrinking a high-res fabric photo to 512×512 before Kontext
    smears motif detail. Prefer native-scale center crop when the upload is
    already large enough; tile small swatches; only LANCZOS-cover when needed.
    """
    fabric_rgb = fabric.convert("RGB")
    fw, fh = fabric_rgb.size
    if fw < 8 or fh < 8:
        return fabric_rgb.resize((width, height), Image.Resampling.LANCZOS)

    # Tile small pattern swatches to preserve motif scale relative to garment
    if fw <= width // 2 and fh <= height // 2:
        tiled = Image.new("RGB", (width, height))
        for x in range(0, width, fw):
            for y in range(0, height, fh):
                tiled.paste(fabric_rgb, (x, y))
        return tiled

    # Large fabric: center-crop at native resolution (no downscale smear).
    if fw >= width and fh >= height:
        left = max(0, (fw - width) // 2)
        top = max(0, (fh - height) // 2)
        return fabric_rgb.crop((left, top, left + width, top + height))

    # One dimension smaller than canvas — cover-scale (unavoidable resize).
    scale = max(width / float(fw), height / float(fh))
    nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
    resized = fabric_rgb.resize((nw, nh), Image.Resampling.LANCZOS)
    left = max(0, (nw - width) // 2)
    top = max(0, (nh - height) // 2)
    return resized.crop((left, top, left + width, top + height))


def _shirt_polygon(w: int, h: int, long_sleeve: bool = False) -> Sequence[Tuple[int, int]]:
    if long_sleeve:
        return [
            (int(w * 0.34), int(h * 0.12)),
            (int(w * 0.42), int(h * 0.18)),
            (int(w * 0.50), int(h * 0.16)),
            (int(w * 0.58), int(h * 0.18)),
            (int(w * 0.66), int(h * 0.12)),
            (int(w * 0.95), int(h * 0.22)),
            (int(w * 0.92), int(h * 0.48)),
            (int(w * 0.72), int(h * 0.40)),
            (int(w * 0.70), int(h * 0.88)),
            (int(w * 0.30), int(h * 0.88)),
            (int(w * 0.28), int(h * 0.40)),
            (int(w * 0.08), int(h * 0.48)),
            (int(w * 0.05), int(h * 0.22)),
        ]
    return [
        (int(w * 0.32), int(h * 0.12)),
        (int(w * 0.40), int(h * 0.18)),
        (int(w * 0.50), int(h * 0.16)),
        (int(w * 0.60), int(h * 0.18)),
        (int(w * 0.68), int(h * 0.12)),
        (int(w * 0.88), int(h * 0.28)),
        (int(w * 0.78), int(h * 0.40)),
        (int(w * 0.70), int(h * 0.34)),
        (int(w * 0.70), int(h * 0.88)),
        (int(w * 0.30), int(h * 0.88)),
        (int(w * 0.30), int(h * 0.34)),
        (int(w * 0.22), int(h * 0.40)),
        (int(w * 0.12), int(h * 0.28)),
    ]


def _dress_polygon(w: int, h: int) -> Sequence[Tuple[int, int]]:
    return [
        (int(w * 0.34), int(h * 0.08)),
        (int(w * 0.42), int(h * 0.14)),
        (int(w * 0.50), int(h * 0.12)),
        (int(w * 0.58), int(h * 0.14)),
        (int(w * 0.66), int(h * 0.08)),
        (int(w * 0.82), int(h * 0.22)),
        (int(w * 0.72), int(h * 0.30)),
        (int(w * 0.78), int(h * 0.94)),
        (int(w * 0.22), int(h * 0.94)),
        (int(w * 0.28), int(h * 0.30)),
        (int(w * 0.18), int(h * 0.22)),
    ]


def _jacket_polygon(w: int, h: int) -> Sequence[Tuple[int, int]]:
    return [
        (int(w * 0.30), int(h * 0.10)),
        (int(w * 0.40), int(h * 0.16)),
        (int(w * 0.50), int(h * 0.18)),
        (int(w * 0.60), int(h * 0.16)),
        (int(w * 0.70), int(h * 0.10)),
        (int(w * 0.94), int(h * 0.28)),
        (int(w * 0.82), int(h * 0.40)),
        (int(w * 0.72), int(h * 0.34)),
        (int(w * 0.74), int(h * 0.90)),
        (int(w * 0.52), int(h * 0.82)),
        (int(w * 0.48), int(h * 0.82)),
        (int(w * 0.26), int(h * 0.90)),
        (int(w * 0.28), int(h * 0.34)),
        (int(w * 0.18), int(h * 0.40)),
        (int(w * 0.06), int(h * 0.28)),
    ]


def _kurta_polygon(w: int, h: int) -> Sequence[Tuple[int, int]]:
    return [
        (int(w * 0.34), int(h * 0.08)),
        (int(w * 0.42), int(h * 0.14)),
        (int(w * 0.50), int(h * 0.12)),
        (int(w * 0.58), int(h * 0.14)),
        (int(w * 0.66), int(h * 0.08)),
        (int(w * 0.90), int(h * 0.24)),
        (int(w * 0.78), int(h * 0.34)),
        (int(w * 0.72), int(h * 0.34)),
        (int(w * 0.74), int(h * 0.94)),
        (int(w * 0.26), int(h * 0.94)),
        (int(w * 0.28), int(h * 0.34)),
        (int(w * 0.22), int(h * 0.34)),
        (int(w * 0.10), int(h * 0.24)),
    ]


def _is_long_sleeve(sleeve: str) -> bool:
    s = (sleeve or "").lower().replace(" ", "_").replace("-", "_")
    return any(
        k in s
        for k in ("long", "full", "three_quarter", "threequarter", "bishop", "puff")
    )


def _polygon_for_garment(
    garment_type: str, w: int, h: int, sleeve: str = ""
) -> Sequence[Tuple[int, int]]:
    key = (garment_type or "shirt").lower().replace(" ", "_")
    long_sleeve = _is_long_sleeve(sleeve)
    if key in {"dress", "gown", "frock"}:
        return _dress_polygon(w, h)
    if key in {"jacket", "blazer", "coat", "hoodie"}:
        return _jacket_polygon(w, h)
    if key in {"kurta", "kurti", "kameez", "tunic", "top"}:
        # Kurti/top: sleeve-aware shirt silhouette (kurta poly ignores sleeve)
        if key in {"kurta"}:
            return _kurta_polygon(w, h)
        return _shirt_polygon(w, h, long_sleeve=long_sleeve)
    return _shirt_polygon(w, h, long_sleeve=long_sleeve)


def build_garment_conditioning_image(
    fabric_image: Image.Image,
    garment_type: str = "shirt",
    width: int = 512,
    height: int = 512,
    sleeve: str = "",
    background: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Create a white-studio garment mockup filled with the uploaded fabric.

    Hard edges (no mask Gaussian blur): soft masks caused measurable edge halos
    in generated outputs. Kontext still invents folds from the silhouette.
    """
    if fabric_image is None:
        raise ValueError("fabric_image is required")

    w, h = int(width), int(height)
    fabric_fill = _cover_fabric(fabric_image, w, h)
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    polygon = _polygon_for_garment(garment_type, w, h, sleeve=sleeve)
    draw.polygon(list(polygon), fill=255)
    # NO GaussianBlur — preserves crisp garment/background boundary for Kontext.

    canvas = Image.new("RGB", (w, h), background)
    canvas.paste(fabric_fill, (0, 0), mask=mask)

    logger.info(
        "Built garment conditioning image: garment=%s sleeve=%s size=%sx%s blur=none",
        garment_type,
        sleeve or "default",
        w,
        h,
    )
    return canvas
