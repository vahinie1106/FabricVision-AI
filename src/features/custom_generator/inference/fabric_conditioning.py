"""Build FLUX Kontext conditioning images for fabric→garment synthesis.

FLUX.1 Kontext preserves input composition. A raw fabric swatch therefore
reproduces a textile fill. We build a garment-shaped mockup filled with the
uploaded fabric so Kontext edits clothing structure while keeping material identity.

When the UI selects an explicit Color (not Match Fabric), we tint the fabric
fill to the target hue while preserving luminance/pattern so Kontext is not
locked to the upload's original palette.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

from PIL import Image, ImageDraw

logger = logging.getLogger("fabricvision.garment_generation.fabric_conditioning")

# UI / vocabulary color names → RGB for lightweight luminance tinting.
_COLOR_NAME_RGB: dict[str, Tuple[int, int, int]] = {
    "black": (20, 20, 20),
    "white": (245, 245, 245),
    "red": (190, 35, 40),
    "blue": (35, 90, 200),
    "green": (40, 140, 70),
    "yellow": (230, 200, 40),
    "navy_blue": (20, 40, 110),
    "navy": (20, 40, 110),
    "royal_blue": (40, 80, 200),
    "maroon": (128, 30, 40),
    "beige": (210, 190, 160),
    "olive_green": (110, 120, 50),
    "olive": (110, 120, 50),
    "pastel_pink": (240, 170, 190),
    "pink": (230, 120, 160),
    "lavender": (180, 160, 210),
    "cream": (245, 235, 210),
    "orange": (220, 120, 40),
    "purple": (120, 60, 160),
    "brown": (120, 75, 45),
    "gray": (140, 140, 140),
    "grey": (140, 140, 140),
}


def normalize_color_key(color: Optional[str]) -> str:
    """Normalize UI/API color strings to snake_case tokens."""
    if color is None:
        return ""
    return str(color).strip().lower().replace("-", " ").replace(" ", "_")


def is_match_fabric_color(color: Optional[str]) -> bool:
    """True when Color means preserve uploaded textile colors."""
    key = normalize_color_key(color)
    return key in ("", "match_fabric", "matchfabric")


def resolve_target_rgb(color: Optional[str]) -> Optional[Tuple[int, int, int]]:
    """Map a color name to RGB, or None if unknown / Match Fabric."""
    key = normalize_color_key(color)
    if is_match_fabric_color(key):
        return None
    if key in _COLOR_NAME_RGB:
        return _COLOR_NAME_RGB[key]
    # Soft match: pastel_pink → pink, olive_green → olive
    for name, rgb in _COLOR_NAME_RGB.items():
        if name in key or key in name:
            return rgb
    return None


def tint_fabric_preserving_texture(
    fabric_image: Image.Image,
    target_color: str,
) -> Image.Image:
    """
    Recolor fabric to ``target_color`` while keeping print/texture luminance.

    Multiplies per-pixel luminance by the target RGB so floral/print contrast
    survives; chrominance follows the UI color. Pure PIL (no extra model).
    """
    rgb = resolve_target_rgb(target_color)
    if rgb is None:
        return fabric_image.convert("RGB")

    src = fabric_image.convert("RGB")
    # Luminance via ITU-R BT.601; scale each channel by target/255.
    gray = src.convert("L")
    tr, tg, tb = rgb
    # Build solid color then multiply with luminance (ImageChops.multiply).
    solid = Image.new("RGB", src.size, (tr, tg, tb))
    # Expand gray to RGB so multiply works channel-wise.
    lum_rgb = Image.merge("RGB", (gray, gray, gray))
    from PIL import ImageChops

    tinted = ImageChops.multiply(solid, lum_rgb)
    logger.info(
        "Tinted fabric for explicit color=%s rgb=%s (luminance-preserving)",
        normalize_color_key(target_color),
        rgb,
    )
    return tinted


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
    target_color: Optional[str] = None,
) -> Image.Image:
    """
    Create a white-studio garment mockup filled with the uploaded fabric.

    Hard edges (no mask Gaussian blur): soft masks caused measurable edge halos
    in generated outputs. Kontext still invents folds from the silhouette.

    When ``target_color`` is an explicit UI color (not Match Fabric), the fabric
    fill is luminance-tinted so Kontext is not anchored to the upload palette.
    """
    if fabric_image is None:
        raise ValueError("fabric_image is required")

    w, h = int(width), int(height)
    source = fabric_image
    recolored = False
    if target_color and not is_match_fabric_color(target_color):
        source = tint_fabric_preserving_texture(fabric_image, target_color)
        recolored = True

    fabric_fill = _cover_fabric(source, w, h)
    # Mild unsharp on the fabric fill only (not the silhouette edge) to counter
    # LANCZOS softness before Kontext sees the mockup. Radius kept tiny.
    try:
        from PIL import ImageFilter

        fabric_fill = fabric_fill.filter(
            ImageFilter.UnsharpMask(radius=1.2, percent=110, threshold=2)
        )
    except Exception:
        pass
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    polygon = _polygon_for_garment(garment_type, w, h, sleeve=sleeve)
    draw.polygon(list(polygon), fill=255)
    # NO GaussianBlur — preserves crisp garment/background boundary for Kontext.

    canvas = Image.new("RGB", (w, h), background)
    canvas.paste(fabric_fill, (0, 0), mask=mask)

    logger.info(
        "Built garment conditioning image: garment=%s sleeve=%s size=%sx%s "
        "blur=none unsharp=mild conditioning_recolored=%s target_color=%s",
        garment_type,
        sleeve or "default",
        w,
        h,
        recolored,
        normalize_color_key(target_color) if recolored else "match_fabric",
    )
    return canvas
