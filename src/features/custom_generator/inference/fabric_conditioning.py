"""Build FLUX Kontext conditioning images for fabric→garment synthesis.

FLUX.1 Kontext preserves input composition. A raw fabric swatch therefore
reproduces a textile fill. We build a garment-shaped mockup filled with the
uploaded fabric so Kontext edits clothing structure while keeping material identity.

When the UI selects an explicit Color (not Match Fabric), only the BASE fabric
region is recolored to the target. Print/motif pixels keep their original hues
(e.g. white ground → blue ground, red florals stay red).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

logger = logging.getLogger("fabricvision.garment_generation.fabric_conditioning")

# UI / vocabulary color names → RGB for base-fabric recolor targets.
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


@dataclass
class FabricBaseRecolorResult:
    """Audit bundle for base-only fabric recoloring."""

    image: Image.Image
    original: Image.Image
    base_mask: Image.Image
    recolored_base: Image.Image
    base_rgb: Tuple[int, int, int]
    target_rgb: Tuple[int, int, int]
    base_coverage: float
    target_color: str


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


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB array (..., 3) to CIE LAB (D65)."""
    rgb_f = rgb.astype(np.float64) / 255.0
    mask = rgb_f > 0.04045
    linear = np.empty_like(rgb_f)
    linear[~mask] = rgb_f[~mask] / 12.92
    linear[mask] = ((rgb_f[mask] + 0.055) / 1.055) ** 2.4
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    xyz = linear @ m.T
    xyz[..., 0] /= 0.95047
    xyz[..., 2] /= 1.08883

    def _f(t: np.ndarray) -> np.ndarray:
        return np.where(t > 0.008856, np.cbrt(t), (7.787 * t) + (16.0 / 116.0))

    fx, fy, fz = _f(xyz[..., 0]), _f(xyz[..., 1]), _f(xyz[..., 2])
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1)


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert CIE LAB (D65) array (..., 3) to uint8 RGB."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    def _f_inv(t: np.ndarray) -> np.ndarray:
        t3 = t**3
        return np.where(t3 > 0.008856, t3, (t - 16.0 / 116.0) / 7.787)

    xyz = np.stack([_f_inv(fx) * 0.95047, _f_inv(fy), _f_inv(fz) * 1.08883], axis=-1)
    m_inv = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        dtype=np.float64,
    )
    linear = xyz @ m_inv.T
    linear = np.clip(linear, 0.0, None)
    mask = linear > 0.0031308
    srgb = np.empty_like(linear)
    srgb[~mask] = linear[~mask] * 12.92
    srgb[mask] = 1.055 * np.power(linear[mask], 1.0 / 2.4) - 0.055
    return np.clip(np.round(srgb * 255.0), 0, 255).astype(np.uint8)


def _estimate_base_rgb(arr: np.ndarray) -> Tuple[int, int, int]:
    """Estimate dominant/base fabric RGB via quantized mode (not motif accents)."""
    # Coarse quantization keeps print accents from winning the mode on florals.
    q = (arr.astype(np.int32) // 24) * 24 + 12
    flat = q.reshape(-1, 3)
    # Pack RGB into int keys for a fast mode count.
    keys = flat[:, 0] * 1_000_000 + flat[:, 1] * 1_000 + flat[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    mode_key = int(vals[int(np.argmax(counts))])
    br = mode_key // 1_000_000
    bg = (mode_key // 1_000) % 1_000
    bb = mode_key % 1_000
    # Refine: mean of pixels near the mode cluster.
    mode_rgb = np.array([br, bg, bb], dtype=np.float64)
    dist = np.linalg.norm(flat.astype(np.float64) - mode_rgb, axis=1)
    keep = dist <= max(18.0, float(np.percentile(dist, 25)))
    if not np.any(keep):
        keep = dist <= float(np.percentile(dist, 40))
    mean = flat[keep].mean(axis=0) if np.any(keep) else mode_rgb
    return (
        int(np.clip(round(mean[0]), 0, 255)),
        int(np.clip(round(mean[1]), 0, 255)),
        int(np.clip(round(mean[2]), 0, 255)),
    )


def _chroma(lab: np.ndarray) -> np.ndarray:
    return np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)


def _soft_base_mask(lab: np.ndarray, base_lab: np.ndarray) -> np.ndarray:
    """
    Soft mask in [0,1]: 1 = base fabric, 0 = protected print/motif.

    Uses LAB distance from the base cluster, then further protects high-chroma
    pixels that differ from the base (printed motifs), without hardcoding hues.
    """
    dist = np.linalg.norm(lab - base_lab.reshape(1, 1, 3), axis=-1)
    # Aggressive base membership: most near-base pixels should recolor.
    p35 = float(np.percentile(dist, 35))
    p70 = float(np.percentile(dist, 70))
    span = max(p70 - p35, 6.0)
    alpha = 1.0 - (dist - p35) / span
    alpha = np.clip(alpha, 0.0, 1.0)

    # Protect chromatic print accents that diverge from the base chroma.
    base_c = float(np.sqrt(base_lab[1] ** 2 + base_lab[2] ** 2))
    pix_c = _chroma(lab)
    # Motif-like: much more chromatic than base OR far in ab-plane.
    ab_dist = np.sqrt((lab[..., 1] - base_lab[1]) ** 2 + (lab[..., 2] - base_lab[2]) ** 2)
    motif_score = np.clip((pix_c - (base_c + 8.0)) / 20.0, 0.0, 1.0)
    motif_score = np.maximum(motif_score, np.clip((ab_dist - 18.0) / 25.0, 0.0, 1.0))
    alpha = alpha * (1.0 - 0.95 * motif_score)
    # Harden confident base pixels so explicit colors read clearly after blend.
    hard = (alpha >= 0.40).astype(np.float64)
    alpha = np.clip(0.20 * alpha + 0.80 * hard, 0.0, 1.0)
    return alpha.astype(np.float64)


def _studio_background_mask(arr: np.ndarray) -> np.ndarray:
    """True for border-connected near-pure-white studio background pixels."""
    h, w = arr.shape[:2]
    lum = arr.astype(np.float64).mean(axis=2)
    chroma = arr.astype(np.float64).max(axis=2) - arr.astype(np.float64).min(axis=2)
    # Strict threshold so off-white fabric bases (e.g. 242) are NOT flood-eaten.
    near_bg = (lum >= 250.0) & (chroma <= 12.0)
    visited = np.zeros((h, w), dtype=bool)
    from collections import deque

    q: deque = deque()
    for x in range(w):
        for y in (0, h - 1):
            if near_bg[y, x]:
                visited[y, x] = True
                q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if near_bg[y, x] and not visited[y, x]:
                visited[y, x] = True
                q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and near_bg[ny, nx]:
                visited[ny, nx] = True
                q.append((ny, nx))
    return visited


def recolor_fabric_base_preserving_motifs(
    fabric_image: Image.Image,
    target_color: str,
    *,
    protect_studio_background: bool = False,
    strength: float = 1.0,
) -> FabricBaseRecolorResult:
    """
    Recolor only the dominant/base fabric region to ``target_color``.

    Print/motif pixels (far from the base color in LAB / high chroma delta)
    keep original RGB. Base pixels transfer target chrominance in LAB while
    preserving local lightness (texture/folds).

    When ``protect_studio_background`` is True (post-generation product shots),
    border-connected near-white pixels stay white so the canvas is not tinted.
    """
    target_rgb = resolve_target_rgb(target_color)
    original = fabric_image.convert("RGB")
    if target_rgb is None:
        blank = Image.new("L", original.size, 0)
        return FabricBaseRecolorResult(
            image=original,
            original=original.copy(),
            base_mask=blank,
            recolored_base=original.copy(),
            base_rgb=(0, 0, 0),
            target_rgb=(0, 0, 0),
            base_coverage=0.0,
            target_color=normalize_color_key(target_color),
        )

    arr = np.asarray(original, dtype=np.uint8)
    lab = _rgb_to_lab(arr)
    base_rgb = _estimate_base_rgb(arr)
    base_lab = _rgb_to_lab(np.array(base_rgb, dtype=np.uint8).reshape(1, 1, 3))[0, 0]
    target_lab = _rgb_to_lab(np.array(target_rgb, dtype=np.uint8).reshape(1, 1, 3))[0, 0]

    alpha = _soft_base_mask(lab, base_lab)
    if protect_studio_background:
        bg = _studio_background_mask(arr)
        alpha = np.where(bg, 0.0, alpha)

    # Mild blur softens base↔motif transitions without smearing motif geometry.
    alpha_img = Image.fromarray((alpha * 255.0).astype(np.uint8), mode="L")
    alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=0.6))
    alpha = np.asarray(alpha_img, dtype=np.float64) / 255.0
    strength = float(np.clip(strength, 0.0, 1.0))
    alpha = np.clip(alpha * strength, 0.0, 1.0)

    # LAB chrominance transfer: keep local L (texture), set a/b toward target.
    # Scale L toward target L so black/white bases read correctly.
    out_lab = lab.copy()
    target_L = float(target_lab[0])
    base_L = max(float(base_lab[0]), 1.0)
    L_scaled = np.clip(lab[..., 0] * (target_L / base_L), 2.0, 98.0)
    if target_L < 35.0:
        # Dark targets need a decisive L pull or blends stay gray.
        out_lab[..., 0] = np.clip(0.10 * lab[..., 0] + 0.90 * L_scaled, 2.0, 55.0)
    elif target_L > 85.0:
        out_lab[..., 0] = np.clip(0.45 * lab[..., 0] + 0.55 * L_scaled, 40.0, 98.0)
    else:
        out_lab[..., 0] = 0.30 * lab[..., 0] + 0.70 * L_scaled
    out_lab[..., 1] = target_lab[1]
    out_lab[..., 2] = target_lab[2]
    recolored = _lab_to_rgb(out_lab).astype(np.float64)

    a3 = alpha[..., None]
    out = a3 * recolored + (1.0 - a3) * arr.astype(np.float64)
    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    result = Image.fromarray(out_u8, mode="RGB")

    coverage = float(alpha.mean())
    logger.info(
        "Base-only fabric recolor target=%s base_rgb=%s target_rgb=%s "
        "base_coverage=%.3f protect_bg=%s (motifs protected)",
        normalize_color_key(target_color),
        base_rgb,
        target_rgb,
        coverage,
        protect_studio_background,
    )
    return FabricBaseRecolorResult(
        image=result,
        original=original.copy(),
        base_mask=alpha_img,
        recolored_base=result.copy(),
        base_rgb=base_rgb,
        target_rgb=target_rgb,
        base_coverage=coverage,
        target_color=normalize_color_key(target_color),
    )


def tint_fabric_preserving_texture(
    fabric_image: Image.Image,
    target_color: str,
) -> Image.Image:
    """Backward-compatible alias → base-only recolor (motifs preserved)."""
    return recolor_fabric_base_preserving_motifs(fabric_image, target_color).image


def save_fabric_recolor_audit(
    audit: FabricBaseRecolorResult,
    audit_dir: Any,
    prefix: str,
) -> Dict[str, str]:
    """Persist original / base_mask / recolored_base PNGs for visual QA."""
    from pathlib import Path

    out = Path(audit_dir)
    out.mkdir(parents=True, exist_ok=True)
    key = normalize_color_key(prefix) or "color"
    paths = {
        "original_fabric": str(out / "original_fabric.png"),
        "base_mask": str(out / "base_mask.png"),
        f"{key}_recolored_base": str(out / f"{key}_recolored_base.png"),
    }
    audit.original.save(paths["original_fabric"])
    audit.base_mask.save(paths["base_mask"])
    audit.recolored_base.save(paths[f"{key}_recolored_base"])
    return paths



def _cover_fabric(fabric: Image.Image, width: int, height: int) -> Image.Image:
    """
    Fill the garment canvas with fabric while preserving print/motif scale.

    Never upscale a small or narrow swatch to cover 768×768 — that stretches
    motifs. Native-scale tile those. Center-crop when the upload already covers
    the canvas. Only a modest LANCZOS cover is used when the fabric is already
    near canvas size (≤ ~1.55×) so a 512 swatch on 768 does not get a hard tile seam.
    """
    fabric_rgb = fabric.convert("RGB")
    fw, fh = fabric_rgb.size
    if fw < 8 or fh < 8:
        return fabric_rgb.resize((width, height), Image.Resampling.LANCZOS)

    # Both axes already cover the canvas: native-resolution center crop (no smear).
    if fw >= width and fh >= height:
        left = max(0, (fw - width) // 2)
        top = max(0, (fh - height) // 2)
        logger.info(
            "[FLUX] conditioning fabric fill=center_crop source=%sx%s canvas=%sx%s",
            fw,
            fh,
            width,
            height,
        )
        return fabric_rgb.crop((left, top, left + width, top + height))

    scale = max(width / float(fw), height / float(fh))
    nearly_full = (
        fw >= int(width * 0.6)
        and fh >= int(height * 0.6)
        and scale <= 1.55
    )
    if nearly_full:
        nw, nh = max(1, int(round(fw * scale))), max(1, int(round(fh * scale)))
        resized = fabric_rgb.resize((nw, nh), Image.Resampling.LANCZOS)
        left = max(0, (nw - width) // 2)
        top = max(0, (nh - height) // 2)
        logger.info(
            "[FLUX] conditioning fabric fill=modest_cover scale=%.3f "
            "source=%sx%s canvas=%sx%s (near-canvas; no tile seam)",
            scale,
            fw,
            fh,
            width,
            height,
        )
        return resized.crop((left, top, left + width, top + height))

    # Small or narrow swatch: tile at native pixel scale (do not stretch).
    tiled = Image.new("RGB", (width, height))
    for y in range(0, height, fh):
        for x in range(0, width, fw):
            tiled.paste(fabric_rgb, (x, y))
    logger.info(
        "[FLUX] conditioning fabric fill=native_tile source=%sx%s canvas=%sx%s "
        "(no cover-scale stretch)",
        fw,
        fh,
        width,
        height,
    )
    return tiled


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


def _normalize_neckline_key(neckline: Optional[str]) -> str:
    """Map UI / taxonomy neckline labels to snake_case tokens."""
    token = (neckline or "").strip().lower().replace("-", " ").replace(" ", "_")
    aliases = {
        "round": "round_neck",
        "crew": "round_neck",
        "crew_neck": "round_neck",
        "crewneck": "round_neck",
        "v": "v_neck",
        "vneck": "v_neck",
        "v_neck": "v_neck",
        "u": "u_neck",
        "u_neck": "u_neck",
    }
    return aliases.get(token, token)


def _circular_neck_arc(
    left: Tuple[int, int],
    right: Tuple[int, int],
    sagitta: int,
    samples: int = 11,
) -> list[Tuple[int, int]]:
    """Shallow circular U from left shoulder to right shoulder (y increases down)."""
    x0, y0 = int(left[0]), int(left[1])
    x1, y1 = int(right[0]), int(right[1])
    if x1 < x0:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
        left, right = (x0, y0), (x1, y1)
    span = max(2, x1 - x0)
    sag = max(3, int(sagitta))
    sag = min(sag, max(3, int(span * 0.22)))
    half = span / 2.0
    radius = (half * half + sag * sag) / (2.0 * sag)
    cx = (x0 + x1) / 2.0
    mid_y = (y0 + y1) / 2.0
    cy = mid_y - (radius - sag)
    n = max(7, int(samples))
    pts: list[Tuple[int, int]] = []
    for i in range(n):
        t = i / (n - 1)
        x = x0 + span * t
        dx = x - cx
        residual = radius * radius - dx * dx
        if residual <= 0:
            y = mid_y + sag * (1.0 - abs(2.0 * t - 1.0))
        else:
            y = cy + math.sqrt(residual)
        pts.append((int(round(x)), int(round(y))))
    pts[0] = (int(left[0]), int(left[1]))
    pts[-1] = (int(right[0]), int(right[1]))
    return pts


def _v_neck_points(
    left: Tuple[int, int],
    right: Tuple[int, int],
    depth: int,
) -> list[Tuple[int, int]]:
    """Pointed V: two straight diagonals meeting at center."""
    x0, y0 = int(left[0]), int(left[1])
    x1, y1 = int(right[0]), int(right[1])
    cx = int(round((x0 + x1) / 2.0))
    cy = int(round((y0 + y1) / 2.0)) + max(6, int(depth))
    return [(x0, y0), (cx, cy), (x1, y1)]


def _apply_neckline_geometry(
    polygon: Sequence[Tuple[int, int]],
    neckline: str,
    height: int,
) -> Sequence[Tuple[int, int]]:
    """Replace only the neck opening (first 5 points); keep body/sleeves."""
    if len(polygon) < 6:
        return polygon
    key = _normalize_neckline_key(neckline)
    left = (int(polygon[0][0]), int(polygon[0][1]))
    right = (int(polygon[4][0]), int(polygon[4][1]))
    body = list(polygon[5:])
    if key == "round_neck":
        # Crew: shallow circular U. height*0.06 read as a deep scoop / U-neck.
        span = max(2, int(right[0]) - int(left[0]))
        inset = max(0, int(round(span * 0.05)))
        left_i = (int(left[0]) + inset, int(left[1]))
        right_i = (int(right[0]) - inset, int(right[1]))
        sag = max(3, int(height * 0.040))
        neck = _circular_neck_arc(left_i, right_i, sagitta=sag, samples=13)
        return neck + body
    if key == "v_neck":
        neck = _v_neck_points(left, right, depth=max(6, int(height * 0.11)))
        return neck + body
    return polygon


def _is_long_sleeve(sleeve: str) -> bool:
    s = (sleeve or "").lower().replace(" ", "_").replace("-", "_")
    return any(
        k in s
        for k in ("long", "full", "three_quarter", "threequarter", "bishop", "puff")
    )


def _polygon_for_garment(
    garment_type: str,
    w: int,
    h: int,
    sleeve: str = "",
    neckline: str = "",
) -> Sequence[Tuple[int, int]]:
    key = (garment_type or "shirt").lower().replace(" ", "_")
    long_sleeve = _is_long_sleeve(sleeve)
    if key in {"dress", "gown", "frock"}:
        polygon = _dress_polygon(w, h)
    elif key in {"jacket", "blazer", "coat", "hoodie"}:
        polygon = _jacket_polygon(w, h)
    elif key in {"kurta", "kurti", "kameez", "tunic", "top"}:
        # Kurti/top: sleeve-aware shirt silhouette (kurta poly ignores sleeve)
        if key in {"kurta"}:
            polygon = _kurta_polygon(w, h)
        else:
            polygon = _shirt_polygon(w, h, long_sleeve=long_sleeve)
    else:
        polygon = _shirt_polygon(w, h, long_sleeve=long_sleeve)
    return _apply_neckline_geometry(polygon, neckline, h)


def build_garment_conditioning_image(
    fabric_image: Image.Image,
    garment_type: str = "shirt",
    width: int = 512,
    height: int = 512,
    sleeve: str = "",
    neckline: str = "",
    background: Tuple[int, int, int] = (255, 255, 255),
    target_color: Optional[str] = None,
) -> Image.Image:
    """
    Create a white-studio garment mockup filled with the uploaded fabric.

    Hard edges (no mask Gaussian blur): soft masks caused measurable edge halos
    in generated outputs. Kontext still invents folds from the silhouette.

    When ``target_color`` is an explicit UI color (not Match Fabric), only the
    dominant/base fabric region is recolored; print/motif colors are preserved.

    ``neckline`` selects only the neck opening geometry (round = circular U,
    V = pointed). Fabric pixels and the body/sleeve outline are unchanged.
    """
    if fabric_image is None:
        raise ValueError("fabric_image is required")

    w, h = int(width), int(height)
    source = fabric_image
    recolored = False
    last_audit: Optional[FabricBaseRecolorResult] = None
    if target_color and not is_match_fabric_color(target_color):
        last_audit = recolor_fabric_base_preserving_motifs(fabric_image, target_color)
        source = last_audit.image
        recolored = True
    # Expose latest audit for pipeline QA (None when Match Fabric).
    build_garment_conditioning_image.last_recolor_audit = last_audit  # type: ignore[attr-defined]
    neck_key = _normalize_neckline_key(neckline)
    build_garment_conditioning_image.last_neckline_key = neck_key  # type: ignore[attr-defined]

    fabric_fill = _cover_fabric(source, w, h)
    # Do not UnsharpMask the conditioning image. Sharpening before Kontext
    # invents halo texture that the model bakes into a blurry/melted print.
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    polygon = _polygon_for_garment(
        garment_type, w, h, sleeve=sleeve, neckline=neckline
    )
    draw.polygon(list(polygon), fill=255)
    # NO GaussianBlur — preserves crisp garment/background boundary for Kontext.

    canvas = Image.new("RGB", (w, h), background)
    canvas.paste(fabric_fill, (0, 0), mask=mask)

    logger.info(
        "Built garment conditioning image: garment=%s sleeve=%s neckline=%s "
        "size=%sx%s blur=none unsharp=none conditioning_recolored=%s "
        "target_color=%s mode=%s",
        garment_type,
        sleeve or "default",
        neck_key or "default",
        w,
        h,
        recolored,
        normalize_color_key(target_color) if recolored else "match_fabric",
        "base_only" if recolored else "match_fabric",
    )
    return canvas


# Optional audit from the last explicit-color conditioning build.
build_garment_conditioning_image.last_recolor_audit = None  # type: ignore[attr-defined]
build_garment_conditioning_image.last_neckline_key = ""  # type: ignore[attr-defined]
