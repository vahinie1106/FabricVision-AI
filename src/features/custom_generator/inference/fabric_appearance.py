"""Lightweight fabric appearance descriptors for fabric-conditioned garment prompts."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

from PIL import Image


def _quantize_rgb(pixel: Tuple[int, int, int], step: int = 24) -> Tuple[int, int, int]:
    return tuple(min(255, (c // step) * step + step // 2) for c in pixel)  # type: ignore[return-value]


def _rgb_to_color_name(rgb: Tuple[int, int, int]) -> str:
    """Map a quantized RGB to a fashion color name (print accents + base)."""
    r, g, b = rgb
    brightness = (r + g + b) / 3.0
    mx, mn = max(r, g, b), min(r, g, b)
    sat = (mx - mn) / max(mx, 1)

    # Near-white / cream base (common floral grounds) — check BEFORE warm averaging traps
    if brightness > 200 and sat < 0.18:
        return "white"
    if brightness > 185 and sat < 0.12:
        return "white"
    if brightness < 35:
        return "black"

    # Chromatic accents
    if r > 140 and r >= g + 40 and r >= b + 40:
        if g > 110 and b < 90:
            return "orange"
        return "red"
    if r > 170 and g > 140 and b < 100 and r - b > 50:
        return "yellow"
    if g > r + 15 and g > b + 15 and g > 90:
        return "green"
    if b > r + 20 and b > g + 10:
        return "navy blue" if r < 90 else "blue"
    if r > 110 and b > 110 and g < min(r, b) - 15:
        return "purple"
    if r > 120 and g > 80 and b > 60 and sat < 0.35 and brightness < 160:
        return "brown"
    if abs(r - g) < 18 and abs(g - b) < 18:
        return "gray" if brightness < 180 else "white"
    if sat < 0.15:
        return "white" if brightness > 160 else "gray"
    return "multicolor"


def describe_fabric_appearance(fabric_image: Image.Image, max_colors: int = 4) -> Dict[str, Any]:
    """
    Derive color/pattern cues from fabric pixels for prompt anchoring.

    WHY not 64×64 average-only: fine florals on white collapse to muddy brown/orange
    averages, which then poison the Kontext prompt (observed: white+red+green → "brown"
    or UI "yellow" overriding the print). We count named colors by pixel frequency and
    keep accent colors (red/green) plus base (white) instead of a single mean.
    """
    img = fabric_image.convert("RGB")
    # 128² keeps small floral accents alive vs 64² BOX smear
    sample = img.resize((128, 128), Image.Resampling.BOX)
    pixels = list(sample.getdata())
    name_counts: Counter[str] = Counter()
    for p in pixels:
        name_counts[_rgb_to_color_name(_quantize_rgb(p))] += 1

    total = max(sum(name_counts.values()), 1)
    # Keep colors that cover ≥3% of pixels (accents) or top base
    significant = [
        (name, cnt)
        for name, cnt in name_counts.most_common()
        if cnt / total >= 0.03 and name != "multicolor"
    ]
    color_names: List[str] = []
    for name, _ in significant:
        if name not in color_names:
            color_names.append(name)
        if len(color_names) >= max_colors:
            break

    # Ensure white base is listed when present (even if slightly under threshold after folds)
    white_frac = name_counts.get("white", 0) / total
    if white_frac >= 0.15 and "white" not in color_names:
        color_names.insert(0, "white")

    if not color_names:
        color_names = ["multicolor"]

    unique_q = len({_quantize_rgb(p) for p in pixels})
    arr = pixels
    w, h = sample.size
    contrast_acc = 0
    contrast_n = 0
    for y in range(h):
        for x in range(0, w - 2, 2):
            i = y * w + x
            j = i + 2
            contrast_acc += sum(abs(arr[i][c] - arr[j][c]) for c in range(3))
            contrast_n += 1
    mean_contrast = (contrast_acc / max(contrast_n, 1)) / 3.0

    if mean_contrast > 12 and len(color_names) >= 2:
        pattern = "floral" if any(c in color_names for c in ("red", "pink", "green")) else "printed"
        if mean_contrast > 45 and "green" not in color_names:
            pattern = "polka-dot"
    elif unique_q >= 40:
        pattern = "printed"
    elif unique_q >= 18:
        pattern = "patterned"
    else:
        pattern = "solid"

    # Prefer white/light base first in the list for prompt clarity
    preferred_order = ["white", "cream", "red", "green", "blue", "navy blue", "yellow", "orange"]
    ordered: List[str] = []
    for pref in preferred_order:
        if pref in color_names and pref not in ordered:
            ordered.append(pref)
    for name in color_names:
        if name not in ordered:
            ordered.append(name)
    color_names = ordered[:max_colors]

    palette = ", ".join(color_names)
    summary = f"{palette} {pattern} textile"
    return {
        "dominant_color_names": color_names,
        "pattern_hint": pattern,
        "palette_summary": palette,
        "appearance_summary": summary,
        "mean_contrast": round(mean_contrast, 2),
        "color_fractions": {
            k: round(v / total, 3) for k, v in name_counts.most_common(8)
        },
    }
