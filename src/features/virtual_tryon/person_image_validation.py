"""Lightweight person-image suitability checks for CatVTON quality validation."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


def assess_person_image(person: Image.Image) -> Tuple[bool, str]:
    """
    Return (ok, reason) for whether an image is plausible as a try-on person photo.

    This is intentionally heuristic — not a person detector. It rejects obvious
    placeholders and dense textile swatches that cannot support quality try-on.

    Synthetic unit-test images with a few large contrasting regions are allowed.
    """
    if not isinstance(person, Image.Image):
        return False, "person image must be a PIL Image"

    rgb = person.convert("RGB")
    w, h = rgb.size
    if min(w, h) < 128:
        return False, f"person image too small ({w}x{h}); need shorter side >= 128"

    aspect = w / max(1, h)
    if aspect < 0.35 or aspect > 2.8:
        return False, f"person aspect ratio {aspect:.2f} unsupported for try-on"

    sample = rgb.resize((64, 64), Image.Resampling.BOX)
    arr = np.asarray(sample, dtype=np.float32)
    std = float(arr.std())
    if std < 12.0:
        return False, (
            "person image looks nearly flat/uniform (std<{:.1f}); "
            "use a real full-/half-body photo, not a solid placeholder"
        ).format(std)

    gray = np.asarray(sample.convert("L"), dtype=np.float32)
    global_edge = float(
        np.abs(np.diff(gray, axis=1)).mean() + np.abs(np.diff(gray, axis=0)).mean()
    )
    smooth_frac = float((np.abs(gray - gray.mean()) < 18.0).mean())

    # Geometric stick-figure / icon: very low edge energy + large flat fills.
    if global_edge < 4.0 and smooth_frac > 0.65:
        return False, (
            "person image looks like a geometric placeholder; "
            "supply a real person photograph"
        )

    # Local edge coefficient of variation: real portraits vary (face/bg vs limbs);
    # fabric swatches have dense, relatively uniform texture energy.
    local_edges: list[float] = []
    for i in range(0, 64, 8):
        for j in range(0, 64, 8):
            patch = gray[i : i + 8, j : j + 8]
            local_edges.append(
                float(
                    np.abs(np.diff(patch, axis=1)).mean()
                    + np.abs(np.diff(patch, axis=0)).mean()
                )
            )
    local_mean = float(np.mean(local_edges)) if local_edges else 0.0
    local_std = float(np.std(local_edges)) if local_edges else 0.0
    edge_cv = local_std / (local_mean + 1e-6)

    if local_mean > 10.0 and edge_cv < 0.55:
        return False, (
            "person image resembles an all-over fabric/textile swatch "
            "(uniform dense texture); provide a real person photo for "
            "CatVTON quality validation"
        )

    return True, "ok"
