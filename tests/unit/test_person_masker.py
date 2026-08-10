"""Unit tests for cloth-region person masks (no model inference)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.features.virtual_tryon.person_masker import (
    build_box_mask,
    build_grabcut_mask,
    cloth_region_band,
    resolve_person_mask,
    restrict_mask_to_cloth_region,
    solidify_overall_dress_mask,
)


def _person_stub(w=256, h=512):
    """Simple full-body colored person on light background."""
    img = Image.new("RGB", (w, h), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    # torso + legs silhouette
    draw.ellipse([w * 0.35, h * 0.05, w * 0.65, h * 0.18], fill=(210, 170, 140))
    draw.rectangle([w * 0.30, h * 0.18, w * 0.70, h * 0.55], fill=(40, 140, 60))
    draw.rectangle([w * 0.35, h * 0.55, w * 0.48, h * 0.95], fill=(40, 140, 60))
    draw.rectangle([w * 0.52, h * 0.55, w * 0.65, h * 0.95], fill=(40, 140, 60))
    return img


def test_cloth_region_band_upper_excludes_legs():
    y0, y1 = cloth_region_band("upper", 512)
    assert y0 < y1
    assert y1 <= int(512 * 0.65)


def test_restrict_mask_clears_lower_body_for_upper():
    binary = np.full((512, 256), 255, dtype=np.uint8)
    out = restrict_mask_to_cloth_region(binary, "upper")
    assert out[int(512 * 0.8), 128] == 0
    assert out[int(512 * 0.3), 128] == 255


def test_box_mask_upper_not_full_height():
    mask = build_box_mask((256, 512), cloth_type="upper", blur_radius=0)
    arr = np.asarray(mask)
    # Legs region should be mostly empty for upper box
    assert float((arr[int(512 * 0.8) :, :] > 127).mean()) < 0.05
    assert float((arr[int(512 * 0.2) : int(512 * 0.5), :] > 127).mean()) > 0.2


def test_grabcut_upper_mask_avoids_legs():
    person = _person_stub()
    mask = build_grabcut_mask(person, blur_radius=1, iters=2, cloth_type="upper")
    if mask is None:
        pytest.skip("GrabCut unavailable or rejected on synthetic stub")
    arr = np.asarray(mask.convert("L"))
    leg_fill = float((arr[int(512 * 0.75) :, :] > 127).mean())
    torso_fill = float((arr[int(512 * 0.15) : int(512 * 0.55), :] > 127).mean())
    assert leg_fill < 0.15
    assert torso_fill > 0.02


def test_solidify_overall_fills_pant_split():
    """Pants-split lower mask must become one continuous dress region."""
    h, w = 512, 256
    binary = np.zeros((h, w), dtype=np.uint8)
    # torso
    binary[int(h * 0.12) : int(h * 0.55), int(w * 0.30) : int(w * 0.70)] = 255
    # split legs
    binary[int(h * 0.55) : int(h * 0.92), int(w * 0.32) : int(w * 0.45)] = 255
    binary[int(h * 0.55) : int(h * 0.92), int(w * 0.55) : int(w * 0.68)] = 255
    out = solidify_overall_dress_mask(binary)
    y = int(h * 0.85)
    row = out[y] > 0
    d = np.diff(np.concatenate([[0], row.astype(np.uint8), [0]]))
    assert int((d == 1).sum()) == 1
    assert float((out[int(h * 0.75) : int(h * 0.88), :] > 0).mean()) > 0.15


def test_grabcut_overall_mask_is_continuous_skirt():
    person = _person_stub()
    mask = build_grabcut_mask(person, blur_radius=1, iters=2, cloth_type="overall")
    if mask is None:
        pytest.skip("GrabCut unavailable or rejected on synthetic stub")
    arr = np.asarray(mask.convert("L")) > 127
    h = arr.shape[0]
    y = int(h * 0.82)
    row = arr[y].astype(np.uint8)
    d = np.diff(np.concatenate([[0], row, [0]]))
    assert int((d == 1).sum()) <= 1
    assert float(arr[int(h * 0.70) : int(h * 0.88), :].mean()) > 0.05
    # bottom should reach dress region
    ys = np.where(arr)[0]
    assert ys.max() / h >= 0.72


def test_resolve_reports_mask_source(tmp_path):
    person = _person_stub()
    mask, source = resolve_person_mask(
        person,
        provided_mask=None,
        target_size=(256, 512),
        catvton_path=tmp_path,  # no AutoMasker
        cloth_type="upper",
    )
    assert source in ("grabcut", "box_fallback")
    assert mask.size == (256, 512)
