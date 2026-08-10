"""Person-mask builders for CatVTON under RTX 3050 6GB constraints.

Priority order (when requested):
1. AutoMasker (DensePose+SCHP) — feature-flagged, unload after use
2. OpenCV GrabCut cloth-region mask — no extra deps, better than a plain box
3. Soft rectangular box — last-resort fallback (explicitly labeled)

Never silently claim a box mask is production AutoMasker output.

CatVTON's agnostic mask must mark the *clothing region to replace*, not the
full body silhouette. A full-person GrabCut/box causes garments to bleed onto
legs/background — the failure mode seen in bad production try-ons.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps

logger = logging.getLogger("fabricvision.virtual_tryon.person_masker")


def _mask_blur(mask: Image.Image, radius: int) -> Image.Image:
    if radius <= 0:
        return mask
    return mask.filter(ImageFilter.GaussianBlur(radius=radius))


def _validate_mask(mask: Image.Image, min_fill: float = 0.05, max_fill: float = 0.95) -> bool:
    arr = np.asarray(mask.convert("L"), dtype=np.float32)
    if arr.size == 0:
        return False
    fill = float((arr > 127).mean())
    return min_fill <= fill <= max_fill


def cloth_region_band(cloth_type: str, height: int) -> Tuple[int, int]:
    """
    Vertical band (y0, y1) for the clothing region to replace.

    Upper: torso / sleeves — clear legs so CatVTON cannot paint cloth there.
    Lower: hips / legs.
    Overall / other: most of the body.
    """
    h = max(1, height)
    ct = (cloth_type or "upper").lower()
    if ct == "lower":
        return int(h * 0.42), int(h * 0.96)
    if ct in ("overall", "dress", "outer"):
        return int(h * 0.10), int(h * 0.94)
    # upper / inner / default
    return int(h * 0.10), int(h * 0.62)


def restrict_mask_to_cloth_region(
    binary: np.ndarray,
    cloth_type: str,
) -> np.ndarray:
    """Zero out mask pixels outside the cloth-type vertical band."""
    h, w = binary.shape[:2]
    y0, y1 = cloth_region_band(cloth_type, h)
    out = np.zeros_like(binary)
    out[y0:y1, :] = binary[y0:y1, :]
    return out


def solidify_overall_dress_mask(binary: np.ndarray) -> np.ndarray:
    """
    Convert a pants-split GrabCut silhouette into a continuous overall dress region.

    GrabCut follows the *worn* outfit. On pants/jumpsuits the lower mask becomes
    two leg columns; CatVTON then has no contiguous skirt panel to inpaint, so
    full dresses truncate like upper-body transfers. For cloth_type=overall we
    fill between left/right extremities from hips through the dress band and
    close small holes — still clipped to the cloth band (no background bleed).
    """
    try:
        import cv2
    except ImportError:
        return binary

    h, w = binary.shape[:2]
    out = (binary > 0).astype(np.uint8) * 255
    if int((out > 0).sum()) < 32:
        return binary

    # Fill inter-leg gaps row-wise from mid-torso / hips through the dress band.
    y_fill0 = int(h * 0.40)
    y0, y1 = cloth_region_band("overall", h)
    y_fill0 = max(y_fill0, y0)
    y_fill1 = y1
    for y in range(y_fill0, y_fill1):
        xs = np.where(out[y] > 0)[0]
        if xs.size >= 2:
            out[y, int(xs.min()) : int(xs.max()) + 1] = 255

    # Strong vertical+horizontal close so skirt reads as one clothing blob.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 25))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)

    # If hem stops early (mini-mask), extend using the width of the lowest dense row.
    ys, xs = np.where(out > 0)
    if ys.size:
        bottom = int(ys.max())
        target_bottom = int(h * 0.90)
        if bottom < int(h * 0.78):
            # Width from a stable hip/thigh row
            ref_y = min(bottom, int(h * 0.70))
            ref_xs = np.where(out[ref_y] > 0)[0]
            if ref_xs.size >= 2:
                x_l, x_r = int(ref_xs.min()), int(ref_xs.max())
                # Mild taper toward hem
                for y in range(bottom + 1, min(target_bottom, y1)):
                    t = (y - bottom) / max(1, target_bottom - bottom)
                    shrink = int(0.08 * (x_r - x_l) * t)
                    out[y, x_l + shrink : max(x_l + shrink + 1, x_r - shrink + 1)] = 255

    # Re-apply band + clear head/feet margins explicitly.
    out = restrict_mask_to_cloth_region(out, "overall")
    out[: int(h * 0.10), :] = 0
    out[int(h * 0.94) :, :] = 0

    # Keep largest component after solidify.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    if num > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        out = np.where(labels == largest, 255, 0).astype(np.uint8)
    return out


def build_box_mask(
    size: Tuple[int, int],
    margin_x_frac: float = 0.2,
    margin_y_frac: float = 0.25,
    blur_radius: int = 9,
    cloth_type: str = "upper",
) -> Image.Image:
    """Soft rectangular clothing-region prior — last-resort agnostic mask."""
    w, h = size
    mask = Image.new("L", size, color=0)
    mx = int(w * margin_x_frac)
    y0, y1 = cloth_region_band(cloth_type, h)
    # Keep horizontal margins; vertical band follows cloth_type (not full body).
    mask.paste(255, (mx, y0, w - mx, y1))
    return _mask_blur(mask, blur_radius)


def build_grabcut_mask(
    person_rgb: Image.Image,
    blur_radius: int = 5,
    iters: int = 5,
    cloth_type: str = "upper",
) -> Optional[Image.Image]:
    """
    OpenCV GrabCut restricted to the clothing region for ``cloth_type``.

    Safer default than a hard box: follows person silhouette inside the torso
    (or lower/overall) band without Detectron2/DensePose.
    Returns None if OpenCV fails or mask invalid.
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV unavailable; cannot build GrabCut mask")
        return None

    rgb = np.asarray(person_rgb.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    if h < 32 or w < 32:
        return None

    # Uniform / near-flat images make OpenCV GrabCut extremely slow or stall.
    if float(rgb.std()) < 8.0:
        logger.warning("GrabCut skipped: person image variance too low (std=%.2f)", float(rgb.std()))
        return None

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    y0, y1 = cloth_region_band(cloth_type, h)
    # Horizontal prior: centered torso; vertical prior matches cloth band.
    x0, x1 = int(w * 0.16), int(w * 0.84)
    rect = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))

    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        # Cap iters — GrabCut can thrash on hard images under GPU/CPU contention.
        cv2.grabCut(bgr, mask, rect, bgd, fgd, min(max(1, iters), 5), cv2.GC_INIT_WITH_RECT)
    except Exception as exc:
        logger.warning("GrabCut failed: %s", exc)
        return None

    binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # Some full-body photos yield empty/near-empty FG after RECT init.
    # Retry once with an explicit cloth-band mask prior.
    min_fg_pixels = max(64, int(0.01 * h * w))
    if int((binary > 0).sum()) < min_fg_pixels:
        logger.info("GrabCut RECT produced empty/weak FG; retrying with mask prior")
        mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
        # Probable FG across cloth band; definite FG in torso core.
        mask[y0:y1, int(w * 0.22) : int(w * 0.78)] = cv2.GC_PR_FGD
        core_y0 = y0 + max(1, (y1 - y0) // 5)
        core_y1 = y1 - max(1, (y1 - y0) // 5)
        mask[core_y0:core_y1, int(w * 0.35) : int(w * 0.65)] = cv2.GC_FGD
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(bgr, mask, None, bgd, fgd, min(max(1, iters), 5), cv2.GC_INIT_WITH_MASK)
        except Exception as exc:
            logger.warning("GrabCut mask-prior retry failed: %s", exc)
            return None
        binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        if int((binary > 0).sum()) < min_fg_pixels:
            logger.warning("GrabCut produced empty foreground after retry")
            return None
    # Keep largest connected component — drop background blobs.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    binary = np.where(labels == largest, 255, 0).astype(np.uint8)

    # Critical: never leave a full-body silhouette for upper/lower try-on.
    binary = restrict_mask_to_cloth_region(binary, cloth_type)

    # Overall/dress: solidify split-leg GrabCut into a continuous skirt region.
    if cloth_type in ("overall", "dress", "outer"):
        binary = solidify_overall_dress_mask(binary)
        logger.info(
            "Overall dress mask solidified (bbox_bottom_frac≈%.3f)",
            (
                float(np.where(binary > 0)[0].max()) / max(1, binary.shape[0])
                if (binary > 0).any()
                else 0.0
            ),
        )

    # Light morphological close to fill small holes inside the clothing region.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    out = Image.fromarray(binary, mode="L")
    # Official CatVTON uses relatively soft mask edges (~9px).
    out = _mask_blur(out, max(blur_radius, 5))
    # Cloth-region fill is smaller than full-body — use a lower min_fill.
    # overall/dress masks legitimately cover more of the frame.
    min_fill = 0.03 if cloth_type in ("upper", "lower", "inner") else 0.05
    max_fill = 0.92 if cloth_type in ("overall", "dress", "outer") else 0.85
    if not _validate_mask(out, min_fill=min_fill, max_fill=max_fill):
        logger.warning(
            "GrabCut cloth-region mask fill ratio out of bounds (type=%s); rejecting",
            cloth_type,
        )
        return None
    return out


def try_automasker_mask(
    person_rgb: Image.Image,
    catvton_root: Path,
    cloth_type: str = "upper",
    device: str = "cuda",
) -> Optional[Image.Image]:
    """
    Attempt native CatVTON AutoMasker (DensePose + SCHP).

    Returns None when detectron2/weights/imports are unavailable so callers
    can fall back without crashing. Caller must unload VRAM after use.
    """
    flag = os.environ.get("CATVTON_USE_AUTOMASKER", "false").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return None

    model_dir = catvton_root / "model"
    dense_dir = catvton_root / "DensePose"
    if not dense_dir.exists():
        dense_dir = catvton_root / "densepose"
    schp_dir = catvton_root / "SCHP"
    if not (model_dir / "cloth_masker.py").exists():
        logger.warning("AutoMasker requested but cloth_masker.py missing")
        return None
    if not schp_dir.exists():
        logger.warning("AutoMasker requested but SCHP weights missing")
        return None

    try:
        import detectron2  # noqa: F401
    except ImportError:
        vendor = catvton_root / "detectron2"
        if vendor.exists() and str(catvton_root) not in sys.path:
            sys.path.insert(0, str(catvton_root))
        try:
            import detectron2  # noqa: F401
        except ImportError:
            logger.warning(
                "CATVTON_USE_AUTOMASKER=true but detectron2 is not importable; "
                "falling back (mask_source will not be automasker)."
            )
            return None

    try:
        if str(catvton_root) not in sys.path:
            sys.path.insert(0, str(catvton_root))
        from model.cloth_masker import AutoMasker  # type: ignore

        dense_ckpt = str(dense_dir)
        schp_ckpt = str(schp_dir)
        am = AutoMasker(densepose_ckpt=dense_ckpt, schp_ckpt=schp_ckpt, device=device)
        pre = am.preprocess_image(person_rgb)
        mask = AutoMasker.cloth_agnostic_mask(
            densepose_mask=pre["densepose"],
            schp_lip_mask=pre["schp_lip"],
            schp_atr_mask=pre["schp_atr"],
            part=cloth_type if cloth_type in ("upper", "lower", "overall", "inner", "outer") else "upper",
        )
        if isinstance(mask, Image.Image):
            mask = mask.convert("L")
        else:
            mask = Image.fromarray(np.asarray(mask)).convert("L")
        if not _validate_mask(mask, min_fill=0.02, max_fill=0.98):
            logger.warning("AutoMasker mask failed validation")
            return None
        return mask
    except Exception as exc:
        logger.warning("AutoMasker failed (%s); will fall back", exc)
        return None
    finally:
        try:
            import torch
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def resolve_person_mask(
    person_rgb: Image.Image,
    provided_mask: Optional[Image.Image],
    target_size: Tuple[int, int],
    catvton_path: str | Path = "models/CatVTON",
    cloth_type: str = "upper",
) -> Tuple[Image.Image, str]:
    """
    Resolve an agnostic person mask and return (mask, mask_source).

    mask_source values:
      provided | automasker | grabcut | box_fallback

    Strategy via CATVTON_MASK_STRATEGY:
      auto (default) — provided → automasker (if enabled) → grabcut → box
      automasker     — AutoMasker only (raises if unavailable)
      grabcut        — GrabCut then box_fallback
      provided_only  — require provided_mask

    Attempts are recorded on ``resolve_person_mask.last_attempts`` for metadata.
    """
    blur = int(os.environ.get("CATVTON_MASK_BLUR", "7"))
    catvton_root = Path(catvton_path)
    strategy = os.environ.get("CATVTON_MASK_STRATEGY", "auto").strip().lower() or "auto"
    attempts: list[str] = []

    if provided_mask is not None:
        mask = ImageOps.fit(provided_mask.convert("L"), target_size, method=Image.Resampling.NEAREST)
        attempts.append("provided:ok")
        resolve_person_mask.last_attempts = attempts  # type: ignore[attr-defined]
        resolve_person_mask.last_strategy = strategy  # type: ignore[attr-defined]
        return mask, "provided"

    if strategy == "provided_only":
        raise RuntimeError(
            "CATVTON_MASK_STRATEGY=provided_only but no person mask was supplied."
        )

    allow_automasker = strategy in ("auto", "automasker")
    allow_grabcut = strategy in ("auto", "grabcut")
    allow_box = strategy in ("auto", "grabcut")

    if allow_automasker:
        # Explicit strategy=automasker forces the attempt even if the env flag is off.
        prev_flag = os.environ.get("CATVTON_USE_AUTOMASKER")
        if strategy == "automasker":
            os.environ["CATVTON_USE_AUTOMASKER"] = "true"

        try:
            am = try_automasker_mask(
                person_rgb.resize(target_size, Image.Resampling.LANCZOS)
                if person_rgb.size != target_size
                else person_rgb,
                catvton_root=catvton_root,
                cloth_type=cloth_type,
            )
        finally:
            if strategy == "automasker":
                if prev_flag is None:
                    os.environ.pop("CATVTON_USE_AUTOMASKER", None)
                else:
                    os.environ["CATVTON_USE_AUTOMASKER"] = prev_flag

        if am is not None:
            if am.size != target_size:
                am = ImageOps.fit(am, target_size, method=Image.Resampling.NEAREST)
            attempts.append("automasker:ok")
            resolve_person_mask.last_attempts = attempts  # type: ignore[attr-defined]
            resolve_person_mask.last_strategy = strategy  # type: ignore[attr-defined]
            return am, "automasker"
        attempts.append("automasker:unavailable_or_failed")
        if strategy == "automasker":
            resolve_person_mask.last_attempts = attempts  # type: ignore[attr-defined]
            resolve_person_mask.last_strategy = strategy  # type: ignore[attr-defined]
            raise RuntimeError(
                "CATVTON_MASK_STRATEGY=automasker but AutoMasker/DensePose/detectron2 "
                "is unavailable or failed. Install detectron2+fvcore and ensure "
                "models/CatVTON/{model,SCHP,DensePose|densepose} exist, or use "
                "CATVTON_MASK_STRATEGY=auto|grabcut."
            )

    prefer_grabcut = os.environ.get("CATVTON_USE_GRABCUT", "true").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    if allow_grabcut and prefer_grabcut:
        person_fit = (
            ImageOps.fit(person_rgb.convert("RGB"), target_size, method=Image.Resampling.LANCZOS)
            if person_rgb.size != target_size
            else person_rgb.convert("RGB")
        )
        gc_mask = build_grabcut_mask(
            person_fit,
            blur_radius=min(max(blur, 5), 9),
            cloth_type=cloth_type,
        )
        if gc_mask is not None:
            attempts.append("grabcut:ok")
            resolve_person_mask.last_attempts = attempts  # type: ignore[attr-defined]
            resolve_person_mask.last_strategy = strategy  # type: ignore[attr-defined]
            return gc_mask, "grabcut"
        attempts.append("grabcut:failed")

    if not allow_box:
        resolve_person_mask.last_attempts = attempts  # type: ignore[attr-defined]
        resolve_person_mask.last_strategy = strategy  # type: ignore[attr-defined]
        raise RuntimeError(
            f"No usable CatVTON mask under CATVTON_MASK_STRATEGY={strategy} "
            f"(attempts={attempts})."
        )

    attempts.append("box_fallback:ok")
    resolve_person_mask.last_attempts = attempts  # type: ignore[attr-defined]
    resolve_person_mask.last_strategy = strategy  # type: ignore[attr-defined]
    return (
        build_box_mask(target_size, blur_radius=max(blur, 5), cloth_type=cloth_type),
        "box_fallback",
    )


# Defaults for metadata readers when resolve_person_mask has not run yet.
resolve_person_mask.last_attempts = []  # type: ignore[attr-defined]
resolve_person_mask.last_strategy = "auto"  # type: ignore[attr-defined]
