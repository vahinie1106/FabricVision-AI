from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


@dataclass
class ValidationResult:
    """Represents the outcome of image validation."""

    is_valid: bool
    reasons: list[str]
    width: int = 0
    height: int = 0
    channels: int = 0
    is_blurry: bool = False
    duplicate_hash: Optional[str] = None


class ImageValidator:
    """Validates image quality and usability for preprocessing."""

    def __init__(self, min_width: int = 16, min_height: int = 16, min_resolution: int = 256, blur_threshold: float = 100.0, min_intensity_std: float = 5.0) -> None:
        self.min_width = min_width
        self.min_height = min_height
        self.min_resolution = min_resolution
        self.blur_threshold = blur_threshold
        self.min_intensity_std = min_intensity_std

    def validate_image(self, image_path: str | Path, image_array: Optional[np.ndarray] = None) -> ValidationResult:
        path = Path(image_path)
        reasons: list[str] = []

        if not path.exists():
            return ValidationResult(False, ["File does not exist"])

        try:
            with Image.open(path) as img:
                if img.mode not in {"RGB", "RGBA", "L", "CMYK"}:
                    reasons.append("Unsupported image mode")
                array = np.array(img.convert("RGB"))
        except Exception as exc:  # noqa: BLE001
            return ValidationResult(False, [f"Unreadable image: {exc}"])

        height, width = array.shape[:2]
        if width < self.min_width or height < self.min_height:
            reasons.append("Low resolution")
        if width * height < self.min_resolution:
            reasons.append("Insufficient pixel area")
        if array.size == 0:
            reasons.append("Empty image")
        if array.ndim != 3 or array.shape[2] != 3:
            reasons.append("Invalid channel count")

        # Blur detection uses variance of the Laplacian; a lower score indicates a blurrier image.
        # Solid-color images are intentionally not flagged as blurry because they may be valid inputs
        # for simple garment or background examples even when their edge structure is minimal.
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        intensity_std = float(gray.std())
        is_blurry = bool(variance < self.blur_threshold and intensity_std > self.min_intensity_std)
        if is_blurry:
            reasons.append("Blurry image")

        return ValidationResult(
            is_valid=not reasons,
            reasons=reasons,
            width=width,
            height=height,
            channels=array.shape[2] if array.ndim == 3 else 0,
            is_blurry=is_blurry,
        )

    def compute_duplicate_hash(self, image_array: np.ndarray) -> str:
        """Generate a stable hash for duplicate detection."""
        return str(hash(image_array.tobytes()))
