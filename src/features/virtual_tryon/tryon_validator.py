from __future__ import annotations

import logging
from typing import Any, Dict
from PIL import Image, ImageStat


class TryOnValidator:
    """Validate virtual try-on output quality, resolution, color space, and pixel integrity."""

    def __init__(self, min_resolution: int = 384) -> None:
        """min_resolution applies to the shorter image side (supports 384x512 portrait)."""
        self.min_resolution = min_resolution
        self.logger = logging.getLogger("fabricvision.virtual_tryon.validator")

    def validate(self, result_image: Image.Image) -> Dict[str, Any]:
        """Validate output image quality parameters."""
        issues: list[str] = []

        if not isinstance(result_image, Image.Image):
            return {
                "valid": False,
                "is_valid": False,
                "validation_status": "failed",
                "confidence": 0.0,
                "issues": ["Invalid try-on image object type"],
            }

        width, height = result_image.size
        shorter = min(width, height)

        if shorter < self.min_resolution:
            issues.append(
                f"Try-on image resolution ({width}x{height}) shorter side below "
                f"minimum threshold ({self.min_resolution}px)"
            )

        if result_image.mode != "RGB":
            issues.append(f"Invalid try-on image mode '{result_image.mode}', expected 'RGB'")

        # Check pixel variance
        try:
            stat = ImageStat.Stat(result_image)
            variances = stat.var
            avg_variance = sum(variances) / len(variances) if variances else 0.0
            if avg_variance < 1.0:
                issues.append("Zero/flat pixel variance detected in try-on result")
        except Exception as exc:
            self.logger.warning("Could not calculate try-on pixel variance: %s", exc)

        is_valid = len(issues) == 0
        return {
            "valid": is_valid,
            "is_valid": is_valid,
            "validation_status": "passed" if is_valid else "failed",
            "resolution": (width, height),
            "mode": result_image.mode,
            "confidence": 0.95 if is_valid else 0.0,
            "issues": issues,
        }
