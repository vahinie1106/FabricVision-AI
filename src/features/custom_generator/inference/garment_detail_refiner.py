"""Fashion-specific contour & seam detail refiner for garment outputs.

Applies high-frequency edge enhancement strictly along garment contours
(neckline, sleeves, seams, hem, silhouette) while preserving 100% of internal
fabric print identity without introducing artificial weave/grid textures.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

logger = logging.getLogger("fabricvision.garment_generation.detail_refiner")


class GarmentDetailRefiner:
    """Contour-guided detail refiner for diffusion-generated garment images."""

    def __init__(
        self,
        edge_threshold: float = 25.0,
        edge_blur_radius: float = 1.0,
        sharpen_factor: float = 1.5,
        contrast_factor: float = 1.1,
    ) -> None:
        self.edge_threshold = edge_threshold
        self.edge_blur_radius = edge_blur_radius
        self.sharpen_factor = sharpen_factor
        self.contrast_factor = contrast_factor

    def refine(
        self,
        image: Image.Image,
        mask_fabric_interior: bool = True,
        enabled: bool = True,
    ) -> Image.Image:
        """
        Enhance garment structural contours (edges/seams/neckline/sleeves).

        Args:
            image: Generated raw PIL Image.
            mask_fabric_interior: If True, restricts sharpening strictly to contour edges.
            enabled: Master switch. If False, returns original image intact.

        Returns:
            Refined PIL Image with sharp garment boundaries and intact fabric pattern.
        """
        if not enabled or image is None:
            return image

        try:
            rgb_img = image.convert("RGB")
            w, h = rgb_img.size

            # 1. Create enhanced version of the full image (unsharp mask + slight contrast boost)
            unsharp = rgb_img.filter(
                ImageFilter.UnsharpMask(radius=2.0, percent=120, threshold=3)
            )
            enhancer = ImageEnhance.Contrast(unsharp)
            enhanced = enhancer.enhance(self.contrast_factor)

            if not mask_fabric_interior:
                logger.info("Garment detail refiner applied globally (mask_fabric_interior=False)")
                return enhanced

            # 2. Extract edge mask for structural contours (Sobel/Gradient energy)
            gray = rgb_img.convert("L")
            gray_arr = np.asarray(gray, dtype=np.float32)

            # Horizontal and vertical gradient
            gx = np.abs(gray_arr[:, 1:] - gray_arr[:, :-1])
            gy = np.abs(gray_arr[1:, :] - gray_arr[:-1, :])

            # Pad gradients to match original dimensions
            gx = np.pad(gx, ((0, 0), (0, 1)), mode="edge")
            gy = np.pad(gy, ((0, 1), (0, 0)), mode="edge")

            grad_mag = np.sqrt(gx**2 + gy**2)

            # Normalize gradient to 0..255
            max_g = grad_mag.max()
            if max_g > 0:
                grad_norm = (grad_mag / max_g * 255.0).astype(np.uint8)
            else:
                grad_norm = np.zeros((h, w), dtype=np.uint8)

            # Soft threshold edge mask
            mask_arr = np.where(grad_norm > self.edge_threshold, 255, 0).astype(np.uint8)

            # Smooth mask edges slightly to prevent harsh step artifacts
            edge_mask = Image.fromarray(mask_arr, mode="L")
            if self.edge_blur_radius > 0:
                edge_mask = edge_mask.filter(
                    ImageFilter.GaussianBlur(radius=self.edge_blur_radius)
                )

            # 3. Composite: Original image inside fabric areas, Enhanced image on contour edges
            result = Image.composite(enhanced, rgb_img, edge_mask)
            logger.info("Contour-guided garment detail refiner successfully applied.")
            return result

        except Exception as exc:
            logger.warning("Detail refiner failed (%s); returning original image", exc)
            return image
