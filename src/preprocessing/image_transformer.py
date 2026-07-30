from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class ImageTransformer:
    """Resizes, pads, normalizes, and optionally smooths images."""

    def __init__(self, target_size: Tuple[int, int] = (512, 512), normalize: bool = True, noise_reduction: str | None = None) -> None:
        self.target_size = target_size
        self.should_normalize = normalize
        self.noise_reduction = noise_reduction

    def resize_with_padding(self, image_array: np.ndarray) -> np.ndarray:
        height, width = image_array.shape[:2]
        target_height, target_width = self.target_size

        scale = min(target_width / width, target_height / height)
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))
        resized = cv2.resize(image_array, (new_width, new_height), interpolation=cv2.INTER_AREA)

        canvas = np.full((target_height, target_width, 3), 255, dtype=np.uint8)
        offset_x = (target_width - new_width) // 2
        offset_y = (target_height - new_height) // 2
        canvas[offset_y : offset_y + new_height, offset_x : offset_x + new_width] = resized
        return canvas

    def normalize(self, image_array: np.ndarray) -> np.ndarray:
        image_array = image_array.astype(np.float32) / 255.0
        return image_array

    def apply_noise_reduction(self, image_array: np.ndarray) -> np.ndarray:
        if self.noise_reduction == "gaussian":
            return cv2.GaussianBlur(image_array, (3, 3), 0)
        if self.noise_reduction == "median":
            return cv2.medianBlur(image_array, 3)
        return image_array

    def transform(self, image_array: np.ndarray) -> np.ndarray:
        processed = self.resize_with_padding(image_array)
        if self.noise_reduction:
            processed = self.apply_noise_reduction(processed)
        if self.should_normalize:
            processed = self.normalize(processed)
        return processed
