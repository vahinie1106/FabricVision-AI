from __future__ import annotations

import cv2
import numpy as np


class ImageCleaner:
    """Provides simple image cleaning operations for future segmentation-ready inputs."""

    def __init__(self, denoise: bool = True) -> None:
        self.denoise = denoise

    def clean(self, image_array: np.ndarray) -> np.ndarray:
        cleaned = image_array.astype(np.uint8)
        if self.denoise:
            cleaned = cv2.fastNlMeansDenoisingColored(cleaned, None, 10, 10, 7, 21)
        return cleaned
