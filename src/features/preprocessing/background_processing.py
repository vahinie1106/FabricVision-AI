from __future__ import annotations

from enum import Enum
from typing import Optional

import cv2
import numpy as np


class BackgroundStrategy(str, Enum):
    """Pluggable strategy for background handling."""

    NONE = "none"
    SIMPLE_MASK = "simple_mask"


class BackgroundProcessor:
    """Applies a lightweight background-processing step that can be replaced later by SAM or U2Net."""

    def __init__(self, strategy: BackgroundStrategy = BackgroundStrategy.NONE) -> None:
        self.strategy = strategy

    def process(self, image_array: np.ndarray) -> np.ndarray:
        if self.strategy == BackgroundStrategy.NONE:
            return image_array

        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        result = image_array.copy()
        result[mask > 0] = [255, 255, 255]
        return result
