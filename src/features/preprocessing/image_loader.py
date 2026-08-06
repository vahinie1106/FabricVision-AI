from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


class ImageLoader:
    """Loads and normalizes images from disk into a consistent representation."""

    def __init__(self, supported_formats: Optional[tuple[str, ...]] = None) -> None:
        self.supported_formats = supported_formats or (".jpg", ".jpeg", ".png")

    def load_image(self, image_path: str | Path) -> Tuple[np.ndarray, str]:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image path does not exist: {path}")
        if path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported image format: {path.suffix}")

        try:
            with Image.open(path) as img:
                rgb_image = img.convert("RGB")
                array = np.array(rgb_image, dtype=np.uint8)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError(f"Corrupted or unreadable image: {path}") from exc

        return array, str(path)

    def load_image_bgr(self, image_path: str | Path) -> Tuple[np.ndarray, str]:
        image_array, image_path_str = self.load_image(image_path)
        return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR), image_path_str
