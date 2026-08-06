from __future__ import annotations

from typing import Optional

import albumentations as A
import numpy as np


class AugmentationProcessor:
    """Applies conservative augmentation while preserving garment structure."""

    def __init__(self, enabled: bool = True, config: Optional[dict] = None) -> None:
        self.enabled = enabled
        self.config = config or {
            "horizontal_flip": 0.5,
            "rotation": 10,
            "brightness": 0.1,
            "contrast": 0.1,
            "zoom": 0.05,
        }
        self.transform = self._build_transform()

    def _build_transform(self) -> A.Compose:
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5 if self.enabled else 0.0),
                A.Rotate(limit=self.config["rotation"], p=0.3 if self.enabled else 0.0),
                A.RandomBrightnessContrast(
                    brightness_limit=self.config["brightness"],
                    contrast_limit=self.config["contrast"],
                    p=0.3 if self.enabled else 0.0,
                ),
                A.Affine(scale=(1 - self.config["zoom"], 1 + self.config["zoom"]), p=0.2 if self.enabled else 0.0),
            ],
            is_check_shapes=False,
        )

    def augment(self, image_array: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return image_array
        transformed = self.transform(image=image_array)
        return transformed["image"]
