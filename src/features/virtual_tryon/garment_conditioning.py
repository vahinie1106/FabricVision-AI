from __future__ import annotations

import logging
from typing import Tuple
from PIL import Image, ImageOps

from src.features.virtual_tryon.models import GarmentConditioningInput


class GarmentConditioner:
    """Preprocess and validate FLUX-generated standalone garment images for CatVTON try-on condition maps."""

    def __init__(self, target_resolution: Tuple[int, int] = (1024, 1024)) -> None:
        self.target_resolution = target_resolution
        self.logger = logging.getLogger("fabricvision.virtual_tryon.garment_conditioner")

    def prepare_garment_condition(self, garment_input: GarmentConditioningInput) -> GarmentConditioningInput:
        """Validate standalone garment parameters and compute binary garment silhouette mask."""
        if not isinstance(garment_input.garment_image, Image.Image):
            raise ValueError("Invalid garment_image: must be a PIL Image object.")

        img = garment_input.garment_image.convert("RGB")
        img_resized = ImageOps.fit(img, self.target_resolution, method=Image.Resampling.LANCZOS)

        garment_mask = garment_input.garment_mask
        if garment_mask is None:
            # Threshold non-white background pixels to build garment silhouette mask
            grayscale = img_resized.convert("L")
            garment_mask = Image.eval(grayscale, lambda px: 0 if px > 245 else 255)
        else:
            garment_mask = ImageOps.fit(garment_mask.convert("L"), self.target_resolution, method=Image.Resampling.NEAREST)

        return GarmentConditioningInput(
            garment_image=img_resized,
            garment_type=garment_input.garment_type,
            garment_mask=garment_mask,
        )
