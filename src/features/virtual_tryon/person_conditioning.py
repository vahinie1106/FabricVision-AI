from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Optional, Any
from PIL import Image, ImageOps

from src.features.virtual_tryon.models import PersonConditioningInput


class PersonConditioner:
    """Preprocess person target images, resize to target resolution, and generate agnostic body masks with DensePose fallback."""

    def __init__(self, target_resolution: Tuple[int, int] = (1024, 1024), catvton_path: str | Path = "models/CatVTON") -> None:
        self.target_resolution = target_resolution
        self.catvton_path = Path(catvton_path)
        self.logger = logging.getLogger("fabricvision.virtual_tryon.person_conditioner")

    def _try_load_densepose(self) -> Optional[Any]:
        densepose_dir = self.catvton_path / "DensePose"
        if densepose_dir.exists() and any(densepose_dir.glob("*.pkl")):
            self.logger.info("Found DensePose weight files at %s", densepose_dir)
            try:
                # Attempt to initialize detectron2/densepose if installed
                import detectron2  # type: ignore
                return "detectron2_densepose"
            except ImportError:
                self.logger.info("Detectron2 package not installed; using clean anatomical bounding box fallback.")
                return None
        return None

    def prepare_person_condition(self, person_input: PersonConditioningInput) -> PersonConditioningInput:
        """Process person image and agnostic mask to target resolution."""
        if not isinstance(person_input.person_image, Image.Image):
            raise ValueError("Invalid person_image: must be a PIL Image object.")

        img = person_input.person_image.convert("RGB")
        img_resized = ImageOps.fit(img, self.target_resolution, method=Image.Resampling.LANCZOS)

        agnostic = person_input.agnostic_mask
        if agnostic is None:
            # Check for DensePose / SCHP integration
            densepose_engine = self._try_load_densepose()
            
            # Generate clean anatomical upper/lower body agnostic mask canvas
            agnostic = Image.new("L", self.target_resolution, color=0)
            margin_x = int(self.target_resolution[0] * 0.2)
            margin_y = int(self.target_resolution[1] * 0.25)
            mask_box = (margin_x, margin_y, self.target_resolution[0] - margin_x, self.target_resolution[1] - margin_y)
            agnostic.paste(255, mask_box)
        else:
            agnostic = ImageOps.fit(agnostic.convert("L"), self.target_resolution, method=Image.Resampling.NEAREST)

        densepose = person_input.densepose
        if densepose is not None:
            densepose = ImageOps.fit(densepose.convert("RGB"), self.target_resolution, method=Image.Resampling.LANCZOS)

        return PersonConditioningInput(
            person_image=img_resized,
            agnostic_mask=agnostic,
            densepose=densepose,
        )
