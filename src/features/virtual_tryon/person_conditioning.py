from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from PIL import Image

from src.features.virtual_tryon.models import PersonConditioningInput
from src.features.virtual_tryon.person_masker import (
    resolve_person_mask,
    restrict_mask_to_cloth_region,
    solidify_overall_dress_mask,
)
from src.features.virtual_tryon.catvton_conditioning import (
    analyze_mask,
    resize_person_and_mask,
    soft_blur_mask,
    validate_clothing_mask,
)


class PersonConditioner:
    """
    Preprocess person images and build a cloth-agnostic mask for CatVTON.

    Resize policy matches upstream app.py: resize_and_crop (NOT ImageOps.fit to a
    square, which warps full-body portraits and misaligns upper/lower bands).
    """

    def __init__(
        self,
        target_resolution: Tuple[int, int] = (384, 512),
        catvton_path: str | Path = "models/CatVTON",
    ) -> None:
        # target_resolution is (width, height) to match CatVTONPipeline kwargs.
        self.target_resolution = target_resolution
        self.catvton_path = Path(catvton_path)
        self.logger = logging.getLogger("fabricvision.virtual_tryon.person_conditioner")
        self.last_mask_source: str = "none"
        self.last_mask_stats: dict = {}
        self.last_mask_validation: tuple[bool, str] = (True, "ok")

    def prepare_person_condition(
        self,
        person_input: PersonConditioningInput,
        cloth_type: str = "upper",
    ) -> PersonConditioningInput:
        """Process person image and agnostic mask to target resolution."""
        if not isinstance(person_input.person_image, Image.Image):
            raise ValueError("Invalid person_image: must be a PIL Image object.")

        img = person_input.person_image.convert("RGB")
        w, h = self.target_resolution

        # Build mask on a provisional crop so GrabCut sees the same framing as inference.
        from src.features.virtual_tryon.catvton_conditioning import _import_catvton_utils

        resize_and_crop, _ = _import_catvton_utils(self.catvton_path)
        img_for_mask = resize_and_crop(img, (w, h))

        mask, source = resolve_person_mask(
            person_rgb=img_for_mask,
            provided_mask=person_input.agnostic_mask,
            target_size=(w, h),
            catvton_path=self.catvton_path,
            cloth_type=cloth_type,
        )

        # GrabCut already softens edges; a second radius-9 blur washes hems/sleeves.
        # AutoMasker/provided get a light official-style soften only.
        if source == "grabcut":
            pass
        elif source == "automasker":
            mask = soft_blur_mask(mask, blur_radius=5)
        else:
            mask = soft_blur_mask(mask, blur_radius=9)

        # Re-assert cloth-type vertical band after blur so upper never paints legs
        # and overall keeps continuous dress coverage.
        binary = (np.asarray(mask.convert("L")) > 127).astype(np.uint8) * 255
        binary = restrict_mask_to_cloth_region(binary, cloth_type)
        if cloth_type in ("overall", "dress", "outer"):
            pre_stats = analyze_mask(Image.fromarray(binary, mode="L"))
            if int(pre_stats.get("lower_split_segments") or 0) >= 2:
                binary = solidify_overall_dress_mask(binary)
                self.logger.info(
                    "Applied overall solidify after %s (lower_split_segments=%s)",
                    source,
                    pre_stats.get("lower_split_segments"),
                )
        mask = Image.fromarray(binary, mode="L")

        img_resized, mask = resize_person_and_mask(
            img_for_mask, mask, (w, h), catvton_root=self.catvton_path
        )

        # resize_and_crop / soft edges can reintroduce leg pixels — clip again.
        binary = (np.asarray(mask.convert("L")) > 127).astype(np.uint8) * 255
        binary = restrict_mask_to_cloth_region(binary, cloth_type)
        mask = Image.fromarray(binary, mode="L")

        self.last_mask_source = source
        ok, reason = validate_clothing_mask(mask, cloth_type=cloth_type)
        self.last_mask_validation = (ok, reason)
        self.last_mask_stats = analyze_mask(mask)
        self.logger.info(
            "Person agnostic mask source=%s valid=%s reason=%s stats=%s",
            source,
            ok,
            reason,
            self.last_mask_stats,
        )

        densepose = person_input.densepose
        if densepose is not None:
            densepose = resize_and_crop(densepose.convert("RGB"), (w, h))

        return PersonConditioningInput(
            person_image=img_resized,
            agnostic_mask=mask,
            densepose=densepose,
        )
