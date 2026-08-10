from __future__ import annotations



import logging

from pathlib import Path

from typing import Tuple

from PIL import Image



from src.features.virtual_tryon.models import GarmentConditioningInput

from src.features.virtual_tryon.catvton_conditioning import resize_garment_condition





class GarmentConditioner:

    """

    Preprocess garment images for CatVTON condition maps.



    Upstream uses resize_and_padding (white letterbox) — NOT ImageOps.fit crop —

    so neckline/hem/silhouette are not arbitrarily stretched or cropped away.

    """



    def __init__(

        self,

        target_resolution: Tuple[int, int] = (384, 512),

        catvton_path: str | Path = "models/CatVTON",

    ) -> None:

        self.target_resolution = target_resolution

        self.catvton_path = Path(catvton_path)

        self.logger = logging.getLogger("fabricvision.virtual_tryon.garment_conditioner")



    def prepare_garment_condition(self, garment_input: GarmentConditioningInput) -> GarmentConditioningInput:

        """Validate garment and resize with official padding semantics."""

        if not isinstance(garment_input.garment_image, Image.Image):

            raise ValueError("Invalid garment_image: must be a PIL Image object.")



        img = garment_input.garment_image.convert("RGB")

        w, h = self.target_resolution

        img_resized = resize_garment_condition(img, (w, h), catvton_root=self.catvton_path)



        garment_mask = garment_input.garment_mask

        if garment_mask is None:

            grayscale = img_resized.convert("L")

            garment_mask = Image.eval(grayscale, lambda px: 0 if px > 245 else 255)

        else:

            # Mask follows garment framing (padded canvas).

            garment_mask = resize_garment_condition(

                garment_mask.convert("RGB"), (w, h), catvton_root=self.catvton_path

            ).convert("L")



        self.logger.info(

            "Garment conditioned with resize_and_padding to %sx%s (type=%s)",

            w,

            h,

            garment_input.garment_type,

        )

        return GarmentConditioningInput(

            garment_image=img_resized,

            garment_type=garment_input.garment_type,

            garment_mask=garment_mask,

        )


