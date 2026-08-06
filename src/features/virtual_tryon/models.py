from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from PIL import Image


@dataclass
class PersonConditioningInput:
    """Input encapsulation for person target image and optional masks."""

    person_image: Image.Image
    agnostic_mask: Optional[Image.Image] = None
    densepose: Optional[Image.Image] = None


@dataclass
class GarmentConditioningInput:
    """Input encapsulation for FLUX-generated standalone garment image."""

    garment_image: Image.Image
    garment_type: str = "garment"
    garment_mask: Optional[Image.Image] = None


@dataclass
class TryOnResult:
    """Output encapsulation for CatVTON try-on result image and metadata."""

    output_image: Image.Image
    image_path: str
    metadata_path: str
    status: str = "completed"
    validation: Dict[str, Any] = field(default_factory=dict)
