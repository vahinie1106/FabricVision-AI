from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any


class MetadataBuilder:
    """Construct a complete metadata payload from parsed model output."""

    def __init__(self) -> None:
        self.logger = logging.getLogger("fabricvision.semantic_analysis.metadata")

    def build(self, parsed_response: dict[str, Any], image_path: str | Path) -> dict[str, Any]:
        """Merge parsed content with a standard shape and image metadata."""
        payload = {
            "garment_identity": {
                "name": parsed_response.get("garment_identity", {}).get("name", "unknown"),
                "gender": parsed_response.get("garment_identity", {}).get("gender", "unknown"),
            },
            "classification": {
                "category": parsed_response.get("classification", {}).get("category", "unknown"),
                "subcategory": parsed_response.get("classification", {}).get("subcategory", "unknown"),
                "garment_type": parsed_response.get("classification", {}).get("garment_type", "unknown"),
            },
            "physical_attributes": {
                "material": parsed_response.get("physical_attributes", {}).get("material", "unknown"),
                "construction": parsed_response.get("physical_attributes", {}).get("construction", "unknown"),
            },
            "visual_attributes": {
                "colors": parsed_response.get("visual_attributes", {}).get("colors", []),
                "patterns": parsed_response.get("visual_attributes", {}).get("patterns", []),
                "texture": parsed_response.get("visual_attributes", {}).get("texture", "unknown"),
            },
            "shape_and_fit": {
                "silhouette": parsed_response.get("shape_and_fit", {}).get("silhouette", "unknown"),
                "fit": parsed_response.get("shape_and_fit", {}).get("fit", "unknown"),
                "sleeves": parsed_response.get("shape_and_fit", {}).get("sleeves", "unknown"),
                "neckline": parsed_response.get("shape_and_fit", {}).get("neckline", "unknown"),
            },
            "style": {
                "occasion": parsed_response.get("style", {}).get("occasion", "unknown"),
                "season": parsed_response.get("style", {}).get("season", "unknown"),
            },
            "fabric_behaviour": {
                "drape": parsed_response.get("fabric_behaviour", {}).get("drape", "unknown"),
                "flexibility": parsed_response.get("fabric_behaviour", {}).get("flexibility", "unknown"),
                "thickness": parsed_response.get("fabric_behaviour", {}).get("thickness", "unknown"),
            },
            "virtual_try_on_attributes": {
                "ease": parsed_response.get("virtual_try_on_attributes", {}).get("ease", "unknown"),
                "stretch": parsed_response.get("virtual_try_on_attributes", {}).get("stretch", "unknown"),
            },
            "ai_analysis": {
                "confidence": parsed_response.get("ai_analysis", {}).get("confidence", 0.0),
                "model_info": parsed_response.get("ai_analysis", {}).get("model_info", "qwen2.5-vl"),
            },
        }
        payload["source_image"] = str(image_path)
        return payload
