from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from src.common.utils.utils import ensure_directory, serialize_json


class DatasetOrganizer:
    """Copy approved garments into a curated taxonomy-based dataset layout."""

    def __init__(self, output_root: str | Path) -> None:
        self.output_root = Path(output_root)
        self.logger = logging.getLogger("fabricvision.semantic_analysis.organization")

    def organize(self, image_path: str | Path, metadata: dict[str, Any]) -> dict[str, Any]:
        """Organize the image and JSON metadata into the curated output tree."""
        source_image = Path(image_path)
        if not source_image.exists():
            raise FileNotFoundError(f"Image not found: {source_image}")

        gender = str(metadata.get("garment_identity", {}).get("gender", "unknown")).lower()
        category = str(metadata.get("classification", {}).get("category", "upper_wear")).lower()
        subcategory = str(metadata.get("classification", {}).get("subcategory", "unknown")).lower()
        stem = source_image.stem

        if gender not in {"men", "women", "unisex"}:
            gender = "unisex"

        target_dir = self.output_root / gender / category / subcategory
        ensure_directory(target_dir)
        target_image = target_dir / f"{stem}{source_image.suffix}"
        target_metadata = target_dir / f"{stem}.json"

        shutil.copy2(source_image, target_image)
        serialize_json(metadata, target_metadata)

        return {
            "target_dir": target_dir,
            "image_path": target_image,
            "metadata_path": target_metadata,
        }
