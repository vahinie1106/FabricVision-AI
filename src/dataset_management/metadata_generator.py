from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


class MetadataGenerator:
    """Generate CSV, JSON, and YAML metadata for the managed dataset."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)

    def generate(self, products: List[Dict[str, Any]]) -> Dict[str, Path]:
        """Create metadata artifacts for all products."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.output_dir / "dataset_metadata.csv"
        json_path = self.output_dir / "dataset_metadata.json"
        yaml_path = self.output_dir / "dataset_metadata.yaml"

        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "product_id",
                    "gender",
                    "garment_type",
                    "material",
                    "pattern",
                    "color",
                    "original_path",
                    "new_path",
                    "confidence",
                    "validation_status",
                    "classification_source",
                    "image_paths",
                ],
            )
            writer.writeheader()
            for product in products:
                writer.writerow(product)

        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(products, handle, indent=2)

        with yaml_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(products, handle, sort_keys=False)

        return {"csv": csv_path, "json": json_path, "yaml": yaml_path}
