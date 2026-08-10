from __future__ import annotations

from pathlib import Path
from shutil import copytree
from typing import Any, Dict, List, Sequence

from src.common.utils.utils import ensure_directory


class DatasetReorganizer:
    """Create a production-ready garment dataset layout while preserving product integrity."""

    def __init__(self, output_root: str | Path) -> None:
        self.output_root = Path(output_root)

    def reorganize(self, products: Sequence[Dict[str, Any]], root_dir: str | Path | None = None) -> List[Path]:
        """Copy each product directory into the target structure without splitting products."""
        ensure_directory(self.output_root)
        created_paths: List[Path] = []
        for product in products:
            source_dir = Path(product["original_path"])
            target_dir = self._target_directory(product)
            if source_dir.exists():
                copytree(source_dir, target_dir, dirs_exist_ok=True)
                created_paths.append(target_dir)
        return created_paths

    def _target_directory(self, product: Dict[str, Any]) -> Path:
        gender = product.get("gender", "unknown").lower()
        garment_type = product.get("garment_type", "upperwear")
        if gender == "men":
            gender_dir = "men"
        elif gender == "women":
            gender_dir = "women"
        else:
            gender_dir = "unisex"

        category_dir = self._map_garment_category(garment_type)
        return self.output_root / "garments" / gender_dir / category_dir / product["product_id"]

    def _map_garment_category(self, garment_type: str) -> str:
        mapping = {
            "T-Shirt": "upperwear",
            "Polo Shirt": "upperwear",
            "Formal Shirt": "upperwear",
            "Casual Shirt": "upperwear",
            "Sweater": "upperwear",
            "Cardigan": "upperwear",
            "Hoodie": "upperwear",
            "Sweatshirt": "upperwear",
            "Jacket": "upperwear",
            "Blazer": "upperwear",
            "Vest": "upperwear",
            "Jeans": "lowerwear",
            "Trousers": "lowerwear",
            "Chinos": "lowerwear",
            "Joggers": "lowerwear",
            "Cargo Pants": "lowerwear",
            "Shorts": "lowerwear",
            "Leggings": "lowerwear",
            "Dress": "dresses",
            "Romper": "dresses",
            "Jumpsuit": "dresses",
            "Kurta": "traditional",
            "Kurti": "traditional",
            "Saree": "traditional",
            "Lehenga": "traditional",
        }
        return mapping.get(garment_type, "upperwear")
