from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ProductRecord:
    """Represents a single discovered product directory and its associated media."""

    product_id: str
    current_folder: str
    gender: str
    image_paths: List[Path]
    image_count: int
    file_formats: List[str]
    directory: str
    classification: Dict[str, Any] | None = None
    attributes: Dict[str, Any] | None = None
    validation_status: str = "pending"
    validation_issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "image_paths": [str(path) for path in self.image_paths],
        }


@dataclass
class DatasetIndex:
    """Reusable dataset representation shared across the dataset management workflow."""

    root_dir: Path
    products: List[ProductRecord]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_dir": str(self.root_dir),
            "products": [product.to_dict() for product in self.products],
            "product_count": len(self.products),
        }
