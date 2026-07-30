from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

from PIL import Image

from .dataset_index import DatasetIndex, ProductRecord


@dataclass
class ValidationResult:
    """Validation status for a single product."""

    product_id: str
    is_valid: bool
    issues: List[str] = field(default_factory=list)
    image_count: int = 0
    readable_images: int = 0


@dataclass
class ValidationReport:
    """Validation results for the full dataset."""

    dataset_root: str
    results: List[ValidationResult]
    valid_products: int
    invalid_products: int


class DatasetValidator:
    """Validate discovered products and report problems without stopping the pipeline."""

    def __init__(self, supported_formats: Sequence[str] | None = None, min_width: int = 16, min_height: int = 16) -> None:
        self.supported_formats = list(supported_formats or [".jpg", ".jpeg", ".png", ".webp"])
        self.min_width = min_width
        self.min_height = min_height

    def validate(self, dataset_index: DatasetIndex) -> ValidationReport:
        """Validate each product and collect non-blocking issues."""
        results: List[ValidationResult] = []
        for product in dataset_index.products:
            results.append(self._validate_product(product))

        valid_products = sum(1 for result in results if result.is_valid)
        invalid_products = len(results) - valid_products
        return ValidationReport(
            dataset_root=str(dataset_index.root_dir),
            results=results,
            valid_products=valid_products,
            invalid_products=invalid_products,
        )

    def _validate_product(self, product: ProductRecord) -> ValidationResult:
        issues: List[str] = []
        if not Path(product.directory).exists():
            issues.append("Folder does not exist")
        if not product.product_id:
            issues.append("Missing product ID")
        if not product.image_paths:
            issues.append("Empty folder")

        readable_images = 0
        for image_path in product.image_paths:
            if image_path.suffix.lower().lstrip(".") not in {fmt.lower().lstrip(".") for fmt in self.supported_formats}:
                issues.append(f"Unsupported format: {image_path.name}")
                continue
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
                    if width < self.min_width or height < self.min_height:
                        issues.append(f"Invalid dimensions for {image_path.name}")
                        continue
                    if image.mode not in {"RGB", "RGBA", "L", "P"}:
                        issues.append(f"Invalid channels for {image_path.name}")
                        continue
                    readable_images += 1
            except (OSError, ValueError) as exc:
                issues.append(f"Unreadable image: {image_path.name} ({exc})")

        if readable_images == 0 and product.image_paths:
            issues.append("No readable images")

        if len(product.image_paths) > len({path.name for path in product.image_paths}):
            issues.append("Duplicate images detected")

        return ValidationResult(
            product_id=product.product_id,
            is_valid=not issues,
            issues=issues,
            image_count=len(product.image_paths),
            readable_images=readable_images,
        )
