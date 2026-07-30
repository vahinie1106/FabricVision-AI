from __future__ import annotations

from typing import Any, Dict, List


class QualityChecker:
    """Verify dataset integrity after organization and metadata generation."""

    def __init__(self) -> None:
        self.stats: Dict[str, Any] = {}

    def check(self, products: List[Dict[str, Any]], reorganized_paths: List[Any]) -> Dict[str, Any]:
        """Generate quality metrics for products and produced directories."""
        missing_products = [product["product_id"] for product in products if not product.get("new_path")]
        duplicate_products = [product["product_id"] for product in products if products.count(product) > 1]
        missing_images = [product["product_id"] for product in products if not product.get("image_paths")]
        metadata_complete = all(product.get("validation_status") for product in products)

        self.stats = {
            "product_count": len(products),
            "reorganized_path_count": len(reorganized_paths),
            "missing_products": missing_products,
            "duplicate_products": duplicate_products,
            "missing_images": missing_images,
            "metadata_complete": metadata_complete,
        }
        return self.stats
