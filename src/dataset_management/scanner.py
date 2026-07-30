from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from .dataset_index import DatasetIndex, ProductRecord
from .utils import infer_gender_from_path, infer_product_id, supported_image


class DatasetScanner:
    """Discover products and image files from the raw dataset tree."""

    def __init__(self, dataset_root: str | Path, supported_formats: Sequence[str] | None = None) -> None:
        self.dataset_root = Path(dataset_root)
        self.supported_formats = list(supported_formats or [".jpg", ".jpeg", ".png", ".webp"])

    def scan(self) -> DatasetIndex:
        """Scan the dataset root, identify product directories, and collect image metadata."""
        products: List[ProductRecord] = []
        if not self.dataset_root.exists():
            return DatasetIndex(root_dir=self.dataset_root, products=[])

        for directory in sorted(self.dataset_root.rglob("*")):
            if not directory.is_dir():
                continue
            image_paths = [path for path in sorted(directory.iterdir()) if path.is_file() and supported_image(path, self.supported_formats)]
            if not image_paths:
                continue

            product_id = infer_product_id(directory)
            gender = infer_gender_from_path(directory)
            relative_directory = str(directory.relative_to(self.dataset_root)).replace("\\", "/") if directory != self.dataset_root else ""
            file_formats = sorted({path.suffix.lower().lstrip(".") for path in image_paths})
            products.append(
                ProductRecord(
                    product_id=product_id,
                    current_folder=relative_directory,
                    gender=gender,
                    image_paths=image_paths,
                    image_count=len(image_paths),
                    file_formats=file_formats,
                    directory=str(directory),
                )
            )

        products.sort(key=lambda product: product.product_id)
        return DatasetIndex(root_dir=self.dataset_root, products=products)
