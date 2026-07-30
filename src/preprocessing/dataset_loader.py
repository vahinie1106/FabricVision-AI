from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


@dataclass
class DatasetSample:
    """Represents a single discovered image file and its metadata."""

    file_path: Path
    file_name: str
    category: str
    relative_path: str
    width: int
    height: int
    file_format: str
    size_bytes: int


@dataclass
class DatasetStatistics:
    """Summary statistics for the discovered dataset."""

    total_images: int = 0
    categories: List[str] = field(default_factory=list)
    formats: Dict[str, int] = field(default_factory=dict)
    total_size_bytes: int = 0
    average_width: float = 0.0
    average_height: float = 0.0


class DatasetLoader:
    """Discovers and indexes images from a raw dataset directory."""

    def __init__(self, root_dir: str | os.PathLike[str], supported_formats: Optional[List[str]] = None) -> None:
        self.root_dir = Path(root_dir)
        self.supported_formats = supported_formats or [".jpg", ".jpeg", ".png"]

    def scan_dataset(self) -> "DatasetIndex":
        """Scan the dataset recursively and collect metadata for each supported image."""
        samples: List[DatasetSample] = []
        for image_path in self.root_dir.rglob("*"):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in self.supported_formats:
                continue
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
                sample = DatasetSample(
                    file_path=image_path,
                    file_name=image_path.name,
                    category=self._infer_category(image_path),
                    relative_path=str(image_path.relative_to(self.root_dir)).replace("\\", "/"),
                    width=width,
                    height=height,
                    file_format=image_path.suffix.lower().lstrip("."),
                    size_bytes=image_path.stat().st_size,
                )
                samples.append(sample)
            except (OSError, ValueError):
                continue

        samples.sort(key=lambda sample: sample.relative_path)
        return DatasetIndex(root_dir=self.root_dir, samples=samples)

    def _infer_category(self, image_path: Path) -> str:
        relative_parts = image_path.relative_to(self.root_dir).parts
        if len(relative_parts) < 2:
            return "root"
        return relative_parts[0]


class DatasetIndex:
    """A lightweight index for dataset samples, suitable for future PyTorch/TensorFlow integration."""

    def __init__(self, root_dir: Path, samples: List[DatasetSample]) -> None:
        self.root_dir = root_dir
        self.samples = samples

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_dir": str(self.root_dir),
            "sample_count": len(self.samples),
            "samples": [
                {
                    "file_path": str(sample.file_path),
                    "file_name": sample.file_name,
                    "category": sample.category,
                    "relative_path": sample.relative_path,
                    "width": sample.width,
                    "height": sample.height,
                    "file_format": sample.file_format,
                    "size_bytes": sample.size_bytes,
                }
                for sample in self.samples
            ],
        }

    def statistics(self) -> DatasetStatistics:
        statistics = DatasetStatistics()
        statistics.total_images = len(self.samples)
        statistics.categories = sorted({sample.category for sample in self.samples})
        statistics.formats = {}
        for sample in self.samples:
            statistics.formats[sample.file_format] = statistics.formats.get(sample.file_format, 0) + 1
            statistics.total_size_bytes += sample.size_bytes
        if self.samples:
            statistics.average_width = sum(sample.width for sample in self.samples) / len(self.samples)
            statistics.average_height = sum(sample.height for sample in self.samples) / len(self.samples)
        return statistics
