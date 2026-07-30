from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml
from PIL import Image


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def supported_image(path: Path, supported_formats: Sequence[str]) -> bool:
    """Check whether a file has a supported image extension."""
    return path.is_file() and path.suffix.lower() in {fmt.lower() for fmt in supported_formats}


def discover_images(root_dir: str | Path, supported_formats: Sequence[str]) -> List[Path]:
    """Discover supported image files recursively from a root directory."""
    root_path = Path(root_dir)
    if not root_path.exists():
        return []
    return [path for path in sorted(root_path.rglob("*")) if supported_image(path, supported_formats)]


def infer_gender_from_path(path: Path) -> str:
    """Infer gender from a path by inspecting path segments."""
    normalized_parts = [part.lower() for part in path.parts]
    if any(part in {"men", "male", "mens"} for part in normalized_parts):
        return "men"
    if any(part in {"women", "female", "womens"} for part in normalized_parts):
        return "women"
    if any(part in {"unisex", "neutral"} for part in normalized_parts):
        return "unisex"
    return "unknown"


def infer_product_id(path: Path, fallback: str = "unknown") -> str:
    """Extract a product identifier from the directory name."""
    if path.name and path.name.lower() not in {"datasets", "garments", "images", "data"}:
        return path.name
    return fallback


def read_image_dimensions(image_path: Path) -> tuple[int, int]:
    """Read image dimensions while preserving file readability validation."""
    with Image.open(image_path) as image:
        return image.size


def serialize_json(data: Dict[str, Any], path: str | Path) -> Path:
    """Write a dictionary to a JSON file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)
    return output_path


def normalize_label(value: Optional[str]) -> str:
    """Normalize a label to a user-friendly string."""
    if not value:
        return "Unknown"
    return value.replace("_", " ").strip().title()
