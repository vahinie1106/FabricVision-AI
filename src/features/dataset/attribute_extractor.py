from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .utils import infer_gender_from_path


class AttributeExtractor:
    """Extract garment metadata attributes from the dataset tree and classifier output."""

    def __init__(self, materials_root: str | Path | None = None, patterns_root: str | Path | None = None) -> None:
        self.materials_root = Path(materials_root) if materials_root else None
        self.patterns_root = Path(patterns_root) if patterns_root else None

    def extract(self, product: Any, classification: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Create a metadata payload with color, material, pattern, and gender details."""
        folder_parts = [part.lower() for part in Path(product.directory).parts]
        material = self._infer_from_tree(folder_parts, self.materials_root, "material")
        pattern = self._infer_from_tree(folder_parts, self.patterns_root, "pattern")
        gender = infer_gender_from_path(Path(product.directory))
        return {
            "gender": gender,
            "garment_type": classification.get("garment_type", "Unknown") if classification else "Unknown",
            "material": material,
            "pattern": pattern,
            "color": "Unknown",
            "secondary_color": "Unknown",
            "confidence": classification.get("confidence", 0.0) if classification else 0.0,
            "classification_source": classification.get("classification_source", "unknown") if classification else "unknown",
        }

    def _infer_from_tree(self, folder_parts: List[str], root: Path | None, label: str) -> str:
        if not root or not root.exists():
            return "Unknown"
        for part in folder_parts:
            candidate = root / part
            if candidate.exists() and candidate.is_dir():
                return part.replace("_", " ").title()
        return "Unknown"
