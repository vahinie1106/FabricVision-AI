from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence


@dataclass
class ClassificationResult:
    """Represents a garment classification result."""

    garment_type: str
    confidence: float
    classification_source: str


class GarmentClassifier:
    """Rule-based garment classifier with a pluggable interface for future vision models."""

    def __init__(self, category_hints: Sequence[str] | None = None) -> None:
        self.category_hints = list(category_hints or [])

    def classify(self, product_id: str, gender: str, folder_name: str, image_paths: Sequence[str] | None = None) -> ClassificationResult:
        """Classify a product using simple heuristics based on folder names and path hints."""
        hint_tokens = [token.lower() for token in folder_name.replace("\\", "/").split("/") if token]
        hint_tokens.extend([token.lower() for token in [product_id, gender] if token])
        hint = " ".join(hint_tokens)
        if any(token in hint for token in ["shirt", "top", "tee"]):
            garment_type = "T-Shirt"
        elif any(token in hint for token in ["polo"]):
            garment_type = "Polo Shirt"
        elif any(token in hint for token in ["blazer", "suit"]):
            garment_type = "Blazer"
        elif any(token in hint for token in ["dress", "gown"]):
            garment_type = "Dress"
        elif any(token in hint for token in ["jumpsuit", "romper"]):
            garment_type = "Jumpsuit"
        elif any(token in hint for token in ["kurta", "kurti"]):
            garment_type = "Kurta"
        elif any(token in hint for token in ["saree"]):
            garment_type = "Saree"
        elif any(token in hint for token in ["lehenga"]):
            garment_type = "Lehenga"
        elif any(token in hint for token in ["jacket", "coat"]):
            garment_type = "Jacket"
        elif any(token in hint for token in ["sweater", "cardigan", "hoodie"]):
            garment_type = "Sweater"
        elif any(token in hint for token in ["jean", "trouser", "pant"]):
            garment_type = "Jeans"
        elif any(token in hint for token in ["short"]):
            garment_type = "Shorts"
        elif any(token in hint for token in ["legging"]):
            garment_type = "Leggings"
        else:
            garment_type = "Upperwear"

        return ClassificationResult(
            garment_type=garment_type,
            confidence=0.72,
            classification_source="rule_based_hints",
        )

    def to_dict(self, classification: ClassificationResult) -> Dict[str, Any]:
        return {
            "garment_type": classification.garment_type,
            "confidence": classification.confidence,
            "classification_source": classification.classification_source,
        }
