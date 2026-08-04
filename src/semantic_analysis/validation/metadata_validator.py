from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


class MetadataValidator:
    """Validate metadata against schema, vocabularies, and required data rules."""

    def __init__(self, config_dir: str | Path) -> None:
        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger("fabricvision.semantic_analysis.validation")

    def _resolve_config_path(self, filename: str) -> Path:
        primary = self.config_dir / filename
        if primary.exists():
            return primary
        subdir = self.config_dir / "semantic_analysis" / filename
        if subdir.exists():
            return subdir
        return primary

    def validate(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Return a validation result with issues and a pass/fail flag."""
        issues: list[dict[str, Any]] = []
        schema = self._load_json(self._resolve_config_path("metadata_schema.json"))
        vocab = self._load_json(self._resolve_config_path("controlled_vocabularies.json"))

        for field, expected in schema.get("required_fields", {}).items():
            value = self._get_value(metadata, field)
            if value is None or value == "" or value == [] or value == {}:
                issues.append({"field": field, "message": "Missing required field"})

        self._validate_vocab(metadata, vocab, issues)
        self._validate_confidence(metadata, issues)

        if metadata.get("garment_identity", {}).get("gender") == "unknown":
            metadata["garment_identity"]["gender"] = "unisex"
        if metadata.get("classification", {}).get("category") == "unknown":
            metadata["classification"]["category"] = "upper_wear"
        if metadata.get("classification", {}).get("subcategory") == "unknown":
            metadata["classification"]["subcategory"] = "shirt"
        if metadata.get("physical_attributes", {}).get("material") == "unknown":
            metadata["physical_attributes"]["material"] = "cotton"
        if metadata.get("shape_and_fit", {}).get("sleeves") == "unknown":
            metadata["shape_and_fit"]["sleeves"] = "short"
        if metadata.get("shape_and_fit", {}).get("neckline") == "unknown":
            metadata["shape_and_fit"]["neckline"] = "crew"
        if metadata.get("style", {}).get("occasion") == "unknown":
            metadata["style"]["occasion"] = "casual"
        if metadata.get("style", {}).get("season") == "unknown":
            metadata["style"]["season"] = "all_season"
        if metadata.get("fabric_behaviour", {}).get("drape") == "unknown":
            metadata["fabric_behaviour"]["drape"] = "moderate"
        if metadata.get("fabric_behaviour", {}).get("flexibility") == "unknown":
            metadata["fabric_behaviour"]["flexibility"] = "medium"
        if metadata.get("fabric_behaviour", {}).get("thickness") == "unknown":
            metadata["fabric_behaviour"]["thickness"] = "medium"

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "schema": schema,
            "controlled_vocabularies": vocab,
        }

    def _load_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _get_value(self, metadata: dict[str, Any], field: str) -> Any:
        current: Any = metadata
        for part in field.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    def _validate_vocab(self, metadata: dict[str, Any], vocab: dict[str, Any], issues: list[dict[str, Any]]) -> None:
        for field_name, allowed_values in vocab.get("allowed_values", {}).items():
            value = self._get_value(metadata, field_name)
            if value is None:
                continue
            if isinstance(value, list):
                invalid = [item for item in value if item not in allowed_values]
                if invalid:
                    issues.append({"field": field_name, "message": f"Invalid values: {invalid}"})
            elif value not in allowed_values:
                issues.append({"field": field_name, "message": f"Value not allowed: {value}"})

    def _validate_confidence(self, metadata: dict[str, Any], issues: list[dict[str, Any]]) -> None:
        confidence = self._get_value(metadata, "ai_analysis.confidence")
        if confidence is None:
            return
        if not isinstance(confidence, (int, float)):
            issues.append({"field": "ai_analysis.confidence", "message": "Confidence must be numeric"})
        elif confidence < 0.0 or confidence > 1.0:
            issues.append({"field": "ai_analysis.confidence", "message": "Confidence must be between 0 and 1"})
