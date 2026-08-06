"""
Metadata Validator Module for Phase 8 Data Management.

Validates garment metadata against:
1. Pydantic GarmentMetadata Schema
2. Required fields
3. Controlled Vocabularies in configs/controlled_vocabularies.json
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union
from pydantic import ValidationError

from .schemas import GarmentMetadata


class MetadataValidator:
    """Validates garment metadata against strict Pydantic schemas and controlled vocabularies."""

    def __init__(self, config_dir: Optional[Union[str, Path]] = None) -> None:
        if config_dir is None:
            self.config_dir = Path(__file__).resolve().parents[2] / "configs"
        else:
            self.config_dir = Path(config_dir)
        self.vocabularies = self._load_controlled_vocabularies()

    def _load_controlled_vocabularies(self) -> Dict[str, List[str]]:
        vocab_file = self.config_dir / "controlled_vocabularies.json"
        if not vocab_file.exists():
            vocab_file = self.config_dir / "semantic_analysis" / "controlled_vocabularies.json"
        if not vocab_file.exists():
            return {}
        try:
            with open(vocab_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("allowed_values", {})
        except Exception:
            return {}

    @classmethod
    def validate(cls, data: Dict[str, Any], config_dir: Optional[Union[str, Path]] = None) -> Tuple[bool, Any]:
        """
        Validates a dictionary against GarmentMetadata schema and controlled vocabularies.
        Returns (is_valid, GarmentMetadata model or error string/details).
        """
        instance = cls(config_dir=config_dir)
        return instance.validate_instance(data)

    def validate_instance(self, data: Dict[str, Any]) -> Tuple[bool, Any]:
        """Validate single dict instance against Pydantic schema and vocabulary constraints."""
        try:
            # Copy data to avoid mutating input dict directly
            data_copy = dict(data)
            if "garment_id" not in data_copy or not data_copy["garment_id"]:
                data_copy["garment_id"] = "garment_000000"

            if "identity" in data_copy and isinstance(data_copy["identity"], dict):
                model = GarmentMetadata(**data_copy)
            else:
                model = GarmentMetadata.from_legacy_dict(data_copy)

        except ValidationError as e:
            return False, f"Pydantic Validation Error: {e}"
        except Exception as e:
            return False, f"Schema Parsing Error: {e}"

        # Controlled vocabulary checks
        vocab_errors = self._validate_controlled_vocabularies(model)
        if vocab_errors:
            return False, f"Vocabulary Errors: {'; '.join(vocab_errors)}"

        return True, model

    def _validate_controlled_vocabularies(self, model: GarmentMetadata) -> List[str]:
        errors: List[str] = []
        vocab = self.vocabularies
        if not vocab:
            return errors

        # Check identity category
        allowed_cat = vocab.get("identity.category", vocab.get("classification.category", []))
        if allowed_cat and model.identity.category not in allowed_cat:
            errors.append(f"Invalid identity.category '{model.identity.category}'. Allowed: {allowed_cat[:5]}...")

        # Check gender
        allowed_gender = vocab.get("identity.gender", vocab.get("garment_identity.gender", []))
        if allowed_gender and model.identity.gender not in allowed_gender:
            errors.append(f"Invalid identity.gender '{model.identity.gender}'. Allowed: {allowed_gender}")

        # Check fabric
        allowed_fabric = vocab.get("physical.fabric", vocab.get("physical_attributes.material", []))
        if allowed_fabric and model.physical.fabric not in allowed_fabric:
            errors.append(f"Invalid physical.fabric '{model.physical.fabric}'. Allowed: {allowed_fabric[:5]}...")

        # Check colors
        allowed_colors = vocab.get("physical.color", vocab.get("visual_attributes.colors", []))
        if allowed_colors:
            for color in model.physical.color:
                if color not in allowed_colors:
                    errors.append(f"Invalid physical.color '{color}'. Allowed: {allowed_colors[:5]}...")

        return errors

    @classmethod
    def load_and_validate_file(cls, filepath: str, config_dir: Optional[Union[str, Path]] = None) -> Tuple[bool, Any]:
        """Loads a JSON file and validates its contents."""
        path = Path(filepath)
        if not path.exists():
            return False, f"File not found: {filepath}"

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.validate(data, config_dir=config_dir)
        except json.JSONDecodeError:
            return False, "Invalid JSON format"
        except Exception as e:
            return False, str(e)
