"""
Metadata Store Module for Persistent Garment Records.

Handles:
- Atomic sequential Garment ID generation (`garment_000001`, `garment_000002`)
- Schema & vocabulary validation before saving
- File IO persistence in data/metadata/
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from .schemas import GarmentMetadata
from .validators import MetadataValidator


class MetadataStore:
    """Manages saving, versioning, ID generation, and loading of production garment metadata."""

    def __init__(self, storage_dir: Union[str, Path] = "data/metadata"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.id_counter_file = self.storage_dir / ".id_counter"
        self._init_counter()

    def _init_counter(self) -> None:
        if not self.id_counter_file.exists():
            with open(self.id_counter_file, "w", encoding="utf-8") as f:
                f.write("0")

    def generate_garment_id(self) -> str:
        """Generates the next sequential garment ID in format 'garment_000001'."""
        with open(self.id_counter_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            current = int(content) if content else 0

        next_val = current + 1
        with open(self.id_counter_file, "w", encoding="utf-8") as f:
            f.write(str(next_val))

        return f"garment_{next_val:06d}"

    def save_metadata(self, raw_data: Dict[str, Any]) -> str:
        """
        Validates raw metadata dict, auto-assigns garment ID if missing,
        and saves JSON file to storage_dir. Returns assigned garment ID.
        """
        data_copy = dict(raw_data)
        if "garment_id" not in data_copy or not data_copy["garment_id"]:
            data_copy["garment_id"] = self.generate_garment_id()

        is_valid, result = MetadataValidator.validate(data_copy)
        if not is_valid:
            raise ValueError(f"Invalid metadata: {result}")

        model: GarmentMetadata = result
        filepath = self.storage_dir / f"{model.garment_id}.json"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(model.model_dump_json(indent=2))

        return model.garment_id

    def load_metadata(self, garment_id: str) -> Optional[GarmentMetadata]:
        """Loads and returns a GarmentMetadata model for a given garment_id."""
        return self.get_metadata(garment_id)

    def get_metadata(self, garment_id: str) -> Optional[GarmentMetadata]:
        """Retrieves and validates metadata for a given garment_id."""
        filepath = self.storage_dir / f"{garment_id}.json"
        if not filepath.exists():
            return None

        is_valid, result = MetadataValidator.load_and_validate_file(str(filepath))
        if is_valid:
            return result
        return None

    def list_garment_ids(self) -> List[str]:
        """Returns a list of all stored garment IDs."""
        return [f.stem for f in self.storage_dir.glob("garment_*.json")]
