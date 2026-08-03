from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class PromptBuilder:
    """Create a structured prompt for garment semantic analysis."""

    def __init__(self, config_dir: str | Path) -> None:
        self.config_dir = Path(config_dir)

    def build(self, image_path: str | Path) -> str:
        """Create a prompt that instructs the model to return structured JSON."""
        taxonomy_path = self.config_dir / "garment_taxonomy.json"
        schema_path = self.config_dir / "metadata_schema.json"
        vocab_path = self.config_dir / "controlled_vocabularies.json"

        prompt = (
            "You are analyzing a single garment image. "
            "Identify the garment carefully and return JSON only. "
            "Do not include markdown fences, explanation, or commentary. "
            "Return a single JSON object with no surrounding text. "
            "If uncertain, use 'unknown' values and lower confidence. "
            "Use the following configuration context for structural guidance:"
            f"\nTaxonomy file: {taxonomy_path}"
            f"\nSchema file: {schema_path}"
            f"\nControlled vocabularies: {vocab_path}"
            "\nReturn a JSON object with the following top-level sections: "
            "garment_identity, classification, physical_attributes, visual_attributes, "
            "shape_and_fit, style, fabric_behaviour, virtual_try_on_attributes, ai_analysis. "
            "Include confidence values between 0 and 1. "
            "Use only values that are consistent with the provided controlled vocabularies."
        )
        return prompt
