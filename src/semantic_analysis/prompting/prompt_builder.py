from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class PromptBuilder:
    """Create a structured prompt for garment semantic analysis."""

    def __init__(self, config_dir: str | Path) -> None:
        self.config_dir = Path(config_dir)

    def _resolve_config_path(self, filename: str) -> Path:
        primary = self.config_dir / filename
        if primary.exists():
            return primary
        subdir = self.config_dir / "semantic_analysis" / filename
        if subdir.exists():
            return subdir
        return primary

    def build(self, image_path: str | Path) -> str:
        """Create a prompt that instructs the model to return structured JSON."""
        schema_path = self._resolve_config_path("metadata_schema.json")
        vocab_path = self._resolve_config_path("controlled_vocabularies.json")

        with open(schema_path, 'r', encoding='utf-8') as f:
            schema_content = json.dumps(json.load(f), indent=2)
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_content = json.dumps(json.load(f), indent=2)

        prompt = (
            "You are analyzing a single garment image. "
            "Identify the garment carefully and return JSON only. "
            "Do not include markdown fences, explanation, or commentary. "
            "Return a single JSON object with no surrounding text. "
            "If uncertain, use 'unknown' values and lower confidence. "
            "Use the following configuration context for structural guidance:"
            f"\nSchema context:\n{schema_content}\n"
            f"\nControlled vocabularies:\n{vocab_content}\n"
            "Return a JSON object matching the exact structure defined in the Schema context. "
            "Include confidence values between 0 and 1 in ai_analysis. "
            "CRITICAL RULES FOR VALUES: "
            "1. You MUST use the exact string values from the controlled vocabularies list. "
            "2. Preserve snake_case format exactly as provided (e.g., output 'tank_top' instead of 'tank top'). "
            "3. If a garment feature does not perfectly match one of the allowed vocabulary items, you MUST output 'unknown'."
        )
        return prompt
