from __future__ import annotations

import json
import logging
from typing import Any


class ResponseParser:
    """Parse raw model responses into a structured dictionary."""

    def __init__(self) -> None:
        self.logger = logging.getLogger("fabricvision.semantic_analysis.parsing")

    def parse(self, raw_response: str) -> dict[str, Any]:
        """Parse JSON from the raw model output, falling back to a structured empty payload."""
        if not raw_response or not raw_response.strip():
            return self._empty_payload()

        text = raw_response.strip()
        if "assistant\n" in text:
            text = text.split("assistant\n")[-1].strip()
            
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]

        start = text.find("{")
        if start != -1:
            brace_count = 0
            for i in range(start, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        text = text[start:i + 1]
                        break

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                unflattened = {}
                for k, v in parsed.items():
                    if '.' in k:
                        parts = k.split('.')
                        current = unflattened
                        for part in parts[:-1]:
                            if part not in current:
                                current[part] = {}
                            current = current[part]
                        current[parts[-1]] = v
                    else:
                        unflattened[k] = v
                return unflattened
        except json.JSONDecodeError as exc:
            self.logger.warning("Failed to parse model response: %s", exc)

        return self._empty_payload()

    def _empty_payload(self) -> dict[str, Any]:
        return {
            "garment_identity": {"name": "unknown", "gender": "unknown"},
            "classification": {"category": "unknown", "subcategory": "unknown", "garment_type": "unknown"},
            "physical_attributes": {"material": "unknown", "construction": "unknown"},
            "visual_attributes": {"colors": [], "patterns": [], "texture": "unknown"},
            "shape_and_fit": {"silhouette": "unknown", "fit": "unknown", "sleeves": "unknown", "neckline": "unknown"},
            "style": {"occasion": "unknown", "season": "unknown"},
            "fabric_behaviour": {"drape": "unknown", "flexibility": "unknown", "thickness": "unknown"},
            "virtual_try_on_attributes": {"ease": "unknown", "stretch": "unknown"},
            "ai_analysis": {"confidence": 0.0, "model_info": "qwen2.5-vl"},
        }
