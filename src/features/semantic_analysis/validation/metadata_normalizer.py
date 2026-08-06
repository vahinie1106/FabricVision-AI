import json
from pathlib import Path
from typing import Any

class MetadataNormalizer:
    """Normalize raw metadata by applying lowercasing, snake_case conversion, and synonym mapping."""
    
    def _resolve_config_path(self, filename: str) -> Path:
        primary = self.config_dir / filename
        if primary.exists():
            return primary
        subdir = self.config_dir / "semantic_analysis" / filename
        if subdir.exists():
            return subdir
        return primary

    def __init__(self, config_dir: str | Path) -> None:
        self.config_dir = Path(config_dir)
        self.vocab = self._load_json(self._resolve_config_path("controlled_vocabularies.json"))
        self.synonyms = self._load_json(self._resolve_config_path("synonym_mapping.json"))
        
    def _load_json(self, path: Path) -> dict[str, Any]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
        
    def _normalize_string(self, val: str) -> str:
        # Lowercase
        val = val.lower().strip()
        # Synonym mapping
        if val in self.synonyms:
            val = self.synonyms[val]
        # Spaces to snake_case
        val = val.replace(" ", "_")
        return val

    def _get_fallback(self, field: str) -> str:
        # Return none or unknown depending on what is allowed in vocab
        allowed = self.vocab.get("allowed_values", {}).get(field, [])
        if "unknown" in allowed:
            return "unknown"
        if "none" in allowed:
            return "none"
        return "unknown"
        
    def normalize(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """In-place normalize the metadata dictionary."""
        allowed_values = self.vocab.get("allowed_values", {})
        
        # Traverse metadata according to schema
        for section, fields in metadata.items():
            if not isinstance(fields, dict):
                continue
            for key, val in fields.items():
                vocab_field = f"{section}.{key}"
                
                # If there's no vocab rule, skip normalization
                if vocab_field not in allowed_values:
                    continue
                    
                allowed = allowed_values[vocab_field]
                
                if isinstance(val, str):
                    norm = self._normalize_string(val)
                    if norm not in allowed:
                        norm = self._get_fallback(vocab_field)
                    metadata[section][key] = norm
                    
                elif isinstance(val, list):
                    new_list = []
                    for item in val:
                        if isinstance(item, str):
                            norm = self._normalize_string(item)
                            if norm in allowed:
                                new_list.append(norm)
                    if not new_list:
                        new_list.append(self._get_fallback(vocab_field))
                    metadata[section][key] = new_list
                    
        return metadata
