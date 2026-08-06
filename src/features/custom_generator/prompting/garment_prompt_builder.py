from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Optional


class GarmentPromptBuilder:
    """Build structured, controlled generative prompts for FLUX garment synthesis."""

    def __init__(self, config_dir: str | Path = "configs") -> None:
        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger("fabricvision.garment_generation.prompt_builder")
        
        self.taxonomy = self._load_json("garment_taxonomy.json")
        self.schema = self._load_json("customization_schema.json")
        self.colors_vocab = self._load_json("color_vocabulary.json")
        self.materials_vocab = self._load_json("material_vocabulary.json")
        self.patterns_vocab = self._load_json("pattern_vocabulary.json")
        self.sleeves_vocab = self._load_json("sleeve_vocabulary.json")
        self.necklines_vocab = self._load_json("neckline_vocabulary.json")
        self.templates = self._load_templates()

    def _resolve_config_path(self, filename: str) -> Path:
        primary = self.config_dir / filename
        if primary.exists():
            return primary
        subdir = self.config_dir / "virtual_tryon" / filename
        if subdir.exists():
            return subdir
        return primary

    def _load_json(self, filename: str) -> Dict[str, Any]:
        filepath = self._resolve_config_path(filename)
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, dict):
                    return data
            except Exception as exc:
                self.logger.warning("Failed to load JSON config %s: %s", filename, exc)
        return {}

    def _load_templates(self) -> Dict[str, str]:
        loaded = self._load_json("garment_templates.json")
        if loaded and "positive_template" in loaded:
            return loaded
        return {
            "positive_template": (
                "A high-resolution studio product photograph of a {gender} {garment_type} crafted from {material} fabric "
                "featuring a {pattern} pattern and {texture} texture. Features: {neckline}, "
                "{sleeve_length}, size {size}, {style} style. Color palette: {dominant_colors}. "
                "Clean white studio background, standalone garment, no human model, no mannequin, photorealistic apparel catalog, 8k quality."
            ),
            "negative_template": (
                "blurry, low quality, human body, face, mannequin, person, cropped garment, distorted seams, noise, artifacts"
            ),
        }

    def _normalize_token(self, val: Optional[str], fallback: str) -> str:
        if not val or not str(val).strip():
            return fallback
        clean_str = str(val).strip().lower().replace(" ", "_")
        return clean_str

    def validate_and_normalize_color(self, color_val: Any) -> str:
        if isinstance(color_val, (list, tuple)):
            if not color_val:
                color_val = "blue"
            else:
                return ", ".join(self.validate_and_normalize_color(c) for c in color_val if c)
        
        token = self._normalize_token(str(color_val) if color_val else None, "blue")
        basic = self.colors_vocab.get("basic_colors", [])
        fashion = self.colors_vocab.get("fashion_colors", [])
        allowed_colors = set(basic + fashion)
        if allowed_colors and token not in allowed_colors:
            self.logger.info("Color '%s' not in controlled vocabulary; using token '%s'", color_val, token)
        return token

    def validate_and_normalize_material(self, mat_val: Optional[str]) -> str:
        token = self._normalize_token(mat_val, "cotton")
        natural = self.materials_vocab.get("natural", [])
        synthetic = self.materials_vocab.get("synthetic", [])
        blended = self.materials_vocab.get("blended", [])
        allowed = set(natural + synthetic + blended)
        if allowed and token not in allowed:
            self.logger.info("Material '%s' normalized to '%s'", mat_val, token)
        return token

    def validate_and_normalize_pattern(self, pattern_val: Optional[str]) -> str:
        token = self._normalize_token(pattern_val, "solid")
        allowed = set(self.patterns_vocab.get("allowed_patterns", []))
        if allowed and token not in allowed:
            self.logger.info("Pattern '%s' not in controlled vocabulary; falling back to 'solid'", pattern_val)
            return "solid"
        return token

    def validate_and_normalize_sleeve(self, sleeve_val: Optional[str]) -> str:
        token = self._normalize_token(sleeve_val, "short_sleeve")
        allowed = set(self.sleeves_vocab.get("allowed_sleeves", []))
        if allowed and token not in allowed:
            return "short_sleeve"
        return token

    def validate_and_normalize_neckline(self, neck_val: Optional[str]) -> str:
        token = self._normalize_token(neck_val, "round_neck")
        allowed = set(self.necklines_vocab.get("allowed_necklines", []))
        if allowed and token not in allowed:
            return "round_neck"
        return token

    def validate_and_normalize_size(self, size_val: Optional[str]) -> str:
        if not size_val:
            return "M"
        clean_size = str(size_val).strip().upper()
        allowed_sizes = set(self.schema.get("allowed_sizes", ["XS", "S", "M", "L", "XL"]))
        if clean_size not in allowed_sizes:
            return "M"
        return clean_size

    def build_prompts(
        self,
        fabric_metadata: Dict[str, Any],
        user_customization: Dict[str, Any],
    ) -> Tuple[str, str]:
        """Construct positive and negative FLUX prompts with strict vocabulary validation."""
        # Extract and validate fabric properties
        raw_mat = fabric_metadata.get("material") or user_customization.get("fabric_material")
        material = self.validate_and_normalize_material(raw_mat)
        
        raw_pat = fabric_metadata.get("pattern") or user_customization.get("pattern")
        pattern = self.validate_and_normalize_pattern(raw_pat)
        
        texture = self._normalize_token(fabric_metadata.get("texture"), "smooth")
        style = self._normalize_token(fabric_metadata.get("style"), "casual")

        raw_color = fabric_metadata.get("dominant_colors") or user_customization.get("color")
        if isinstance(raw_color, list):
            colors_str = ", ".join(self.validate_and_normalize_color(c).replace("_", " ") for c in raw_color if c)
        else:
            colors_str = self.validate_and_normalize_color(raw_color).replace("_", " ")
        colors_str = colors_str or "blue"

        # Extract and validate user customization choices
        gender = self._normalize_token(user_customization.get("gender"), "women")
        garment_type = self._normalize_token(user_customization.get("garment_type"), "kurti")
        sleeve_length = self.validate_and_normalize_sleeve(user_customization.get("sleeve"))
        neckline = self.validate_and_normalize_neckline(user_customization.get("neckline"))
        size = self.validate_and_normalize_size(user_customization.get("size"))

        # Additional attributes (e.g. embroidery, embellishments)
        extra_attrs = fabric_metadata.get("additional_attributes") or user_customization.get("additional_attributes")
        extra_str = ""
        if isinstance(extra_attrs, list) and extra_attrs:
            extra_str = ", " + ", ".join(str(a).replace("_", " ") for a in extra_attrs)
        elif isinstance(extra_attrs, str) and extra_attrs.strip():
            extra_str = ", " + extra_attrs.strip().replace("_", " ")

        # Clean string formatting for natural prompt phrasing
        neck_phrase = neckline.replace("_", " ")
        if neck_phrase.endswith(" neck"):
            neck_phrase = neck_phrase[:-5] + " neckline"
        elif not neck_phrase.endswith("neckline") and not neck_phrase.endswith("collar"):
            neck_phrase = f"{neck_phrase} neckline"

        sleeve_phrase = sleeve_length.replace("_", "-")
        if sleeve_phrase.endswith("-sleeve"):
            sleeve_phrase = sleeve_phrase[:-7] + "-sleeves"
        elif not sleeve_phrase.endswith("sleeves") and sleeve_phrase != "sleeveless":
            sleeve_phrase = f"{sleeve_phrase} sleeves"

        context = {
            "gender": gender.replace("_", " "),
            "garment_type": garment_type.replace("_", " "),
            "material": material.replace("_", " "),
            "pattern": pattern.replace("_", " "),
            "texture": texture.replace("_", " "),
            "style": style.replace("_", " "),
            "neckline": neck_phrase,
            "sleeve_length": sleeve_phrase,
            "size": size,
            "dominant_colors": colors_str,
        }

        positive_template = self.templates.get("positive_template", "")
        negative_template = self.templates.get("negative_template", "")

        try:
            positive_prompt = positive_template.format(**context) + extra_str
        except KeyError as exc:
            self.logger.error("Missing template key during prompt formatting: %s", exc)
            positive_prompt = f"A studio product photograph of a {context['gender']} {context['garment_type']} made of {context['material']} fabric, size {context['size']}, white background." + extra_str

        return positive_prompt, negative_template
