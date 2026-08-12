from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


# CLIP text encoder budget for FLUX (includes special tokens → keep content under this).
CLIP_MAX_TOKENS = 77
CLIP_SAFE_CONTENT_TOKENS = 75


class GarmentPromptBuilder:
    """Build compact, CLIP-safe generative prompts for FLUX.1-Kontext garment synthesis."""

    def __init__(self, config_dir: str | Path = "configs") -> None:
        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger("fabricvision.garment_generation.prompt_builder")
        self._clip_tokenizer = None

        self.taxonomy = self._load_json("garment_taxonomy.json")
        self.schema = self._load_json("customization_schema.json")
        self.colors_vocab = self._load_json("color_vocabulary.json")
        self.materials_vocab = self._load_json("material_vocabulary.json")
        self.patterns_vocab = self._load_json("pattern_vocabulary.json")
        self.sleeves_vocab = self._load_json("sleeve_vocabulary.json")
        self.necklines_vocab = self._load_json("neckline_vocabulary.json")
        self.templates = self._load_templates()
        self.last_prompt_stats: Dict[str, Any] = {}

    def _resolve_config_path(self, filename: str) -> Path:
        """Resolve config files across the project's config package layout."""
        candidates = [
            self.config_dir / filename,
            self.config_dir / "custom_generator" / filename,
            self.config_dir / "semantic_analysis" / filename,
            self.config_dir / "virtual_tryon" / filename,
            self.config_dir.parent / "semantic_analysis" / filename,
            self.config_dir.parent / "custom_generator" / filename,
        ]
        for path in candidates:
            if path.exists():
                return path
        return candidates[0]

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
        defaults = {
            "positive_template": (
                "Finished wearable {gender} {garment_type}, {fit}, {neckline}, {sleeve_length}, "
                "{material} with {pattern} pattern, {texture} texture, colors {dominant_colors}. "
                "Sharp detailed photoreal studio product photo, clear seams and folds, "
                "white background, no model."
            ),
            "negative_template": (
                "fabric swatch, textile sample, fabric close-up, repeating textile fill, "
                "abstract fabric, flat cloth only, cropped fabric, pattern tile, blurry, "
                "soft focus, muddy details, low resolution, human model, person, face, "
                "mannequin, busy background, distorted proportions, artifacts, "
                "plastic fabric, cgi render, warped sleeves, malformed neckline, "
                "extra limbs, jewelry, random objects"
            ),
            "kontext_instruction_template": (
                "Edit this fabric-filled {garment_type} mockup into a sharp, highly detailed "
                "photoreal {gender} {garment_type} with {neckline}, {sleeve_length}, {fit}. "
                "Keep exact fabric print, weave, texture, and colors. Clear seams, folds, hem, "
                "and edges. White studio fashion product photo. Not a swatch, no model."
            ),
        }
        if loaded and "positive_template" in loaded:
            merged = dict(defaults)
            merged.update({k: v for k, v in loaded.items() if isinstance(v, str) and v.strip()})
            return merged
        return defaults

    def _get_clip_tokenizer(self) -> Any:
        """Lazy-load CLIP tokenizer for accurate 77-token budget checks (no silent truncation)."""
        if self._clip_tokenizer is not None:
            return self._clip_tokenizer
        try:
            from transformers import CLIPTokenizer

            # Prefer local FLUX CLIP tokenizer if present (offline-safe).
            local_candidates = [
                Path("models/flux-kontext/tokenizer"),
                Path("models/flux/tokenizer"),
            ]
            for path in local_candidates:
                if path.exists() and (path / "vocab.json").exists():
                    self._clip_tokenizer = CLIPTokenizer.from_pretrained(str(path))
                    return self._clip_tokenizer
            self._clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            return self._clip_tokenizer
        except Exception as exc:
            self.logger.warning("CLIP tokenizer unavailable for token counting: %s", exc)
            self._clip_tokenizer = False
            return None

    def count_clip_tokens(self, text: str) -> int:
        """Return CLIP token count including special tokens, or a conservative estimate."""
        tokenizer = self._get_clip_tokenizer()
        if tokenizer is not None and tokenizer is not False:
            encoded = tokenizer(
                text,
                truncation=False,
                add_special_tokens=True,
                return_attention_mask=False,
            )
            return int(len(encoded["input_ids"]))
        # Conservative fallback: ~0.75 words/token → overestimate slightly
        words = max(1, len(text.split()))
        return int(words * 1.35) + 2

    def _normalize_token(self, val: Optional[str], fallback: str) -> str:
        if not val or not str(val).strip():
            return fallback
        return str(val).strip().lower().replace(" ", "_")

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
        aliases = {
            "dotted": "polka_dot",
            "polka-dot": "polka_dot",
            "polkadot": "polka_dot",
            "dots": "polka_dot",
            "patterned": "printed",
        }
        token = aliases.get(token, token)
        allowed = set(self.patterns_vocab.get("allowed_patterns", []))
        if allowed and token not in allowed:
            self.logger.info("Pattern '%s' not in controlled vocabulary; keeping descriptive token", pattern_val)
            return token if token else "solid"
        return token

    def validate_and_normalize_sleeve(self, sleeve_val: Optional[str]) -> str:
        token = self._normalize_token(sleeve_val, "short_sleeve")
        aliases = {
            "long": "full_sleeve",
            "long_sleeve": "full_sleeve",
            "long_sleeves": "full_sleeve",
            "short": "short_sleeve",
            "short_sleeves": "short_sleeve",
            "puff": "puff_sleeve",
            "puff_sleeves": "puff_sleeve",
            "sleeveless": "sleeveless",
        }
        token = aliases.get(token, token)
        allowed = set(self.sleeves_vocab.get("allowed_sleeves", []))
        if allowed and token not in allowed:
            self.logger.info("Sleeve '%s' outside vocabulary; using sanitized token '%s'", sleeve_val, token)
            return token
        return token

    def validate_and_normalize_neckline(self, neck_val: Optional[str]) -> str:
        token = self._normalize_token(neck_val, "round_neck")
        aliases = {
            "collar": "collar_neck",
            "collared": "collar_neck",
            "shirt_collar": "collar_neck",
            "mandarin": "mandarin_collar",
            "mandarin_collar": "mandarin_collar",
            "sweetheart": "sweetheart_neck",
            "round": "round_neck",
            "v_neck": "v_neck",
            "vneck": "v_neck",
        }
        token = aliases.get(token, token)
        allowed = set(self.necklines_vocab.get("allowed_necklines", []))
        if allowed and token not in allowed:
            self.logger.info("Neckline '%s' outside vocabulary; using sanitized token '%s'", neck_val, token)
            return token
        return token

    def validate_and_normalize_size(self, size_val: Optional[str]) -> str:
        if not size_val:
            return "M"
        clean_size = str(size_val).strip().upper()
        allowed_sizes = set(self.schema.get("allowed_sizes", ["XS", "S", "M", "L", "XL"]))
        if clean_size not in allowed_sizes:
            return "M"
        return clean_size

    def _build_context(
        self,
        fabric_metadata: Dict[str, Any],
        user_customization: Dict[str, Any],
    ) -> Dict[str, str]:
        raw_mat = (
            user_customization.get("material")
            or fabric_metadata.get("material")
            or user_customization.get("fabric")
            or fabric_metadata.get("fabric")
        )
        material = self.validate_and_normalize_material(raw_mat)

        raw_pat = fabric_metadata.get("pattern") or user_customization.get("pattern")
        pattern = self.validate_and_normalize_pattern(raw_pat)

        texture = self._normalize_token(
            user_customization.get("texture") or fabric_metadata.get("texture"),
            "smooth",
        )
        style = self._normalize_token(
            user_customization.get("style") or fabric_metadata.get("style"),
            "casual",
        )
        fit = self._normalize_token(
            user_customization.get("fit") or fabric_metadata.get("fit"),
            "regular",
        )
        fit_phrase = fit.replace("_", " ")
        if not fit_phrase.endswith("fit"):
            fit_phrase = f"{fit_phrase} fit"
        occasion = self._normalize_token(
            user_customization.get("occasion") or fabric_metadata.get("occasion"),
            "casual",
        )
        season = self._normalize_token(
            user_customization.get("season") or fabric_metadata.get("season"),
            "all_season",
        )

        raw_color = (
            fabric_metadata.get("dominant_colors")
            or fabric_metadata.get("color")
            or user_customization.get("color")
        )
        color_source = str(fabric_metadata.get("color_source") or "").strip()
        force_recolor = bool(user_customization.get("force_recolor"))
        ui_color = user_customization.get("color") or fabric_metadata.get("color")
        ui_key = (
            str(ui_color).strip().lower().replace(" ", "_").replace("-", "_")
            if ui_color is not None
            else ""
        )
        explicit_recolor = color_source == "ui_recolor" or (
            force_recolor and ui_key not in ("", "match_fabric", "matchfabric")
        )
        # When fabric pixels provided colors, ignore UI recolor unless force_recolor
        if explicit_recolor and ui_key not in ("", "match_fabric", "matchfabric"):
            raw_color = user_customization.get("color") or fabric_metadata.get(
                "dominant_colors"
            ) or raw_color
            color_mode = "explicit"
        elif color_source == "fabric_pixels":
            raw_color = fabric_metadata.get("dominant_colors") or raw_color
            color_mode = "match_fabric"
        else:
            color_mode = "match_fabric"
        if isinstance(raw_color, list):
            # Keep up to 3 names so white+red+green florals survive CLIP budget
            colors_str = ", ".join(
                self.validate_and_normalize_color(c).replace("_", " ") for c in raw_color[:3] if c
            )
        else:
            colors_str = self.validate_and_normalize_color(raw_color).replace("_", " ")
        colors_str = colors_str or "multicolor"
        # Never leave Match Fabric as a "color" token in the prompt.
        if colors_str.replace(" ", "_") in ("match_fabric", "matchfabric"):
            colors_str = "multicolor"

        gender = self._normalize_token(user_customization.get("gender"), "women")
        garment_type = self._normalize_token(user_customization.get("garment_type"), "kurti")
        sleeve_length = self.validate_and_normalize_sleeve(user_customization.get("sleeve"))
        neckline = self.validate_and_normalize_neckline(user_customization.get("neckline"))
        size = self.validate_and_normalize_size(user_customization.get("size"))

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

        appearance = str(
            fabric_metadata.get("fabric_appearance")
            or fabric_metadata.get("appearance_summary")
            or ""
        ).strip()
        # Keep appearance short enough for CLIP; strip redundant filler words.
        if len(appearance) > 90:
            appearance = appearance[:87].rsplit(" ", 1)[0] + "…"

        motif_raw = (
            fabric_metadata.get("motif_colors")
            or fabric_metadata.get("source_palette")
            or []
        )
        if isinstance(motif_raw, list):
            # Drop the target/base name if it leaked into motif list.
            motif_names = [
                self.validate_and_normalize_color(c).replace("_", " ")
                for c in motif_raw[:3]
                if c
                and str(c).lower().replace(" ", "_")
                not in ("match_fabric", colors_str.replace(" ", "_"))
            ]
            # Prefer secondary palette entries (skip first dominant when possible).
            if (
                fabric_metadata.get("source_palette")
                and not fabric_metadata.get("motif_colors")
                and len(motif_names) > 1
            ):
                motif_names = motif_names[1:]
            motif_str = ", ".join(motif_names) if motif_names else "original print"
        else:
            motif_str = str(motif_raw).replace("_", " ") or "original print"

        return {
            "gender": gender.replace("_", " "),
            "garment_type": garment_type.replace("_", " "),
            "material": material.replace("_", " "),
            "pattern": pattern.replace("_", " "),
            "texture": texture.replace("_", " "),
            "style": style.replace("_", " "),
            "fit": fit_phrase,
            "occasion": occasion.replace("_", " "),
            "season": season.replace("_", " "),
            "neckline": neck_phrase,
            "sleeve_length": sleeve_phrase,
            "size": size,
            "dominant_colors": colors_str,
            "fabric_appearance": appearance,
            "color_mode": color_mode,
            "motif_colors": motif_str,
        }

    def _build_compact_kontext_layers(self, context: Dict[str, str]) -> list[str]:
        """
        Layered prompt fragments ordered by importance.

        Why layers: CLIP truncates at 77 tokens. Garment construction and fabric
        identity stay first; style is optional and drops under budget pressure.

        Color modes:
        - match_fabric: preserve uploaded textile colors (do not recolor).
        - explicit: change BASE fabric color only; keep original print/motif colors.
        """
        primary = (
            f"Edit fabric-filled {context['garment_type']} mockup into realistic "
            f"wearable {context['gender']} {context['garment_type']}: "
            f"{context['neckline']}, {context['sleeve_length']}, {context['fit']}, "
            f"natural drape."
        )
        appearance = (context.get("fabric_appearance") or "").strip()
        color_mode = (context.get("color_mode") or "match_fabric").strip()
        target = (context.get("dominant_colors") or "multicolor").strip()
        motifs = (context.get("motif_colors") or "original print").strip()
        if color_mode == "explicit":
            # Base-only recolor — keep compact so CLIP budget retains this layer.
            fabric = (
                f"Change only the base fabric color to {target}; keep original "
                f"{context['pattern']} print colors ({motifs}) and print scale. "
                f"Do not recolor the printed motifs."
            )
        elif appearance:
            fabric = (
                f"Preserve source fabric look ({appearance}; "
                f"{context['dominant_colors']}, {context['pattern']}, "
                f"{context['material']}). Same print scale; do not recolor."
            )
        else:
            fabric = (
                f"Preserve source fabric print/colors ({context['dominant_colors']}; "
                f"{context['material']}, {context['pattern']}, {context['texture']}). "
                f"Same print scale; do not recolor."
            )
        quality = (
            "Natural folds following the print, clean seams and hem, sharp silhouette "
            "on white. No model, not a swatch, not plastic CGI."
        )
        # Avoid "casual casual" when style == occasion
        if context["style"] == context["occasion"]:
            style = f"{context['style']}."
        else:
            style = f"{context['style']} {context['occasion']}."
        return [primary, fabric, quality, style]

    def fit_to_clip_budget(
        self,
        layers: list[str],
        max_tokens: int = CLIP_SAFE_CONTENT_TOKENS,
        fabric_fallback: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Deterministically compact prompt layers to fit CLIP budget.

        Never silently truncate mid-sentence via the tokenizer — drop optional
        trailing layers first so garment attributes stay intact.
        """
        compacted = False
        used: list[str] = []
        for layer in layers:
            candidate = " ".join(used + [layer]).strip()
            count = self.count_clip_tokens(candidate)
            if count <= max_tokens:
                used.append(layer)
            else:
                compacted = True
                break

        if not used:
            # Extreme fallback: garment identity only (still structured, not generic)
            used = [layers[0]]
            compacted = True

        prompt = " ".join(used).strip()
        token_count = self.count_clip_tokens(prompt)

        # If primary+fabric overflow, keep a grounded fabric clause (not a vague slogan).
        if token_count > max_tokens and len(layers) >= 2:
            compacted = True
            short_fabric = fabric_fallback or (
                "Preserve source fabric print, colors, and pattern scale; do not recolor."
            )
            prompt = f"{layers[0]} {short_fabric}".strip()
            if len(layers) > 2:
                with_quality = f"{prompt} {layers[2]}".strip()
                if self.count_clip_tokens(with_quality) <= max_tokens:
                    prompt = with_quality
            token_count = self.count_clip_tokens(prompt)
            while token_count > max_tokens and " " in prompt:
                # Drop trailing words from the last clause only
                parts = prompt.rsplit(" ", 1)
                prompt = parts[0]
                token_count = self.count_clip_tokens(prompt)

        truncated = token_count > CLIP_MAX_TOKENS
        stats = {
            "token_count": token_count,
            "token_budget": CLIP_MAX_TOKENS,
            "prompt_compacted": compacted,
            "truncated": truncated,
        }
        self.logger.info(
            "[FLUX PROMPT]\nToken count: %s / %s\nPrompt compacted: %s\nTruncated: %s",
            token_count,
            CLIP_MAX_TOKENS,
            compacted,
            truncated,
        )
        if truncated:
            self.logger.warning(
                "Prompt still exceeds CLIP budget (%s > %s); tokenizer may truncate.",
                token_count,
                CLIP_MAX_TOKENS,
            )
        return prompt, stats

    def build_prompts(
        self,
        fabric_metadata: Dict[str, Any],
        user_customization: Dict[str, Any],
    ) -> Tuple[str, str]:
        """Construct positive and negative FLUX prompts with vocabulary validation."""
        context = self._build_context(fabric_metadata, user_customization)

        extra_attrs = fabric_metadata.get("additional_attributes") or user_customization.get("additional_attributes")
        extra_str = ""
        if isinstance(extra_attrs, list) and extra_attrs:
            extra_str = ", " + ", ".join(str(a).replace("_", " ") for a in extra_attrs[:3])
        elif isinstance(extra_attrs, str) and extra_attrs.strip():
            extra_str = ", " + extra_attrs.strip().replace("_", " ")

        positive_template = self.templates.get("positive_template", "")
        negative_template = self.templates.get("negative_template", "")

        try:
            positive_prompt = positive_template.format(**context) + extra_str
        except KeyError as exc:
            self.logger.error("Missing template key during prompt formatting: %s", exc)
            positive_prompt = (
                f"Finished wearable {context['gender']} {context['garment_type']}, "
                f"{context['neckline']}, {context['sleeve_length']}, "
                f"{context['material']}, white studio background."
            ) + extra_str

        positive_prompt, stats = self.fit_to_clip_budget([positive_prompt])
        self.last_prompt_stats = stats
        self.logger.info("Parsed garment context: %s", context)
        self.logger.info("Positive prompt: %s", positive_prompt)
        self.logger.info("Negative prompt: %s", negative_template)
        return positive_prompt, negative_template

    def build_kontext_prompt(
        self,
        fabric_metadata: Dict[str, Any],
        user_customization: Dict[str, Any],
    ) -> Tuple[str, str]:
        """
        Build a CLIP-safe FLUX.1-Kontext edit instruction for fabric→garment.

        Prioritizes: garment identity → construction → fabric fidelity → clarity → studio presentation.
        Avoids redundant aesthetic adjectives that waste the 77-token CLIP budget.
        """
        context = self._build_context(fabric_metadata, user_customization)
        negative_prompt = self.templates.get("negative_template", "")

        # Prefer layered compact builder (guarantees attribute priority under CLIP budget).
        # Template remains available for Gradio/legacy callers that want full wording.
        layers = self._build_compact_kontext_layers(context)
        if context.get("color_mode") == "explicit":
            target = context.get("dominant_colors") or "target"
            fabric_fallback = (
                f"Change only the base fabric color to {target}; keep original "
                f"print motif colors."
            )
        else:
            fabric_fallback = (
                "Preserve source fabric print, colors, and pattern scale; do not recolor."
            )
        final_positive, stats = self.fit_to_clip_budget(
            layers, fabric_fallback=fabric_fallback
        )
        self.last_prompt_stats = {
            **stats,
            "color_mode": context.get("color_mode"),
            "dominant_colors": context.get("dominant_colors"),
            "color_instruction": layers[1] if len(layers) > 1 else "",
        }
        self.logger.info(
            "[FLUX COLOR DEBUG] FINAL COLOR INSTRUCTION:\n%s",
            layers[1] if len(layers) > 1 else final_positive,
        )
        self.logger.info("Kontext final prompt: %s", final_positive)
        return final_positive, negative_prompt

    def build_material_garment_prompt(
        self,
        fabric_metadata: Dict[str, Any],
        user_customization: Dict[str, Any],
        fabric_appearance: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Legacy text-only prompt path (not used by current Kontext pipeline)."""
        context = self._build_context(fabric_metadata, user_customization)
        product_prompt, negative_prompt = self.build_prompts(fabric_metadata, user_customization)

        appearance = (fabric_appearance or "").strip() or (
            f"{context['dominant_colors']} {context['pattern']} {context['material']}"
        )
        context_with_fabric = {**context, "fabric_appearance": appearance}

        instruction_template = self.templates.get("material_instruction_template", "")
        try:
            instruction = instruction_template.format(**context_with_fabric)
        except KeyError:
            instruction = (
                f"Create finished wearable {context['gender']} {context['garment_type']} "
                f"from {appearance}. Not a fabric swatch."
            )

        final_positive, stats = self.fit_to_clip_budget([instruction, product_prompt])
        self.last_prompt_stats = stats
        self.logger.info("Material garment prompt: %s", final_positive)
        return final_positive, negative_prompt
