import json
from pathlib import Path

from src.features.custom_generator.prompting.garment_prompt_builder import (
    DEFAULT_NECKLINE_VISUAL_PHRASES,
    GarmentPromptBuilder,
)


def _builder() -> GarmentPromptBuilder:
    return GarmentPromptBuilder(Path("configs"))


def test_neckline_vocabulary_validation():
    builder = _builder()
    neck_path = Path("configs/semantic_analysis/neckline_vocabulary.json")
    assert neck_path.exists()

    with open(neck_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "allowed_necklines" in data
    assert "round_neck" in data["allowed_necklines"]
    assert "mandarin_collar" in data["allowed_necklines"]
    assert "visual_phrases" in data
    assert set(data["visual_phrases"]) == set(data["allowed_necklines"])

    assert builder.validate_and_normalize_neckline("Mandarin Collar") == "mandarin_collar"
    # Empty/missing input falls back to the documented default.
    assert builder.validate_and_normalize_neckline(None) == "round_neck"
    # Unknown-but-descriptive values are sanitized and kept (not silently
    # collapsed to a default) — see garment_prompt_builder.validate_and_normalize_neckline.
    assert builder.validate_and_normalize_neckline("unknown_neck") == "unknown_neck"


def test_ui_neckline_labels_normalize_to_vocab_tokens():
    builder = _builder()
    expected = {
        "Round Neck": "round_neck",
        "V Neck": "v_neck",
        "U Neck": "u_neck",
        "Square Neck": "square_neck",
        "Boat Neck": "boat_neck",
        "High Neck": "high_neck",
        "Collar Neck": "collar_neck",
        "Mandarin Collar": "mandarin_collar",
        "Off Shoulder": "off_shoulder",
        "Sweetheart Neck": "sweetheart_neck",
        "Halter Neck": "halter_neck",
        "Keyhole Neck": "keyhole_neck",
        "v-neck": "v_neck",
        "crew neck": "round_neck",
    }
    for label, token in expected.items():
        assert builder.validate_and_normalize_neckline(label) == token


def test_each_supported_neckline_maps_to_visual_description():
    builder = _builder()
    vocab = builder.necklines_vocab.get("visual_phrases") or {}
    for token, phrase in DEFAULT_NECKLINE_VISUAL_PHRASES.items():
        visual = builder.neckline_visual_phrase(token)
        assert visual == phrase
        assert vocab.get(token) == phrase
