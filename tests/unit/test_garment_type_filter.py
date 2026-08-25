"""Gender-filtered Garment Type options must follow garment_taxonomy.json."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND_TAXONOMY = ROOT / "configs" / "semantic_analysis" / "garment_taxonomy.json"
FRONTEND_TAXONOMY = ROOT / "frontend" / "src" / "lib" / "garmentTaxonomy.json"
FILTER_TS = ROOT / "frontend" / "src" / "lib" / "garmentTypeFilter.ts"
PAGE_TSX = ROOT / "frontend" / "src" / "app" / "studio" / "custom-garment" / "page.tsx"

DISPLAY_OVERRIDES = {
    "t_shirt": "T-Shirt",
    "polo_shirt": "Polo Shirt",
}


def _display(key: str) -> str:
    if key in DISPLAY_OVERRIDES:
        return DISPLAY_OVERRIDES[key]
    return " ".join(part.capitalize() for part in key.split("_") if part)


def _keys_for_gender(gender: str) -> list[str]:
    data = json.loads(FRONTEND_TAXONOMY.read_text(encoding="utf-8"))
    cats = data["gender_categories"]
    key = gender.strip().lower().replace("-", "_").replace(" ", "_")
    if key in {"men", "male", "mens", "man"}:
        cat = "men"
    elif key in {"women", "female", "womens", "woman"}:
        cat = "women"
    elif key in {"unisex", "neutral"}:
        cat = "unisex"
    else:
        cat = "women"
    block = cats[cat]
    if "items" in block:
        keys = list(block["items"])
    else:
        keys = list(block.get("traditional") or []) + list(block.get("western") or [])
    seen: set[str] = set()
    out: list[str] = []
    for item in keys:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _options(gender: str) -> list[str]:
    return [_display(k) for k in _keys_for_gender(gender)]


GENDER_DEFAULTS = {"men": "Shirt", "women": "Dress", "unisex": "T-Shirt"}


def _resolve(gender: str, current: str) -> str:
    options = _options(gender)
    if current and current in options:
        return current
    key = gender.strip().lower()
    if key in {"men", "male", "mens", "man"}:
        cat = "men"
    elif key in {"unisex", "neutral"}:
        cat = "unisex"
    else:
        cat = "women"
    preferred = GENDER_DEFAULTS[cat]
    if preferred in options:
        return preferred
    return options[0] if options else ""


def test_frontend_taxonomy_matches_backend_gender_categories():
    backend = json.loads(BACKEND_TAXONOMY.read_text(encoding="utf-8"))
    frontend = json.loads(FRONTEND_TAXONOMY.read_text(encoding="utf-8"))
    assert frontend["gender_categories"] == backend["gender_categories"]


def test_women_options_are_women_only():
    options = _options("Women")
    assert "Dress" in options
    assert "Kurti" in options
    assert "Saree" in options
    assert "Blouse" in options
    assert "Top" in options
    assert "Skirt" in options
    assert "Shirt" not in options
    assert "Kurta" not in options
    assert "Polo Shirt" not in options
    assert "T-Shirt" not in options
    assert options[0] == "Saree"


def test_men_options_are_men_only():
    options = _options("Men")
    assert "Shirt" in options
    assert "T-Shirt" in options
    assert "Kurta" in options
    assert "Polo Shirt" in options
    assert "Dress" not in options
    assert "Saree" not in options
    assert "Kurti" not in options
    assert "Lehenga" not in options
    assert options[0] == "Shirt"


def test_unisex_options_come_from_taxonomy():
    options = _options("Unisex")
    assert "T-Shirt" in options
    assert "Hoodie" in options
    assert "Dress" not in options
    assert "Shirt" not in options


def test_invalid_garment_resets_when_gender_changes():
    assert _resolve("Women", "Dress") == "Dress"
    assert _resolve("Men", "Dress") == "Shirt"
    assert _resolve("Men", "Jacket") == "Jacket"
    assert _resolve("Women", "Jacket") == "Jacket"
    assert _resolve("Women", "Shirt") == "Dress"
    assert _resolve("Unisex", "Dress") == "T-Shirt"


def test_filter_module_is_data_driven():
    text = FILTER_TS.read_text(encoding="utf-8")
    assert 'from "./garmentTaxonomy.json"' in text
    assert "export function garmentsForGender" in text
    assert "export function resolveGarmentTypeForGender" in text
    assert '["Dress", "Shirt"' not in text


def test_custom_garment_page_filters_by_gender():
    page = PAGE_TSX.read_text(encoding="utf-8")
    assert "garmentsForGender" in page
    assert "resolveGarmentTypeForGender" in page
    assert "handleGenderChange" in page
    assert "options={garmentOptions}" in page
    assert '["Dress", "Shirt", "Trousers", "Jacket", "Kurti"' not in page
    assert 'from "@/lib/garmentTypeFilter"' in page or "from '@/lib/garmentTypeFilter'" in page
