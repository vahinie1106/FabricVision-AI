import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger("fabricvision.taxonomy")
workspace_root = Path(__file__).resolve().parents[3]

def load_fashion_taxonomy() -> Dict[str, List[str]]:
    """Dynamically load controlled fashion taxonomy vocabularies from config file."""
    taxonomy_file = workspace_root / "configs" / "fashion_taxonomy.json"
    if taxonomy_file.exists():
        try:
            with open(taxonomy_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Failed to load fashion taxonomy JSON (%s); using defaults", exc)

    return {
        "genders": ["Female", "Male", "Unisex"],
        "garment_types": [
            "T-Shirt", "Shirt", "Kurta", "Hoodie", "Blazer",
            "Dress", "Saree", "Lehenga", "Skirt", "Jeans", "Trousers", "Top"
        ],
        "color_palettes": [
            "Black", "White", "Red", "Blue", "Green", "Yellow", "Pink",
            "Purple", "Orange", "Brown", "Beige", "Grey", "Navy Blue",
            "Pastel", "Monochrome", "Multicolor", "Metallic"
        ],
        "necklines": [
            "Round Neck", "V Neck", "U Neck", "Boat Neck", "Square Neck",
            "Sweetheart Neck", "Halter Neck", "High Neck", "Collar Neck",
            "Off Shoulder", "One Shoulder"
        ],
        "sleeve_lengths": [
            "Sleeveless", "Cap Sleeve", "Short Sleeve", "Half Sleeve",
            "Three Quarter Sleeve", "Full Sleeve", "Bell Sleeve", "Puff Sleeve", "Balloon Sleeve"
        ],
        "fabrics": [
            "Cotton", "Silk", "Linen", "Denim", "Wool", "Velvet",
            "Chiffon", "Polyester", "Leather", "Satin", "Rayon"
        ],
        "styles": [
            "Casual", "Formal", "Traditional", "Party Wear",
            "Streetwear", "Office Wear", "Wedding Wear", "Luxury Fashion"
        ],
        "fits": [
            "Slim Fit", "Regular Fit", "Oversized", "Loose Fit", "Body Fit"
        ],
        "occasions": [
            "Casual", "Party", "Work", "Festival", "Wedding", "Sport"
        ],
    }
