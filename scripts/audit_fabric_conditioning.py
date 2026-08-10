"""Audit fabric→conditioning→prompt stages WITHOUT running FLUX (fast).

Saves stage images + stats for one fabric so we can see color/detail loss
before Kontext.

Usage:
  python scripts/audit_fabric_conditioning.py --image path/to/fabric.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _stats(img: Image.Image) -> dict:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    gx = np.abs(arr[:, 1:, :] - arr[:, :-1, :]).mean()
    gy = np.abs(arr[1:, :, :] - arr[:-1, :, :]).mean()
    return {
        "size": list(img.size),
        "mode": img.mode,
        "mean_rgb": [round(float(x), 2) for x in arr.mean(axis=(0, 1))],
        "std_rgb": [round(float(x), 2) for x in arr.std(axis=(0, 1))],
        "min_rgb": [int(x) for x in arr.reshape(-1, 3).min(axis=0)],
        "max_rgb": [int(x) for x in arr.reshape(-1, 3).max(axis=0)],
        "edge_energy": round(float((gx + gy) / 2), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--garment-type", default="dress")
    parser.add_argument("--sleeve", default="short_sleeve")
    parser.add_argument("--ui-color", default="white", help="Simulated UI color field")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--out", default=str(ROOT / "outputs" / "audit_conditioning"))
    args = parser.parse_args()

    from src.features.custom_generator.inference.fabric_appearance import (
        describe_fabric_appearance,
    )
    from src.features.custom_generator.inference.fabric_conditioning import (
        build_garment_conditioning_image,
    )
    from src.features.custom_generator.prompting.garment_prompt_builder import (
        GarmentPromptBuilder,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fabric = Image.open(args.image).convert("RGB")
    fabric.save(out / "A_original_fabric.png")
    appearance = describe_fabric_appearance(fabric)

    cond = build_garment_conditioning_image(
        fabric,
        garment_type=args.garment_type,
        width=args.size,
        height=args.size,
        sleeve=args.sleeve,
    )
    cond.save(out / "B_conditioning.png")

    # Prompt with CURRENT (buggy) priority: UI color wins
    builder = GarmentPromptBuilder(ROOT / "configs")
    meta_ui = {
        "material": "cotton",
        "dominant_colors": [args.ui_color],
        "pattern": appearance.get("pattern_hint"),
    }
    user_ui = {
        "gender": "women",
        "garment_type": args.garment_type,
        "color": args.ui_color,
        "sleeve": args.sleeve,
        "neckline": "round_neck",
        "fit": "regular",
        "style": "casual",
        "occasion": "casual",
        "season": "summer",
        "material": "cotton",
    }
    # Simulate pipeline enrichment that only fills colors when missing
    meta_enriched = dict(meta_ui)
    if appearance.get("dominant_color_names") and not meta_enriched.get("dominant_colors"):
        meta_enriched["dominant_colors"] = appearance["dominant_color_names"]
    if appearance.get("pattern_hint"):
        meta_enriched["pattern"] = appearance["pattern_hint"]

    prompt_current, _ = builder.build_kontext_prompt(meta_enriched, user_ui)
    stats_current = dict(builder.last_prompt_stats)

    # FIXED priority: pixel appearance colors win; UI color ignored for fabric identity
    meta_fix = {
        "material": "cotton",
        "dominant_colors": appearance.get("dominant_color_names") or [args.ui_color],
        "pattern": appearance.get("pattern_hint") or "printed",
        "color_source": "fabric_pixels",
    }
    user_fix = {**user_ui}
    # Do not let UI color override fabric identity
    user_fix.pop("color", None)
    prompt_fixed, _ = builder.build_kontext_prompt(meta_fix, user_fix)
    stats_fixed = dict(builder.last_prompt_stats)

    report = {
        "fabric_path": str(args.image),
        "appearance": appearance,
        "A_original": _stats(fabric),
        "B_conditioning": _stats(cond),
        "ui_color_simulated": args.ui_color,
        "prompt_current_ui_wins": prompt_current,
        "prompt_current_stats": stats_current,
        "prompt_fixed_fabric_wins": prompt_fixed,
        "prompt_fixed_stats": stats_fixed,
        "diagnosis": {
            "ui_color_overrides_fabric": True,
            "mask_gaussian_blur_radius": 0.4,
            "cover_resize": "LANCZOS cover-crop to generation size",
            "flux_input_size": args.size,
        },
    }
    (out / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote stage images to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
