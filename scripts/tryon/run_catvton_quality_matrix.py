"""CatVTON quality matrix — real person + garment only; one axis at a time.

Usage:
  python scripts/tryon/run_catvton_quality_matrix.py \\
    --person path/to/real_person.jpg \\
    --garment path/to/garment.png \\
    --garment-type dress

  python scripts/tryon/run_catvton_quality_matrix.py --plan-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("fabricvision.catvton_quality_matrix")

MATRIX: list[dict[str, Any]] = [
    {
        "id": "A_baseline_grabcut_30",
        "mask_strategy": "grabcut",
        "steps": 30,
        "guidance": 2.5,
        "width": 384,
        "height": 512,
        "requires_automasker": False,
    },
    {
        "id": "B_automasker_30",
        "mask_strategy": "automasker",
        "steps": 30,
        "guidance": 2.5,
        "width": 384,
        "height": 512,
        "requires_automasker": True,
    },
    {
        "id": "C_automasker_40",
        "mask_strategy": "automasker",
        "steps": 40,
        "guidance": 2.5,
        "width": 384,
        "height": 512,
        "requires_automasker": True,
        "optional": True,
        "note": "Higher steps only after B succeeds with real AutoMasker",
    },
]

MAPPING_CHECKLIST = [
    "upper_shoulders",
    "upper_neckline",
    "upper_sleeves",
    "upper_collar",
    "body_torso_alignment",
    "body_width",
    "body_length",
    "body_side_boundaries",
    "body_hem",
    "fabric_print",
    "fabric_color",
    "fabric_texture",
    "person_face",
    "person_skin",
    "person_hair",
    "person_proportions",
    "artifact_doubled_limbs",
    "artifact_floating_garment",
    "artifact_hands",
    "artifact_face_warp",
    "artifact_background",
    "artifact_bleed",
]


def automasker_dependency_report() -> dict[str, Any]:
    """Probe AutoMasker stack without installing packages."""
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "catvton_requirements": "models/CatVTON/requirements.txt",
        "documented_packages": [
            "fvcore==0.1.5.post20221221",
            "cloudpickle",
            "omegaconf",
            "pycocotools==2.0.8",
        ],
        "do_not": "Blindly pip install random detectron2 wheels; match CUDA/torch carefully.",
    }
    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda"] = torch.version.cuda
        report["gpu"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
        report["physical_vram_mb"] = (
            round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 1)
            if torch.cuda.is_available()
            else 0.0
        )
    except Exception as exc:
        report["torch_error"] = str(exc)

    for mod in ("fvcore", "cloudpickle", "omegaconf"):
        try:
            __import__(mod)
            report[f"{mod}_import"] = "ok"
        except Exception as exc:
            report[f"{mod}_import"] = f"FAIL: {type(exc).__name__}: {exc}"

    # Vendored detectron2 lives under models/CatVTON (not site-packages).
    vendor_root = ROOT / "models" / "CatVTON"
    vendor = vendor_root / "detectron2"
    report["vendored_detectron2_path_exists"] = vendor.exists()
    report["detectron2_import"] = "not_probed"
    if vendor.exists():
        inserted = False
        root_s = str(vendor_root)
        if root_s not in sys.path:
            sys.path.insert(0, root_s)
            inserted = True
        try:
            import detectron2  # noqa: F401

            report["detectron2_import"] = "ok"
            report["detectron2_file"] = getattr(detectron2, "__file__", None)
        except Exception as exc:
            report["detectron2_import"] = f"FAIL: {type(exc).__name__}: {exc}"
        finally:
            if inserted:
                try:
                    sys.path.remove(root_s)
                except ValueError:
                    pass
    else:
        try:
            import detectron2  # noqa: F401

            report["detectron2_import"] = "ok"
        except Exception as exc:
            report["detectron2_import"] = f"FAIL: {type(exc).__name__}: {exc}"

    report["automasker_ready"] = (
        report.get("fvcore_import") == "ok" and report.get("detectron2_import") == "ok"
    )
    if not report["automasker_ready"]:
        report["action"] = (
            "Install CatVTON-documented fvcore/cloudpickle/omegaconf into the project "
            "venv and ensure models/CatVTON/detectron2 is present. Do not downgrade "
            "project torch 2.6+cu124 to CatVTON's pinned 2.4.0 unless using a dedicated env."
        )
    else:
        report["action"] = (
            "Import probe OK. Runtime AutoMasker still needs DensePose/SCHP weights "
            "and may need additional detectron2 native ops — verify with cell B."
        )
    return report


def run_cell(
    cell: dict[str, Any],
    person: Image.Image,
    garment: Image.Image,
    person_path: Path,
    garment_path: Path,
    garment_type: str,
) -> dict[str, Any]:
    from src.features.virtual_tryon.models import (
        GarmentConditioningInput,
        PersonConditioningInput,
    )
    from src.features.virtual_tryon.tryon_pipeline import TryOnConfig, VirtualTryOnPipeline

    os.environ["CATVTON_REQUIRE_REAL"] = "1"
    os.environ["CATVTON_ALLOW_FALLBACK"] = "0"
    os.environ["CATVTON_MASK_STRATEGY"] = cell["mask_strategy"]
    if cell["mask_strategy"] == "automasker":
        os.environ["CATVTON_USE_AUTOMASKER"] = "true"
    else:
        os.environ["CATVTON_USE_AUTOMASKER"] = "false"

    record: dict[str, Any] = {
        "id": cell["id"],
        "configuration": deepcopy(cell),
        "person_path": str(person_path),
        "garment_path": str(garment_path),
        "garment_type": garment_type,
        "mapping_checklist_pending_manual_review": MAPPING_CHECKLIST,
    }
    try:
        config = TryOnConfig(
            height=int(cell["height"]),
            width=int(cell["width"]),
            allow_fallback=False,
            num_inference_steps=int(cell["steps"]),
            guidance_scale=float(cell["guidance"]),
            attn_ckpt_version="vitonhd",
            output_root=str(ROOT / "outputs" / "virtual_tryon"),
            experiments_root=str(ROOT / "experiments"),
        )
        pipeline = VirtualTryOnPipeline(config=config)
        t0 = time.perf_counter()
        result = pipeline.run(
            person_input=PersonConditioningInput(person_image=person),
            garment_input=GarmentConditioningInput(
                garment_image=garment,
                garment_type=garment_type,
            ),
            output_filename=f"catvton_quality_{cell['id']}",
            person_filename=person_path.name,
            garment_filename=garment_path.name,
        )
        infer_s = round(time.perf_counter() - t0, 3)
        meta: dict[str, Any] = {}
        if result.metadata_path and Path(result.metadata_path).exists():
            meta = json.loads(Path(result.metadata_path).read_text(encoding="utf-8"))
        record.update(
            {
                "status": result.status,
                "inference_time_s": infer_s,
                "load_time_s": meta.get("load_time_s"),
                "image_path": result.image_path,
                "metadata_path": result.metadata_path,
                "mask_strategy": meta.get("mask_strategy") or cell["mask_strategy"],
                "mask_attempts": meta.get("mask_attempts"),
                "mask_source": meta.get("mask_source"),
                "resolution": [cell["width"], cell["height"]],
                "steps": cell["steps"],
                "guidance": cell["guidance"],
                "peak_vram_mb": meta.get("peak_vram_mb"),
                "was_real_catvton_used": meta.get("was_real_catvton_used"),
                "was_fallback_used": meta.get("was_fallback_used"),
                "inference_backend": meta.get("inference_backend"),
                "precision": meta.get("precision") or meta.get("dtype"),
            }
        )
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("CatVTON cell %s failed", cell["id"])
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="CatVTON quality matrix")
    parser.add_argument("--person", help="Real person photograph (required unless --plan-only)")
    parser.add_argument("--garment", help="Garment image (required unless --plan-only)")
    parser.add_argument("--garment-type", default="dress")
    parser.add_argument("--only", default="", help="Comma-separated cell ids")
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--output",
        default=str(ROOT / "experiments" / "tryon_results" / "catvton_quality_matrix.json"),
    )
    args = parser.parse_args()

    deps = automasker_dependency_report()
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    planned: list[dict[str, Any]] = []
    for cell in MATRIX:
        if only and cell["id"] not in only:
            continue
        if cell.get("optional") and not args.include_optional and not (
            only and cell["id"] in only
        ):
            planned.append({**cell, "decision": "skipped_optional"})
            continue
        if cell.get("requires_automasker") and not deps.get("automasker_ready"):
            planned.append(
                {
                    **cell,
                    "decision": "skipped_automasker_unavailable",
                    "reason": deps.get("action")
                    or "AutoMasker dependencies not importable",
                }
            )
            continue
        planned.append({**cell, "decision": "run"})

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.plan_only:
        payload = {
            "status": "plan_only",
            "automasker_dependencies": deps,
            "cells": planned,
            "mapping_checklist": MAPPING_CHECKLIST,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 0

    if not args.person or not args.garment:
        print(
            "ERROR: --person and --garment are required for quality runs.\n"
            "Provide a REAL full-/half-body person photo and a garment image.\n"
            "Do not use fabric swatches or examples/person placeholders.\n"
            "Use --plan-only to inspect the matrix without inference.",
            file=sys.stderr,
        )
        # Still write dependency probe for operators.
        out_path.write_text(
            json.dumps(
                {
                    "status": "blocked_missing_person_or_garment",
                    "automasker_dependencies": deps,
                    "cells": planned,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return 2

    person_path = Path(args.person)
    garment_path = Path(args.garment)
    if not person_path.is_file() or not garment_path.is_file():
        print(
            f"ERROR: missing inputs person={person_path.is_file()} garment={garment_path.is_file()}",
            file=sys.stderr,
        )
        return 2

    from src.features.virtual_tryon.person_image_validation import assess_person_image

    person = Image.open(person_path).convert("RGB")
    garment = Image.open(garment_path).convert("RGB")
    ok, reason = assess_person_image(person)
    if not ok:
        print(
            f"ERROR: invalid person image for quality validation: {reason}\n"
            "Refuse fabric sheets / geometric placeholders.",
            file=sys.stderr,
        )
        return 3

    results: list[dict[str, Any]] = []
    for cell in planned:
        if cell.get("decision") != "run":
            results.append(
                {
                    "id": cell["id"],
                    "status": cell["decision"],
                    "reason": cell.get("reason"),
                    "configuration": cell,
                }
            )
            continue
        logger.info("=== Running CatVTON cell %s ===", cell["id"])
        results.append(
            run_cell(
                cell=cell,
                person=person,
                garment=garment,
                person_path=person_path,
                garment_path=garment_path,
                garment_type=args.garment_type,
            )
        )
        out_path.write_text(
            json.dumps(
                {
                    "status": "partial",
                    "person_assessment": reason,
                    "automasker_dependencies": deps,
                    "results": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    payload = {
        "status": "completed",
        "person_path": str(person_path),
        "garment_path": str(garment_path),
        "person_assessment": reason,
        "automasker_dependencies": deps,
        "results": results,
        "mapping_checklist": MAPPING_CHECKLIST,
        "mapping_observations": (
            "Fill after manual inspection of output images; do not claim mapping "
            "improved without viewing the PNGs."
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
