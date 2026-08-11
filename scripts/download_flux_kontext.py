"""Download FLUX.1-Kontext weights for FabricVision-AI.

Default: NF4 Diffusers package suitable for RTX 3050 6GB (eramth/flux-kontext-4bit).
Official full weights: black-forest-labs/FLUX.1-Kontext-dev (requires HF license accept).

Usage:
  .\\venv\\Scripts\\python.exe scripts\\download_flux_kontext.py
  .\\venv\\Scripts\\python.exe scripts\\download_flux_kontext.py --official
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official",
        action="store_true",
        help="Download official black-forest-labs/FLUX.1-Kontext-dev (large, gated).",
    )
    parser.add_argument(
        "--out",
        default="models/flux-kontext",
        help="Local directory for weights",
    )
    args = parser.parse_args()

    # Avoid flaky XET CDN failures on some networks
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    from huggingface_hub import snapshot_download

    repo = (
        "black-forest-labs/FLUX.1-Kontext-dev"
        if args.official
        else "eramth/flux-kontext-4bit"
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo} -> {out.resolve()}")
    path = snapshot_download(
        repo_id=repo,
        local_dir=str(out),
        max_workers=2,
        resume_download=True,
    )
    print("DONE", path)

    # Fail loudly if the tree is still incomplete (e.g. LFS pointers only).
    import sys

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.features.custom_generator.model.flux_model_loader import FLUXModelLoader

    loader = FLUXModelLoader(model_path=out, hf_model_id=repo)
    report = loader.preflight_validate_package(out, source=repo)
    if not report["ready"]:
        print("PREFLIGHT_FAIL", report)
        return 1
    print("PREFLIGHT_PASS", report["path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
