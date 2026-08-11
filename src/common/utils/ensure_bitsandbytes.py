"""Ensure bitsandbytes is importable with valid package metadata.

Diffusers NF4 packages (e.g. ``eramth/flux-kontext-4bit``) call
``importlib.metadata.version("bitsandbytes")``. A missing install raises
``PackageNotFoundError`` even when CUDA/FLUX weights are fine — common on
fresh Kaggle images.
"""

from __future__ import annotations

import importlib
import logging
import os
import subprocess
import sys
from typing import Optional, Tuple

logger = logging.getLogger("fabricvision.deps.bitsandbytes")

# Latest stable on PyPI as of this fix; wheels cover CUDA 11.8–13.x and pick
# the backend from the installed torch CUDA runtime (Kaggle: torch 2.10+cu128).
DEFAULT_SPEC = os.environ.get("FABRICVISION_BITSANDBYTES_SPEC", "bitsandbytes>=0.45.0").strip()


def probe_bitsandbytes() -> Tuple[bool, str, Optional[str]]:
    """
    Return (ok, detail, version_or_none).

    ``ok`` requires both import and importlib.metadata package version.
    """
    try:
        import bitsandbytes as bnb  # noqa: F401
    except Exception as exc:
        return False, f"import_failed: {type(exc).__name__}: {exc}", None

    try:
        import importlib.metadata as md

        meta_ver = md.version("bitsandbytes")
    except Exception as exc:
        return False, f"metadata_failed: {type(exc).__name__}: {exc}", None

    mod_ver = getattr(sys.modules.get("bitsandbytes"), "__version__", None) or meta_ver
    return True, f"ok import={mod_ver} metadata={meta_ver}", str(meta_ver)


def _purge_partial_imports() -> None:
    for name in list(sys.modules):
        if name == "bitsandbytes" or name.startswith("bitsandbytes."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()


def install_bitsandbytes(spec: Optional[str] = None) -> None:
    """Install/upgrade bitsandbytes via pip for the current interpreter."""
    package_spec = (spec or DEFAULT_SPEC).strip() or "bitsandbytes>=0.45.0"
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        package_spec,
    ]
    logger.info("Installing bitsandbytes: %s", " ".join(cmd))
    print(f"[ensure_bitsandbytes] Running: {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd)
    _purge_partial_imports()


def ensure_bitsandbytes(
    *,
    auto_install: bool = True,
    spec: Optional[str] = None,
) -> str:
    """
    Verify bitsandbytes; optionally pip-install when missing/broken.

    Returns the installed package version string.
    Raises RuntimeError when verification still fails after install.
    """
    ok, detail, version = probe_bitsandbytes()
    if ok and version:
        logger.info("bitsandbytes ready (%s)", detail)
        print(f"[ensure_bitsandbytes] READY {detail}", flush=True)
        return version

    print(f"[ensure_bitsandbytes] MISSING/BROKEN ({detail})", flush=True)
    if not auto_install:
        raise RuntimeError(
            "MODEL_DEPENDENCY_ERROR: bitsandbytes is required for FLUX NF4 "
            f"({detail}). Install with: pip install -U bitsandbytes"
        )

    try:
        install_bitsandbytes(spec=spec)
    except Exception as exc:
        raise RuntimeError(
            "MODEL_DEPENDENCY_ERROR: failed to install bitsandbytes "
            f"({type(exc).__name__}: {exc})"
        ) from exc

    ok, detail, version = probe_bitsandbytes()
    if not ok or not version:
        raise RuntimeError(
            "MODEL_DEPENDENCY_ERROR: bitsandbytes still unavailable after install "
            f"({detail})"
        )

    print(f"[ensure_bitsandbytes] INSTALLED_OK {detail}", flush=True)
    logger.info("bitsandbytes installed (%s)", detail)
    return version
