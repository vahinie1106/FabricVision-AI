"""Portable Hugging Face cache location for Windows + Kaggle.

Fresh Kaggle clones do not include gitignored ``models/`` weights. Without a
stable HF cache root, every notebook restart re-downloads FLUX into a volatile
default cache. Prefer ``/kaggle/working/.cache/huggingface`` on Kaggle so
snapshot blobs persist for the session/working tree.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def is_kaggle_environment() -> bool:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE"):
        return True
    return Path("/kaggle/working").is_dir()


def default_hf_home() -> Optional[Path]:
    """Return an explicit HF_HOME when we should override the platform default."""
    if os.environ.get("HF_HOME", "").strip():
        return Path(os.environ["HF_HOME"]).expanduser()
    if is_kaggle_environment():
        return Path("/kaggle/working/.cache/huggingface")
    return None


def ensure_huggingface_cache_env() -> dict[str, str]:
    """
    Ensure HF_HOME / HUGGINGFACE_HUB_CACHE point at a durable location.

    Does not override variables the operator already set. Safe on Windows (no-op
    when not on Kaggle and env unset — hub keeps using ~/.cache/huggingface).
    """
    applied: dict[str, str] = {}
    home = default_hf_home()
    if home is None:
        # Local/dev: leave defaults (typically ~/.cache/huggingface).
        hub = os.environ.get("HUGGINGFACE_HUB_CACHE", "").strip()
        applied["HF_HOME"] = os.environ.get("HF_HOME", "")
        applied["HUGGINGFACE_HUB_CACHE"] = hub
        return applied

    home.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("HF_HOME", "").strip():
        os.environ["HF_HOME"] = str(home)
        applied["HF_HOME"] = str(home)
    else:
        applied["HF_HOME"] = os.environ["HF_HOME"]

    hub_dir = Path(os.environ["HF_HOME"]) / "hub"
    hub_dir.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("HUGGINGFACE_HUB_CACHE", "").strip():
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_dir)
        applied["HUGGINGFACE_HUB_CACHE"] = str(hub_dir)
    else:
        applied["HUGGINGFACE_HUB_CACHE"] = os.environ["HUGGINGFACE_HUB_CACHE"]

    # Transformers still honors TRANSFORMERS_CACHE on older stacks.
    if not os.environ.get("TRANSFORMERS_CACHE", "").strip():
        tf_dir = Path(os.environ["HF_HOME"]) / "transformers"
        tf_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TRANSFORMERS_CACHE"] = str(tf_dir)
        applied["TRANSFORMERS_CACHE"] = str(tf_dir)

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    return applied


def huggingface_hub_cache_dir() -> Path:
    """Resolved hub cache directory (after ensure_huggingface_cache_env)."""
    ensure_huggingface_cache_env()
    explicit = os.environ.get("HUGGINGFACE_HUB_CACHE", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def hub_repo_cache_dir(repo_id: str) -> Path:
    """``models--org--name`` folder under the hub cache."""
    safe = (repo_id or "").strip().replace("/", "--")
    return huggingface_hub_cache_dir() / f"models--{safe}"
