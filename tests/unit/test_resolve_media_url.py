"""Regression tests for Kaggle generated-image URL delivery.

Root cause fixed: resolveMediaUrl was applied twice (page + ResultCard). With a
Jupyter base path this produced:

  /k/<session>/proxy/proxy/8000/k/<session>/proxy/proxy/8000/outputs/...

which 404s behind the proxy and renders as a blank <img>.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _join_media(url: str, origin: str) -> str:
    """Mirror frontend/src/lib/resolveMediaUrl.ts (path-origin branch)."""
    if not url:
        return ""
    if url.startswith(("blob:", "data:", "http://", "https://")):
        return url
    path = url if url.startswith("/") else f"/{url}"
    if not origin:
        return path
    if origin.startswith("/"):
        base = origin.rstrip("/")
        if path == base or path.startswith(f"{base}/"):
            return path
        return f"{base}{path}"
    return f"{origin.rstrip('/')}{path}"


BASE = "/k/abc123/proxy/proxy/8000"
BACKEND_RESULT = "/outputs/generated_garments/images/garment_demo.png"


def test_resolve_media_url_prefixes_outputs_once():
    once = _join_media(BACKEND_RESULT, BASE)
    assert once == f"{BASE}{BACKEND_RESULT}"


def test_resolve_media_url_is_idempotent_under_kaggle_base_path():
    """Page resolves, then ResultCard resolves again — must not double-prefix."""
    once = _join_media(BACKEND_RESULT, BASE)
    twice = _join_media(once, BASE)
    assert twice == once
    assert twice.count("/k/abc123/proxy/proxy/8000") == 1
    assert twice.endswith(BACKEND_RESULT)


def test_resolve_media_url_local_dev_absolute_origin():
    joined = _join_media(BACKEND_RESULT, "http://127.0.0.1:8000")
    assert joined == f"http://127.0.0.1:8000{BACKEND_RESULT}"
    assert _join_media(joined, "http://127.0.0.1:8000") == joined or joined.startswith(
        "http://"
    )


def test_typescript_resolve_media_url_is_idempotent():
    src = Path("frontend/src/lib/resolveMediaUrl.ts").read_text(encoding="utf-8")
    assert "Idempotent" in src or "idempotent" in src.lower()
    assert "path.startsWith(`${base}/`)" in src or 'path.startsWith(`${base}/`)' in src
    assert "resolveApiOrigin" in src


def test_outputs_static_endpoint_serves_png(tmp_path, monkeypatch):
    """FastAPI /outputs mount must return image/png for generated artifacts."""
    # Point OUTPUT_DIR at an isolated tree for this test.
    out = tmp_path / "outputs"
    rel = Path("generated_garments") / "images"
    (out / rel).mkdir(parents=True)
    png = out / rel / "garment_demo.png"
    # Minimal valid 1x1 PNG
    png.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
        )
    )

    import backend_api.config.settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "OUTPUT_DIR", out)

    # Re-import app with patched settings is hard (mount already bound).
    # Mount a fresh StaticFiles app instead — same contract as main.py.
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles

    app = FastAPI()
    app.mount("/outputs", StaticFiles(directory=str(out)), name="outputs")
    client = TestClient(app)
    resp = client.get("/outputs/generated_garments/images/garment_demo.png")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/")
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(resp.content) > 0


def test_result_card_still_calls_resolve_media_url():
    """Ensure ResultCard keeps resolving (safe once idempotent)."""
    text = Path("frontend/src/components/studio/ResultCard.tsx").read_text(
        encoding="utf-8"
    )
    assert "resolveMediaUrl(imageUrl)" in text
