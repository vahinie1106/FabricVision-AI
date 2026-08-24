"""Gateway readiness must probe Next.js / — never /health."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from backend_api.gateway import (
    _should_skip_proxy,
    check_frontend_upstream,
    frontend_root_url,
)


def test_health_is_not_proxied_to_next():
    assert _should_skip_proxy("/health") is True
    assert _should_skip_proxy("/health/") is True
    assert _should_skip_proxy("/about") is False


def test_frontend_root_url_uses_slash_not_health(monkeypatch):
    monkeypatch.delenv("NEXT_PUBLIC_BASE_PATH", raising=False)
    monkeypatch.setenv("FRONTEND_UPSTREAM", "http://127.0.0.1:3000")
    assert frontend_root_url() == "http://127.0.0.1:3000/"
    assert not frontend_root_url().endswith("/health")


def test_frontend_root_url_respects_base_path(monkeypatch):
    monkeypatch.setenv("FRONTEND_UPSTREAM", "http://127.0.0.1:3000")
    monkeypatch.setenv("NEXT_PUBLIC_BASE_PATH", "/proxy/proxy/8000")
    assert frontend_root_url() == "http://127.0.0.1:3000/proxy/proxy/8000/"


def test_check_frontend_upstream_gets_root(monkeypatch):
    import asyncio

    monkeypatch.delenv("NEXT_PUBLIC_BASE_PATH", raising=False)
    monkeypatch.setenv("FRONTEND_UPSTREAM", "http://127.0.0.1:3000")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("backend_api.gateway.httpx.AsyncClient", return_value=mock_client):
        ok, detail = asyncio.run(check_frontend_upstream())

    assert ok is True
    mock_client.get.assert_awaited_once()
    called_url = mock_client.get.await_args.args[0]
    assert called_url == "http://127.0.0.1:3000/"
    assert "/health" not in called_url
    assert "200" in detail


def test_gateway_health_returns_200_when_frontend_root_ok(monkeypatch):
    monkeypatch.setenv("FLUX_WARMUP_ON_STARTUP", "false")
    from backend_api.main import app

    async def _ok():
        return True, "http://127.0.0.1:3000/ → HTTP 200"

    with patch("backend_api.gateway.check_frontend_upstream", new=_ok):
        client = TestClient(app)
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_gateway_health_returns_502_when_frontend_down(monkeypatch):
    monkeypatch.setenv("FLUX_WARMUP_ON_STARTUP", "false")
    from backend_api.main import app

    async def _bad():
        return False, "Frontend upstream unreachable at http://127.0.0.1:3000"

    with patch("backend_api.gateway.check_frontend_upstream", new=_bad):
        client = TestClient(app)
        resp = client.get("/health")
    assert resp.status_code == 502
    assert b"Frontend upstream unreachable" in resp.content


def test_api_v1_health_still_liveness_without_frontend(monkeypatch):
    """API liveness must not require Next (used by unit/integration clients)."""
    monkeypatch.setenv("FLUX_WARMUP_ON_STARTUP", "false")
    from backend_api.main import app

    client = TestClient(app)
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy", "version": "1.0.0"}


def test_cors_regex_allows_kaggle_and_googleusercontent_hosts():
    import re

    from backend_api.main import CORS_ORIGIN_REGEX

    rx = re.compile(CORS_ORIGIN_REGEX)
    assert rx.fullmatch("https://kkb-production.jupyter-proxy.kaggle.net")
    assert rx.fullmatch("https://foo.kaggleusercontent.com")
    assert rx.fullmatch("https://123.notebooks.googleusercontent.com")
    assert rx.fullmatch("https://example.jupyter-proxy.kaggle.net")
    assert not rx.fullmatch("http://evil.example")
    assert not rx.fullmatch("https://evil.com")


def test_openapi_exposes_real_ai_routes(monkeypatch):
    monkeypatch.setenv("FLUX_WARMUP_ON_STARTUP", "false")
    from backend_api.main import app

    client = TestClient(app)
    spec = client.get("/openapi.json").json()
    paths = spec.get("paths") or {}
    for route in (
        "/api/v1/health",
        "/api/v1/analyze",
        "/api/v1/generate",
        "/api/v1/tryon",
        "/api/v1/status/{job_id}",
    ):
        assert route in paths, route
    assert "post" in paths["/api/v1/generate"]
    assert "post" in paths["/api/v1/tryon"]
    assert "post" in paths["/api/v1/analyze"]
    assert "get" in paths["/api/v1/status/{job_id}"]


def test_rewrite_strips_loopback_location_headers():
    import httpx

    from backend_api.gateway import _rewrite_response_headers

    out = _rewrite_response_headers(
        httpx.Headers({"location": "http://127.0.0.1:3000/about"})
    )
    assert out["location"] == "/about"
    out = _rewrite_response_headers(
        httpx.Headers({"location": "http://127.0.0.1:8000/studio/custom-garment"})
    )
    assert out["location"] == "/studio/custom-garment"

