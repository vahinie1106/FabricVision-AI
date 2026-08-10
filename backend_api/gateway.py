"""Reverse-proxy frontend traffic from FastAPI (:8000) to Next.js (:3000).

Keeps API routes, OpenAPI, and /outputs on FastAPI. Used for Kaggle where only
port 8000 is publicly reachable via the Jupyter proxy.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable
from urllib.parse import urljoin

import httpx
from fastapi import FastAPI, Request, Response

logger = logging.getLogger("fabricvision.gateway")

# Hop-by-hop headers must not be forwarded (RFC 7230).
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


def frontend_upstream() -> str:
    return os.environ.get("FRONTEND_UPSTREAM", "http://127.0.0.1:3000").rstrip("/")


def frontend_base_path() -> str:
    """Next.js basePath used at build time (e.g. /proxy/8000 on Kaggle)."""
    return (os.environ.get("NEXT_PUBLIC_BASE_PATH") or "").rstrip("/")


def _should_skip_proxy(path: str) -> bool:
    """Paths owned by FastAPI — never forward to Next.js."""
    if path.startswith("/api/"):
        return True
    if path.startswith("/outputs"):
        return True
    if path in ("/docs", "/redoc", "/openapi.json") or path.startswith("/docs/") or path.startswith("/redoc/"):
        return True
    # FastAPI custom openapi under /api/v1/openapi.json already covered by /api/
    return False


def _forward_path(path: str) -> str:
    """
    Map the path FastAPI received to the path Next.js expects.

    Kaggle's Jupyter proxy strips `/proxy/8000` before the request reaches the
    app, while a Next.js build with basePath=`/proxy/8000` still expects that
    prefix. Re-attach it when configured.
    """
    base = frontend_base_path()
    if not base:
        return path if path else "/"
    if path == base or path.startswith(base + "/"):
        return path
    if path == "/":
        return base + "/"
    return f"{base}{path}"


def _filter_request_headers(headers: Iterable[tuple[str, str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in headers:
        if key.lower() in _HOP_BY_HOP:
            continue
        out[key] = value
    return out


def _filter_response_headers(headers: httpx.Headers) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in headers.items():
        lower = key.lower()
        if lower in _HOP_BY_HOP:
            continue
        # httpx decompresses bodies by default — never forward these.
        if lower in {"content-encoding", "content-length", "content-md5"}:
            continue
        out[key] = value
    return out


def register_frontend_gateway(app: FastAPI) -> None:
    """
    Register a catch-all proxy to Next.js.

    Must be called AFTER API routers and /outputs mount so those keep priority.
    """

    @app.api_route(
        "/{full_path:path}",
        methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        include_in_schema=False,
    )
    async def proxy_frontend(full_path: str, request: Request) -> Response:
        path = request.url.path or "/"
        if _should_skip_proxy(path):
            return Response(status_code=404, content=b"Not Found")

        upstream = frontend_upstream()
        target_path = _forward_path(path)
        query = request.url.query
        url = urljoin(upstream + "/", target_path.lstrip("/"))
        if query:
            url = f"{url}?{query}"

        body = await request.body()
        headers = _filter_request_headers(request.headers.items())
        # Tell Next the original host when useful; keep upstream Host for local Next.
        headers.pop("host", None)

        timeout = httpx.Timeout(120.0, connect=10.0)
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=False,
            ) as client:
                upstream_resp = await client.request(
                    method=request.method,
                    url=url,
                    headers=headers,
                    content=body,
                )
        except httpx.ConnectError:
            logger.warning(
                "Frontend upstream unreachable at %s (is Next.js running on :3000?)",
                upstream,
            )
            return Response(
                status_code=502,
                content=(
                    b"Frontend unavailable. Start Next.js (port 3000) or run "
                    b"scripts/run_kaggle.py so FastAPI can proxy the UI."
                ),
                media_type="text/plain",
            )
        except httpx.RequestError as exc:
            logger.warning("Frontend proxy error for %s: %s", url, exc)
            return Response(
                status_code=502,
                content=f"Frontend proxy error: {exc}".encode("utf-8"),
                media_type="text/plain",
            )

        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers=_filter_response_headers(upstream_resp.headers),
            media_type=upstream_resp.headers.get("content-type"),
        )

    logger.info(
        "Registered frontend gateway → %s (basePath=%s)",
        frontend_upstream(),
        frontend_base_path() or "(none)",
    )
