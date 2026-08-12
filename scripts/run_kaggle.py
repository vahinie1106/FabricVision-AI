#!/usr/bin/env python3
"""Start FabricVision-AI for Kaggle / local deployment.

Default Kaggle mode = SPLIT notebook ports (recommended):
  Browser UI  → Kaggle /proxy/3000/  → Next.js :3000
  Browser API → Kaggle /proxy/8000/  → FastAPI  :8000
  API base    → /proxy/8000/api/v1

Optional gateway mode (--gateway):
  Browser → Kaggle …/proxy/proxy/8000 → FastAPI gateway → Next :3000
  (Jupyter base_url already ends in /proxy/, so jupyter-server-proxy
   yields the intentional two-layer /proxy/proxy/<port> path.)

Startup order (always):
  1. Configure env + HF cache + deps
  2. Start FastAPI (FLUX warms in-process in the background)
  3. Wait until /health + /api/v1/health respond
  4. Start Next.js
  5. Wait until frontend / returns 200
  6. Monitor /api/v1/flux-status until READY
  7. Print final URLs and keep services alive

Usage (from repo root):
  python scripts/run_kaggle.py
  python scripts/run_kaggle.py --skip-build
  python scripts/run_kaggle.py --gateway          # single-port public gateway
  python scripts/run_kaggle.py --prefetch-flux   # optional disk prep BEFORE services
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
BASE_PATH_MARKER = FRONTEND / ".next" / "fabricvision-base-path.txt"
KAGGLE_API_LOG = ROOT / "kaggle_api.log"
BACKEND_LOG = Path(os.environ.get("FABRICVISION_BACKEND_LOG", "/tmp/fabricvision_backend.log"))
FRONTEND_LOG = Path(os.environ.get("FABRICVISION_FRONTEND_LOG", "/tmp/fabricvision_frontend.log"))
# On Windows /tmp may not exist — fall back to repo logs/.
if os.name == "nt":
    _log_dir = ROOT / "logs"
    _log_dir.mkdir(parents=True, exist_ok=True)
    if str(BACKEND_LOG).replace("\\", "/").startswith("/tmp/"):
        BACKEND_LOG = _log_dir / "fabricvision_backend.log"
    if str(FRONTEND_LOG).replace("\\", "/").startswith("/tmp/"):
        FRONTEND_LOG = _log_dir / "fabricvision_frontend.log"
KAGGLE_PROXY_HOST_DEFAULT = "https://kkb-production.jupyter-proxy.kaggle.net"


def _log(msg: str) -> None:
    print(f"[run_kaggle] {msg}", flush=True)


def is_kaggle_environment() -> bool:
    if Path("/kaggle").exists():
        return True
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return True
    if os.environ.get("KAGGLE_URL"):
        return True
    return False


def read_jupyter_base_url() -> Optional[str]:
    """Return the active Jupyter server base_url, or None if unavailable."""
    try:
        from jupyter_server.serverapp import list_running_servers
    except Exception as exc:
        _log(f"jupyter_server not importable ({exc})")
        return None

    try:
        servers = list(list_running_servers())
    except Exception as exc:
        _log(f"list_running_servers() failed ({exc})")
        return None

    if not servers:
        _log("No running Jupyter servers discovered")
        return None

    # Prefer a server whose base_url looks like a Kaggle notebook proxy path.
    ranked = sorted(
        servers,
        key=lambda s: (
            0 if "/k/" in str(s.get("base_url") or "") else 1,
            str(s.get("base_url") or ""),
        ),
    )
    base = str(ranked[0].get("base_url") or "").strip()
    if not base:
        return None
    if not base.startswith("/"):
        base = "/" + base
    if not base.endswith("/"):
        base += "/"
    return base


def _normalize_jupyter_base_url(jupyter_base_url: str) -> str:
    """Normalize Jupyter base_url to an absolute directory path ending with '/'."""
    base = (jupyter_base_url or "/").strip() or "/"
    if not base.startswith("/"):
        base = "/" + base
    if not base.endswith("/"):
        base += "/"
    return base


def _normalize_public_path(path: str) -> str:
    """Normalize a public path; keep intentional /proxy/proxy/<port> intact."""
    out = (path or "").strip() or "/"
    if not out.startswith("/"):
        out = "/" + out
    # Collapse accidental triple+ proxy segments only.
    while "/proxy/proxy/proxy/" in out:
        out = out.replace("/proxy/proxy/proxy/", "/proxy/proxy/")
    return out.rstrip("/")


def read_jupyter_token() -> Optional[str]:
    """Return the Jupyter server token if available (helps authenticated public probes)."""
    try:
        from jupyter_server.serverapp import list_running_servers
    except Exception:
        return None
    try:
        servers = list(list_running_servers())
    except Exception:
        return None
    if not servers:
        return None
    ranked = sorted(
        servers,
        key=lambda s: (
            0 if "/k/" in str(s.get("base_url") or "") else 1,
            str(s.get("base_url") or ""),
        ),
    )
    token = str(ranked[0].get("token") or "").strip()
    return token or None


def candidate_public_paths(jupyter_base_url: str, port: int = 8000) -> List[str]:
    """
    Ordered public path candidates for a local port behind Kaggle/Jupyter.

    jupyter-server-proxy exposes ports at ``{base_url}proxy/{port}/``.
    When Kaggle's Jupyter ``base_url`` is already ``/k/<session>/proxy/``, that
    yields the intentional two-layer path:

      /k/<session>/proxy/proxy/<port>

    Single-layer ``/k/<session>/proxy/<port>`` is also probed in case the edge
    maps differently. Host-root ``/proxy/<port>`` is last (legacy).
    """
    port_s = str(int(port))
    base = _normalize_jupyter_base_url(jupyter_base_url).rstrip("/")
    ordered: List[str] = []

    def add(path: str) -> None:
        norm = _normalize_public_path(path)
        if norm and norm not in ordered:
            ordered.append(norm)

    if base.endswith("/proxy") or base == "/proxy":
        # Primary: jupyter-server-proxy under the Jupyter tunnel prefix.
        add(f"{base}/proxy/{port_s}")
        # Fallback previously used (often 404 on current Kaggle).
        add(f"{base}/{port_s}")
    else:
        add(f"{base}/proxy/{port_s}")

    # Session path without the Jupyter /proxy segment: /k/<session>/<port>
    if base.endswith("/proxy"):
        session = base[: -len("/proxy")]
        if session:
            add(f"{session}/{port_s}")

    add(f"/proxy/{port_s}")
    add(f"/proxy/proxy/{port_s}")
    return ordered


def _urls_for_public_path(host: str, public_path: str) -> Dict[str, str]:
    host = host.rstrip("/")
    path = _normalize_public_path(public_path)
    return {
        "public_path": path,
        "public_url": f"{host}{path}/",
        "about_url": f"{host}{path}/about",
        "docs_url": f"{host}{path}/docs",
        "health_url": f"{host}{path}/api/v1/health",
        "projects_url": f"{host}{path}/projects",
        "studio_garment_url": f"{host}{path}/studio/custom-garment",
        "studio_tryon_url": f"{host}{path}/studio/virtual-tryon",
        "studio_semantic_url": f"{host}{path}/studio/semantic-analysis",
    }


def build_public_proxy_info(
    jupyter_base_url: str,
    port: int = 8000,
    proxy_host: Optional[str] = None,
    public_path: Optional[str] = None,
) -> Dict[str, str]:
    """
    Construct browser-facing path/URL for a port behind Jupyter Server Proxy.

    Default (before live probe) prefers jupyter-server-proxy semantics:

      base_url    = /k/<session>/proxy/
      public_path = /k/<session>/proxy/proxy/8000

    Live discovery may override ``public_path`` after probing the public host.
    """
    host = (
        proxy_host
        or os.environ.get("KAGGLE_JUPYTER_PROXY_HOST")
        or KAGGLE_PROXY_HOST_DEFAULT
    ).rstrip("/")

    override = (os.environ.get("KAGGLE_PUBLIC_PATH") or "").strip()
    if public_path:
        path = _normalize_public_path(public_path)
    elif override:
        path = _normalize_public_path("/" + override.strip("/"))
    else:
        candidates = candidate_public_paths(jupyter_base_url, port=port)
        path = candidates[0] if candidates else f"/proxy/{int(port)}"

    info = _urls_for_public_path(host, path)
    info["jupyter_base_url"] = _normalize_jupyter_base_url(jupyter_base_url)
    info["proxy_host"] = host
    info["candidates"] = ",".join(candidate_public_paths(jupyter_base_url, port=port))
    return info


def _is_kaggle_edge_404(status: int, body: str) -> bool:
    text = (body or "").strip().lower()
    return status == 404 and ("404 page not found" in text or text == "not found")


def _looks_like_app_response(status: int, body: str, content_hint: str = "") -> bool:
    if status != 200:
        return False
    lower = (body or "").lower()
    hint = (content_hint or "").lower()
    if "404 page not found" in lower:
        return False
    if "application/json" in hint and ("status" in lower or "ok" in lower or "{" in lower):
        return True
    if "text/html" in hint or "<!doctype html>" in lower or "<html" in lower:
        return True
    if "fastapi" in lower or "swagger" in lower or "fabricvision" in lower:
        return True
    # Health JSON without content-type in our thin probe
    if '"status"' in lower or lower.strip().startswith("{"):
        return True
    return False


def _http_probe(url: str, timeout: float = 12.0, token: Optional[str] = None) -> Dict[str, Any]:
    """GET url; return status/body/headers (does not follow forever)."""
    target = url
    if token and "token=" not in url:
        sep = "&" if "?" in url else "?"
        target = f"{url}{sep}token={token}"
    out: Dict[str, Any] = {
        "url": target,
        "status": 0,
        "body": "",
        "content_type": "",
        "location": None,
        "error": None,
    }
    try:
        req = urllib.request.Request(
            target,
            method="GET",
            headers={"User-Agent": "FabricVision-run_kaggle/1.0", "Accept": "*/*"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out["status"] = int(getattr(resp, "status", 200))
            out["content_type"] = resp.headers.get("Content-Type") or ""
            out["location"] = resp.headers.get("Location")
            out["body"] = resp.read(2_000_000).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        out["status"] = int(exc.code)
        out["content_type"] = (exc.headers.get("Content-Type") if exc.headers else "") or ""
        out["location"] = exc.headers.get("Location") if exc.headers else None
        try:
            out["body"] = exc.read(64_000).decode("utf-8", errors="replace")
        except Exception:
            out["body"] = str(exc.reason)
        out["error"] = f"HTTPError {exc.code}"
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def discover_working_public_path(
    jupyter_base_url: str,
    port: int = 8000,
    proxy_host: Optional[str] = None,
    token: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Probe public candidate URLs and return the first path that reaches our app
    on /, /about, /docs, and /api/v1/health (all HTTP 200).
    """
    host = (
        proxy_host
        or os.environ.get("KAGGLE_JUPYTER_PROXY_HOST")
        or KAGGLE_PROXY_HOST_DEFAULT
    ).rstrip("/")
    token = token or read_jupyter_token()
    suffixes = [
        ("root", ""),
        ("about", "about"),
        ("docs", "docs"),
        ("health", "api/v1/health"),
    ]
    report: List[Dict[str, Any]] = []

    for path in candidate_public_paths(jupyter_base_url, port=port):
        urls = _urls_for_public_path(host, path)
        _log(f"PUBLIC probe candidate: {urls['public_url']}")
        endpoint_status: Dict[str, int] = {}
        ok_all = True
        details: Dict[str, Any] = {}
        for name, suffix in suffixes:
            base = urls["public_url"] if not suffix else f"{host}{path}/{suffix}"
            probe = _http_probe(base, timeout=15.0, token=token)
            endpoint_status[name] = int(probe.get("status") or 0)
            details[name] = {
                "status": probe.get("status"),
                "content_type": probe.get("content_type"),
                "location": probe.get("location"),
                "body_preview": (probe.get("body") or "")[:160],
                "edge_404": _is_kaggle_edge_404(
                    int(probe.get("status") or 0), probe.get("body") or ""
                ),
            }
            status = int(probe.get("status") or 0)
            body = probe.get("body") or ""
            ctype = probe.get("content_type") or ""
            if status != 200 or _is_kaggle_edge_404(status, body):
                ok_all = False
            elif name == "health" and not (
                _looks_like_app_response(status, body, ctype) or "{" in body
            ):
                ok_all = False
            elif name in {"root", "about", "docs"} and not _looks_like_app_response(
                status, body, ctype
            ):
                # Some Swagger pages are HTML without doctype in first bytes — allow 200 docs.
                if name != "docs":
                    ok_all = False
            _log(
                f"  {name}: HTTP {status} "
                f"edge404={details[name]['edge_404']} "
                f"ctype={ctype!r}"
            )

        entry = {
            "path": path,
            "urls": urls,
            "endpoint_status": endpoint_status,
            "ok": ok_all,
            "details": details,
        }
        report.append(entry)
        if ok_all:
            _log(f"PUBLIC winner: {path}")
            return {
                "public_path": path,
                "proxy_host": host,
                "urls": urls,
                "endpoint_status": endpoint_status,
                "report": report,
            }

    _log("No public candidate reached the app on all required endpoints.")
    return {"public_path": None, "proxy_host": host, "urls": None, "report": report}


def detect_deployment(port: int = 8000) -> Dict[str, Any]:
    """
    Decide base_path + optional public Kaggle URLs for this runtime.
    Never defaults to a hard-coded '/proxy/8000'.
    """
    info: Dict[str, Any] = {
        "is_kaggle": is_kaggle_environment(),
        "jupyter_base_url": None,
        "base_path": "",
        "public_url": None,
        "docs_url": None,
        "about_url": None,
        "health_url": None,
        "proxy_host": None,
        "candidates": [],
    }

    if not info["is_kaggle"]:
        return info

    # Prefer env hints when Jupyter introspection is unavailable.
    for env_key in (
        "JUPYTERHUB_BASE_URL",
        "JUPYTER_BASE_URL",
        "NB_PREFIX",
        "KAGGLE_BASE_URL",
    ):
        val = (os.environ.get(env_key) or "").strip()
        if val and "/k/" in val:
            _log(f"Found env {env_key}={val!r}")

    jupyter_base = read_jupyter_base_url()
    if not jupyter_base:
        for env_key in ("JUPYTERHUB_BASE_URL", "JUPYTER_BASE_URL", "NB_PREFIX"):
            val = (os.environ.get(env_key) or "").strip()
            if val:
                jupyter_base = _normalize_jupyter_base_url(val)
                _log(f"Using {env_key} as jupyter base: {jupyter_base!r}")
                break

    if not jupyter_base:
        _log(
            "WARNING: On Kaggle but could not read Jupyter base_url. "
            "Set KAGGLE_PUBLIC_PATH or NEXT_PUBLIC_BASE_PATH manually."
        )
        env_path = (
            os.environ.get("NEXT_PUBLIC_BASE_PATH")
            or os.environ.get("KAGGLE_PUBLIC_PATH")
            or ""
        ).strip()
        if env_path:
            info["base_path"] = "/" + env_path.strip("/")
        return info

    public = build_public_proxy_info(jupyter_base, port=port)
    info.update(
        {
            "jupyter_base_url": public["jupyter_base_url"],
            "base_path": public["public_path"],
            "public_url": public["public_url"],
            "docs_url": public["docs_url"],
            "about_url": public["about_url"],
            "health_url": public["health_url"],
            "proxy_host": public["proxy_host"],
            "candidates": candidate_public_paths(jupyter_base, port=port),
        }
    )
    return info


def apply_public_path(deploy: Dict[str, Any], public_path: str) -> Dict[str, Any]:
    """Update deploy dict + return new base_path for a discovered public path."""
    host = (deploy.get("proxy_host") or KAGGLE_PROXY_HOST_DEFAULT).rstrip("/")
    urls = _urls_for_public_path(host, public_path)
    deploy["base_path"] = urls["public_path"]
    deploy["public_url"] = urls["public_url"]
    deploy["about_url"] = urls["about_url"]
    deploy["docs_url"] = urls["docs_url"]
    deploy["health_url"] = urls["health_url"]
    deploy["projects_url"] = urls["projects_url"]
    deploy["studio_garment_url"] = urls["studio_garment_url"]
    deploy["studio_tryon_url"] = urls["studio_tryon_url"]
    deploy["studio_semantic_url"] = urls["studio_semantic_url"]
    return deploy


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def _http_status(url: str, timeout: float = 8.0) -> tuple[int, str]:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = int(getattr(resp, "status", 200))
            # Need enough bytes to see /_next/static script tags (not just <head>).
            body = resp.read(2_000_000).decode("utf-8", errors="replace")
            return code, body
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read(64_000).decode("utf-8", errors="replace")
        except Exception:
            body = str(exc.reason)
        return int(exc.code), body
    except Exception as exc:
        return 0, str(exc)


def _http_ok(url: str, timeout: float = 5.0) -> tuple[bool, int, str]:
    code, body = _http_status(url, timeout=timeout)
    return 200 <= code < 400, code, body


def _pids_on_port(port: int) -> List[int]:
    pids: List[int] = []
    if sys.platform.startswith("win"):
        try:
            out = subprocess.check_output(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    (
                        f"Get-NetTCPConnection -LocalPort {port} -State Listen "
                        f"-ErrorAction SilentlyContinue | "
                        f"Select-Object -ExpandProperty OwningProcess"
                    ),
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            for line in out.splitlines():
                line = line.strip()
                if line.isdigit():
                    pids.append(int(line))
        except Exception:
            pass
        return sorted(set(pids))

    try:
        out = subprocess.check_output(
            ["bash", "-lc", f"lsof -t -iTCP:{port} -sTCP:LISTEN 2>/dev/null || true"],
            text=True,
        )
        for line in out.splitlines():
            line = line.strip()
            if line.isdigit():
                pids.append(int(line))
    except Exception:
        pass
    if not pids:
        try:
            out = subprocess.check_output(
                ["bash", "-lc", f"fuser -n tcp {port} 2>/dev/null || true"],
                text=True,
            )
            for tok in out.replace("\n", " ").split():
                tok = tok.strip().rstrip("m")
                if tok.isdigit():
                    pids.append(int(tok))
        except Exception:
            pass
    return sorted(set(pids))


def _kill_pids(pids: List[int]) -> None:
    for pid in pids:
        if pid == os.getpid():
            continue
        try:
            if sys.platform.startswith("win"):
                subprocess.check_call(
                    ["taskkill", "/PID", str(pid), "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                os.kill(pid, signal.SIGTERM)
        except Exception as exc:
            _log(f"Could not stop PID {pid}: {exc}")


def stop_port(port: int, label: str) -> None:
    pids = _pids_on_port(port)
    if not pids:
        _log(f"{label} port {port}: free")
        return
    _log(f"{label} port {port}: stopping PIDs {pids}")
    _kill_pids(pids)
    for _ in range(20):
        if not _port_open("127.0.0.1", port):
            break
        time.sleep(0.25)


def _frontend_env(base_path: str, *, split_proxy: bool = False) -> dict:
    """Env for Next build/start — never bake loopback API URLs into the client."""
    env = os.environ.copy()
    if split_proxy or not base_path:
        # Split Kaggle ports: UI on :3000 (/proxy/3000), API on :8000 (/proxy/8000).
        # apiConfig.ts preserves /proxy/8000 when the page is on /proxy/3000.
        env["NEXT_PUBLIC_USE_SAME_ORIGIN"] = "false"
        env["NEXT_PUBLIC_API_URL"] = "/proxy/8000/api/v1"
        env["NEXT_PUBLIC_API_BASE_URL"] = "/proxy/8000/api/v1"
        env["NEXT_PUBLIC_BACKEND_URL"] = "/proxy/8000"
        env["NEXT_PUBLIC_API_ORIGIN"] = "/proxy/8000"
        env.pop("NEXT_PUBLIC_BASE_PATH", None)
    else:
        # Gateway mode: UI+API same public prefix (…/proxy/proxy/8000).
        env["NEXT_PUBLIC_USE_SAME_ORIGIN"] = "true"
        env["NEXT_PUBLIC_API_URL"] = "/api/v1"
        env["NEXT_PUBLIC_API_BASE_URL"] = "/api/v1"
        env["NEXT_PUBLIC_API_ORIGIN"] = ""
        env["NEXT_PUBLIC_BACKEND_URL"] = ""
        env["NEXT_PUBLIC_BASE_PATH"] = base_path
    return env


def write_frontend_dotenv(base_path: str, *, split_proxy: bool) -> Path:
    """Write frontend/.env.local so Next embeds the correct public API base."""
    env = _frontend_env(base_path, split_proxy=split_proxy)
    keys = (
        "NEXT_PUBLIC_API_URL",
        "NEXT_PUBLIC_API_BASE_URL",
        "NEXT_PUBLIC_BACKEND_URL",
        "NEXT_PUBLIC_API_ORIGIN",
        "NEXT_PUBLIC_USE_SAME_ORIGIN",
        "NEXT_PUBLIC_BASE_PATH",
    )
    lines = [
        "# Generated by scripts/run_kaggle.py — do not commit secrets.",
        "# NEXT_PUBLIC_* values are embedded into the client bundle at build time.",
    ]
    for key in keys:
        if key in env and env[key] is not None:
            lines.append(f"{key}={env[key]}")
    path = FRONTEND / ".env.local"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log(
        f"Wrote {path} "
        f"(split_proxy={split_proxy} API_URL={env.get('NEXT_PUBLIC_API_URL')!r} "
        f"BASE_PATH={env.get('NEXT_PUBLIC_BASE_PATH', '')!r})"
    )
    return path


def http_ok(url: str, *, timeout: float = 2.5) -> bool:
    code, _ = _http_status(url, timeout=timeout)
    return code == 200


def backend_already_healthy() -> bool:
    return http_ok("http://127.0.0.1:8000/api/v1/health") and http_ok(
        "http://127.0.0.1:8000/docs"
    )


def frontend_already_healthy(base_path: str = "") -> bool:
    root = f"http://127.0.0.1:3000{base_path}/" if base_path else "http://127.0.0.1:3000/"
    return http_ok(root)


def extract_next_static_refs(html: str) -> List[str]:
    """Collect /_next/static asset paths referenced by HTML."""
    import re

    refs: List[str] = []
    for match in re.finditer(
        r"""(?:src|href)=["']([^"']*?/_next/static/[^"']+)["']""",
        html,
        flags=re.IGNORECASE,
    ):
        ref = match.group(1).strip()
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def assert_html_next_assets_use_base_path(url: str, base_path: str) -> List[str]:
    """
    Ensure HTML script/link tags target {basePath}/_next/static/... when a
    public prefix is configured. Bare /_next/... behind Kaggle proxy loads from
    the wrong host root → JS never runs → blank/infinite load UI.
    """
    code, body = _http_status(url, timeout=15.0)
    if code != 200:
        raise RuntimeError(f"HTML asset check failed for {url}: HTTP {code}")
    refs = extract_next_static_refs(body)
    if not refs:
        # App Router may stream; still require at least one static ref for / pages.
        raise RuntimeError(
            f"HTML from {url} contains no /_next/static asset references — "
            "Next.js client bundle will not load in the browser."
        )
    base = (base_path or "").rstrip("/")
    if base:
        bare = [r for r in refs if r.startswith("/_next/")]
        if bare:
            raise RuntimeError(
                f"HTML from {url} references bare /_next assets without basePath "
                f"{base!r}: {bare[:3]}. Rebuild with NEXT_PUBLIC_BASE_PATH set — "
                "otherwise the public Kaggle site appears to load forever."
            )
        mismatched = [
            r
            for r in refs
            if r.startswith("/") and not r.startswith(base + "/") and not r.startswith("http")
        ]
        if mismatched:
            raise RuntimeError(
                f"HTML from {url} has /_next assets outside basePath {base!r}: "
                f"{mismatched[:3]}"
            )
        _log(
            f"OK HTML /_next assets use basePath {base}: "
            f"{len(refs)} refs (e.g. {refs[0]})"
        )
    else:
        _log(f"OK HTML /_next assets present ({len(refs)} refs, no basePath)")
    return refs


def assert_public_next_static_reachable(
    *,
    public_url: str,
    base_path: str,
    proxy_host: str,
    token: Optional[str] = None,
) -> None:
    """Fetch public HTML, then GET the first JS/CSS chunk via the public host."""
    probe = _http_probe(public_url, timeout=25.0, token=token)
    status = int(probe.get("status") or 0)
    body = str(probe.get("body") or "")
    if status != 200:
        raise RuntimeError(
            f"Public HTML not reachable for asset check: {public_url} → HTTP {status}"
        )
    refs = extract_next_static_refs(body)
    if not refs:
        # Fall back to local gateway HTML (same Next build) if public body is truncated.
        refs = assert_html_next_assets_use_base_path("http://127.0.0.1:8000/", base_path)
        _log("PUBLIC HTML had no /_next refs in body — using local gateway HTML refs")
    else:
        base = (base_path or "").rstrip("/")
        if base:
            bare = [r for r in refs if r.startswith("/_next/")]
            if bare:
                raise RuntimeError(
                    f"PUBLIC HTML references bare /_next without basePath {base!r}: "
                    f"{bare[:3]}"
                )
            _log(
                f"OK PUBLIC HTML /_next assets use basePath {base}: "
                f"{len(refs)} refs (e.g. {refs[0]})"
            )

    host = proxy_host.rstrip("/")
    checked = 0
    for ref in refs:
        if ref.startswith("http://") or ref.startswith("https://"):
            asset_url = ref
        elif ref.startswith("/"):
            asset_url = f"{host}{ref}"
        else:
            asset_url = f"{public_url.rstrip('/')}/{ref.lstrip('/')}"
        if "/_next/static/" not in asset_url:
            continue
        asset_probe = _http_probe(asset_url, timeout=25.0, token=token)
        asset_status = int(asset_probe.get("status") or 0)
        _log(f"PUBLIC /_next asset: HTTP {asset_status} ({asset_url})")
        if asset_status != 200:
            raise RuntimeError(
                f"Public Next.js asset not reachable (browser will hang): "
                f"{asset_url} → HTTP {asset_status}. "
                "basePath/assetPrefix mismatch or gateway not forwarding /_next."
            )
        checked += 1
        if checked >= 3:
            break
    if checked == 0:
        raise RuntimeError(
            "Could not verify any public /_next/static assets — refusing to claim "
            "the website is ready."
        )
    _log(f"OK PUBLIC /_next static verification ({checked} assets)")


def build_frontend(base_path: str, *, split_proxy: bool = False) -> None:
    write_frontend_dotenv(base_path, split_proxy=split_proxy)
    env = _frontend_env(base_path, split_proxy=split_proxy)

    _log("Installing frontend deps if needed...")
    if not (FRONTEND / "node_modules").exists():
        if (FRONTEND / "package-lock.json").exists():
            subprocess.check_call(["npm", "ci"], cwd=str(FRONTEND), env=env)
        else:
            subprocess.check_call(["npm", "install"], cwd=str(FRONTEND), env=env)
    else:
        _log("frontend/node_modules present — skipping npm install")

    _log(
        f"Building Next.js (split_proxy={split_proxy} "
        f"BASE_PATH={base_path or '(none)'}, "
        f"API_URL={env.get('NEXT_PUBLIC_API_URL')})..."
    )
    subprocess.check_call(["npm", "run", "build"], cwd=str(FRONTEND), env=env)
    BASE_PATH_MARKER.parent.mkdir(parents=True, exist_ok=True)
    marker = "SPLIT:/proxy/8000" if split_proxy else (base_path or "")
    BASE_PATH_MARKER.write_text(marker, encoding="utf-8")


def read_built_base_path() -> Optional[str]:
    if not BASE_PATH_MARKER.exists():
        return None
    return BASE_PATH_MARKER.read_text(encoding="utf-8").strip()


def start_next(base_path: str, *, split_proxy: bool = False) -> subprocess.Popen:
    write_frontend_dotenv(base_path, split_proxy=split_proxy)
    env = _frontend_env(base_path, split_proxy=split_proxy)
    env["PORT"] = "3000"
    env["HOSTNAME"] = "127.0.0.1"
    FRONTEND_LOG.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(FRONTEND_LOG, "w", encoding="utf-8", errors="replace", buffering=1)
    _log(f"Starting Next.js production server on 127.0.0.1:3000 (log={FRONTEND_LOG}) ...")
    proc = subprocess.Popen(
        ["npm", "run", "start", "--", "-H", "127.0.0.1", "-p", "3000"],
        cwd=str(FRONTEND),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_fh.write(f"[run_kaggle] Next.js PID: {proc.pid}\n")
    log_fh.flush()
    _log(f"Next.js PID: {proc.pid}")
    proc._fabricvision_fe_log_fh = log_fh  # type: ignore[attr-defined]
    if proc.stdout is not None:
        t = threading.Thread(
            target=_drain_pipe_to_log,
            args=(proc.stdout, log_fh),
            name="next-log-drain",
            daemon=True,
        )
        t.start()
    return proc


def _tail_text(path: Path, *, max_chars: int = 12_000) -> str:
    """Return the last max_chars of a log file (best-effort)."""
    try:
        if not path.exists():
            return f"(log missing: {path})"
        data = path.read_text(encoding="utf-8", errors="replace")
        if len(data) <= max_chars:
            return data if data.strip() else "(log empty)"
        return data[-max_chars:]
    except Exception as exc:
        return f"(could not read {path}: {exc})"


def _drain_pipe_to_log(pipe, log_fh: TextIO, echo: bool = True) -> None:
    """Background reader so PIPE never fills and kaggle_api.log always has output."""
    try:
        for line in iter(pipe.readline, b""):
            if not line:
                break
            text = line.decode("utf-8", errors="replace")
            try:
                log_fh.write(text)
                log_fh.flush()
            except Exception:
                pass
            if echo:
                try:
                    sys.stdout.write(text)
                    sys.stdout.flush()
                except Exception:
                    pass
    except Exception:
        pass
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def start_fastapi(base_path: str) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTHONUNBUFFERED"] = "1"
    env["FRONTEND_UPSTREAM"] = env.get("FRONTEND_UPSTREAM", "http://127.0.0.1:3000")
    # Ensure HF cache is set in the child before any model import.
    try:
        from src.common.utils.hf_cache_env import ensure_huggingface_cache_env

        ensure_huggingface_cache_env()
        for key in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_HUB_DISABLE_XET"):
            if os.environ.get(key):
                env[key] = os.environ[key]
    except Exception as exc:
        _log(f"HF cache env for FastAPI child skipped: {exc}")
    if base_path:
        env["NEXT_PUBLIC_BASE_PATH"] = base_path
    else:
        env.pop("NEXT_PUBLIC_BASE_PATH", None)

    py = sys.executable
    KAGGLE_API_LOG.parent.mkdir(parents=True, exist_ok=True)
    BACKEND_LOG.parent.mkdir(parents=True, exist_ok=True)
    # Truncate previous run so the first crash traceback is unambiguous.
    log_fh = open(KAGGLE_API_LOG, "w", encoding="utf-8", errors="replace", buffering=1)
    backend_fh = open(BACKEND_LOG, "w", encoding="utf-8", errors="replace", buffering=1)
    header = f"[run_kaggle] Starting FastAPI python={py} cwd={ROOT}\n"
    log_fh.write(header)
    backend_fh.write(header)
    log_fh.flush()
    backend_fh.flush()

    _log(f"Starting FastAPI/Uvicorn on 0.0.0.0:8000 (log={BACKEND_LOG}) ...")
    proc = subprocess.Popen(
        [
            py,
            "-m",
            "uvicorn",
            "backend_api.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    for fh in (log_fh, backend_fh):
        fh.write(f"[run_kaggle] FastAPI PID: {proc.pid}\n")
        fh.flush()
    _log(f"FastAPI PID: {proc.pid}")

    # Tee stdout into both log files + console.
    class _Tee:
        def __init__(self, *files: TextIO) -> None:
            self._files = files

        def write(self, text: str) -> None:
            for f in self._files:
                try:
                    f.write(text)
                    f.flush()
                except Exception:
                    pass

        def flush(self) -> None:
            for f in self._files:
                try:
                    f.flush()
                except Exception:
                    pass

    tee = _Tee(log_fh, backend_fh)
    proc._fabricvision_api_log_fh = log_fh  # type: ignore[attr-defined]
    proc._fabricvision_backend_log_fh = backend_fh  # type: ignore[attr-defined]
    proc._fabricvision_api_log_path = BACKEND_LOG  # type: ignore[attr-defined]
    if proc.stdout is not None:
        t = threading.Thread(
            target=_drain_pipe_to_log,
            args=(proc.stdout, tee),
            name="fastapi-log-drain",
            daemon=True,
        )
        t.start()
        proc._fabricvision_api_log_thread = t  # type: ignore[attr-defined]
    return proc


def wait_http(
    url: str,
    label: str,
    attempts: int = 60,
    *,
    proc: Optional[subprocess.Popen] = None,
) -> int:
    last_detail = ""
    for i in range(attempts):
        if proc is not None:
            code = proc.poll()
            if code is not None:
                log_path = getattr(proc, "_fabricvision_api_log_path", KAGGLE_API_LOG)
                tail = _tail_text(Path(log_path))
                _log(f"❌ {label}: process exited early (exit={code})")
                _log(f"--- latest backend output ({log_path}) ---")
                print(tail, flush=True)
                raise RuntimeError(
                    f"FastAPI process exited before becoming live "
                    f"(exit={code}). See {log_path}. Last output:\n{tail}"
                )
            if i == 0 or i % 5 == 0:
                _log(f"⏳ FastAPI still starting... {i}s elapsed")
        ok, http_code, detail = _http_ok(url)
        last_detail = detail
        if ok:
            _log(f"OK {label}: {url} → HTTP {http_code}")
            return http_code
        time.sleep(1.0)
    if proc is not None:
        log_path = getattr(proc, "_fabricvision_api_log_path", KAGGLE_API_LOG)
        tail = _tail_text(Path(log_path))
        raise RuntimeError(
            f"{label} failed validation: {url} ({last_detail}). "
            f"Backend log ({log_path}):\n{tail}"
        )
    raise RuntimeError(f"{label} failed validation: {url} ({last_detail})")


def assert_html_has_no_stale_proxy_prefix(url: str) -> None:
    """Reject builds that hard-code the obsolete host-root /proxy/8000 prefix."""
    code, body = _http_status(url, timeout=10.0)
    if code != 200:
        raise RuntimeError(f"HTML check failed for {url}: HTTP {code}")
    needles = (
        'href="/proxy/8000',
        "href='/proxy/8000",
        'src="/proxy/8000',
        "src='/proxy/8000",
        '"/proxy/8000/',
        "'/proxy/8000/",
    )
    # Allow intentional .../proxy/proxy/8000/... on Kaggle.
    if "/proxy/proxy/8000" in body:
        _log(f"OK HTML check: {url} uses /proxy/proxy/8000 base (Kaggle jsp path)")
        return
    if any(n in body for n in needles):
        raise RuntimeError(
            f"HTML from {url} still contains hard-coded host-root '/proxy/8000' paths"
        )
    _log(f"OK HTML check: {url} has no hard-coded host-root /proxy/8000 links")


def print_banner(
    *,
    health_code: int,
    root_code: int,
    next_code: int,
    deploy: Dict[str, Any],
    public_status: Optional[Dict[str, int]] = None,
    flux_status: Optional[Dict[str, Any]] = None,
    application_ready: bool = False,
) -> None:
    print("", flush=True)
    print("=" * 60, flush=True)
    print("FABRICVISION-AI KAGGLE", flush=True)
    print("=" * 60, flush=True)
    print("", flush=True)
    print("Environment:", flush=True)
    print(f"    Kaggle: {'YES' if deploy.get('is_kaggle') else 'NO'}", flush=True)
    print(f"    Mode: {deploy.get('deploy_mode') or 'unknown'}", flush=True)
    print(f"    GPU: {deploy.get('gpu_name') or '(see logs)'}", flush=True)
    print("", flush=True)
    print("Cache:", flush=True)
    print(f"    HF_HOME: {os.environ.get('HF_HOME') or '(default)'}", flush=True)
    print(
        f"    HUGGINGFACE_HUB_CACHE: {os.environ.get('HUGGINGFACE_HUB_CACHE') or '(default)'}",
        flush=True,
    )
    if flux_status:
        print(
            f"    Disk cache: {flux_status.get('cache_status') or ((flux_status.get('result') or {}).get('cache_status'))}",
            flush=True,
        )
    print("", flush=True)
    print("Backend:", flush=True)
    print(f"    http://127.0.0.1:8000/api/v1/health → {health_code}", flush=True)
    print(f"    http://127.0.0.1:8000/docs", flush=True)
    print(f"    log: {BACKEND_LOG}", flush=True)
    print("", flush=True)
    print("Frontend:", flush=True)
    print(f"    http://127.0.0.1:3000/ → {next_code}", flush=True)
    print(f"    gateway / → {root_code}", flush=True)
    print(f"    log: {FRONTEND_LOG}", flush=True)
    print("", flush=True)
    print("Proxy:", flush=True)
    print("    Frontend: /proxy/3000/", flush=True)
    print("    Backend:  /proxy/8000/", flush=True)
    print("    API:      /proxy/8000/api/v1/", flush=True)
    if deploy.get("base_path"):
        print(
            f"    Gateway basePath (optional): {deploy.get('base_path')}",
            flush=True,
        )
        print(
            "    NOTE: /proxy/proxy/<port> is intentional when Jupyter base_url "
            "already ends with /proxy/ (jupyter-server-proxy nesting).",
            flush=True,
        )
    print("", flush=True)
    print("FLUX:", flush=True)
    if flux_status:
        print(
            f"    State: {flux_status.get('state')} "
            f"ready={flux_status.get('ready')} "
            f"progress={flux_status.get('progress')}% "
            f"stage={flux_status.get('stage') or flux_status.get('current_step')}",
            flush=True,
        )
        print("    http://127.0.0.1:8000/api/v1/flux-status", flush=True)
    else:
        print("    (not confirmed)", flush=True)
    print("", flush=True)
    if deploy.get("public_url"):
        print("PUBLIC WEBSITE (gateway discovery, if available):", flush=True)
        print(f"    {deploy.get('public_url')}", flush=True)
        if public_status:
            print(f"    HTTP {public_status.get('root', 'n/a')}", flush=True)
        print("", flush=True)
    print(f"Next/FastAPI base_path in use: {deploy.get('base_path') or '(none — split proxy)'}", flush=True)
    print("", flush=True)
    if application_ready:
        print("=" * 60, flush=True)
        print("FABRICVISION-AI READY", flush=True)
        print("=" * 60, flush=True)
        print("    Open Kaggle PORT 3000 for the UI.", flush=True)
        print("    Open Kaggle PORT 8000 for API/docs/outputs.", flush=True)
        print(
            "    Next.js + FastAPI liveness + FLUX residency confirmed.",
            flush=True,
        )
    else:
        print("APPLICATION NOT READY", flush=True)
        print(
            "    Do not treat API /health 200 as FLUX-ready.",
            flush=True,
        )
    print("=" * 60, flush=True)
    print("", flush=True)


def configure_kaggle_flux_runtime() -> None:
    """Set completion-first FLUX env defaults and warm CUDA on the main thread.

    Do NOT force 768×12 GPU-resident — that path OOMs on T4-class cards when
    NF4 + Kontext activations exceed free headroom. Prefer measured policy:
    512 Standard + VAE tiling; opt into 768 only via FLUX_ALLOW_HIGH_RES.
    """
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Durable HF cache under /kaggle/working so restarts reuse blobs (CACHE HIT).
    try:
        from src.common.utils.hf_cache_env import ensure_huggingface_cache_env

        applied = ensure_huggingface_cache_env()
        _log(
            "HF cache env: "
            f"HF_HOME={applied.get('HF_HOME') or os.environ.get('HF_HOME')!r} "
            f"HUGGINGFACE_HUB_CACHE="
            f"{applied.get('HUGGINGFACE_HUB_CACHE') or os.environ.get('HUGGINGFACE_HUB_CACHE')!r}"
        )
    except Exception as exc:
        _log(f"HF cache env setup skipped: {exc}")
    try:
        import torch

        if not torch.cuda.is_available():
            _log("CUDA not available — FLUX will fail on this runtime")
            return
        # Initialize CUDA context on the main process before worker threads.
        _ = torch.zeros(1, device="cuda")
        props = torch.cuda.get_device_properties(0)
        vram_mb = props.total_memory / (1024**2)
        name = torch.cuda.get_device_name(0)
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        free = max(0.0, vram_mb - reserved)
        _log(
            f"GPU={name} total_mb={vram_mb:.0f} alloc_mb={allocated:.0f} "
            f"reserved_mb={reserved:.0f} free_mb={free:.0f} torch={torch.__version__}"
        )
        if vram_mb >= 14000:
            # Completion-first defaults. Operators may set FLUX_ALLOW_HIGH_RES=true
            # and/or FLUX_GENERATION_RESOLUTION=768 after a successful smoke.
            os.environ.setdefault("FLUX_VAE_TILING", "true")
            os.environ.setdefault("FLUX_GENERATION_RESOLUTION", "512")
            os.environ.setdefault("FLUX_STANDARD_STEPS", "8")
            # Tesla T4 (sm_75): prefer model_cpu_offload + FP16. GPU-resident NF4
            # + park_on_cpu left the VAE on CPU and triggered CUDA_ERROR at ~50%.
            try:
                major, _ = torch.cuda.get_device_capability(0)
            except Exception:
                major = 8
            if int(major) < 8:
                os.environ.setdefault("FLUX_MODEL_CPU_OFFLOAD", "true")
                os.environ.setdefault("FLUX_TORCH_DTYPE", "float16")
                _log(
                    "Pre-Ampere T4-class defaults: FLUX_MODEL_CPU_OFFLOAD=true "
                    "FLUX_TORCH_DTYPE=float16 (transformer/compute); "
                    "VAE is upcast to float32 at load/generate to avoid fp16 NaN/black"
                )
            # Demote the previous unsafe notebook default (768 + GPU-resident) unless
            # the operator explicitly opts into high-res.
            allow_hi = os.environ.get("FLUX_ALLOW_HIGH_RES", "").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if (
                not allow_hi
                and os.environ.get("FLUX_GENERATION_RESOLUTION", "").strip() == "768"
            ):
                os.environ["FLUX_GENERATION_RESOLUTION"] = "512"
                _log(
                    "Demoted legacy FLUX_GENERATION_RESOLUTION=768 → 512 "
                    "(set FLUX_ALLOW_HIGH_RES=true to keep 768)"
                )
            if (
                not allow_hi
                and os.environ.get("FLUX_STANDARD_STEPS", "").strip() == "12"
                and os.environ.get("FLUX_GENERATION_RESOLUTION", "").strip() in ("", "512")
            ):
                # 12 steps @ 512 is fine; leave it. Only demote when paired with 768 intent.
                pass
            _log(
                "High-VRAM completion-first defaults: "
                f"FLUX_GENERATION_RESOLUTION={os.environ.get('FLUX_GENERATION_RESOLUTION')} "
                f"FLUX_STANDARD_STEPS={os.environ.get('FLUX_STANDARD_STEPS')} "
                f"FLUX_MODEL_CPU_OFFLOAD={os.environ.get('FLUX_MODEL_CPU_OFFLOAD', 'auto')} "
                "FLUX_VAE_TILING=true (768 gated behind free headroom / FLUX_ALLOW_HIGH_RES)"
            )
        else:
            os.environ.setdefault("FLUX_MODEL_CPU_OFFLOAD", "true")
            os.environ.setdefault("FLUX_VAE_TILING", "true")
            _log("Low-VRAM defaults: FLUX_MODEL_CPU_OFFLOAD=true FLUX_VAE_TILING=true")
    except Exception as exc:
        _log(f"GPU runtime probe skipped: {exc}")


def prefetch_flux_weights() -> None:
    """Ensure FLUX weights exist on disk for the FastAPI child process.

    IMPORTANT: This runs in the run_kaggle PARENT process. Loading a full
    FluxKontextPipeline here does NOT share memory with uvicorn. We therefore
    only resolve/download the package to disk, then leave GPU init to the API
    child's ``flux_warmup`` (the only in-memory load Generate can reuse).
    """
    _log(
        "Prefetching FLUX weights to disk only "
        "(API child will load into memory — parent pipeline is not shared)..."
    )
    t0 = time.perf_counter()
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from src.common.utils.utils import load_yaml_config
        from src.features.custom_generator.model.flux_model_loader import FLUXModelLoader

        hf_id = os.environ.get("FLUX_KONTEXT_MODEL_ID", "").strip() or None
        if not hf_id:
            flux_yaml = (
                load_yaml_config(ROOT / "configs" / "custom_generator" / "flux_config.yaml")
                or {}
            )
            hf_id = (flux_yaml.get("hf_model_id") or "").strip() or None

        loader = FLUXModelLoader(
            model_path=ROOT / "models" / "flux-kontext",
            allow_fallback=False,
            hf_model_id=hf_id,
        )
        print(
            f"[FLUX LOAD TRACE] process_id={os.getpid()} request_id=prefetch_parent "
            f"pipeline_load_start (disk resolve only)",
            flush=True,
        )
        pipeline_root, transformer_root = loader._resolve_model_source()
        print(
            f"[FLUX LOAD TRACE] process_id={os.getpid()} request_id=prefetch_parent "
            f"pipeline_load_end model_reused=false "
            f"cache_state={getattr(loader, '_cache_status', None)} "
            f"pipeline_root={pipeline_root} transformer_root={transformer_root}",
            flush=True,
        )
        _log(
            f"FLUX disk prefetch done in {time.perf_counter() - t0:.1f}s "
            f"cache={getattr(loader, '_cache_status', None)} "
            f"path={pipeline_root} "
            f"(no parent GPU residency — API warmup owns in-memory load)"
        )
    except Exception as exc:
        _log(f"ERROR: FLUX prefetch failed: {type(exc).__name__}: {exc}")
        raise


def wait_api_flux_ready(*, timeout_s: float = 900.0) -> Dict[str, Any]:
    """Poll /api/v1/flux-status until the API child has FLUX in memory.

    Raises on FAILED or timeout — callers must NOT declare APPLICATION READY
    while FLUX is still STARTING.
    """
    url = "http://127.0.0.1:8000/api/v1/flux-status"
    deadline = time.perf_counter() + timeout_s
    last: Dict[str, Any] = {}
    consecutive_errors = 0

    def _useful_status(payload: Dict[str, Any]) -> bool:
        st = str((payload or {}).get("state") or "").upper()
        return bool(payload) and st not in ("", "UNKNOWN")

    while time.perf_counter() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            # Large downloads can slow the event loop briefly; allow a longer read.
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                status = int(getattr(resp, "status", 200))
                ctype = resp.headers.get("Content-Type") or ""
                raw = resp.read(256_000)
                body = raw.decode("utf-8", errors="replace")
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError as exc:
                consecutive_errors += 1
                preview = body[:240].replace("\n", "\\n")
                _log(
                    "⚠️ Could not parse FLUX status JSON: "
                    f"{type(exc).__name__}: {exc} | HTTP {status} "
                    f"ctype={ctype!r} bytes={len(raw)} preview={preview!r}"
                )
                # Preserve last useful STARTING/READY/FAILED — never overwrite
                # a real FLUX state with silent UNKNOWN/warming.
                if not _useful_status(last):
                    last = {
                        "error": f"JSONDecodeError: {exc}",
                        "ready": False,
                        "state": "UNKNOWN",
                        "current_step": "warming",
                        "poll_error": True,
                    }
                else:
                    last = dict(last)
                    last["poll_error"] = f"JSONDecodeError: {exc}"
            else:
                consecutive_errors = 0
                last = parsed
        except Exception as exc:
            consecutive_errors += 1
            _log(
                f"⚠️ FLUX status poll failed: {type(exc).__name__}: {exc} "
                f"(consecutive={consecutive_errors})"
            )
            if not _useful_status(last):
                last = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "ready": False,
                    "state": "UNKNOWN",
                    "current_step": "warming",
                    "poll_error": True,
                }
            else:
                last = dict(last)
                last["poll_error"] = f"{type(exc).__name__}: {exc}"
        state = str(last.get("state") or "").upper()
        if last.get("ready") or last.get("in_memory") or state == "READY":
            _log(
                f"API FLUX READY pid={last.get('api_pid') or last.get('pid')} "
                f"state={state} disk_cache={last.get('cache_status') or ((last.get('result') or {}).get('cache_status'))} "
                f"duration_s={last.get('load_duration_s')} "
                f"pipeline_exists={last.get('pipeline_exists')}"
            )
            return last
        if state == "FAILED":
            err = last.get("error") or "API FLUX warmup failed"
            _log(f"❌ FLUX WARMUP FAILED: {err}")
            raise RuntimeError(err)
        if state == "SKIPPED":
            _log(
                "API FLUX warmup SKIPPED "
                f"(reason={(last.get('result') or {}).get('reason')}) — "
                "Generate may cold-load"
            )
            return last
        step = last.get("current_step") or last.get("stage") or "warming"
        pe = last.get("poll_error")
        _log(
            f"Waiting for API-process FLUX... state={state} "
            f"progress={last.get('progress')}% step={step!r} "
            f"elapsed_s={last.get('load_duration_s')}"
            + (f" poll_error={pe!r}" if pe else "")
        )
        time.sleep(2.0)
    raise TimeoutError(
        f"API FLUX warmup not ready within {timeout_s:.0f}s last={last}"
    )


def require_api_flux_ready(*, timeout_s: float = 900.0) -> Dict[str, Any]:
    """Hard gate: FLUX must be READY (or intentionally SKIPPED) before app READY."""
    status = wait_api_flux_ready(timeout_s=timeout_s)
    state = str(status.get("state") or "").upper()
    if state == "FAILED":
        raise RuntimeError(status.get("error") or "FLUX warmup failed")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="FabricVision-AI Kaggle / gateway launcher")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip npm build when the existing frontend/.next matches the target mode",
    )
    parser.add_argument(
        "--no-base-path",
        action="store_true",
        help="Force empty Next basePath (same as default split-proxy mode)",
    )
    parser.add_argument(
        "--gateway",
        action="store_true",
        help="Single-port gateway mode (Next basePath = discovered …/proxy/proxy/8000)",
    )
    parser.add_argument(
        "--base-path",
        default=None,
        help="Override public base path (implies --gateway)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Public FastAPI / gateway port (default 8000)",
    )
    parser.add_argument(
        "--prefetch-flux",
        action="store_true",
        help="Optional PREPARATION: download FLUX weights BEFORE starting services",
    )
    parser.add_argument(
        "--no-prefetch-flux",
        action="store_true",
        help="Skip parent-process FLUX prefetch (default; API warmup downloads)",
    )
    args = parser.parse_args()

    os.chdir(ROOT)

    deploy = detect_deployment(port=args.port)
    gateway_mode = bool(args.gateway or (args.base_path is not None and not args.no_base_path))
    split_proxy = (not gateway_mode) or bool(args.no_base_path)

    if args.no_base_path or split_proxy:
        base_path = ""
        deploy["base_path"] = ""
        deploy["deploy_mode"] = "split_proxy"
    elif args.base_path is not None:
        base_path = "/" + args.base_path.strip("/") if args.base_path.strip() else ""
        deploy["base_path"] = base_path
        deploy["deploy_mode"] = "gateway"
        if deploy.get("proxy_host") and base_path:
            deploy["public_url"] = f"{deploy['proxy_host']}{base_path}/"
            deploy["about_url"] = f"{deploy['proxy_host']}{base_path}/about"
            deploy["docs_url"] = f"{deploy['proxy_host']}{base_path}/docs"
            deploy["health_url"] = f"{deploy['proxy_host']}{base_path}/api/v1/health"
        split_proxy = False
    else:
        base_path = str(deploy.get("base_path") or "")
        deploy["deploy_mode"] = "gateway" if base_path else "local"
        split_proxy = not bool(base_path)
        if split_proxy:
            base_path = ""
            deploy["base_path"] = ""
            deploy["deploy_mode"] = "split_proxy"

    if split_proxy:
        base_path = ""
        deploy["base_path"] = ""
        deploy["deploy_mode"] = "split_proxy"

    _log(
        f"Deployment mode: kaggle={deploy['is_kaggle']} "
        f"deploy_mode={deploy.get('deploy_mode')!r} "
        f"split_proxy={split_proxy} "
        f"jupyter_base_url={deploy.get('jupyter_base_url')!r} "
        f"base_path={base_path!r}"
    )

    if (
        gateway_mode
        and deploy.get("is_kaggle")
        and not args.no_base_path
        and not base_path
    ):
        _log(
            "ERROR: --gateway on Kaggle requires a non-empty public basePath. "
            "Could not detect Jupyter base_url. Set KAGGLE_PUBLIC_PATH or "
            "pass --base-path /k/<session>/proxy/proxy/8000. "
            "Or omit --gateway to use split /proxy/3000 + /proxy/8000 mode."
        )
        return 1

    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from src.common.utils.ensure_bitsandbytes import ensure_bitsandbytes

        bnb_ver = ensure_bitsandbytes(auto_install=True)
        _log(f"bitsandbytes verified: {bnb_ver}")
    except Exception as exc:
        _log(f"ERROR: bitsandbytes prerequisite failed: {exc}")
        return 1

    try:
        import importlib.util

        if importlib.util.find_spec("python_multipart") is None and importlib.util.find_spec(
            "multipart"
        ) is None:
            _log("python-multipart missing — installing...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "python-multipart>=0.0.18"]
            )
        from python_multipart import __version__ as _mp_ver  # type: ignore

        _log(f"python-multipart verified: {_mp_ver}")
    except Exception as exc:
        _log(
            f"ERROR: python-multipart prerequisite failed: {exc}. "
            "FastAPI generation/try-on Form routes will crash uvicorn on import."
        )
        return 1

    configure_kaggle_flux_runtime()

    do_prefetch = bool(args.prefetch_flux) and not bool(args.no_prefetch_flux)
    if do_prefetch:
        _log(
            "PREPARATION PHASE: FLUX disk prefetch BEFORE services "
            "(backend/frontend are not started yet — connection refused is expected)"
        )
        try:
            prefetch_flux_weights()
        except Exception:
            return 1
        _log("PREPARATION PHASE complete — starting services next")
    else:
        _log(
            "Skipping parent FLUX prefetch (default). "
            "API warmup will CHECKING_CACHE / DOWNLOAD / READY in-process."
        )

    children: List[subprocess.Popen] = []
    next_proc: Optional[subprocess.Popen] = None
    api_proc: Optional[subprocess.Popen] = None

    backend_up = backend_already_healthy()
    frontend_up = frontend_already_healthy(base_path if not split_proxy else "")
    if backend_up:
        _log("Backend already healthy on :8000 — reusing (not spawning another uvicorn)")
    if frontend_up:
        _log("Frontend already healthy on :3000 — reusing (not spawning another Next)")

    if not backend_up:
        stop_port(args.port, "backend")
    if not frontend_up:
        stop_port(3000, "frontend")

    built = read_built_base_path()
    target_marker = "SPLIT:/proxy/8000" if split_proxy else (base_path or "")
    need_build = not args.skip_build
    if args.skip_build:
        if not (FRONTEND / ".next").exists():
            _log("No frontend/.next found; building...")
            need_build = True
        elif (built or "") != target_marker:
            _log(
                f"Built marker {built!r} != target {target_marker!r}; rebuilding "
                "(required for correct Kaggle public API routing)."
            )
            need_build = True

    if need_build:
        build_frontend(base_path=base_path, split_proxy=split_proxy)

    def _shutdown(*_args) -> None:
        _log("Shutting down...")
        for proc in children:
            if proc.poll() is None:
                proc.terminate()
        time.sleep(1.0)
        for proc in children:
            if proc.poll() is None:
                proc.kill()

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown)

    try:
        _log("Backend starting...")
        if not backend_up:
            api_proc = start_fastapi(base_path=base_path)
            children.append(api_proc)
        health_code = wait_http(
            "http://127.0.0.1:8000/api/v1/health",
            "API health",
            proc=api_proc,
            attempts=90,
        )
        wait_http("http://127.0.0.1:8000/docs", "Swagger /docs")
        wait_http("http://127.0.0.1:8000/openapi.json", "OpenAPI JSON")
        _log("Backend READY (liveness) — FLUX may still be warming in background")

        _log("Frontend starting...")
        if not frontend_up:
            next_proc = start_next(base_path=base_path, split_proxy=split_proxy)
            children.append(next_proc)
        next_root = (
            f"http://127.0.0.1:3000{base_path}/"
            if (base_path and not split_proxy)
            else "http://127.0.0.1:3000/"
        )
        next_code = wait_http(next_root, "Next.js root", attempts=90)
        wait_http("http://127.0.0.1:8000/health", "Gateway health", attempts=60)
        wait_http("http://127.0.0.1:8000/api/v1/openapi.json", "OpenAPI JSON (v1)")
        _log("Frontend READY")

        flux_status: Optional[Dict[str, Any]] = None
        if os.environ.get("FLUX_WARMUP_ON_STARTUP", "true").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        ):
            _log("Monitoring FLUX via /api/v1/flux-status (services stay up)...")
            flux_status = require_api_flux_ready(
                timeout_s=float(os.environ.get("FLUX_WARMUP_WAIT_S", "900"))
            )
        else:
            _log("FLUX_WARMUP_ON_STARTUP disabled — skipping FLUX readiness gate")
            flux_status = {"state": "SKIPPED", "ready": False}

        root_code = wait_http("http://127.0.0.1:8000/", "Gateway frontend /")
        wait_http("http://127.0.0.1:8000/about", "Gateway frontend /about")

        if not split_proxy:
            assert_html_has_no_stale_proxy_prefix("http://127.0.0.1:8000/")
            assert_html_next_assets_use_base_path("http://127.0.0.1:8000/", base_path)
        else:
            _log(
                "Split-proxy mode: skipping gateway basePath asset assert "
                "(UI is /proxy/3000, API is /proxy/8000/api/v1)"
            )

        next_code, _ = _http_status(next_root)

        public_status: Dict[str, int] = {}
        discovery_report = None
        if (
            gateway_mode
            and deploy.get("is_kaggle")
            and deploy.get("jupyter_base_url")
            and not args.no_base_path
        ):
            _log("Discovering working PUBLIC Kaggle proxy route via live probes...")
            discovery = discover_working_public_path(
                str(deploy["jupyter_base_url"]),
                port=args.port,
                proxy_host=deploy.get("proxy_host"),
            )
            discovery_report = discovery.get("report") if discovery else None
            winner = (discovery or {}).get("public_path")
            if winner:
                if winner != base_path:
                    _log(
                        f"Public winner {winner!r} differs from build base_path "
                        f"{base_path!r} — rebuilding frontend and restarting."
                    )
                    apply_public_path(deploy, winner)
                    base_path = winner
                    for proc in list(children):
                        if proc.poll() is None:
                            proc.terminate()
                    time.sleep(1.5)
                    for proc in list(children):
                        if proc.poll() is None:
                            proc.kill()
                    children.clear()
                    stop_port(3000, "frontend")
                    stop_port(args.port, "backend")
                    build_frontend(base_path=base_path, split_proxy=False)
                    next_proc = start_next(base_path=base_path, split_proxy=False)
                    children.append(next_proc)
                    next_root = (
                        f"http://127.0.0.1:3000{base_path}/"
                        if base_path
                        else "http://127.0.0.1:3000/"
                    )
                    wait_http(next_root, "Next.js root (rebuild)", attempts=90)
                    api_proc = start_fastapi(base_path=base_path)
                    children.append(api_proc)
                    health_code = wait_http(
                        "http://127.0.0.1:8000/api/v1/health",
                        "API health (rebuild)",
                        proc=api_proc,
                    )
                    wait_http("http://127.0.0.1:8000/health", "Gateway health (rebuild)")
                    if os.environ.get("FLUX_WARMUP_ON_STARTUP", "true").strip().lower() not in (
                        "0",
                        "false",
                        "no",
                        "off",
                    ):
                        flux_status = require_api_flux_ready(
                            timeout_s=float(os.environ.get("FLUX_WARMUP_WAIT_S", "900"))
                        )
                    root_code = wait_http(
                        "http://127.0.0.1:8000/", "Gateway frontend / (rebuild)"
                    )
                    wait_http("http://127.0.0.1:8000/about", "Gateway /about (rebuild)")
                    assert_html_has_no_stale_proxy_prefix("http://127.0.0.1:8000/")
                    assert_html_next_assets_use_base_path(
                        "http://127.0.0.1:8000/", base_path
                    )
                    next_code, _ = _http_status(next_root)
                else:
                    apply_public_path(deploy, winner)

                token = read_jupyter_token()
                for key, url_key in (
                    ("root", "public_url"),
                    ("about", "about_url"),
                    ("docs", "docs_url"),
                    ("health", "health_url"),
                    ("projects", "projects_url"),
                    ("studio_garment", "studio_garment_url"),
                    ("studio_tryon", "studio_tryon_url"),
                    ("studio_semantic", "studio_semantic_url"),
                ):
                    url = deploy.get(url_key)
                    if not url:
                        continue
                    probe = _http_probe(url, timeout=20.0, token=token)
                    public_status[key] = int(probe.get("status") or 0)
                    _log(f"PUBLIC verify {key}: HTTP {public_status[key]} ({url})")

                required_ok = all(
                    public_status.get(k) == 200 for k in ("root", "about", "docs", "health")
                )
                if not required_ok:
                    print_banner(
                        health_code=health_code,
                        root_code=root_code,
                        next_code=next_code,
                        deploy=deploy,
                        public_status=public_status,
                        flux_status=flux_status,
                        application_ready=False,
                    )
                    _log(
                        "ERROR: Winning public path failed final verification. "
                        "Services left running — Ctrl+C to stop."
                    )
                    while True:
                        for proc, name in ((next_proc, "Next.js"), (api_proc, "FastAPI")):
                            if proc is not None and proc.poll() is not None:
                                _log(f"{name} exited")
                                _shutdown()
                                return 1
                        time.sleep(1.0)
                else:
                    try:
                        assert_public_next_static_reachable(
                            public_url=str(deploy.get("public_url") or ""),
                            base_path=base_path,
                            proxy_host=str(
                                deploy.get("proxy_host") or KAGGLE_PROXY_HOST_DEFAULT
                            ),
                            token=token,
                        )
                        public_status["next_static"] = 200
                    except Exception as asset_exc:
                        public_status["next_static"] = 0
                        print_banner(
                            health_code=health_code,
                            root_code=root_code,
                            next_code=next_code,
                            deploy=deploy,
                            public_status=public_status,
                            flux_status=flux_status,
                            application_ready=False,
                        )
                        _log(
                            f"ERROR: Public /_next static verification failed: {asset_exc}"
                        )
                        _log("Services left running for debugging — Ctrl+C to stop.")
                        while True:
                            for proc, name in (
                                (next_proc, "Next.js"),
                                (api_proc, "FastAPI"),
                            ):
                                if proc is not None and proc.poll() is not None:
                                    _log(f"{name} exited")
                                    _shutdown()
                                    return 1
                            time.sleep(1.0)
            else:
                print_banner(
                    health_code=health_code,
                    root_code=root_code,
                    next_code=next_code,
                    deploy=deploy,
                    public_status=public_status or None,
                    flux_status=flux_status,
                    application_ready=False,
                )
                _log(
                    "ERROR: No public Kaggle proxy candidate reached the app. "
                    "Local gateway is healthy. Services left running — Ctrl+C to stop."
                )
                if discovery_report:
                    _log(json.dumps(discovery_report, indent=2, default=str)[:4000])
                while True:
                    for proc, name in ((next_proc, "Next.js"), (api_proc, "FastAPI")):
                        if proc is not None and proc.poll() is not None:
                            _log(f"{name} exited")
                            _shutdown()
                            return 1
                    time.sleep(1.0)

        print_banner(
            health_code=health_code,
            root_code=root_code,
            next_code=next_code,
            deploy=deploy,
            public_status=public_status or None,
            flux_status=flux_status,
            application_ready=True,
        )
        _log("APPLICATION READY — Press Ctrl+C to stop.")

        while True:
            for proc, name in ((next_proc, "Next.js"), (api_proc, "FastAPI")):
                if proc is None:
                    continue
                code = proc.poll()
                if code is not None:
                    _log(f"{name} exited with code {code}")
                    _shutdown()
                    return code or 1
            time.sleep(1.0)
    except Exception as exc:
        _log(f"Startup validation failed: {exc}")
        _shutdown()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
