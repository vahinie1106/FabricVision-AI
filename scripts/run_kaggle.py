#!/usr/bin/env python3
"""Start FabricVision-AI for Kaggle / single-port public deployment.

Architecture:
  Browser → Kaggle HTTPS proxy → :8000 (FastAPI gateway)
    ├── /api/v1/* , /docs , /openapi.json , /outputs/*  → FastAPI
    └── /*  → reverse-proxy → Next.js on 127.0.0.1:3000

On Kaggle the public path is derived from the live Jupyter ``base_url`` and
then confirmed by probing the public jupyter-proxy host. Typical result:

  jupyter base_url = /k/<session>/proxy/
  public_path      = /k/<session>/proxy/proxy/8000

Usage (from repo root):
  python scripts/run_kaggle.py
  python scripts/run_kaggle.py --skip-build
  python scripts/run_kaggle.py --no-base-path   # local gateway, no public prefix
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
BASE_PATH_MARKER = FRONTEND / ".next" / "fabricvision-base-path.txt"
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


def _frontend_env(base_path: str) -> dict:
    """Env for Next build/start — never bake loopback API URLs into the client."""
    env = os.environ.copy()
    env["NEXT_PUBLIC_USE_SAME_ORIGIN"] = "true"
    # Relative API path; browser resolves with live detectRuntimeBasePath().
    # Do NOT bake http://127.0.0.1:8000 (frontend/.env.local local-dev default) —
    # that makes public Kaggle tabs hang calling the user's own machine.
    env["NEXT_PUBLIC_API_URL"] = "/api/v1"
    env["NEXT_PUBLIC_API_ORIGIN"] = ""
    if base_path:
        env["NEXT_PUBLIC_BASE_PATH"] = base_path
    else:
        env.pop("NEXT_PUBLIC_BASE_PATH", None)
    return env


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


def build_frontend(base_path: str) -> None:
    env = _frontend_env(base_path)

    _log("Installing frontend deps if needed...")
    if not (FRONTEND / "node_modules").exists():
        if (FRONTEND / "package-lock.json").exists():
            subprocess.check_call(["npm", "ci"], cwd=str(FRONTEND), env=env)
        else:
            subprocess.check_call(["npm", "install"], cwd=str(FRONTEND), env=env)
    else:
        _log("frontend/node_modules present — skipping npm install")

    _log(
        f"Building Next.js (BASE_PATH={base_path or '(none)'}, "
        f"API_URL={env.get('NEXT_PUBLIC_API_URL')})..."
    )
    subprocess.check_call(["npm", "run", "build"], cwd=str(FRONTEND), env=env)
    BASE_PATH_MARKER.parent.mkdir(parents=True, exist_ok=True)
    BASE_PATH_MARKER.write_text(base_path or "", encoding="utf-8")


def read_built_base_path() -> Optional[str]:
    if not BASE_PATH_MARKER.exists():
        return None
    return BASE_PATH_MARKER.read_text(encoding="utf-8").strip()


def start_next(base_path: str) -> subprocess.Popen:
    env = _frontend_env(base_path)
    env["PORT"] = "3000"
    env["HOSTNAME"] = "127.0.0.1"
    _log("Starting Next.js production server on 127.0.0.1:3000 ...")
    return subprocess.Popen(
        ["npm", "run", "start", "--", "-H", "127.0.0.1", "-p", "3000"],
        cwd=str(FRONTEND),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def start_fastapi(base_path: str) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["FRONTEND_UPSTREAM"] = env.get("FRONTEND_UPSTREAM", "http://127.0.0.1:3000")
    if base_path:
        env["NEXT_PUBLIC_BASE_PATH"] = base_path
    else:
        env.pop("NEXT_PUBLIC_BASE_PATH", None)

    py = sys.executable
    _log("Starting FastAPI/Uvicorn on 0.0.0.0:8000 ...")
    return subprocess.Popen(
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
    )


def wait_http(url: str, label: str, attempts: int = 60) -> int:
    last_detail = ""
    for _ in range(attempts):
        ok, code, detail = _http_ok(url)
        last_detail = detail
        if ok:
            _log(f"OK {label}: {url} → HTTP {code}")
            return code
        time.sleep(1.0)
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
) -> None:
    print("", flush=True)
    print("=" * 60, flush=True)
    print("FABRICVISION-AI KAGGLE DEPLOYMENT", flush=True)
    print("=" * 60, flush=True)
    print("", flush=True)
    print("Backend:", flush=True)
    print(f"    http://127.0.0.1:8000/api/v1/health → {health_code}", flush=True)
    print("", flush=True)
    print("Gateway:", flush=True)
    print(f"    http://127.0.0.1:8000/ → {root_code}", flush=True)
    print("", flush=True)
    print("Frontend:", flush=True)
    print(f"    http://127.0.0.1:3000 → {next_code}", flush=True)
    print("", flush=True)
    print("Kaggle Jupyter base URL:", flush=True)
    print(f"    {deploy.get('jupyter_base_url') or '(not detected — local/non-Kaggle)'}", flush=True)
    print("", flush=True)
    print("PUBLIC WEBSITE (open this exact URL in your browser):", flush=True)
    print(f"    {deploy.get('public_url') or '(n/a outside Kaggle)'}", flush=True)
    if public_status and deploy.get("public_url"):
        print(f"    HTTP {public_status.get('root', 'n/a')}", flush=True)
        if public_status.get("next_static") is not None:
            print(
                f"    /_next/static assets: HTTP {public_status.get('next_static')}",
                flush=True,
            )
    print("", flush=True)
    print(
        "NOTE: Page HTML returning 200 is not enough — JS under /_next/static "
        "must also load through this same public prefix.",
        flush=True,
    )
    print("PUBLIC ABOUT:", flush=True)
    print(f"    {deploy.get('about_url') or '(n/a)'}", flush=True)
    if public_status and deploy.get("about_url"):
        print(f"    HTTP {public_status.get('about', 'n/a')}", flush=True)
    print("", flush=True)
    print("PUBLIC SWAGGER:", flush=True)
    print(f"    {deploy.get('docs_url') or '(n/a)'}", flush=True)
    if public_status and deploy.get("docs_url"):
        print(f"    HTTP {public_status.get('docs', 'n/a')}", flush=True)
    print("", flush=True)
    print("PUBLIC HEALTH:", flush=True)
    print(f"    {deploy.get('health_url') or '(n/a)'}", flush=True)
    if public_status and deploy.get("health_url"):
        print(f"    HTTP {public_status.get('health', 'n/a')}", flush=True)
    print("", flush=True)
    print(f"Next/FastAPI base_path in use: {deploy.get('base_path') or '(none)'}", flush=True)
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
            # Leave FLUX_MODEL_CPU_OFFLOAD unset → loader auto (GPU-resident on
            # ≥14GB for speed) while generation stays at 512 unless headroom allows 768.
            _log(
                "High-VRAM completion-first defaults: "
                f"FLUX_GENERATION_RESOLUTION={os.environ.get('FLUX_GENERATION_RESOLUTION')} "
                f"FLUX_STANDARD_STEPS={os.environ.get('FLUX_STANDARD_STEPS')} "
                "FLUX_VAE_TILING=true (768 gated behind free headroom / FLUX_ALLOW_HIGH_RES)"
            )
        else:
            os.environ.setdefault("FLUX_MODEL_CPU_OFFLOAD", "true")
            os.environ.setdefault("FLUX_VAE_TILING", "true")
            _log("Low-VRAM defaults: FLUX_MODEL_CPU_OFFLOAD=true FLUX_VAE_TILING=true")
    except Exception as exc:
        _log(f"GPU runtime probe skipped: {exc}")


def prefetch_flux_weights() -> None:
    """Download FLUX weights at startup so first Generate is not a multi-GB cold download.

    Loads once to verify the package, then unloads so the FastAPI child process
    can initialize CUDA cleanly (parent must not keep a resident pipeline).
    """
    _log("Prefetching FLUX.1-Kontext weights (CACHE HIT/MISS logged by loader)...")
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

        def _cb(step: str, pct: int) -> None:
            _log(f"prefetch {pct}% {step}")

        loader = FLUXModelLoader(
            model_path=ROOT / "models" / "flux-kontext",
            allow_fallback=False,
            hf_model_id=hf_id,
        )
        loader.set_progress_callback(_cb)
        pipe = loader.load()
        if pipe is None:
            raise RuntimeError("FLUX prefetch returned no pipeline")
        info = loader.get_runtime_info()
        # Free VRAM before spawning FastAPI (separate process will reload from disk).
        try:
            loader.park_on_cpu()
        except Exception:
            pass
        loader._pipeline = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        _log(
            f"FLUX prefetch done in {time.perf_counter() - t0:.1f}s "
            f"cache={info.get('cache_status')} init_s={info.get('init_time_s')} "
            f"download_s={info.get('download_time_s')} offload={info.get('offload_strategy')} "
            f"(pipeline unloaded for API child)"
        )
    except Exception as exc:
        _log(f"ERROR: FLUX prefetch failed: {type(exc).__name__}: {exc}")
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="FabricVision-AI Kaggle / gateway launcher")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip npm build when the existing frontend/.next matches the target base path",
    )
    parser.add_argument(
        "--no-base-path",
        action="store_true",
        help="Force empty base path (local single-port gateway)",
    )
    parser.add_argument(
        "--base-path",
        default=None,
        help="Override public base path (otherwise detected on Kaggle)",
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
        help="Force FLUX weight download/init before serving traffic",
    )
    parser.add_argument(
        "--no-prefetch-flux",
        action="store_true",
        help="Skip FLUX prefetch even on Kaggle (first Generate pays cold download)",
    )
    args = parser.parse_args()

    os.chdir(ROOT)

    deploy = detect_deployment(port=args.port)

    if args.no_base_path:
        base_path = ""
        deploy["base_path"] = ""
        deploy["public_url"] = None
    elif args.base_path is not None:
        base_path = "/" + args.base_path.strip("/") if args.base_path.strip() else ""
        deploy["base_path"] = base_path
        if deploy.get("proxy_host") and base_path:
            deploy["public_url"] = f"{deploy['proxy_host']}{base_path}/"
            deploy["about_url"] = f"{deploy['proxy_host']}{base_path}/about"
            deploy["docs_url"] = f"{deploy['proxy_host']}{base_path}/docs"
            deploy["health_url"] = f"{deploy['proxy_host']}{base_path}/api/v1/health"
    else:
        # Auto: Kaggle → dynamic path; local → no prefix (never hard-code /proxy/8000).
        base_path = str(deploy.get("base_path") or "")

    _log(
        f"Deployment mode: kaggle={deploy['is_kaggle']} "
        f"jupyter_base_url={deploy.get('jupyter_base_url')!r} "
        f"base_path={base_path!r}"
    )

    if (
        deploy.get("is_kaggle")
        and not args.no_base_path
        and not base_path
    ):
        _log(
            "ERROR: Kaggle deployment requires a non-empty public basePath. "
            "Could not detect Jupyter base_url. Set KAGGLE_PUBLIC_PATH or "
            "pass --base-path /k/<session>/proxy/proxy/8000. "
            "Building with empty basePath makes /_next assets miss the proxy "
            "and the PUBLIC WEBSITE loads forever in the browser."
        )
        return 1

    # FLUX NF4 (eramth/flux-kontext-4bit) needs bitsandbytes before any load.
    # Fresh Kaggle images often ship without it → PackageNotFoundError in Diffusers.
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from src.common.utils.ensure_bitsandbytes import ensure_bitsandbytes

        bnb_ver = ensure_bitsandbytes(auto_install=True)
        _log(f"bitsandbytes verified: {bnb_ver}")
    except Exception as exc:
        _log(f"ERROR: bitsandbytes prerequisite failed: {exc}")
        return 1

    configure_kaggle_flux_runtime()

    do_prefetch = args.prefetch_flux or (
        bool(deploy.get("is_kaggle")) and not args.no_prefetch_flux
    )
    if do_prefetch:
        try:
            prefetch_flux_weights()
        except Exception:
            return 1
    else:
        _log("Skipping FLUX prefetch (first Generate may download weights)")

    stop_port(3000, "frontend")
    stop_port(args.port, "backend")

    built = read_built_base_path()
    need_build = not args.skip_build
    if args.skip_build:
        if not (FRONTEND / ".next").exists():
            _log("No frontend/.next found; building...")
            need_build = True
        elif (built or "") != base_path:
            _log(
                f"Built base path {built!r} != target {base_path!r}; rebuilding "
                "(required for correct Kaggle public assets)."
            )
            need_build = True

    if need_build:
        build_frontend(base_path=base_path)

    next_proc = start_next(base_path=base_path)
    try:
        next_root = f"http://127.0.0.1:3000{base_path}/" if base_path else "http://127.0.0.1:3000/"
        wait_http(next_root, "Next.js root", attempts=90)
    except Exception:
        _log("Next.js did not become ready")
        try:
            next_proc.terminate()
        except Exception:
            pass
        raise

    api_proc = start_fastapi(base_path=base_path)
    children = [next_proc, api_proc]

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
        health_code = wait_http("http://127.0.0.1:8000/api/v1/health", "API health")
        wait_http("http://127.0.0.1:8000/docs", "Swagger /docs")
        wait_http("http://127.0.0.1:8000/openapi.json", "OpenAPI JSON")
        wait_http("http://127.0.0.1:8000/api/v1/openapi.json", "OpenAPI JSON (v1)")
        root_code = wait_http("http://127.0.0.1:8000/", "Gateway frontend /")
        wait_http("http://127.0.0.1:8000/about", "Gateway frontend /about")

        # Local gateway builds must not bake the obsolete host-root /proxy/8000 prefix.
        assert_html_has_no_stale_proxy_prefix("http://127.0.0.1:8000/")
        # Critical: HTML must reference {basePath}/_next/static (not bare /_next).
        assert_html_next_assets_use_base_path("http://127.0.0.1:8000/", base_path)

        next_code, _ = _http_status(
            f"http://127.0.0.1:3000{base_path}/" if base_path else "http://127.0.0.1:3000/"
        )

        public_status: Dict[str, int] = {}
        discovery_report = None
        if deploy.get("is_kaggle") and deploy.get("jupyter_base_url") and not args.no_base_path:
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
                    # Restart Next + FastAPI with the winning base path.
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
                    build_frontend(base_path=base_path)
                    next_proc = start_next(base_path=base_path)
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
                        "http://127.0.0.1:8000/api/v1/health", "API health (rebuild)"
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

                # Final verification of the winning public URL (all required endpoints).
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
                    )
                    _log(
                        "ERROR: Winning public path failed final verification. "
                        "Services left running — Ctrl+C to stop."
                    )
                    while True:
                        for proc, name in ((next_proc, "Next.js"), (api_proc, "FastAPI")):
                            if proc.poll() is not None:
                                _log(f"{name} exited")
                                _shutdown()
                                return 1
                        time.sleep(1.0)
                else:
                    # HTML 200 is not enough — browser hang is usually /_next/static.
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
                        )
                        _log(
                            f"ERROR: Public /_next static verification failed: {asset_exc}"
                        )
                        _log(
                            "Services left running for debugging — Ctrl+C to stop."
                        )
                        while True:
                            for proc, name in (
                                (next_proc, "Next.js"),
                                (api_proc, "FastAPI"),
                            ):
                                if proc.poll() is not None:
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
                )
                _log(
                    "ERROR: No public Kaggle proxy candidate reached the app. "
                    "Local gateway is healthy. Services left running — Ctrl+C to stop."
                )
                if discovery_report:
                    _log(json.dumps(discovery_report, indent=2, default=str)[:4000])
                while True:
                    for proc, name in ((next_proc, "Next.js"), (api_proc, "FastAPI")):
                        if proc.poll() is not None:
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
        )
        _log("Press Ctrl+C to stop.")

        while True:
            for proc, name in ((next_proc, "Next.js"), (api_proc, "FastAPI")):
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
