#!/usr/bin/env python3
"""Start FabricVision-AI for Kaggle / single-port public deployment.

Architecture:
  Browser → Kaggle HTTPS proxy → :8000 (FastAPI gateway)
    ├── /api/v1/* , /docs , /openapi.json , /outputs/*  → FastAPI
    └── /*  → reverse-proxy → Next.js on 127.0.0.1:3000

On Kaggle the public path is derived dynamically from the live Jupyter
server ``base_url`` (never hard-coded as only ``/proxy/8000``).

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
from urllib.parse import urljoin

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


def build_public_proxy_info(
    jupyter_base_url: str,
    port: int = 8000,
    proxy_host: Optional[str] = None,
) -> Dict[str, str]:
    """
    Construct the browser-facing path/URL for a port behind Jupyter Server Proxy.

    Uses urljoin against the live Jupyter ``base_url`` so we do not assume a
    fixed ``/proxy/8000``-only shape. Typical Kaggle result:

      base_url   = /k/<kernel>/<token>/proxy/
      public_path = /k/<kernel>/<token>/proxy/proxy/8000
    """
    host = (
        proxy_host
        or os.environ.get("KAGGLE_JUPYTER_PROXY_HOST")
        or KAGGLE_PROXY_HOST_DEFAULT
    ).rstrip("/")

    # Explicit override if an operator already knows the correct public path.
    override = (os.environ.get("KAGGLE_PUBLIC_PATH") or "").strip()
    if override:
        public_path = "/" + override.strip("/") 
    else:
        # urljoin('/k/x/y/proxy/', 'proxy/8000') → '/k/x/y/proxy/proxy/8000'
        public_path = urljoin(jupyter_base_url, f"proxy/{port}").rstrip("/")
        if not public_path.startswith("/"):
            public_path = "/" + public_path

    public_url = f"{host}{public_path}/"
    return {
        "jupyter_base_url": jupyter_base_url,
        "public_path": public_path,
        "public_url": public_url,
        "proxy_host": host,
        "docs_url": f"{host}{public_path}/docs",
        "health_url": f"{host}{public_path}/api/v1/health",
    }


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
        "health_url": None,
        "proxy_host": None,
    }

    if not info["is_kaggle"]:
        return info

    jupyter_base = read_jupyter_base_url()
    if not jupyter_base:
        _log(
            "WARNING: On Kaggle but could not read Jupyter base_url. "
            "Set KAGGLE_PUBLIC_PATH or NEXT_PUBLIC_BASE_PATH manually."
        )
        env_path = (os.environ.get("NEXT_PUBLIC_BASE_PATH") or os.environ.get("KAGGLE_PUBLIC_PATH") or "").strip()
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
            "health_url": public["health_url"],
            "proxy_host": public["proxy_host"],
        }
    )
    return info


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
            body = resp.read(400).decode("utf-8", errors="replace")
            return code, body
    except urllib.error.HTTPError as exc:
        return int(exc.code), str(exc.reason)
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
    env = os.environ.copy()
    env["NEXT_PUBLIC_USE_SAME_ORIGIN"] = "true"
    if base_path:
        env["NEXT_PUBLIC_BASE_PATH"] = base_path
        env["NEXT_PUBLIC_API_URL"] = f"{base_path}/api/v1"
        env["NEXT_PUBLIC_API_ORIGIN"] = base_path
    else:
        env.pop("NEXT_PUBLIC_BASE_PATH", None)
        env["NEXT_PUBLIC_API_URL"] = "/api/v1"
        env["NEXT_PUBLIC_API_ORIGIN"] = ""
    return env


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


def assert_html_has_no_stale_proxy_prefix(url: str, forbidden: str = "/proxy/8000") -> None:
    """Ensure gateway HTML is not still baking the obsolete fixed prefix."""
    code, body = _http_status(url, timeout=10.0)
    if code != 200:
        raise RuntimeError(f"HTML check failed for {url}: HTTP {code}")
    # Only flag when it appears as a path prefix in asset/API links.
    needles = (
        f'"{forbidden}/',
        f"'{forbidden}/",
        f'href="{forbidden}',
        f"href='{forbidden}",
        f'src="{forbidden}',
        f"src='{forbidden}",
    )
    if any(n in body for n in needles):
        raise RuntimeError(
            f"HTML from {url} still contains hard-coded '{forbidden}' asset/API paths"
        )
    _log(f"OK HTML check: {url} has no hard-coded {forbidden} links")


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
    print("PUBLIC WEBSITE:", flush=True)
    print(f"    {deploy.get('public_url') or '(n/a outside Kaggle)'}", flush=True)
    if public_status and deploy.get("public_url"):
        print(f"    HTTP {public_status.get('root', 'n/a')}", flush=True)
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

        # Local gateway builds must not bake the obsolete fixed prefix.
        if not base_path or base_path != "/proxy/8000":
            assert_html_has_no_stale_proxy_prefix("http://127.0.0.1:8000/")

        next_code, _ = _http_status(
            f"http://127.0.0.1:3000{base_path}/" if base_path else "http://127.0.0.1:3000/"
        )

        public_status: Dict[str, int] = {}
        if deploy.get("public_url"):
            _log(f"Probing PUBLIC website: {deploy['public_url']}")
            public_status["root"], _ = _http_status(deploy["public_url"], timeout=20.0)
            if deploy.get("docs_url"):
                public_status["docs"], _ = _http_status(deploy["docs_url"], timeout=20.0)
            if deploy.get("health_url"):
                public_status["health"], _ = _http_status(deploy["health_url"], timeout=20.0)

            # Persist probe result for operators / CI logs.
            probe_path = ROOT / "experiments" / "generation_results" / "kaggle_public_probe.json"
            probe_path.parent.mkdir(parents=True, exist_ok=True)
            probe_path.write_text(
                json.dumps(
                    {
                        "deploy": deploy,
                        "public_status": public_status,
                        "local": {
                            "health": health_code,
                            "root": root_code,
                            "next": next_code,
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            if public_status.get("root") != 200:
                print_banner(
                    health_code=health_code,
                    root_code=root_code,
                    next_code=next_code,
                    deploy=deploy,
                    public_status=public_status,
                )
                _log(
                    f"ERROR: PUBLIC website returned HTTP {public_status.get('root')} "
                    f"for {deploy['public_url']}. Local gateway is healthy; fix the "
                    f"public path mapping. Services left running — Ctrl+C to stop."
                )
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
