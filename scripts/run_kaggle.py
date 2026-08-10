#!/usr/bin/env python3
"""Start FabricVision-AI for Kaggle / single-port public deployment.

Architecture:
  Browser → :8000 (FastAPI)
    ├── /api/v1/* , /docs , /openapi.json , /outputs/*  → FastAPI
    └── /*  → reverse-proxy → Next.js on 127.0.0.1:3000

Usage (from repo root):
  python scripts/run_kaggle.py
  python scripts/run_kaggle.py --skip-build   # reuse existing frontend/.next
  python scripts/run_kaggle.py --no-base-path  # local gateway without /proxy/8000

Environment:
  NEXT_PUBLIC_BASE_PATH   default /proxy/8000 on this script (Kaggle Jupyter proxy)
  FRONTEND_UPSTREAM       default http://127.0.0.1:3000
  FABRICVISION_CORS_ORIGINS  optional comma-separated extra origins
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"


def _log(msg: str) -> None:
    print(f"[run_kaggle] {msg}", flush=True)


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def _http_ok(url: str, timeout: float = 5.0) -> tuple[bool, int, str]:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = getattr(resp, "status", 200)
            body = resp.read(200).decode("utf-8", errors="replace")
            return 200 <= int(code) < 400, int(code), body
    except urllib.error.HTTPError as exc:
        return False, int(exc.code), str(exc.reason)
    except Exception as exc:
        return False, 0, str(exc)


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

    # Linux / Kaggle
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


def build_frontend(base_path: str, same_origin: bool) -> None:
    env = os.environ.copy()
    env["NEXT_PUBLIC_USE_SAME_ORIGIN"] = "true" if same_origin else env.get(
        "NEXT_PUBLIC_USE_SAME_ORIGIN", "true"
    )
    if base_path:
        env["NEXT_PUBLIC_BASE_PATH"] = base_path
        env["NEXT_PUBLIC_API_URL"] = f"{base_path}/api/v1"
        env["NEXT_PUBLIC_API_ORIGIN"] = base_path
    else:
        env.pop("NEXT_PUBLIC_BASE_PATH", None)
        env["NEXT_PUBLIC_API_URL"] = "/api/v1"
        env["NEXT_PUBLIC_API_ORIGIN"] = ""

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


def start_next(base_path: str) -> subprocess.Popen:
    env = os.environ.copy()
    if base_path:
        env["NEXT_PUBLIC_BASE_PATH"] = base_path
        env["NEXT_PUBLIC_API_URL"] = f"{base_path}/api/v1"
        env["NEXT_PUBLIC_API_ORIGIN"] = base_path
    else:
        env.pop("NEXT_PUBLIC_BASE_PATH", None)
        env["NEXT_PUBLIC_API_URL"] = "/api/v1"
        env["NEXT_PUBLIC_USE_SAME_ORIGIN"] = "true"
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


def wait_http(url: str, label: str, attempts: int = 60) -> None:
    for i in range(attempts):
        ok, code, detail = _http_ok(url)
        if ok:
            _log(f"OK {label}: {url} → HTTP {code}")
            return
        time.sleep(1.0)
    raise RuntimeError(f"{label} failed validation: {url} ({detail})")


def main() -> int:
    parser = argparse.ArgumentParser(description="FabricVision-AI Kaggle / gateway launcher")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip npm build (reuse frontend/.next)",
    )
    parser.add_argument(
        "--no-base-path",
        action="store_true",
        help="Do not set NEXT_PUBLIC_BASE_PATH (local single-port gateway)",
    )
    parser.add_argument(
        "--base-path",
        default=None,
        help="Override public base path (default: /proxy/8000)",
    )
    args = parser.parse_args()

    os.chdir(ROOT)

    if args.no_base_path:
        base_path = ""
    elif args.base_path is not None:
        base_path = args.base_path.rstrip("/")
    else:
        base_path = (os.environ.get("NEXT_PUBLIC_BASE_PATH") or "/proxy/8000").rstrip("/")

    stop_port(3000, "frontend")
    stop_port(8000, "backend")

    if not args.skip_build:
        build_frontend(base_path=base_path, same_origin=True)
    elif not (FRONTEND / ".next").exists():
        _log("No frontend/.next found; building...")
        build_frontend(base_path=base_path, same_origin=True)

    next_proc = start_next(base_path=base_path)
    try:
        # Warm Next before FastAPI starts proxying
        wait_http(
            f"http://127.0.0.1:3000{base_path}/",
            "Next.js root",
            attempts=90,
        )
    except Exception:
        _log("Next.js did not become ready; dumping recent output...")
        if next_proc.stdout:
            try:
                # Non-blocking-ish read of whatever is buffered
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
        wait_http("http://127.0.0.1:8000/api/v1/health", "API health")
        wait_http("http://127.0.0.1:8000/docs", "Swagger /docs")
        wait_http("http://127.0.0.1:8000/openapi.json", "OpenAPI JSON")
        wait_http("http://127.0.0.1:8000/api/v1/openapi.json", "OpenAPI JSON (v1)")
        # Gateway frontend: FastAPI receives stripped paths; Next expects basePath.
        wait_http("http://127.0.0.1:8000/", "Gateway frontend /")
        wait_http("http://127.0.0.1:8000/about", "Gateway frontend /about")

        _log("")
        _log("FabricVision-AI is up.")
        _log("Local entry:  http://127.0.0.1:8000/")
        _log("API health:   http://127.0.0.1:8000/api/v1/health")
        _log("Swagger:      http://127.0.0.1:8000/docs")
        if base_path:
            _log(
                "Kaggle proxy: https://<jupyter-proxy-host>"
                f"{base_path}/"
            )
            _log(f"Frontend API base (built-in): {base_path}/api/v1")
        else:
            _log("Frontend API base (built-in): /api/v1")
        _log("Press Ctrl+C to stop.")

        # Keep running; surface child exits
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
