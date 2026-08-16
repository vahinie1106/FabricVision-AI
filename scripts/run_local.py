#!/usr/bin/env python3
"""Start FabricVision-AI for local development (Windows / Linux).

This is the local counterpart to scripts/run_kaggle.py. It restores
frontend/.env.local to loopback FastAPI URLs and starts:

  Next.js  http://127.0.0.1:3000
  FastAPI  http://127.0.0.1:8000
  API      http://127.0.0.1:8000/api/v1

Usage (from repo root, venv activated):

  python scripts/run_local.py

Do not run this on Kaggle — use scripts/run_kaggle.py there.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
ENV_LOCAL = FRONTEND / ".env.local"

LOCAL_DOTENV = """# Local Next.js development — do not commit.
# scripts/run_kaggle.py overwrites this file; re-run scripts/run_local.py to restore.
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000/api/v1
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000/api/v1
NEXT_PUBLIC_BACKEND_URL=http://127.0.0.1:8000
NEXT_PUBLIC_API_ORIGIN=http://127.0.0.1:8000
NEXT_PUBLIC_USE_SAME_ORIGIN=false
NEXT_PUBLIC_FORBID_LOOPBACK=false
NEXT_PUBLIC_DEFAULT_GENERATION_MODE=Production
"""


def _log(msg: str) -> None:
    print(f"[run_local] {msg}", flush=True)


def write_local_dotenv() -> None:
    ENV_LOCAL.write_text(LOCAL_DOTENV, encoding="utf-8")
    _log(f"Wrote {ENV_LOCAL} (API=http://127.0.0.1:8000/api/v1)")


def _http_status(url: str, timeout: float = 5.0) -> int:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return int(getattr(resp, "status", 200))
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except Exception:
        return 0


def wait_http(url: str, label: str, attempts: int = 90) -> int:
    last = 0
    for _ in range(attempts):
        last = _http_status(url)
        if last == 200:
            _log(f"OK {label}: {url} -> HTTP {last}")
            return last
        time.sleep(1.0)
    raise RuntimeError(f"{label} failed: {url} → HTTP {last}")


def main() -> int:
    if Path("/kaggle").exists() or os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        _log("ERROR: On Kaggle — use python scripts/run_kaggle.py instead.")
        return 1

    os.chdir(ROOT)
    if not (FRONTEND / "package.json").exists():
        _log(f"ERROR: Next.js app not found at {FRONTEND}")
        return 1

    write_local_dotenv()

    children: list[subprocess.Popen] = []

    def _shutdown(*_args) -> None:
        _log("Shutting down...")
        for proc in children:
            if proc.poll() is not None:
                continue
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                proc.terminate()
        time.sleep(1.0)
        for proc in children:
            if proc.poll() is None:
                proc.kill()

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONPATH", str(ROOT))
    env.setdefault("FLUX_WARMUP_ON_STARTUP", "false")
    env.pop("HOST", None)
    env.pop("HOSTNAME", None)
    env["PORT"] = "3000"

    py = sys.executable
    _log("Starting FastAPI on 127.0.0.1:8000 ...")
    api = subprocess.Popen(
        [py, str(ROOT / "run_api.py")],
        cwd=str(ROOT),
        env=env,
    )
    children.append(api)

    fe_env = env.copy()
    _log("Starting Next.js on 127.0.0.1:3000 (frontend/) ...")
    npm = "npm.cmd" if os.name == "nt" else "npm"
    next_proc = subprocess.Popen(
        [npm, "run", "dev"],
        cwd=str(FRONTEND),
        env=fe_env,
    )
    children.append(next_proc)

    try:
        wait_http("http://127.0.0.1:8000/api/v1/health", "API health")
        wait_http("http://127.0.0.1:3000/", "Next.js root")
    except Exception as exc:
        _log(f"Startup validation failed: {exc}")
        _shutdown()
        return 1

    print("", flush=True)
    print("LOCAL READY", flush=True)
    print("  Frontend:  http://127.0.0.1:3000", flush=True)
    print("  Backend:   http://127.0.0.1:8000", flush=True)
    print("  API:       http://127.0.0.1:8000/api/v1", flush=True)
    print("  Health:    http://127.0.0.1:8000/api/v1/health", flush=True)
    print("Press Ctrl+C to stop.", flush=True)

    try:
        while True:
            for proc, name in ((next_proc, "Next.js"), (api, "FastAPI")):
                code = proc.poll()
                if code is not None:
                    _log(f"{name} exited with code {code}")
                    _shutdown()
                    return code or 1
            time.sleep(1.0)
    except KeyboardInterrupt:
        _shutdown()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
