"""Helpers to keep the FabricVision API port bindable on Windows."""

from __future__ import annotations

import socket
import subprocess
import sys
from typing import Optional


def is_port_free(host: str, port: int) -> bool:
    """Return True if we can bind TCP (host, port)."""
    family = socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _windows_listener_pid(port: int) -> Optional[int]:
    """Return OwningProcess for a LISTENING socket on port, or None."""
    try:
        # Prefer PowerShell for reliable PID on modern Windows.
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                f"(Get-NetTCPConnection -LocalPort {port} -State Listen "
                f"-ErrorAction SilentlyContinue | Select-Object -First 1 "
                f"-ExpandProperty OwningProcess)"
            ),
        ]
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        if out.isdigit():
            return int(out)
    except Exception:
        pass

    try:
        out = subprocess.check_output(["netstat", "-ano"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if f":{port}" not in line or "LISTENING" not in line.upper():
                continue
            parts = line.split()
            if not parts:
                continue
            pid_s = parts[-1]
            if pid_s.isdigit():
                return int(pid_s)
    except Exception:
        pass
    return None


def _process_command_line(pid: int) -> str:
    try:
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                f"(Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\" "
                f"| Select-Object -ExpandProperty CommandLine)"
            ),
        ]
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _looks_like_fabricvision_uvicorn(cmdline: str) -> bool:
    lower = cmdline.lower()
    return "uvicorn" in lower and (
        "backend_api.main:app" in lower or "backend_api.main" in lower
    )


def free_api_port(port: int = 8000, host: str = "127.0.0.1") -> dict:
    """
    Ensure the API port can be bound.

    If a leftover FabricVision uvicorn still holds the port (common after a
    killed terminal / crashed reload child), terminate that process only.
    Does not change the port number (frontend defaults to :8000).
    """
    result = {
        "port": port,
        "was_free": True,
        "freed_pid": None,
        "action": "none",
        "message": f"Port {port} is available",
    }

    # Check both localhost and wildcard — either blocks a new bind.
    if is_port_free("127.0.0.1", port) and is_port_free("0.0.0.0", port):
        return result

    result["was_free"] = False
    pid = _windows_listener_pid(port)
    if pid is None:
        result["action"] = "blocked_unknown"
        result["message"] = (
            f"Port {port} is not free, but no listener PID was found. "
            f"Close any process using port {port} and retry."
        )
        return result

    cmdline = _process_command_line(pid)
    result["freed_pid"] = pid
    if not _looks_like_fabricvision_uvicorn(cmdline):
        result["action"] = "blocked_other"
        result["message"] = (
            f"Port {port} is in use by PID {pid} "
            f"({cmdline[:160] or 'unknown process'}). "
            f"Stop that process, then retry. "
            f"Do not change the API port — the frontend expects :{port}."
        )
        return result

    try:
        subprocess.check_call(
            ["taskkill", "/PID", str(pid), "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        result["action"] = "kill_failed"
        result["message"] = (
            f"Found leftover FabricVision uvicorn PID {pid} on port {port}, "
            f"but could not stop it: {exc}"
        )
        return result

    # Brief settle for Windows TIME_WAIT / handle release
    import time

    for _ in range(20):
        if is_port_free("127.0.0.1", port):
            break
        time.sleep(0.25)

    if is_port_free("127.0.0.1", port):
        result["action"] = "killed_orphan"
        result["message"] = (
            f"Stopped leftover FabricVision uvicorn PID {pid} holding port {port}."
        )
    else:
        result["action"] = "still_busy"
        result["message"] = (
            f"Stopped PID {pid}, but port {port} is still busy. Wait a moment and retry."
        )
    return result


def main(argv: Optional[list[str]] = None) -> int:
    port = 8000
    if argv and len(argv) > 1 and argv[1].isdigit():
        port = int(argv[1])
    info = free_api_port(port=port)
    print(info["message"])
    if info["action"] in ("blocked_other", "blocked_unknown", "kill_failed", "still_busy"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
