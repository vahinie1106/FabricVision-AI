"""Kaggle startup must not declare APPLICATION READY until FLUX is READY."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


def test_require_api_flux_ready_accepts_ready(monkeypatch):
    import scripts.run_kaggle as rk

    payload = {
        "state": "READY",
        "ready": True,
        "in_memory": True,
        "pipeline_exists": True,
        "api_pid": 99,
        "load_duration_s": 12.5,
        "cache_status": "hybrid",
    }

    class _Resp:
        headers = {"Content-Type": "application/json"}
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, _n):
            return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(
        rk.urllib.request,
        "urlopen",
        lambda *a, **k: _Resp(),
    )
    out = rk.require_api_flux_ready(timeout_s=2.0)
    assert out["state"] == "READY"


def test_require_api_flux_ready_raises_on_failed(monkeypatch):
    import scripts.run_kaggle as rk

    payload = {"state": "FAILED", "ready": False, "error": "cuda oom"}

    class _Resp:
        headers = {"Content-Type": "application/json"}
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, _n):
            return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(rk.urllib.request, "urlopen", lambda *a, **k: _Resp())
    with pytest.raises(RuntimeError, match="cuda oom"):
        rk.require_api_flux_ready(timeout_s=2.0)


def test_wait_api_flux_ready_does_not_treat_starting_as_ready(monkeypatch):
    import scripts.run_kaggle as rk

    calls = {"n": 0}
    payloads = [
        {
            "state": "STARTING",
            "ready": False,
            "in_memory": False,
            "progress": 8,
            "current_step": "Initializing FLUX",
        },
        {
            "state": "READY",
            "ready": True,
            "in_memory": True,
            "pipeline_exists": True,
            "api_pid": 1,
        },
    ]

    class _Resp:
        def __init__(self, body: bytes):
            self._body = body
            self.headers = {"Content-Type": "application/json"}
            self.status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, _n):
            return self._body

    def _open(*_a, **_k):
        idx = min(calls["n"], len(payloads) - 1)
        calls["n"] += 1
        return _Resp(json.dumps(payloads[idx]).encode("utf-8"))

    monkeypatch.setattr(rk.urllib.request, "urlopen", _open)
    monkeypatch.setattr(rk.time, "sleep", lambda *_: None)
    out = rk.wait_api_flux_ready(timeout_s=5.0)
    assert out["state"] == "READY"
    assert calls["n"] >= 2


def test_wait_api_flux_ready_logs_parse_failure_not_silent_unknown(monkeypatch, capsys):
    import scripts.run_kaggle as rk

    class _Resp:
        headers = {"Content-Type": "text/html"}
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, _n):
            return b"<html>not json</html>"

    n = {"i": 0}

    def _open(*_a, **_k):
        n["i"] += 1
        if n["i"] < 3:
            return _Resp()
        class _Ok:
            headers = {"Content-Type": "application/json"}
            status = 200
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def read(self, _n):
                return json.dumps(
                    {"state": "READY", "ready": True, "in_memory": True}
                ).encode()
        return _Ok()

    monkeypatch.setattr(rk.urllib.request, "urlopen", _open)
    monkeypatch.setattr(rk.time, "sleep", lambda *_: None)
    out = rk.wait_api_flux_ready(timeout_s=5.0)
    assert out["state"] == "READY"
    logged = capsys.readouterr().out
    assert "Could not parse FLUX status JSON" in logged
    assert "preview=" in logged


def test_wait_api_flux_ready_preserves_last_useful_state_on_poll_error(monkeypatch, capsys):
    """Transient poll/parse errors must not wipe a known STARTING status."""
    import scripts.run_kaggle as rk

    n = {"i": 0}

    def _open(*_a, **_k):
        n["i"] += 1

        class _Resp:
            status = 200

            def __init__(self):
                if n["i"] == 1:
                    self.headers = {"Content-Type": "application/json"}
                    self._body = json.dumps(
                        {
                            "state": "STARTING",
                            "ready": False,
                            "progress": 42,
                            "current_step": "Downloading FLUX weights",
                            "load_duration_s": 12.5,
                        }
                    ).encode()
                elif n["i"] == 2:
                    self.headers = {"Content-Type": "text/html"}
                    self._body = b"<html>bad gateway</html>"
                else:
                    self.headers = {"Content-Type": "application/json"}
                    self._body = json.dumps(
                        {"state": "READY", "ready": True, "in_memory": True}
                    ).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, _n):
                return self._body

        return _Resp()

    logs: list[str] = []

    def _capture(msg: str) -> None:
        logs.append(msg)

    monkeypatch.setattr(rk.urllib.request, "urlopen", _open)
    monkeypatch.setattr(rk.time, "sleep", lambda *_: None)
    monkeypatch.setattr(rk, "_log", _capture)
    out = rk.wait_api_flux_ready(timeout_s=5.0)
    assert out["state"] == "READY"
    mid = [m for m in logs if "Waiting for API-process FLUX" in m and "progress=42" in m]
    assert mid, f"expected preserved STARTING progress in logs, got: {logs}"
    assert any("Could not parse FLUX status JSON" in m for m in logs)
    assert any("state=STARTING" in m and "poll_error=" in m for m in logs)