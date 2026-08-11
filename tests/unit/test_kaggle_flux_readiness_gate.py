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
