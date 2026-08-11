"""Unit tests for API-process FLUX warmup coordination."""

from __future__ import annotations

import backend_api.services.flux_warmup as warm


def test_get_warmup_status_shape(monkeypatch):
    monkeypatch.setattr(warm, "flux_in_memory_ready", lambda: False)
    with warm._lock:
        warm._state.update(
            {
                "state": "idle",
                "pid": None,
                "error": None,
                "result": None,
                "started_at": None,
                "finished_at": None,
            }
        )
        warm._ready.clear()
    status = warm.get_warmup_status()
    assert status["state"] == "idle"
    assert status["ready"] is False
    assert status["in_memory"] is False


def test_wait_returns_when_skipped_disabled(monkeypatch):
    monkeypatch.setattr(warm, "flux_in_memory_ready", lambda: False)
    with warm._lock:
        warm._state.update(
            {
                "state": "skipped",
                "result": {"skipped": True, "reason": "disabled"},
                "error": None,
            }
        )
        warm._ready.set()
    out = warm.wait_until_flux_ready(timeout_s=1.0)
    assert out["state"] == "skipped"


def test_wait_returns_on_failed_so_generate_can_retry(monkeypatch):
    monkeypatch.setattr(warm, "flux_in_memory_ready", lambda: False)
    with warm._lock:
        warm._state.update(
            {
                "state": "failed",
                "error": "boom",
                "result": None,
            }
        )
        warm._ready.set()
    out = warm.wait_until_flux_ready(timeout_s=1.0)
    assert out["state"] == "failed"
    assert out["error"] == "boom"
