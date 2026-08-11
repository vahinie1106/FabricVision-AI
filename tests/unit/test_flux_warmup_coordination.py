"""Unit tests for API-process FLUX warmup coordination and readiness."""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import backend_api.services.flux_warmup as warm


def _reset():
    warm.reset_warmup_state()


def test_get_warmup_status_exposes_public_states(monkeypatch):
    monkeypatch.setattr(warm, "flux_in_memory_ready", lambda: False)
    _reset()
    status = warm.get_warmup_status()
    assert status["state"] == "IDLE"
    assert status["ready"] is False
    assert status["in_memory"] is False
    assert "progress" in status
    assert "pipeline_exists" in status


def test_status_transitions_starting_to_ready(monkeypatch):
    monkeypatch.setattr(warm, "flux_in_memory_ready", lambda: False)
    _reset()
    with warm._lock:
        warm._state.update(
            {
                "state": "loading",
                "pid": 42,
                "started_at": time.perf_counter(),
                "progress": 14,
                "current_step": "Loading FluxKontextPipeline",
                "stage": "LOADING",
            }
        )
    mid = warm.get_warmup_status()
    assert mid["state"] == "STARTING"
    assert mid["progress"] == 14
    assert mid["current_step"] == "Loading FluxKontextPipeline"

    monkeypatch.setattr(warm, "flux_in_memory_ready", lambda: True)
    with warm._lock:
        warm._state.update(
            {
                "state": "ready",
                "progress": 100,
                "current_step": "FLUX READY",
                "stage": "READY",
                "pipeline_exists": True,
                "finished_at": time.perf_counter(),
            }
        )
        warm._ready.set()
    done = warm.get_warmup_status()
    assert done["state"] == "READY"
    assert done["ready"] is True
    assert done["pipeline_exists"] is True


def test_status_transitions_starting_to_failed(monkeypatch):
    monkeypatch.setattr(warm, "flux_in_memory_ready", lambda: False)
    _reset()
    with warm._lock:
        warm._state.update(
            {
                "state": "failed",
                "error": "boom",
                "stage": "FAILED",
                "progress": 8,
            }
        )
        warm._ready.set()
    out = warm.get_warmup_status()
    assert out["state"] == "FAILED"
    assert out["error"] == "boom"
    assert out["ready"] is False


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
    assert out["state"] == "SKIPPED"


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
    assert out["state"] == "FAILED"
    assert out["error"] == "boom"


def test_wait_blocks_while_starting_then_reuses(monkeypatch):
    """Generate must wait on in-flight warmup — not start a second load."""
    monkeypatch.setattr(warm, "flux_in_memory_ready", lambda: False)
    _reset()
    with warm._lock:
        warm._state.update(
            {
                "state": "loading",
                "pid": 7,
                "started_at": time.perf_counter(),
                "progress": 10,
            }
        )
        warm._ready.clear()

    seen = []

    def _flip_ready():
        time.sleep(0.15)
        monkeypatch.setattr(warm, "flux_in_memory_ready", lambda: True)
        with warm._lock:
            warm._state["state"] = "ready"
            warm._state["pipeline_exists"] = True
        warm._ready.set()
        seen.append("flipped")

    threading.Thread(target=_flip_ready, daemon=True).start()
    out = warm.wait_until_flux_ready(timeout_s=2.0, poll_s=0.05)
    assert seen == ["flipped"]
    assert out["state"] == "READY"
    assert out["in_memory"] is True


def test_warmup_single_flight_second_caller_waits(monkeypatch):
    """Second warm_flux call while loading must wait — not start another load."""
    _reset()
    monkeypatch.setenv("FLUX_WARMUP_ON_STARTUP", "force")
    monkeypatch.setattr(warm, "flux_in_memory_ready", lambda: False)
    with warm._lock:
        warm._state.update(
            {
                "state": "loading",
                "pid": 9,
                "started_at": time.perf_counter(),
            }
        )
        warm._ready.clear()

    def finish():
        time.sleep(0.12)
        monkeypatch.setattr(warm, "flux_in_memory_ready", lambda: True)
        with warm._lock:
            warm._state["state"] = "ready"
            warm._state["pipeline_exists"] = True
        warm._ready.set()

    threading.Thread(target=finish, daemon=True).start()
    # force bypasses pytest skip; already_loading → wait_until_flux_ready
    out = warm.warm_flux_in_api_process()
    assert out["state"] == "READY"
    assert out.get("in_memory") is True


def test_prefetch_parent_does_not_set_pipeline():
    """Parent disk resolve must not create a resident pipeline."""
    from src.features.custom_generator.model.flux_model_loader import FLUXModelLoader

    loader = FLUXModelLoader(model_path="models/flux-kontext", allow_fallback=True)
    assert getattr(loader, "pipeline", None) is None
    assert loader._pipeline is None
