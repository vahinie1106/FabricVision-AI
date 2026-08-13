"""Regression: Custom Garment must not treat IDLE/UNKNOWN as FLUX warming."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
UI_TS = ROOT / "frontend" / "src" / "lib" / "fluxWarmupUi.ts"
HOOK_TS = ROOT / "frontend" / "src" / "hooks" / "useFluxWarmupStatus.ts"
PAGE_TSX = ROOT / "frontend" / "src" / "app" / "studio" / "custom-garment" / "page.tsx"


def _derive(state: str | None, *, ready=False, in_memory=False, progress=0, poll_error=None, error=None):
    """Mirror of frontend/src/lib/fluxWarmupUi.ts deriveFluxWarmupUi (keep in sync)."""
    raw = (state or "UNKNOWN").upper()
    if raw == "LOADING":
        raw = "STARTING"
    if raw not in {"IDLE", "STARTING", "READY", "FAILED", "SKIPPED", "UNKNOWN"}:
        raw = "UNKNOWN"
    is_ready = raw == "READY" or bool(ready) or bool(in_memory)
    warming = raw == "STARTING" and not is_ready
    failed = raw == "FAILED"
    return {
        "state": raw,
        "warming": warming,
        "ready": is_ready,
        "failed": failed,
        "generateEnabledByFlux": not warming,
        "showWarmupBanner": warming,
        "progress": max(1, min(100, int(progress) or 1)) if warming else max(0, min(100, int(progress) or 0)),
        "error": (error or poll_error or None),
    }


def test_idle_is_not_warming_and_generate_enabled():
    ui = _derive("IDLE", progress=0)
    assert ui["warming"] is False
    assert ui["showWarmupBanner"] is False
    assert ui["generateEnabledByFlux"] is True
    assert ui["progress"] == 0


def test_unknown_is_not_warming_and_generate_enabled():
    ui = _derive("UNKNOWN", progress=0)
    assert ui["warming"] is False
    assert ui["showWarmupBanner"] is False
    assert ui["generateEnabledByFlux"] is True


def test_temporary_poll_failure_is_not_warming():
    ui = _derive(None, poll_error="Failed to fetch", progress=0)
    assert ui["state"] == "UNKNOWN"
    assert ui["warming"] is False
    assert ui["showWarmupBanner"] is False
    assert ui["generateEnabledByFlux"] is True
    assert ui["error"] == "Failed to fetch"


def test_starting_shows_real_warmup_banner_and_blocks_generate():
    ui = _derive("STARTING", progress=17)
    assert ui["warming"] is True
    assert ui["showWarmupBanner"] is True
    assert ui["generateEnabledByFlux"] is False
    assert ui["progress"] == 17


def test_loading_alias_maps_to_starting():
    ui = _derive("LOADING", progress=9)
    assert ui["state"] == "STARTING"
    assert ui["warming"] is True
    assert ui["progress"] == 9


def test_failed_is_recoverable_generate_enabled():
    ui = _derive("FAILED", progress=12, error="CUDA OOM")
    assert ui["warming"] is False
    assert ui["failed"] is True
    assert ui["showWarmupBanner"] is False
    assert ui["generateEnabledByFlux"] is True
    assert ui["error"] == "CUDA OOM"


def test_ready_generate_immediately_usable():
    ui = _derive("READY", ready=True, in_memory=True, progress=100)
    assert ui["warming"] is False
    assert ui["ready"] is True
    assert ui["generateEnabledByFlux"] is True
    assert ui["showWarmupBanner"] is False


def test_flux_warmup_ui_source_encodes_starting_only_rule():
    text = UI_TS.read_text(encoding="utf-8")
    assert "state === \"STARTING\"" in text or 'state === "STARTING"' in text
    assert "generateEnabledByFlux: !warming" in text
    assert "showWarmupBanner: warming" in text
    # Must not treat IDLE/UNKNOWN as warming in the derive helper.
    warming_line = [
        ln for ln in text.splitlines() if "const warming" in ln or "warming =" in ln
    ]
    assert warming_line, "expected warming assignment in fluxWarmupUi.ts"
    joined = "\n".join(warming_line)
    assert "IDLE" not in joined
    assert "UNKNOWN" not in joined


def test_custom_garment_page_uses_generateEnabledByFlux_not_failed_lock():
    page = PAGE_TSX.read_text(encoding="utf-8")
    hook = HOOK_TS.read_text(encoding="utf-8")
    assert "deriveFluxWarmupUi" in hook
    assert "generateEnabledByFlux" in page
    assert "showWarmupBanner" in page
    assert "fluxWarmup.failed" not in page.split("disabled=")[1].split("}")[0]
    assert "AI engine is warming up" in page
    # Do not permanently disable Generate on FAILED.
    assert 'fluxWarmup.failed\n            }' not in page
    assert "AI engine unavailable" not in page
