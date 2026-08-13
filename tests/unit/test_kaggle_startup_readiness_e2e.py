"""End-to-end readiness-gate validation for Kaggle startup semantics."""

from __future__ import annotations

import inspect
import io
import json
from contextlib import redirect_stdout
from unittest.mock import patch

import pytest


def test_banner_application_ready_only_when_flag_true():
    """Health 200 alone must NOT print APPLICATION READY ✓."""
    import scripts.run_kaggle as rk

    deploy = {
        "jupyter_base_url": None,
        "public_url": None,
        "about_url": None,
        "docs_url": None,
        "health_url": None,
        "base_path": "",
    }
    buf = io.StringIO()
    with redirect_stdout(buf):
        rk.print_banner(
            health_code=200,
            root_code=200,
            next_code=200,
            deploy=deploy,
            flux_status={
                "state": "STARTING",
                "ready": False,
                "pipeline_exists": False,
                "in_memory": False,
                "progress": 8,
            },
            application_ready=False,
        )
    text = buf.getvalue()
    assert "APPLICATION READY ✓" not in text
    assert "FABRICVISION-AI READY" not in text
    assert "APPLICATION NOT READY" in text
    assert "state=STARTING" in text or "State: STARTING" in text


def test_banner_application_ready_requires_flux_ready_payload():
    import scripts.run_kaggle as rk

    deploy = {
        "jupyter_base_url": None,
        "public_url": "https://example/proxy/",
        "about_url": None,
        "docs_url": None,
        "health_url": None,
        "base_path": "/proxy",
    }
    buf = io.StringIO()
    with redirect_stdout(buf):
        rk.print_banner(
            health_code=200,
            root_code=200,
            next_code=200,
            deploy=deploy,
            flux_status={
                "state": "READY",
                "ready": True,
                "pipeline_exists": True,
                "in_memory": True,
                "api_pid": 988,
                "load_duration_s": 36.0,
            },
            application_ready=True,
        )
    text = buf.getvalue()
    assert "FABRICVISION-AI READY" in text
    assert "State: READY" in text or "state=READY" in text
    assert "pipeline_exists=True" in text or "ready=True" in text


def test_wait_logs_progress_while_starting(monkeypatch):
    import scripts.run_kaggle as rk

    calls = {"n": 0}
    logs: list[str] = []
    payloads = [
        {
            "state": "STARTING",
            "ready": False,
            "in_memory": False,
            "pipeline_exists": False,
            "progress": 14,
            "current_step": "Loading FluxKontextPipeline (T5/CLIP/VAE)",
            "load_duration_s": 20,
        },
        {
            "state": "READY",
            "ready": True,
            "in_memory": True,
            "pipeline_exists": True,
            "api_pid": 988,
            "cache_status": "hybrid",
            "load_duration_s": 36,
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
    monkeypatch.setattr(rk, "_log", lambda msg: logs.append(str(msg)))

    out = rk.wait_api_flux_ready(timeout_s=5.0)
    assert out["state"] == "READY"
    assert out["ready"] is True
    assert out["pipeline_exists"] is True
    assert out["in_memory"] is True
    assert any("STARTING" in m and "14%" in m for m in logs)
    assert any("Loading FluxKontextPipeline" in m for m in logs)
    assert any("API FLUX READY" in m for m in logs)


def test_failed_flux_raises_before_application_ready(monkeypatch):
    import scripts.run_kaggle as rk

    payload = {
        "state": "FAILED",
        "ready": False,
        "in_memory": False,
        "pipeline_exists": False,
        "error": "RuntimeError: boom",
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

    logs: list[str] = []
    monkeypatch.setattr(rk.urllib.request, "urlopen", lambda *a, **k: _Resp())
    monkeypatch.setattr(rk, "_log", lambda msg: logs.append(str(msg)))
    with pytest.raises(RuntimeError, match="boom"):
        rk.require_api_flux_ready(timeout_s=2.0)
    assert any("FLUX WARMUP FAILED" in m for m in logs)


def test_main_source_requires_flux_ready_on_initial_and_rebuild_paths():
    """Static invariant: rebuild path must also call require_api_flux_ready."""
    import scripts.run_kaggle as rk

    src = inspect.getsource(rk.main)
    assert src.count("require_api_flux_ready(") >= 2
    banner = inspect.getsource(rk.print_banner)
    assert "FABRICVISION-AI READY" in banner
    assert "APPLICATION NOT READY" in banner
    # Final success banner must pass application_ready=True
    assert "application_ready=True" in src
    # Failure/public-error banners must not claim ready
    assert src.count("application_ready=False") >= 2


def test_health_liveness_does_not_imply_flux_ready(monkeypatch):
    """ /api/v1/health stays 200 even when FLUX is STARTING. """
    monkeypatch.setenv("FLUX_WARMUP_ON_STARTUP", "false")
    import backend_api.services.flux_warmup as warm

    warm.reset_warmup_state()
    with warm._lock:
        warm._state.update(
            {
                "state": "loading",
                "progress": 8,
                "current_step": "Initializing FLUX",
                "started_at": 1.0,
            }
        )
        warm._ready.clear()

    from fastapi.testclient import TestClient
    from backend_api.main import app

    client = TestClient(app)
    h = client.get("/api/v1/health")
    assert h.status_code == 200
    assert h.json()["status"] == "healthy"

    fs = client.get("/api/v1/flux-status")
    assert fs.status_code == 200
    body = fs.json()
    assert body["state"] == "STARTING"
    assert body["ready"] is False


def test_generate_reuses_resident_pipeline_no_second_from_pretrained(monkeypatch):
    """After warmup READY, FluxManager.load must reuse (no second from_pretrained)."""
    from src.integrations.flux.flux_manager import FluxManager

    mgr = FluxManager(model_path="models/flux-kontext", allow_fallback=True)
    fake_loader = type(
        "L",
        (),
        {
            "pipeline": object(),
            "_cache_status": "hybrid",
            "load": lambda self, progress_callback=None: self.pipeline,
            "hf_model_id": "test",
        },
    )()
    mgr.loader = fake_loader
    loads = {"n": 0}

    def _boom(*_a, **_k):
        loads["n"] += 1
        raise AssertionError("from_pretrained must not run on reuse")

    with patch(
        "src.features.custom_generator.model.flux_model_loader.FLUXModelLoader",
        side_effect=_boom,
    ):
        out = mgr.load()
    assert out is fake_loader.pipeline
    assert loads["n"] == 0


def test_vram_policy_unchanged_t4_safe():
    from src.features.custom_generator.inference.flux_vram_policy import (
        select_standard_generation_policy,
    )
    import os

    for k in (
        "FLUX_GENERATION_RESOLUTION",
        "FLUX_PRODUCTION_SIZE",
        "FLUX_STANDARD_STEPS",
        "FLUX_ALLOW_HIGH_RES",
        "FLUX_MODEL_CPU_OFFLOAD",
    ):
        os.environ.pop(k, None)
    p = select_standard_generation_policy(
        physical_mb=15109.0,
        free_mb=2500.0,
        offload_strategy="gpu_resident",
    )
    assert p.profile == "standard_t4_safe"
    assert p.height == 512 and p.width == 512
    assert p.num_inference_steps == 8
    assert p.guidance_scale == 3.0
    assert p.enable_vae_tiling is True


def test_custom_garment_page_gates_generate_on_flux_warmup():
    from pathlib import Path

    page = Path("frontend/src/app/studio/custom-garment/page.tsx").read_text(
        encoding="utf-8"
    )
    hook = Path("frontend/src/hooks/useFluxWarmupStatus.ts").read_text(encoding="utf-8")
    ui = Path("frontend/src/lib/fluxWarmupUi.ts").read_text(encoding="utf-8")
    assert "useFluxWarmupStatus" in page
    assert "generateEnabledByFlux" in page
    assert "showWarmupBanner" in page
    assert "AI engine is warming up" in page
    assert "AI engine ready" in page
    assert "AI engine failed to initialize" in page
    assert "Waiting for AI engine" in page
    assert "deriveFluxWarmupUi" in hook
    assert "state === \"STARTING\"" in ui or 'state === "STARTING"' in ui
    assert "IDLE" not in [
        ln for ln in ui.splitlines() if "warming" in ln and "=" in ln
    ][0]


def test_frontend_env_split_proxy_preserves_backend_port():
    import scripts.run_kaggle as rk

    env = rk._frontend_env("", split_proxy=True)
    assert env["NEXT_PUBLIC_API_URL"] == "/proxy/8000/api/v1"
    assert env["NEXT_PUBLIC_BACKEND_URL"] == "/proxy/8000"
    assert env["NEXT_PUBLIC_USE_SAME_ORIGIN"] == "false"
    assert "NEXT_PUBLIC_BASE_PATH" not in env


def test_frontend_env_gateway_uses_relative_api():
    import scripts.run_kaggle as rk

    env = rk._frontend_env("/k/abc/proxy/proxy/8000", split_proxy=False)
    assert env["NEXT_PUBLIC_API_URL"] == "/api/v1"
    assert env["NEXT_PUBLIC_BASE_PATH"] == "/k/abc/proxy/proxy/8000"
    assert env["NEXT_PUBLIC_USE_SAME_ORIGIN"] == "true"


def test_main_default_skips_blocking_prefetch_before_services():
    """Parent prefetch must not be the default (health must come up first)."""
    import inspect
    import scripts.run_kaggle as rk

    src = inspect.getsource(rk.main)
    assert "do_prefetch = bool(args.prefetch_flux)" in src
    assert "Backend starting..." in src
    assert "Frontend starting..." in src
    assert src.index("Backend starting...") < src.index("Frontend starting...")
    assert "Skipping parent FLUX prefetch" in src


def test_proxy_proxy_path_is_intentional_jsp_nesting():
    """Document that /proxy/proxy/PORT is valid when Jupyter base_url ends with /proxy/."""
    import scripts.run_kaggle as rk

    paths = rk.candidate_public_paths("/k/sess/proxy/", port=8000)
    assert "/k/sess/proxy/proxy/8000" in paths
    assert "/proxy/8000" in paths


def test_write_frontend_dotenv_rejects_loopback(tmp_path, monkeypatch):
    import scripts.run_kaggle as rk

    fe = tmp_path / "frontend"
    fe.mkdir()
    monkeypatch.setattr(rk, "FRONTEND", fe)

    def bad_env(base_path, *, split_proxy=False):
        return {
            "NEXT_PUBLIC_API_URL": "http://127.0.0.1:8000/api/v1",
            "NEXT_PUBLIC_API_BASE_URL": "http://127.0.0.1:8000/api/v1",
            "NEXT_PUBLIC_BACKEND_URL": "http://127.0.0.1:8000",
            "NEXT_PUBLIC_API_ORIGIN": "http://127.0.0.1:8000",
            "NEXT_PUBLIC_USE_SAME_ORIGIN": "false",
        }

    monkeypatch.setattr(rk, "_frontend_env", bad_env)
    try:
        rk.write_frontend_dotenv("", split_proxy=True)
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "127.0.0.1" in str(exc) or "loopback" in str(exc).lower()


def test_frontend_dotenv_is_kaggle_safe(tmp_path, monkeypatch):
    import scripts.run_kaggle as rk

    fe = tmp_path / "frontend"
    fe.mkdir()
    monkeypatch.setattr(rk, "FRONTEND", fe)
    dotenv = fe / ".env.local"
    dotenv.write_text(
        "NEXT_PUBLIC_API_URL=/proxy/8000/api/v1\n"
        "NEXT_PUBLIC_USE_SAME_ORIGIN=false\n"
        "NEXT_PUBLIC_FORBID_LOOPBACK=true\n",
        encoding="utf-8",
    )
    assert rk.frontend_dotenv_is_kaggle_safe(split_proxy=True) is True
    dotenv.write_text(
        "NEXT_PUBLIC_API_URL=http://127.0.0.1:8000/api/v1\n",
        encoding="utf-8",
    )
    assert rk.frontend_dotenv_is_kaggle_safe(split_proxy=True) is False


def test_home_page_contains_fabricvision_20_markers():
    from pathlib import Path

    page = Path("frontend/src/app/page.tsx").read_text(encoding="utf-8")
    for marker in (
        "FabricVision-AI 2.0",
        "Transform Fabrics Into",
        "Intelligent Fashion",
        "Start Creating",
        "Explore Technology",
        "Input Concept",
        "Raw Linen Fabric",
        "FLUX Synthesis Active",
        "Generated AI Garment",
        "AI Fashion Intelligence",
        "Virtual Try-On",
        "Semantic Extraction",
    ):
        assert marker in page


def test_studio_index_is_redirect_only():
    from pathlib import Path

    page = Path("frontend/src/app/studio/page.tsx").read_text(encoding="utf-8")
    assert "redirect(" in page
    assert "/studio/custom-garment" in page
    assert "dashboard" not in page.lower()
    assert "sidebar" not in page.lower()


def test_kaggle_benchmark_prefers_tracked_test_image():
    from pathlib import Path

    src = Path("scripts/kaggle_flux_production_benchmark.py").read_text(encoding="utf-8")
    e2e = Path("scripts/kaggle_e2e_integration.py").read_text(encoding="utf-8")
    tracked = 'ROOT / "tests" / "test_images" / "test_img_1.jpg"'
    assert tracked in src
    assert tracked in e2e
    # Tracked sample must be listed before gitignored uploads in the benchmark.
    assert src.index(tracked) < src.index('ROOT / "data" / "uploads"')


def test_run_api_does_not_enable_reload_by_default():
    from pathlib import Path

    src = Path("run_api.py").read_text(encoding="utf-8")
    assert "UVICORN_RELOAD" in src
    assert 'reload=True' not in src.replace("reload=reload_flag", "")
