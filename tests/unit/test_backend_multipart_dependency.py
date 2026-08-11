"""FastAPI File()/Form() routes require python-multipart at import time."""

from __future__ import annotations

import importlib
import importlib.util


def test_python_multipart_is_importable():
    """Bare `fastapi` does not install this — requirements.txt must pin it."""
    spec = importlib.util.find_spec("python_multipart") or importlib.util.find_spec(
        "multipart"
    )
    assert spec is not None, (
        "python-multipart is missing. FastAPI Form()/File() routes will crash "
        "uvicorn during backend_api.main import on a fresh Kaggle install."
    )
    try:
        from python_multipart import __version__ as version
    except Exception:
        import multipart

        version = getattr(multipart, "__version__", "unknown")
    assert version


def test_backend_main_imports_with_generation_form_routes():
    """Regression: importing generation Form routes must not raise multipart RuntimeError."""
    # Fresh import of the generation router (File/Form) exercises ensure_multipart.
    import backend_api.routes.generation as gen

    importlib.reload(gen)
    assert gen.router is not None
    route_paths = [getattr(r, "path", "") for r in gen.router.routes]
    assert any("generate" in p for p in route_paths)

    import backend_api.main as main_mod

    app = main_mod.app
    assert app.title
    assert any(
        getattr(r, "path", "") in ("/api/v1/health", "/health", "/health/")
        for r in app.routes
    )
