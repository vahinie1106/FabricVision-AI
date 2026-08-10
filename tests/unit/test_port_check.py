"""Unit tests for API port availability helpers."""

from backend_api.utils.port_check import (
    _looks_like_fabricvision_uvicorn,
    is_port_free,
)


def test_is_port_free_ephemeral():
    # Binding port 0 asks the OS for a free ephemeral port — always free to probe.
    assert is_port_free("127.0.0.1", 0) is True


def test_detects_fabricvision_uvicorn_cmdline():
    assert _looks_like_fabricvision_uvicorn(
        r'"C:\Python\python.exe" -m uvicorn backend_api.main:app --host 127.0.0.1 --port 8000'
    )
    assert not _looks_like_fabricvision_uvicorn(
        r'"C:\Python\python.exe" -m http.server 8000'
    )
