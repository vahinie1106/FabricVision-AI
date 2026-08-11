"""Tests for Kaggle public HTML /_next asset path checks."""

from __future__ import annotations

import pytest

from scripts.run_kaggle import (
    extract_next_static_refs,
    assert_html_next_assets_use_base_path,
)


def test_extract_next_static_refs_finds_js_and_css():
    html = """
    <html><head>
      <link rel="stylesheet" href="/k/abc/proxy/proxy/8000/_next/static/css/app.css"/>
      <script src="/k/abc/proxy/proxy/8000/_next/static/chunks/main.js"></script>
    </head></html>
    """
    refs = extract_next_static_refs(html)
    assert len(refs) == 2
    assert refs[0].endswith("app.css")
    assert refs[1].endswith("main.js")


def test_assert_rejects_bare_next_when_base_path_set(monkeypatch):
    html = '<script src="/_next/static/chunks/main.js"></script>'

    def fake_status(url: str, timeout: float = 8.0):
        return 200, html

    monkeypatch.setattr("scripts.run_kaggle._http_status", fake_status)
    with pytest.raises(RuntimeError, match="bare /_next"):
        assert_html_next_assets_use_base_path(
            "http://127.0.0.1:8000/",
            "/k/abc/proxy/proxy/8000",
        )


def test_assert_accepts_prefixed_assets(monkeypatch):
    base = "/k/abc/proxy/proxy/8000"
    html = f'<script src="{base}/_next/static/chunks/main.js"></script>'

    def fake_status(url: str, timeout: float = 8.0):
        return 200, html

    monkeypatch.setattr("scripts.run_kaggle._http_status", fake_status)
    refs = assert_html_next_assets_use_base_path("http://127.0.0.1:8000/", base)
    assert refs == [f"{base}/_next/static/chunks/main.js"]
