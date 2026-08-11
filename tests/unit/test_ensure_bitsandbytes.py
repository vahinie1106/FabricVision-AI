"""Tests for bitsandbytes prerequisite helper."""

from src.common.utils.ensure_bitsandbytes import probe_bitsandbytes, ensure_bitsandbytes


def test_probe_bitsandbytes_reports_tuple():
    ok, detail, version = probe_bitsandbytes()
    assert isinstance(ok, bool)
    assert isinstance(detail, str)
    if ok:
        assert version
        assert version[0].isdigit()


def test_ensure_bitsandbytes_when_present():
    ok, _, _ = probe_bitsandbytes()
    if not ok:
        # Environment without GPU wheels still may install; skip hard requirement in CI CPU-only.
        return
    ver = ensure_bitsandbytes(auto_install=False)
    assert ver
