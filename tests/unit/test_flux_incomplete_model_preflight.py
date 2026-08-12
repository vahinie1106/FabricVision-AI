"""Regression: incomplete models/flux-kontext must not reach from_pretrained."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.features.custom_generator.model.flux_model_loader import FLUXModelLoader


def _write(path: Path, data: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_bytes(data)


def _fake_component_weights(root: Path, name: str, size: int = 2 * 1024 * 1024) -> None:
    _write(root / name / "config.json", "{}")
    _write(root / name / "diffusion_pytorch_model.safetensors", b"\0" * size)


def _make_complete_package(root: Path) -> None:
    _write(root / "model_index.json", '{"_class_name": "FluxKontextPipeline"}')
    for name in ("tokenizer", "tokenizer_2", "scheduler"):
        (root / name).mkdir(parents=True, exist_ok=True)
        _write(root / name / "config.json", "{}")
    _fake_component_weights(root, "vae")
    _fake_component_weights(root, "text_encoder")
    _fake_component_weights(root, "text_encoder_2")
    _fake_component_weights(
        root, "transformer", size=FLUXModelLoader._MIN_TRANSFORMER_WEIGHT_BYTES
    )


def test_missing_directory_is_invalid(tmp_path: Path):
    root = tmp_path / "flux-kontext"
    loader = FLUXModelLoader(model_path=root, hf_model_id="eramth/flux-kontext-4bit")
    report = loader.preflight_validate_package(root)
    assert report["ready"] is False
    assert root.exists() is False


def test_only_model_index_is_invalid(tmp_path: Path):
    root = tmp_path / "flux-kontext"
    _write(root / "model_index.json", '{"_class_name": "FluxKontextPipeline"}')
    loader = FLUXModelLoader(model_path=root)
    report = loader.preflight_validate_package(root)
    assert report["ready"] is False
    assert report["transformer_weights"] is False


def test_missing_transformer_weights_invalid(tmp_path: Path):
    root = tmp_path / "flux-kontext"
    _write(root / "model_index.json", '{"_class_name": "FluxKontextPipeline"}')
    for name in FLUXModelLoader.REQUIRED_COMPONENT_DIRS:
        (root / name).mkdir(parents=True, exist_ok=True)
        _write(root / name / "config.json", "{}")
    _fake_component_weights(root, "vae")
    _fake_component_weights(root, "text_encoder")
    _fake_component_weights(root, "text_encoder_2")
    # transformer dir exists but no weights
    loader = FLUXModelLoader(model_path=root, hf_model_id="eramth/flux-kontext-4bit")
    report = loader.preflight_validate_package(root)
    assert report["ready"] is False
    assert report["transformer_weights"] is False


def test_lfs_pointer_transformer_invalid(tmp_path: Path):
    root = tmp_path / "flux-kontext"
    _make_complete_package(root)
    pointer = (
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:fc1bf10b2d1c9a104f63464ab799b429d737e1e6b6e7c7b473bc82599e940976\n"
        "size 6699380895\n"
    )
    _write(root / "transformer" / "diffusion_pytorch_model.safetensors", pointer)
    loader = FLUXModelLoader(model_path=root)
    report = loader.preflight_validate_package(root)
    assert report["ready"] is False
    assert report["transformer_lfs_pointer"] is True


def test_tiny_transformer_file_invalid(tmp_path: Path):
    root = tmp_path / "flux-kontext"
    _make_complete_package(root)
    _write(root / "transformer" / "diffusion_pytorch_model.safetensors", b"\0" * 4096)
    loader = FLUXModelLoader(model_path=root)
    assert loader._has_transformer_weights(root) is False
    assert loader.preflight_validate_package(root)["ready"] is False


def test_complete_package_cache_hit(tmp_path: Path):
    root = tmp_path / "flux-kontext"
    _make_complete_package(root)
    loader = FLUXModelLoader(model_path=root, hf_model_id="eramth/flux-kontext-4bit")
    report = loader.preflight_validate_package(root)
    assert report["ready"] is True
    with patch.object(loader, "_ensure_hub_package") as ensure:
        pipeline_root, transformer_root = loader._resolve_model_source()
    ensure.assert_not_called()
    assert Path(pipeline_root) == root
    assert transformer_root is None
    assert loader._cache_status == "hit"


def test_incomplete_package_triggers_purge_and_download(tmp_path: Path, monkeypatch):
    root = tmp_path / "flux-kontext"
    hub = tmp_path / "hf-hub"
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    _write(root / "model_index.json", '{"_class_name": "FluxKontextPipeline"}')
    _fake_component_weights(root, "vae")
    _fake_component_weights(root, "text_encoder")
    _fake_component_weights(root, "text_encoder_2")
    (root / "transformer").mkdir(parents=True, exist_ok=True)
    _write(root / "transformer" / "config.json", "{}")
    # Tiny stub weight (not a real transformer) → purge allowed.
    _write(root / "transformer" / "diffusion_pytorch_model.safetensors", b"\0" * 4096)

    loader = FLUXModelLoader(model_path=root, hf_model_id="eramth/flux-kontext-4bit")

    def _fake_snapshot(*, repo_id, local_dir=None, cache_dir=None, **_kwargs):
        # Default path: hub cache (no local_dir).
        assert local_dir is None
        dest = Path(cache_dir) / "models--eramth--flux-kontext-4bit" / "snapshots" / "abc"
        _make_complete_package(dest)
        return str(dest)

    with patch("huggingface_hub.snapshot_download", side_effect=_fake_snapshot) as mocked:
        path = loader._ensure_hub_package("eramth/flux-kontext-4bit")

    mocked.assert_called_once()
    kwargs = mocked.call_args.kwargs
    assert "cache_dir" in kwargs
    assert "local_dir" not in kwargs
    assert Path(path).exists()
    assert loader._package_ready_for_pipeline(Path(path)) is True


def test_valid_partial_transformer_is_not_purged(tmp_path: Path, monkeypatch):
    """Interrupted downloads with a real transformer must resume via local_dir, not wipe GiB."""
    root = tmp_path / "flux-kontext"
    hub = tmp_path / "hf-hub"
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    _write(root / "model_index.json", '{"_class_name": "FluxKontextPipeline"}')
    (root / "transformer").mkdir(parents=True, exist_ok=True)
    _write(root / "transformer" / "config.json", "{}")
    _write(
        root / "transformer" / "diffusion_pytorch_model.safetensors",
        b"\0" * FLUXModelLoader._MIN_TRANSFORMER_WEIGHT_BYTES,
    )
    loader = FLUXModelLoader(model_path=root, hf_model_id="eramth/flux-kontext-4bit")
    report = loader.preflight_validate_package(root)
    assert report["transformer_weights"] is True
    assert loader._should_purge_before_download(root, report) is False

    def _fake_snapshot(*, repo_id, local_dir=None, cache_dir=None, **_kwargs):
        # Resume must use local_dir so existing transformer is kept.
        assert local_dir is not None
        dest = Path(local_dir)
        _make_complete_package(dest)
        return str(dest)

    with patch("huggingface_hub.snapshot_download", side_effect=_fake_snapshot) as mocked:
        path = loader._ensure_hub_package("eramth/flux-kontext-4bit")

    mocked.assert_called_once()
    assert Path(path) == root
    assert loader._package_ready_for_pipeline(root)


def test_hub_snapshot_cache_hit_skips_download(tmp_path: Path, monkeypatch):
    hub = tmp_path / "hf-hub"
    snap = hub / "models--eramth--flux-kontext-4bit" / "snapshots" / "abc123"
    _make_complete_package(snap)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    local = tmp_path / "models" / "flux-kontext"
    loader = FLUXModelLoader(model_path=local, hf_model_id="eramth/flux-kontext-4bit")

    with patch("huggingface_hub.snapshot_download") as mocked:
        path = loader._ensure_hub_package("eramth/flux-kontext-4bit")

    mocked.assert_not_called()
    assert Path(path) == snap
    assert loader._cache_status == "hit"


def test_cache_miss_downloads_into_hub_cache_not_local_dir(tmp_path: Path, monkeypatch):
    root = tmp_path / "flux-kontext"
    hub = tmp_path / "hf-hub"
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    loader = FLUXModelLoader(model_path=root, hf_model_id="eramth/flux-kontext-4bit")

    def _fake_snapshot(*, repo_id, local_dir=None, cache_dir=None, **_kwargs):
        assert local_dir is None
        dest = Path(cache_dir) / "models--eramth--flux-kontext-4bit" / "snapshots" / "rev1"
        _make_complete_package(dest)
        return str(dest)

    with patch("huggingface_hub.snapshot_download", side_effect=_fake_snapshot) as mocked:
        first = loader._ensure_hub_package("eramth/flux-kontext-4bit")
        assert mocked.call_count == 1
        assert "cache_dir" in mocked.call_args.kwargs
        assert Path(first).exists()

        loader2 = FLUXModelLoader(model_path=root, hf_model_id="eramth/flux-kontext-4bit")
        second = loader2._ensure_hub_package("eramth/flux-kontext-4bit")
        assert mocked.call_count == 1  # no second download — hub snapshot HIT
        assert loader2._cache_status == "hit"
        assert Path(second) == Path(first)


def test_hf_cache_env_kaggle_default(tmp_path: Path, monkeypatch):
    from src.common.utils import hf_cache_env as mod

    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    monkeypatch.setattr(mod, "is_kaggle_environment", lambda: True)
    monkeypatch.setattr(mod, "default_hf_home", lambda: tmp_path / "kaggle-hf")

    applied = mod.ensure_huggingface_cache_env()
    assert applied["HF_HOME"] == str(tmp_path / "kaggle-hf")
    assert (tmp_path / "kaggle-hf" / "hub").is_dir()
    assert os.environ["HUGGINGFACE_HUB_CACHE"].endswith("hub")


def test_download_failure_never_ready(tmp_path: Path, monkeypatch):
    root = tmp_path / "flux-kontext"
    hub = tmp_path / "hf-hub"
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    loader = FLUXModelLoader(
        model_path=root, hf_model_id="eramth/flux-kontext-4bit", allow_fallback=False
    )

    with patch(
        "huggingface_hub.snapshot_download",
        side_effect=RuntimeError("network down"),
    ):
        with pytest.raises(RuntimeError, match="MODEL_DOWNLOAD_FAILED"):
            loader._ensure_hub_package("eramth/flux-kontext-4bit")

    assert loader._package_ready_for_pipeline(root) is False


def test_valid_package_allows_assert_loadable(tmp_path: Path):
    root = tmp_path / "flux-kontext"
    _make_complete_package(root)
    loader = FLUXModelLoader(model_path=root, hf_model_id="eramth/flux-kontext-4bit")
    loader._assert_local_package_loadable(str(root), source="eramth/flux-kontext-4bit")


def test_invalid_package_blocks_from_pretrained(tmp_path: Path):
    root = tmp_path / "flux-kontext"
    _write(root / "model_index.json", "{}")
    (root / "transformer").mkdir(parents=True, exist_ok=True)
    loader = FLUXModelLoader(model_path=root, hf_model_id="eramth/flux-kontext-4bit")
    with pytest.raises(RuntimeError, match="refusing from_pretrained"):
        loader._assert_local_package_loadable(str(root), source="eramth/flux-kontext-4bit")


def test_resolve_incomplete_triggers_hub_download(tmp_path: Path, monkeypatch):
    root = tmp_path / "flux-kontext"
    hub = tmp_path / "hf-hub"
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    _write(root / "model_index.json", '{"_class_name": "FluxKontextPipeline"}')
    (root / "transformer").mkdir(parents=True, exist_ok=True)

    loader = FLUXModelLoader(
        model_path=root,
        hf_model_id="eramth/flux-kontext-4bit",
        allow_fallback=False,
    )

    def _fake_snapshot(*, repo_id, local_dir=None, cache_dir=None, **_kwargs):
        dest = Path(cache_dir) / "models--eramth--flux-kontext-4bit" / "snapshots" / "r"
        _make_complete_package(dest)
        return str(dest)

    with patch("huggingface_hub.snapshot_download", side_effect=_fake_snapshot):
        pipeline_root, transformer_root = loader._resolve_model_source()

    assert transformer_root is None
    assert loader._package_ready_for_pipeline(Path(pipeline_root)) is True


def test_progress_is_monotonic_across_heartbeats(tmp_path: Path):
    loader = FLUXModelLoader(model_path=tmp_path / "flux", hf_model_id="eramth/flux-kontext-4bit")
    seen: list[int] = []
    loader.set_progress_callback(lambda _s, p: seen.append(p))
    loader._progress("Downloading FLUX weights (2.00 GiB in hub_cache, 10s elapsed)", 29)
    loader._progress("Loading FLUX (15s elapsed)", 17)  # must not regress
    assert seen[-1] >= 29
    assert loader._load_phase_pct >= 29


def test_refs_main_alone_is_not_cache_hit(tmp_path: Path, monkeypatch):
    hub = tmp_path / "hf-hub"
    repo = hub / "models--eramth--flux-kontext-4bit"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text("deadbeef", encoding="utf-8")
    (repo / "snapshots").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    loader = FLUXModelLoader(
        model_path=tmp_path / "models" / "flux-kontext",
        hf_model_id="eramth/flux-kontext-4bit",
    )
    assert loader._find_complete_hub_snapshot("eramth/flux-kontext-4bit") is None


def test_force_local_dir_download_still_supported(tmp_path: Path, monkeypatch):
    root = tmp_path / "flux-kontext"
    hub = tmp_path / "hf-hub"
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.setenv("FLUX_DOWNLOAD_TO_LOCAL_DIR", "1")
    loader = FLUXModelLoader(model_path=root, hf_model_id="eramth/flux-kontext-4bit")

    def _fake_snapshot(*, repo_id, local_dir=None, cache_dir=None, **_kwargs):
        assert local_dir is not None
        dest = Path(local_dir)
        _make_complete_package(dest)
        return str(dest)

    with patch("huggingface_hub.snapshot_download", side_effect=_fake_snapshot) as mocked:
        path = loader._ensure_hub_package("eramth/flux-kontext-4bit")
    assert Path(path) == root
    assert "local_dir" in mocked.call_args.kwargs


def test_load_path_never_calls_from_pretrained_when_assert_fails(tmp_path: Path):
    """Guard: incomplete local package must fail before Diffusers from_pretrained."""
    root = tmp_path / "flux-kontext"
    _write(root / "model_index.json", "{}")
    for name in FLUXModelLoader.REQUIRED_COMPONENT_DIRS:
        (root / name).mkdir(parents=True, exist_ok=True)

    loader = FLUXModelLoader(
        model_path=root,
        hf_model_id="eramth/flux-kontext-4bit",
        allow_fallback=False,
    )

    # Pretend resolve returned the broken local path (the old bug).
    with patch.object(
        loader, "_resolve_model_source", return_value=(str(root), None)
    ), patch(
        "diffusers.FluxKontextPipeline.from_pretrained"
    ) as from_pretrained, patch(
        "diffusers.FluxTransformer2DModel.from_pretrained"
    ) as transformer_from_pretrained:
        with pytest.raises(RuntimeError, match="refusing from_pretrained|Failed to load FLUX"):
            loader._load_after_heartbeat_start(t0=0.0, target_device="cpu")

    from_pretrained.assert_not_called()
    transformer_from_pretrained.assert_not_called()
