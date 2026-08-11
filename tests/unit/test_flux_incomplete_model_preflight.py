"""Regression: incomplete models/flux-kontext must not reach from_pretrained."""

from __future__ import annotations

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


def test_incomplete_package_triggers_purge_and_download(tmp_path: Path):
    root = tmp_path / "flux-kontext"
    _write(root / "model_index.json", '{"_class_name": "FluxKontextPipeline"}')
    _fake_component_weights(root, "vae")
    _fake_component_weights(root, "text_encoder")
    _fake_component_weights(root, "text_encoder_2")
    (root / "transformer").mkdir(parents=True, exist_ok=True)
    _write(root / "transformer" / "config.json", "{}")

    loader = FLUXModelLoader(model_path=root, hf_model_id="eramth/flux-kontext-4bit")

    def _fake_snapshot(*, repo_id, local_dir, **_kwargs):
        dest = Path(local_dir)
        _make_complete_package(dest)
        return str(dest)

    with patch("huggingface_hub.snapshot_download", side_effect=_fake_snapshot) as mocked:
        path = loader._ensure_hub_package("eramth/flux-kontext-4bit")

    mocked.assert_called_once()
    assert Path(path) == root
    assert loader._package_ready_for_pipeline(root) is True
    # Incomplete tree must have been purged before download.
    assert not (root / "transformer" / "config.json").exists() or (
        root / "transformer" / "diffusion_pytorch_model.safetensors"
    ).exists()


def test_download_failure_never_ready(tmp_path: Path):
    root = tmp_path / "flux-kontext"
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


def test_resolve_incomplete_triggers_hub_download(tmp_path: Path):
    root = tmp_path / "flux-kontext"
    _write(root / "model_index.json", '{"_class_name": "FluxKontextPipeline"}')
    (root / "transformer").mkdir(parents=True, exist_ok=True)

    loader = FLUXModelLoader(
        model_path=root,
        hf_model_id="eramth/flux-kontext-4bit",
        allow_fallback=False,
    )

    def _fake_snapshot(*, repo_id, local_dir, **_kwargs):
        dest = Path(local_dir)
        _make_complete_package(dest)
        return str(dest)

    with patch("huggingface_hub.snapshot_download", side_effect=_fake_snapshot):
        pipeline_root, transformer_root = loader._resolve_model_source()

    assert transformer_root is None
    assert Path(pipeline_root) == root
    assert loader._package_ready_for_pipeline(root) is True


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
