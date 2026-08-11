"""Regression: incomplete models/flux-kontext must not be treated as loadable."""

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


def test_incomplete_dir_without_transformer_fails_preflight(tmp_path: Path):
    root = tmp_path / "flux-kontext"
    _write(root / "model_index.json", '{"_class_name": "FluxKontextPipeline"}')
    _fake_component_weights(root, "vae")
    _fake_component_weights(root, "text_encoder")
    _fake_component_weights(root, "text_encoder_2")
    # transformer/ present but empty — the Kaggle failure mode.
    (root / "transformer").mkdir(parents=True, exist_ok=True)
    _write(root / "transformer" / "config.json", "{}")

    loader = FLUXModelLoader(model_path=root, hf_model_id="eramth/flux-kontext-4bit")
    report = loader.preflight_validate_package(root, source="eramth/flux-kontext-4bit")

    assert report["ready"] is False
    assert report["transformer_weights"] is False
    assert loader._package_ready_for_pipeline(root) is False
    assert loader._is_complete_local_dir(root) is False


def test_lfs_pointer_transformer_is_not_ready(tmp_path: Path):
    root = tmp_path / "flux-kontext"
    _write(root / "model_index.json", '{"_class_name": "FluxKontextPipeline"}')
    _fake_component_weights(root, "vae")
    _fake_component_weights(root, "text_encoder")
    _fake_component_weights(root, "text_encoder_2")
    pointer = (
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:fc1bf10b2d1c9a104f63464ab799b429d737e1e6b6e7c7b473bc82599e940976\n"
        "size 6699380895\n"
    )
    _write(root / "transformer" / "diffusion_pytorch_model.safetensors", pointer)

    loader = FLUXModelLoader(model_path=root)
    assert loader._has_transformer_weights(root) is False
    assert loader._package_ready_for_pipeline(root) is False
    report = loader.preflight_validate_package(root)
    assert report["ready"] is False
    assert report["transformer_weights"] is False
    assert loader._is_lfs_pointer_file(
        root / "transformer" / "diffusion_pytorch_model.safetensors"
    )


def test_tiny_transformer_file_rejected(tmp_path: Path):
    root = tmp_path / "flux-kontext"
    _write(root / "model_index.json", "{}")
    _fake_component_weights(root, "vae")
    _fake_component_weights(root, "text_encoder")
    _fake_component_weights(root, "text_encoder_2")
    # Non-pointer but truncated download (a few KB).
    _write(root / "transformer" / "diffusion_pytorch_model.safetensors", b"\0" * 4096)

    loader = FLUXModelLoader(model_path=root)
    assert loader._has_transformer_weights(root) is False


def test_complete_package_passes_preflight(tmp_path: Path):
    root = tmp_path / "flux-kontext"
    _write(root / "model_index.json", '{"_class_name": "FluxKontextPipeline"}')
    _fake_component_weights(root, "vae")
    _fake_component_weights(root, "text_encoder")
    _fake_component_weights(root, "text_encoder_2")
    _fake_component_weights(
        root, "transformer", size=FLUXModelLoader._MIN_TRANSFORMER_WEIGHT_BYTES
    )

    loader = FLUXModelLoader(model_path=root, hf_model_id="eramth/flux-kontext-4bit")
    report = loader.preflight_validate_package(root)
    assert report["ready"] is True
    assert report["transformer_weights"] is True
    assert loader._package_ready_for_pipeline(root) is True


def test_ensure_hub_does_not_cache_hit_incomplete_tree(tmp_path: Path):
    """Exact bug: incomplete local tree must trigger download, not CACHE HIT."""
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
        _fake_component_weights(
            dest, "transformer", size=FLUXModelLoader._MIN_TRANSFORMER_WEIGHT_BYTES
        )
        return str(dest)

    with patch(
        "huggingface_hub.snapshot_download", side_effect=_fake_snapshot
    ) as mocked:
        path = loader._ensure_hub_package("eramth/flux-kontext-4bit")

    mocked.assert_called_once()
    assert Path(path) == root
    assert loader._cache_status == "miss" or loader._package_ready_for_pipeline(root)
    assert loader._package_ready_for_pipeline(root) is True


def test_resolve_incomplete_triggers_hub_download(tmp_path: Path):
    root = tmp_path / "flux-kontext"
    _write(root / "model_index.json", '{"_class_name": "FluxKontextPipeline"}')
    _fake_component_weights(root, "vae")
    _fake_component_weights(root, "text_encoder")
    _fake_component_weights(root, "text_encoder_2")
    (root / "transformer").mkdir(parents=True, exist_ok=True)

    loader = FLUXModelLoader(
        model_path=root,
        hf_model_id="eramth/flux-kontext-4bit",
        allow_fallback=False,
    )

    def _fake_snapshot(*, repo_id, local_dir, **_kwargs):
        dest = Path(local_dir)
        _fake_component_weights(
            dest, "transformer", size=FLUXModelLoader._MIN_TRANSFORMER_WEIGHT_BYTES
        )
        return str(dest)

    with patch("huggingface_hub.snapshot_download", side_effect=_fake_snapshot):
        pipeline_root, transformer_root = loader._resolve_model_source()

    assert transformer_root is None
    assert Path(pipeline_root) == root
    assert loader._package_ready_for_pipeline(root) is True
