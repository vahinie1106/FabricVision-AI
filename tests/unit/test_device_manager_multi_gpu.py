"""Unit tests for multi-GPU device resolution (Kaggle T4×2)."""

from __future__ import annotations

from src.common.models.device_manager import DeviceManager


def test_is_cuda_device_variants():
    assert DeviceManager.is_cuda_device("cuda")
    assert DeviceManager.is_cuda_device("cuda:0")
    assert DeviceManager.is_cuda_device("CUDA:1")
    assert not DeviceManager.is_cuda_device("cpu")
    assert not DeviceManager.is_cuda_device(None)


def test_cuda_device_index():
    assert DeviceManager.cuda_device_index("cuda") == 0
    assert DeviceManager.cuda_device_index("cuda:1") == 1
    assert DeviceManager.cuda_device_index("cpu") is None


def test_resolve_role_device_from_env(monkeypatch):
    monkeypatch.setenv("FLUX_CUDA_DEVICE", "0")
    monkeypatch.setenv("CATVTON_CUDA_DEVICE", "1")
    assert DeviceManager.resolve_role_device("flux") == "cuda:0"
    assert DeviceManager.resolve_role_device("catvton") == "cuda:1"


def test_resolve_role_device_full_string(monkeypatch):
    monkeypatch.setenv("FLUX_CUDA_DEVICE", "cuda:0")
    assert DeviceManager.resolve_role_device("flux") == "cuda:0"


def test_dual_gpu_residency_when_roles_differ(monkeypatch):
    monkeypatch.delenv("FABRICVISION_DUAL_GPU", raising=False)
    monkeypatch.setenv("FLUX_CUDA_DEVICE", "0")
    monkeypatch.setenv("CATVTON_CUDA_DEVICE", "1")
    assert DeviceManager.dual_gpu_residency_enabled() is True


def test_dual_gpu_residency_forced_off(monkeypatch):
    monkeypatch.setenv("FABRICVISION_DUAL_GPU", "0")
    monkeypatch.setenv("FLUX_CUDA_DEVICE", "0")
    monkeypatch.setenv("CATVTON_CUDA_DEVICE", "1")
    assert DeviceManager.dual_gpu_residency_enabled() is False


def test_model_manager_keeps_peer_on_dual_gpu(monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    monkeypatch.setenv("FABRICVISION_DUAL_GPU", "1")
    monkeypatch.setenv("FLUX_CUDA_DEVICE", "0")
    monkeypatch.setenv("CATVTON_CUDA_DEVICE", "1")

    from src.common.models.model_manager import ModelManager

    mm = ModelManager()
    assert mm._can_keep_peer_resident("flux", "catvton") is True
    assert mm._can_keep_peer_resident("catvton", "flux") is True
    # Force sequential when same device
    monkeypatch.setenv("CATVTON_CUDA_DEVICE", "0")
    mm2 = ModelManager()
    assert mm2._can_keep_peer_resident("flux", "catvton") is False


def test_resolve_device_keeps_cuda_index():
    dm = DeviceManager()
    assert dm.resolve_device("cuda:1") == "cuda:1"
    assert dm.resolve_device("1") == "cuda:1"
