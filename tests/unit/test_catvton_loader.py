import pytest
from pathlib import Path
from src.features.virtual_tryon.catvton_loader import CatVTONModelLoader

def test_catvton_loader_initialization():
    loader = CatVTONModelLoader(model_path="models/CatVTON", allow_fallback=True)
    assert loader.model_path == Path("models/CatVTON")
    assert loader.precision == "bfloat16"
    assert loader.pipeline is None

def test_catvton_loader_dry_run_fallback():
    loader = CatVTONModelLoader(model_path="non_existent_catvton", allow_fallback=True)
    pipeline = loader.load()
    assert pipeline is None
