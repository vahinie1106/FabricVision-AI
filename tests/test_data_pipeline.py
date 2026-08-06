import os
import shutil
import pytest
from pathlib import Path

from src.data_management.dataset_loader import (
    DatasetLoaderFactory,
    DeepFashionLoader,
    DeepFashion2Loader,
    FashionpediaLoader,
    LocalDatasetLoader,
)
from src.data_management.schemas import GarmentMetadata
from src.data_management.validators import MetadataValidator
from src.data_management.metadata_store import MetadataStore


# ============================================================================
# 1. Dataset Loader Adapter Tests
# ============================================================================

def test_dataset_loader_factory():
    df_loader = DatasetLoaderFactory.get_loader("deepfashion")
    assert isinstance(df_loader, DeepFashionLoader)

    df2_loader = DatasetLoaderFactory.get_loader("deepfashion2")
    assert isinstance(df2_loader, DeepFashion2Loader)

    fp_loader = DatasetLoaderFactory.get_loader("fashionpedia")
    assert isinstance(fp_loader, FashionpediaLoader)

    local_loader = DatasetLoaderFactory.get_loader("local")
    assert isinstance(local_loader, LocalDatasetLoader)


def test_mock_dataset_loading():
    loader = DatasetLoaderFactory.get_loader("deepfashion")
    records = loader.load_dataset("data/raw")
    assert isinstance(records, list)
    assert len(records) > 0
    assert "identity" in records[0]
    assert records[0]["identity"]["category"] == "upper_wear"
    assert records[0]["source_dataset"] == "DeepFashion"


def test_deepfashion2_mock_loading():
    loader = DatasetLoaderFactory.get_loader("deepfashion2")
    records = loader.load_dataset("data/raw")
    assert len(records) > 0
    assert records[0]["source_dataset"] == "DeepFashion2"


def test_fashionpedia_mock_loading():
    loader = DatasetLoaderFactory.get_loader("fashionpedia")
    records = loader.load_dataset("data/raw")
    assert len(records) > 0
    assert records[0]["source_dataset"] == "Fashionpedia"


# ============================================================================
# 2. Metadata Schema & Validation Tests
# ============================================================================

def test_metadata_validation_success():
    valid_data = {
        "garment_id": "garment_000001",
        "identity": {
            "category": "upper_wear",
            "gender": "women",
            "season": "summer",
            "occasion": "casual"
        },
        "physical": {
            "fabric": "cotton",
            "texture": "smooth",
            "color": ["white"],
            "pattern": "solid"
        },
        "construction": {
            "neckline": "crew",
            "sleeve": "short",
            "silhouette": "regular",
            "fit": "regular"
        },
        "style": {
            "aesthetic": "minimalist",
            "trend": "classic",
            "fashion_category": "basics"
        }
    }
    is_valid, result = MetadataValidator.validate(valid_data)
    assert is_valid, f"Expected valid, got error: {result}"
    assert isinstance(result, GarmentMetadata)
    assert result.garment_id == "garment_000001"
    assert result.identity.category == "upper_wear"


def test_metadata_validation_failure_missing_fields():
    invalid_data = {
        "garment_id": "garment_000002",
        "identity": {
            "category": "upper_wear"
            # Missing gender, season, occasion
        }
    }
    is_valid, result = MetadataValidator.validate(invalid_data)
    assert not is_valid
    assert "Validation Error" in str(result) or "Parsing Error" in str(result)


def test_metadata_validation_invalid_vocabulary():
    invalid_vocab_data = {
        "garment_id": "garment_000003",
        "identity": {
            "category": "non_existent_category",
            "gender": "women",
            "season": "summer",
            "occasion": "casual"
        },
        "physical": {
            "fabric": "cotton",
            "texture": "smooth",
            "color": ["white"],
            "pattern": "solid"
        },
        "construction": {
            "neckline": "crew",
            "sleeve": "short",
            "silhouette": "regular",
            "fit": "regular"
        },
        "style": {
            "aesthetic": "minimalist",
            "trend": "classic",
            "fashion_category": "basics"
        }
    }
    is_valid, result = MetadataValidator.validate(invalid_vocab_data)
    assert not is_valid
    assert "Vocabulary Errors" in str(result) or "Invalid identity.category" in str(result)


def test_legacy_qwen_output_adaptation():
    legacy_data = {
        "garment_identity": {"gender": "women"},
        "classification": {"category": "upper_wear", "subcategory": "shirt"},
        "physical_attributes": {"material": "cotton", "fabric_textures": "smooth"},
        "visual_attributes": {"colors": ["blue"], "patterns": "striped"},
        "shape_and_fit": {"silhouette": "fitted", "fit": "slim", "sleeves": "short", "neckline": "v_neck"},
        "style": {"occasion": "casual", "season": "summer"}
    }
    is_valid, result = MetadataValidator.validate(legacy_data)
    assert is_valid, f"Legacy adaptation failed: {result}"
    assert isinstance(result, GarmentMetadata)
    assert result.physical.fabric == "cotton"
    assert result.identity.category == "upper_wear"


# ============================================================================
# 3. Garment ID Generation & Storage Tests
# ============================================================================

def test_id_generation_and_metadata_store():
    test_dir = Path("tests/test_metadata_store_temp")
    if test_dir.exists():
        shutil.rmtree(test_dir)

    store = MetadataStore(storage_dir=test_dir)

    # Test explicit ID generation
    id1 = store.generate_garment_id()
    assert id1 == "garment_000001"

    id2 = store.generate_garment_id()
    assert id2 == "garment_000002"

    valid_data = {
        "identity": {
            "category": "lower_wear",
            "gender": "men",
            "season": "all_season",
            "occasion": "casual"
        },
        "physical": {
            "fabric": "denim",
            "texture": "rough",
            "color": ["blue"],
            "pattern": "solid"
        },
        "construction": {
            "neckline": "none",
            "sleeve": "none",
            "silhouette": "regular",
            "fit": "straight"
        },
        "style": {
            "aesthetic": "casual",
            "trend": "classic",
            "fashion_category": "jeans"
        }
    }

    garment_id = store.save_metadata(valid_data)
    assert garment_id == "garment_000003"

    loaded = store.load_metadata(garment_id)
    assert loaded is not None
    assert loaded.garment_id == "garment_000003"
    assert loaded.physical.fabric == "denim"
    assert loaded.identity.category == "lower_wear"

    # Cleanup
    if test_dir.exists():
        shutil.rmtree(test_dir)
