"""
FabricVision-AI Data Management Package.
Provides data loaders, schema validators, persistent metadata storage, and dataset adapters.
"""

from .dataset_loader import (
    BaseDatasetLoader,
    DeepFashionLoader,
    DeepFashion2Loader,
    FashionpediaLoader,
    LocalDatasetLoader,
    DatasetLoaderFactory,
)
from .schemas import GarmentMetadata, GarmentIdentity, PhysicalAttributes, Construction, Style
from .validators import MetadataValidator
from .metadata_store import MetadataStore

__all__ = [
    "BaseDatasetLoader",
    "DeepFashionLoader",
    "DeepFashion2Loader",
    "FashionpediaLoader",
    "LocalDatasetLoader",
    "DatasetLoaderFactory",
    "GarmentMetadata",
    "GarmentIdentity",
    "PhysicalAttributes",
    "Construction",
    "Style",
    "MetadataValidator",
    "MetadataStore",
]
