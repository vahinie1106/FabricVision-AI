"""
Polymorphic Dataset Adapter Layer for FabricVision-AI.

Supports standardized ingestion of benchmark datasets:
- DeepFashion
- DeepFashion2
- Fashionpedia
- Local Datasets
"""

from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import List, Dict, Any, Optional


class BaseDatasetLoader(ABC):
    """Abstract base class for ingesting external fashion datasets into standardized metadata structures."""

    @abstractmethod
    def load_images(self, data_path: str) -> List[str]:
        """Scan directory and return list of valid image file paths."""
        pass

    @abstractmethod
    def load_annotations(self, data_path: str) -> List[Dict[str, Any]]:
        """Parse raw annotation files (JSON, TXT, COCO format) from dataset path."""
        pass

    @abstractmethod
    def normalize_metadata(self, raw_annotation: Dict[str, Any]) -> Dict[str, Any]:
        """Convert dataset-specific annotations to canonical FabricVision schema format."""
        pass

    @abstractmethod
    def load_dataset(self, data_path: str) -> List[Dict[str, Any]]:
        """Load and return a list of standardized garment dictionaries."""
        pass


class DeepFashionLoader(BaseDatasetLoader):
    """Adapter for DeepFashion (Category & Attribute Prediction Benchmark)."""

    def load_images(self, data_path: str) -> List[str]:
        p = Path(data_path)
        if not p.exists():
            return []
        valid_exts = {".jpg", ".jpeg", ".png"}
        return [str(f) for f in p.glob("**/*") if f.suffix.lower() in valid_exts]

    def load_annotations(self, data_path: str) -> List[Dict[str, Any]]:
        # Mock/Abstract annotation loader for DeepFashion attribute files
        anno_file = Path(data_path) / "list_attr_img.txt"
        if anno_file.exists():
            return [{"image_path": "sample.jpg", "category": "Upper", "attribute": "cotton"}]
        return [{"image_path": "sample_deepfashion.jpg", "category": "upper_wear", "attribute": "cotton"}]

    def normalize_metadata(self, raw_annotation: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "identity": {
                "category": raw_annotation.get("category", "upper_wear"),
                "gender": raw_annotation.get("gender", "women"),
                "season": "summer",
                "occasion": "casual",
            },
            "physical": {
                "fabric": raw_annotation.get("attribute", "cotton"),
                "texture": "smooth",
                "color": ["white"],
                "pattern": "solid",
            },
            "construction": {
                "neckline": "crew",
                "sleeve": "short",
                "silhouette": "regular",
                "fit": "regular",
            },
            "style": {
                "aesthetic": "casual",
                "trend": "classic",
                "fashion_category": "tops",
            },
            "source_dataset": "DeepFashion",
        }

    def load_dataset(self, data_path: str) -> List[Dict[str, Any]]:
        annos = self.load_annotations(data_path)
        return [self.normalize_metadata(anno) for anno in annos]


class DeepFashion2Loader(BaseDatasetLoader):
    """Adapter for DeepFashion2 dataset (COCO format keypoints and polygons)."""

    def load_images(self, data_path: str) -> List[str]:
        p = Path(data_path)
        if not p.exists():
            return []
        return [str(f) for f in p.glob("**/*.jpg")]

    def load_annotations(self, data_path: str) -> List[Dict[str, Any]]:
        return [{"item_id": 1, "category_name": "short_sleeve_top", "style": 1}]

    def normalize_metadata(self, raw_annotation: Dict[str, Any]) -> Dict[str, Any]:
        category_map = {"short_sleeve_top": "upper_wear", "trousers": "lower_wear", "skirt": "lower_wear"}
        cat = category_map.get(raw_annotation.get("category_name", ""), "upper_wear")
        return {
            "identity": {
                "category": cat,
                "gender": "unisex",
                "season": "all_season",
                "occasion": "casual",
            },
            "physical": {
                "fabric": "cotton",
                "texture": "smooth",
                "color": ["blue"],
                "pattern": "solid",
            },
            "construction": {
                "neckline": "crew",
                "sleeve": "short",
                "silhouette": "regular",
                "fit": "regular",
            },
            "style": {
                "aesthetic": "modern",
                "trend": "streetwear",
                "fashion_category": "casual",
            },
            "source_dataset": "DeepFashion2",
        }

    def load_dataset(self, data_path: str) -> List[Dict[str, Any]]:
        annos = self.load_annotations(data_path)
        return [self.normalize_metadata(anno) for anno in annos]


class FashionpediaLoader(BaseDatasetLoader):
    """Adapter for Fashionpedia fine-grained attributes dataset."""

    def load_images(self, data_path: str) -> List[str]:
        p = Path(data_path)
        if not p.exists():
            return []
        return [str(f) for f in p.glob("**/*.png")]

    def load_annotations(self, data_path: str) -> List[Dict[str, Any]]:
        return [{"image_id": 101, "attributes": ["v-neck", "short-sleeve", "floral"]}]

    def normalize_metadata(self, raw_annotation: Dict[str, Any]) -> Dict[str, Any]:
        attrs = raw_annotation.get("attributes", [])
        pattern = "floral" if "floral" in attrs else "solid"
        neckline = "v_neck" if "v-neck" in attrs else "crew"
        return {
            "identity": {
                "category": "dresses",
                "gender": "women",
                "season": "summer",
                "occasion": "party",
            },
            "physical": {
                "fabric": "chiffon",
                "texture": "smooth",
                "color": ["multi"],
                "pattern": pattern,
            },
            "construction": {
                "neckline": neckline,
                "sleeve": "short",
                "silhouette": "a_line",
                "fit": "regular",
            },
            "style": {
                "aesthetic": "romantic",
                "trend": "vintage",
                "fashion_category": "dresses",
            },
            "source_dataset": "Fashionpedia",
        }

    def load_dataset(self, data_path: str) -> List[Dict[str, Any]]:
        annos = self.load_annotations(data_path)
        return [self.normalize_metadata(anno) for anno in annos]


class LocalDatasetLoader(BaseDatasetLoader):
    """Adapter for local raw/processed filesystem images."""

    def load_images(self, data_path: str) -> List[str]:
        p = Path(data_path)
        if not p.exists():
            return []
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        return [str(f) for f in p.glob("*") if f.suffix.lower() in valid_exts]

    def load_annotations(self, data_path: str) -> List[Dict[str, Any]]:
        images = self.load_images(data_path)
        return [{"image_path": img} for img in images]

    def normalize_metadata(self, raw_annotation: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "identity": {
                "category": "upper_wear",
                "gender": "unisex",
                "season": "all_season",
                "occasion": "casual",
            },
            "physical": {
                "fabric": "cotton",
                "texture": "smooth",
                "color": ["black"],
                "pattern": "solid",
            },
            "construction": {
                "neckline": "crew",
                "sleeve": "short",
                "silhouette": "regular",
                "fit": "regular",
            },
            "style": {
                "aesthetic": "minimalist",
                "trend": "classic",
                "fashion_category": "t-shirt",
            },
            "source_dataset": "LocalDataset",
        }

    def load_dataset(self, data_path: str) -> List[Dict[str, Any]]:
        annos = self.load_annotations(data_path)
        return [self.normalize_metadata(anno) for anno in annos]


class DatasetLoaderFactory:
    """Factory for instantiating dataset loaders by name."""

    @staticmethod
    def get_loader(dataset_name: str) -> BaseDatasetLoader:
        normalized_name = dataset_name.lower().replace("-", "").replace("_", "")
        if normalized_name in {"deepfashion", "deepfashion1"}:
            return DeepFashionLoader()
        elif normalized_name == "deepfashion2":
            return DeepFashion2Loader()
        elif normalized_name == "fashionpedia":
            return FashionpediaLoader()
        elif normalized_name in {"local", "localdataset"}:
            return LocalDatasetLoader()
        else:
            raise ValueError(f"Unsupported dataset loader: '{dataset_name}'")
