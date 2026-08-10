from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .attribute_extractor import AttributeExtractor
from .dataset_index import DatasetIndex, ProductRecord
from .garment_classifier import GarmentClassifier
from .metadata_generator import MetadataGenerator
from .quality_checker import QualityChecker
from .reorganizer import DatasetReorganizer
from .report_generator import ReportGenerator
from .scanner import DatasetScanner
from src.common.utils.utils import ensure_directory, load_yaml_config
from .validator import DatasetValidator


@dataclass
class DatasetManagementConfig:
    """Configuration for the dataset management layer."""

    dataset_root: str = "datasets"
    output_root: str = "datasets/garments"
    reports_dir: str = "reports"
    supported_formats: List[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp"])
    min_width: int = 16
    min_height: int = 16
    materials_root: Optional[str] = None
    patterns_root: Optional[str] = None
    config_path: Optional[str] = None


class DatasetManager:
    """Orchestrate scanning, validation, classification, metadata, reorganization, and reporting."""

    def __init__(self, config: Optional[DatasetManagementConfig] = None, config_path: Optional[str] = None) -> None:
        self.config = config or DatasetManagementConfig()
        if config_path:
            self.config.config_path = config_path
        self.logger = logging.getLogger("fabricvision.dataset_management")
        self._load_config_file()
        self.scanner = DatasetScanner(self.config.dataset_root, self.config.supported_formats)
        self.validator = DatasetValidator(self.config.supported_formats, self.config.min_width, self.config.min_height)
        self.classifier = GarmentClassifier()
        self.attribute_extractor = AttributeExtractor(self.config.materials_root, self.config.patterns_root)
        self.metadata_generator = MetadataGenerator(self.config.output_root)
        self.reorganizer = DatasetReorganizer(self.config.output_root)
        self.quality_checker = QualityChecker()
        self.report_generator = ReportGenerator(self.config.reports_dir)

    def run(self) -> Dict[str, Any]:
        """Run the complete dataset management workflow."""
        dataset_index = self.scanner.scan()
        validation_report = self.validator.validate(dataset_index)
        products: List[Dict[str, Any]] = []

        for product in dataset_index.products:
            classification = self.classifier.classify(
                product_id=product.product_id,
                gender=product.gender,
                folder_name=product.directory,
                image_paths=[str(path) for path in product.image_paths],
            )
            product.classification = {"garment_type": classification.garment_type, "confidence": classification.confidence, "classification_source": classification.classification_source}
            product.attributes = self.attribute_extractor.extract(product, product.classification)
            validation_status = "valid" if validation_report.results and any(result.product_id == product.product_id and result.is_valid for result in validation_report.results) else "invalid"
            product.validation_status = validation_status
            product.validation_issues = next((result.issues for result in validation_report.results if result.product_id == product.product_id), [])
            metadata_entry = {
                "product_id": product.product_id,
                "gender": product.attributes["gender"],
                "garment_type": product.attributes["garment_type"],
                "material": product.attributes["material"],
                "pattern": product.attributes["pattern"],
                "color": product.attributes["color"],
                "original_path": product.directory,
                "new_path": "",
                "confidence": product.classification["confidence"],
                "validation_status": product.validation_status,
                "classification_source": product.classification["classification_source"],
                "image_paths": [str(path) for path in product.image_paths],
            }
            products.append(metadata_entry)

        metadata_files = self.metadata_generator.generate(products)
        reorganized_paths = self.reorganizer.reorganize(products, self.config.dataset_root)
        for product in products:
            target_dir = self._build_target_path(product)
            product["new_path"] = str(target_dir)
            if target_dir.exists():
                product["validation_status"] = "valid"

        quality_stats = self.quality_checker.check(products, reorganized_paths)
        report_paths = self.report_generator.generate_reports(products, quality_stats, validation_report)

        return {
            "dataset_index": dataset_index,
            "validation_report": validation_report,
            "metadata_files": metadata_files,
            "reorganized_paths": reorganized_paths,
            "quality_stats": quality_stats,
            "reports": report_paths,
            "products_indexed": len(dataset_index.products),
            "valid_products": validation_report.valid_products,
        }

    def _load_config_file(self) -> None:
        if self.config.config_path:
            loaded_config = load_yaml_config(self.config.config_path)
            if loaded_config:
                self.config.dataset_root = loaded_config.get("dataset_root", self.config.dataset_root)
                self.config.output_root = loaded_config.get("output_root", self.config.output_root)
                self.config.reports_dir = loaded_config.get("reports_dir", self.config.reports_dir)
                self.config.supported_formats = loaded_config.get("supported_formats", self.config.supported_formats)
                self.config.min_width = loaded_config.get("min_width", self.config.min_width)
                self.config.min_height = loaded_config.get("min_height", self.config.min_height)
                self.config.materials_root = loaded_config.get("materials_root", self.config.materials_root)
                self.config.patterns_root = loaded_config.get("patterns_root", self.config.patterns_root)

    def _build_target_path(self, product: Dict[str, Any]) -> Path:
        gender = product.get("gender", "unknown").lower()
        garment_type = product.get("garment_type", "upperwear")
        if gender == "men":
            gender_dir = "men"
        elif gender == "women":
            gender_dir = "women"
        else:
            gender_dir = "unisex"
        category_dir = {
            "T-Shirt": "upperwear",
            "Polo Shirt": "upperwear",
            "Formal Shirt": "upperwear",
            "Casual Shirt": "upperwear",
            "Sweater": "upperwear",
            "Cardigan": "upperwear",
            "Hoodie": "upperwear",
            "Sweatshirt": "upperwear",
            "Jacket": "upperwear",
            "Blazer": "upperwear",
            "Vest": "upperwear",
            "Jeans": "lowerwear",
            "Trousers": "lowerwear",
            "Chinos": "lowerwear",
            "Joggers": "lowerwear",
            "Cargo Pants": "lowerwear",
            "Shorts": "lowerwear",
            "Leggings": "lowerwear",
            "Dress": "dresses",
            "Romper": "dresses",
            "Jumpsuit": "dresses",
            "Kurta": "traditional",
            "Kurti": "traditional",
            "Saree": "traditional",
            "Lehenga": "traditional",
        }.get(garment_type, "upperwear")
        return Path(self.config.output_root) / "garments" / gender_dir / category_dir / product["product_id"]
