from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .augmentation import AugmentationProcessor
from .background_processing import BackgroundProcessor, BackgroundStrategy
from .dataset_loader import DatasetIndex, DatasetLoader
from .image_cleaner import ImageCleaner
from .image_loader import ImageLoader
from .image_transformer import ImageTransformer
from .image_validator import ImageValidator, ValidationResult


@dataclass
class PreprocessingConfig:
    """Centralized configuration for the preprocessing pipeline."""

    input_dir: str | Path = "data/raw"
    output_dir: str | Path = "data/processed"
    target_size: Tuple[int, int] = (512, 512)
    supported_formats: List[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png"])
    min_width: int = 16
    min_height: int = 16
    min_resolution: int = 256
    blur_threshold: float = 100.0
    enable_background_processing: bool = True
    background_strategy: str = "none"
    enable_noise_reduction: bool = False
    noise_reduction_type: Optional[str] = None
    enable_augmentation: bool = False
    augmentation_config: Optional[Dict[str, float]] = None
    normalize: bool = True


class PreprocessingPipeline:
    """Orchestrates dataset discovery, validation, transformation, and output generation."""

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        self.config = config or PreprocessingConfig()
        self.logger = logging.getLogger("fabricvision.preprocessing")
        self.image_loader = ImageLoader(supported_formats=tuple(self.config.supported_formats))
        self.image_validator = ImageValidator(
            min_width=self.config.min_width,
            min_height=self.config.min_height,
            min_resolution=self.config.min_resolution,
            blur_threshold=self.config.blur_threshold,
        )
        self.image_cleaner = ImageCleaner(denoise=self.config.enable_noise_reduction)
        self.background_processor = BackgroundProcessor(
            strategy=BackgroundStrategy(self.config.background_strategy)
            if self.config.background_strategy in {item.value for item in BackgroundStrategy}
            else BackgroundStrategy.NONE
        )
        self.image_transformer = ImageTransformer(
            target_size=self.config.target_size,
            normalize=self.config.normalize,
            noise_reduction=self.config.noise_reduction_type,
        )
        self.augmentation_processor = AugmentationProcessor(
            enabled=self.config.enable_augmentation,
            config=self.config.augmentation_config,
        )
        self.dataset_loader = DatasetLoader(
            root_dir=self.config.input_dir,
            supported_formats=self.config.supported_formats,
        )

    def process_dataset(self) -> Dict[str, object]:
        """Process an entire input dataset and save preprocessed images to the output directory."""
        start_time = perf_counter()
        dataset_index = self.dataset_loader.scan_dataset()
        self._ensure_output_directory()

        processed_count = 0
        failed_count = 0
        skipped_count = 0
        output_paths: List[str] = []

        for sample in tqdm(dataset_index.samples, desc="Preprocessing images", unit="image"):
            try:
                validation_result = self._validate_and_prepare(sample.file_path)
                if not validation_result.is_valid:
                    failed_count += 1
                    self.logger.warning("Skipping %s due to validation reasons: %s", sample.file_path, validation_result.reasons)
                    continue

                output_path = self._write_processed_image(sample.file_path)
                processed_count += 1
                output_paths.append(str(output_path))
            except Exception as exc:  # noqa: BLE001
                failed_count += 1
                self.logger.exception("Failed to process %s: %s", sample.file_path, exc)

        elapsed_seconds = perf_counter() - start_time
        return {
            "dataset_index": dataset_index,
            "processed_count": processed_count,
            "failed_count": failed_count,
            "skipped_count": skipped_count,
            "processing_time_seconds": round(elapsed_seconds, 3),
            "output_dir": str(self.config.output_dir),
            "output_paths": output_paths,
        }

    def _validate_and_prepare(self, image_path: Path) -> ValidationResult:
        image_array, _ = self.image_loader.load_image(image_path)
        validation_result = self.image_validator.validate_image(image_path, image_array)
        if not validation_result.is_valid:
            return validation_result
        return validation_result

    def _write_processed_image(self, image_path: Path) -> Path:
        image_array, _ = self.image_loader.load_image(image_path)
        cleaned_image = self.image_cleaner.clean(image_array)

        if self.config.enable_background_processing:
            cleaned_image = self.background_processor.process(cleaned_image)

        transformed_image = self.image_transformer.transform(cleaned_image)
        augmented_image = self.augmentation_processor.augment(transformed_image)

        output_path = self._build_output_path(image_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image = self._to_uint8_image(augmented_image)
        image.save(output_path)
        return output_path

    def _build_output_path(self, image_path: Path) -> Path:
        relative_path = image_path.relative_to(self.config.input_dir)
        return Path(self.config.output_dir) / relative_path

    def _ensure_output_directory(self) -> None:
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def _to_uint8_image(self, image_array: np.ndarray) -> object:
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        from PIL import Image as PILImage

        return PILImage.fromarray(image_array)
