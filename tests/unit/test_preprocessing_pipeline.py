from pathlib import Path

from PIL import Image

from src.features.preprocessing.dataset_loader import DatasetLoader
from src.features.preprocessing.preprocessing_pipeline import PreprocessingConfig, PreprocessingPipeline


def test_dataset_loader_discovers_images_and_metadata(tmp_path: Path) -> None:
    garments_dir = tmp_path / "garments" / "shirts"
    garments_dir.mkdir(parents=True)
    person_dir = tmp_path / "persons"
    person_dir.mkdir(parents=True)

    Image.new("RGB", (64, 64), color="red").save(garments_dir / "shirt_001.jpg")
    Image.new("RGB", (48, 48), color="blue").save(person_dir / "person_001.png")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")

    loader = DatasetLoader(root_dir=tmp_path)
    dataset = loader.scan_dataset()

    assert len(dataset.samples) == 2
    assert {sample.category for sample in dataset.samples} == {"garments", "persons"}
    assert any(sample.file_name == "shirt_001.jpg" for sample in dataset.samples)
    assert all(sample.width > 0 for sample in dataset.samples)


def test_preprocessing_pipeline_processes_sample_dataset(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    garments_dir = raw_dir / "garments" / "shirts"
    garments_dir.mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (128, 96), color="green").save(garments_dir / "shirt_001.jpg")
    (garments_dir / "broken.jpg").write_bytes(b"not-an-image")

    config = PreprocessingConfig(
        input_dir=raw_dir,
        output_dir=output_dir,
        target_size=(64, 64),
        supported_formats=[".jpg", ".jpeg", ".png"],
        min_width=16,
        min_height=16,
        min_resolution=16,
        enable_augmentation=False,
        enable_background_processing=False,
        enable_noise_reduction=False,
    )

    pipeline = PreprocessingPipeline(config=config)
    stats = pipeline.process_dataset()

    assert stats["processed_count"] == 1
    assert stats["failed_count"] == 0
    assert output_dir.joinpath("garments", "shirts", "shirt_001.jpg").exists()
