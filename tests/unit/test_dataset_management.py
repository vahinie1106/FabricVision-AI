from pathlib import Path

from PIL import Image

from src.dataset_management.dataset_manager import DatasetManagementConfig, DatasetManager


def test_dataset_management_pipeline_creates_metadata_and_reports(tmp_path: Path) -> None:
    dataset_root = tmp_path / "datasets"
    output_root = tmp_path / "prepared"
    reports_dir = tmp_path / "reports"
    dataset_root.mkdir(parents=True, exist_ok=True)

    men_product = dataset_root / "men" / "shirts" / "prod_001"
    men_product.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (128, 160), color="red").save(men_product / "front.jpg")

    women_product = dataset_root / "women" / "dresses" / "prod_002"
    women_product.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (160, 200), color="blue").save(women_product / "front.png")

    config = DatasetManagementConfig(
        dataset_root=str(dataset_root),
        output_root=str(output_root),
        reports_dir=str(reports_dir),
        supported_formats=[".jpg", ".jpeg", ".png", ".webp"],
    )

    manager = DatasetManager(config=config)
    result = manager.run()

    assert result["products_indexed"] == 2
    assert result["valid_products"] == 2
    assert (output_root / "garments" / "men" / "upperwear" / "prod_001").exists()
    assert (output_root / "garments" / "women" / "dresses" / "prod_002").exists()
    assert (reports_dir / "classification_report.csv").exists()
    assert (reports_dir / "dataset_summary.md").exists()
    assert (reports_dir / "validation_report.md").exists()
    assert (reports_dir / "quality_report.md").exists()
