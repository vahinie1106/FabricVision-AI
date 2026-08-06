from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List


class ReportGenerator:
    """Generate markdown and CSV reports for dataset management results."""

    def __init__(self, reports_dir: str | Path) -> None:
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_reports(self, metadata: List[Dict[str, Any]], quality_stats: Dict[str, Any], validation_report: Any) -> Dict[str, Path]:
        """Write classification, summary, validation, and quality reports."""
        classification_report = self.reports_dir / "classification_report.csv"
        dataset_summary = self.reports_dir / "dataset_summary.md"
        validation_report_path = self.reports_dir / "validation_report.md"
        quality_report_path = self.reports_dir / "quality_report.md"

        with classification_report.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["product_id", "garment_type", "confidence", "classification_source"])
            writer.writeheader()
            for product in metadata:
                writer.writerow({
                    "product_id": product["product_id"],
                    "garment_type": product["garment_type"],
                    "confidence": product["confidence"],
                    "classification_source": product["classification_source"],
                })

        dataset_summary.write_text(
            "# Dataset Summary\n\n"
            f"- Products: {len(metadata)}\n"
            f"- Validation: {len([item for item in metadata if item['validation_status'] == 'valid'])} valid products\n"
            f"- Reorganized paths: {quality_stats.get('reorganized_path_count', 0)}\n",
            encoding="utf-8",
        )

        validation_report_path.write_text(
            "# Validation Report\n\n"
            f"- Valid products: {validation_report.valid_products}\n"
            f"- Invalid products: {validation_report.invalid_products}\n",
            encoding="utf-8",
        )

        quality_report_path.write_text(
            "# Quality Report\n\n"
            f"- Product count: {quality_stats.get('product_count', 0)}\n"
            f"- Missing products: {len(quality_stats.get('missing_products', []))}\n"
            f"- Missing images: {len(quality_stats.get('missing_images', []))}\n"
            f"- Metadata complete: {quality_stats.get('metadata_complete', False)}\n",
            encoding="utf-8",
        )

        return {
            "classification_report": classification_report,
            "dataset_summary": dataset_summary,
            "validation_report": validation_report_path,
            "quality_report": quality_report_path,
        }
