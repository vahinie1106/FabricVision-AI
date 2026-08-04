import os
import shutil
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent
    
    # 1. Create structure
    dirs_to_create = [
        "datasets/original",
        "datasets/validation/sample_10",
        "datasets/scale_testing/sample_500",
        "data/processed",
        "data/processed_validation",
        "data/processed_scale_test",
        "outputs/semantic_analysis/validation_10",
        "outputs/semantic_analysis/scale_test_500",
        "reports/validation",
        "reports/scale_testing"
    ]
    
    for d in dirs_to_create:
        (root / d).mkdir(parents=True, exist_ok=True)
        
    # 2. Move datasets/validation_sample -> datasets/validation/sample_10
    val_sample_old = root / "datasets" / "validation_sample"
    val_sample_new = root / "datasets" / "validation" / "sample_10"
    if val_sample_old.exists():
        for item in val_sample_old.iterdir():
            shutil.move(str(item), str(val_sample_new / item.name))
        val_sample_old.rmdir()
        
    # 3. Move curated_dataset -> outputs/semantic_analysis/validation_10
    curated_old = root / "curated_dataset"
    curated_new = root / "outputs" / "semantic_analysis" / "validation_10"
    if curated_old.exists():
        for item in curated_old.iterdir():
            shutil.move(str(item), str(curated_new / item.name))
        curated_old.rmdir()
        
    # 4. Clean up old temporary scale_test_sample and outputs if they exist
    scale_old = root / "datasets" / "scale_test_sample"
    if scale_old.exists():
        shutil.rmtree(scale_old)
        
    data_scale_old = root / "data" / "scale_test_processed"
    if data_scale_old.exists():
        shutil.rmtree(data_scale_old)
        
    scale_curated_old = root / "scale_test_curated"
    if scale_curated_old.exists():
        shutil.rmtree(scale_curated_old)
        
    # 5. Move reports to reports/validation
    reports_dir = root / "reports"
    reports_val_dir = root / "reports" / "validation"
    for rfile in ["classification_report.csv", "dataset_summary.md", "quality_report.md", "validation_report.md"]:
        rpath = reports_dir / rfile
        if rpath.exists():
            shutil.move(str(rpath), str(reports_val_dir / rfile))
            
    final_report = root / "Final_Validation_Report.md"
    if final_report.exists():
        shutil.move(str(final_report), str(reports_val_dir / "Final_Validation_Report.md"))
        
    print("Folder organization complete.")

if __name__ == "__main__":
    main()
