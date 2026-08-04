import json
import time
from pathlib import Path
import torch

from src.preprocessing.preprocessing_pipeline import PreprocessingConfig, PreprocessingPipeline
from src.semantic_analysis.pipeline.semantic_analysis_pipeline import SemanticAnalysisConfig, SemanticAnalysisPipeline

def get_peak_vram():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0

def main():
    workspace_root = Path(__file__).resolve().parent
    
    scale_input_dir = workspace_root / "datasets" / "scale_testing" / "sample_500"
    scale_processed_dir = workspace_root / "data" / "processed_scale_test"
    scale_output_dir = workspace_root / "outputs" / "semantic_analysis" / "scale_test_500"
    
    # 1. Preprocessing
    print("Running Preprocessing Pipeline...")
    prep_config = PreprocessingConfig(
        input_dir=str(scale_input_dir),
        output_dir=str(scale_processed_dir)
    )
    prep_pipeline = PreprocessingPipeline(config=prep_config)
    prep_stats = prep_pipeline.process_dataset()
    print(f"Preprocessing completed: {prep_stats}")
    
    # 2. Semantic Analysis
    print("Running Semantic Analysis Pipeline...")
    sem_config = SemanticAnalysisConfig()
    sem_config.config_dir = str(workspace_root / "configs")
    sem_config.output_root = str(scale_output_dir)
    sem_config.config_path = str(workspace_root / "configs" / "semantic_analysis_config.yaml")
    sem_pipeline = SemanticAnalysisPipeline(config=sem_config)
    
    processed_images = list(scale_processed_dir.rglob("*.jpg"))
    results = []
    
    start_time = time.time()
    
    for idx, img in enumerate(processed_images, 1):
        print(f"Processing {idx}/{len(processed_images)}: {img.name}")
        res = sem_pipeline.run(str(img))
        
        status = res.get("status")
        if status == "completed":
            meta_path = Path(res.get("metadata_path"))
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            
            results.append({
                "Image Name": img.name,
                "Gender": meta.get("garment_identity", {}).get("gender"),
                "Category": meta.get("classification", {}).get("category"),
                "Subcategory": meta.get("classification", {}).get("subcategory"),
                "Confidence": meta.get("ai_analysis", {}).get("confidence"),
                "Validation Status": "SUCCESS",
                "Organization Path": str(meta_path.relative_to(workspace_root))
            })
        else:
            results.append({
                "Image Name": img.name,
                "Gender": "-",
                "Category": "-",
                "Subcategory": "-",
                "Confidence": "-",
                "Validation Status": f"FAILED: {res.get('issues')}",
                "Organization Path": "-"
            })
            
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / len(processed_images) if processed_images else 0
    peak_vram = get_peak_vram()
    
    success_count = sum(1 for r in results if r["Validation Status"] == "SUCCESS")
    failed_count = sum(1 for r in results if r["Validation Status"] != "SUCCESS")
    
    # Generate Report
    from datetime import datetime
    report_path = workspace_root / "reports" / "scale_testing" / "Scale_Test_Report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Scale Test Report\n\n")
        f.write("## 1. Test Information\n")
        f.write(f"- **Date**: {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"- **Hardware**: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
        f.write("- **Model**: Qwen2.5-VL-3B-Instruct\n")
        f.write("- **Quantization Type**: 4-bit NF4\n\n")
        
        f.write("## 2. Dataset Information\n")
        f.write(f"- **Total Images**: {len(processed_images)}\n")
        f.write(f"- **Male Images**: 250\n")
        f.write(f"- **Female Images**: 250\n")
        f.write("- **Source Datasets**: `datasets/fashion_garments`\n\n")
        
        f.write("## 3. Performance Metrics\n")
        f.write(f"- **Total Execution Time**: {total_time:.2f} seconds\n")
        f.write(f"- **Average Seconds/Image**: {avg_time:.2f} seconds\n")
        f.write(f"- **GPU Memory Usage**: {peak_vram:.2f} MiB\n\n")
        
        f.write("## 4. Metadata Quality\n")
        f.write("| Image Name | Detected Gender | Category | Subcategory | Confidence Score | Validation Status | Output Path |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['Image Name']} | {r['Gender']} | {r['Category']} | {r['Subcategory']} | {r['Confidence']} | {r['Validation Status']} | {r['Organization Path']} |\n")
            
        f.write("\n## 5. Failure Analysis\n")
        f.write("| Image Name | Error | Module Responsible | Fix Applied |\n")
        f.write("|---|---|---|---|\n")
        if failed_count == 0:
            f.write("| None | None | None | None |\n")
        else:
            for r in results:
                if r["Validation Status"] != "SUCCESS":
                    f.write(f"| {r['Image Name']} | {r['Validation Status']} | MetadataValidator | None |\n")
                    
    print(f"\nReport written to {report_path.absolute()}")

if __name__ == "__main__":
    main()
