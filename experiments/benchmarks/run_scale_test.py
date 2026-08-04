import json
import random
import shutil
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
    source_dir = workspace_root / "datasets" / "fashion_garments"
    if not source_dir.exists():
        source_dir = workspace_root / "datasets" / "garments"
    
    scale_input_dir = workspace_root / "datasets" / "scale_test_sample"
    scale_processed_dir = workspace_root / "data" / "scale_test_processed"
    scale_output_dir = workspace_root / "scale_test_curated"
    
    # 1. Clean up old scale test dirs
    for d in [scale_input_dir, scale_processed_dir, scale_output_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        
    # 2. Select 500 random images
    all_images = list(source_dir.rglob("*.jpg")) + list(source_dir.rglob("*.jpeg")) + list(source_dir.rglob("*.png"))
    selected = random.sample(all_images, min(500, len(all_images)))
    print(f"Selected {len(selected)} images for scale test.")
    
    for img in selected:
        shutil.copy(img, scale_input_dir / img.name)
        
    # 3. Preprocessing
    print("Running Preprocessing Pipeline...")
    prep_config = PreprocessingConfig(
        input_dir=str(scale_input_dir),
        output_dir=str(scale_processed_dir)
    )
    prep_pipeline = PreprocessingPipeline(config=prep_config)
    prep_stats = prep_pipeline.process_dataset()
    print(f"Preprocessing completed: {prep_stats}")
    
    # 4. Semantic Analysis
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
                "Detected Category": meta.get("classification", {}).get("category"),
                "Confidence": meta.get("ai_analysis", {}).get("confidence"),
                "Validation Status": "SUCCESS",
                "Organization Path": str(meta_path.relative_to(workspace_root))
            })
        else:
            results.append({
                "Image Name": img.name,
                "Detected Category": "-",
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
    
    # Category Distribution
    cat_dist = {}
    for r in results:
        if r["Validation Status"] == "SUCCESS":
            cat = r["Detected Category"]
            cat_dist[cat] = cat_dist.get(cat, 0) + 1
            
    # Generate Report
    report_path = workspace_root / "Scale_Test_Report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Scale Test Report (500 Images)\n\n")
        f.write("## Metrics\n")
        f.write(f"- **Total processing time:** {total_time:.2f} seconds\n")
        f.write(f"- **Average seconds/image:** {avg_time:.2f} seconds\n")
        f.write(f"- **GPU VRAM usage (Peak):** {peak_vram:.2f} MiB\n")
        f.write(f"- **Successful metadata count:** {success_count}\n")
        f.write(f"- **Failed metadata count:** {failed_count}\n")
        f.write(f"- **Validation failures:** {failed_count}\n\n")
        
        f.write("## Category Distribution\n")
        for cat, count in cat_dist.items():
            f.write(f"- {cat}: {count}\n")
        f.write("\n")
        
        f.write("## Image Details\n")
        f.write("| Image Name | Detected Category | Confidence | Validation Status | Organization Path |\n")
        f.write("|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['Image Name']} | {r['Detected Category']} | {r['Confidence']} | {r['Validation Status']} | {r['Organization Path']} |\n")
            
    print(f"\nReport written to {report_path.absolute()}")

if __name__ == "__main__":
    main()
