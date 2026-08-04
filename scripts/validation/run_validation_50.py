import json
import time
import os
import psutil
from pathlib import Path
import torch

from src.preprocessing.preprocessing_pipeline import PreprocessingConfig, PreprocessingPipeline
from src.semantic_analysis.pipeline.semantic_analysis_pipeline import SemanticAnalysisConfig, SemanticAnalysisPipeline

def get_sys_metrics():
    metrics = {}
    if torch.cuda.is_available():
        metrics["gpu_memory_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        # Note: torch.cuda.utilization() isn't standard, using placeholder for utilization
        metrics["gpu_utilization"] = "N/A"
    else:
        metrics["gpu_memory_mb"] = 0
        metrics["gpu_utilization"] = "N/A"
        
    metrics["cpu_percent"] = psutil.cpu_percent(interval=None)
    metrics["ram_percent"] = psutil.virtual_memory().percent
    return metrics

def main():
    workspace_root = Path(__file__).resolve().parent
    
    input_dir = workspace_root / "datasets" / "validation" / "sample_50"
    processed_dir = workspace_root / "data" / "processed_validation_50"
    output_dir = workspace_root / "outputs" / "semantic_analysis" / "validation_50"
    report_dir = workspace_root / "reports" / "validation_50"
    
    for d in [processed_dir, output_dir, report_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    print("Running Preprocessing Pipeline...")
    t0 = time.time()
    prep_config = PreprocessingConfig(
        input_dir=str(input_dir),
        output_dir=str(processed_dir)
    )
    prep_pipeline = PreprocessingPipeline(config=prep_config)
    prep_stats = prep_pipeline.process_dataset()
    t1 = time.time()
    prep_time = t1 - t0
    
    print("Running Semantic Analysis Pipeline...")
    sem_config = SemanticAnalysisConfig()
    sem_config.config_dir = str(workspace_root / "configs")
    sem_config.output_root = str(output_dir)
    sem_config.config_path = str(workspace_root / "configs" / "semantic_analysis_config.yaml")
    
    # Overwrite the semantic analysis output_root at runtime to avoid the yaml overriding it back to curated_dataset
    loaded_pipeline = SemanticAnalysisPipeline(config=sem_config)
    loaded_pipeline.config.output_root = str(output_dir)
    loaded_pipeline.organizer.output_root = Path(output_dir)
    
    processed_images = list(processed_dir.rglob("*.jpg"))
    
    results = []
    inference_times = []
    
    start_time = time.time()
    
    for idx, img in enumerate(processed_images, 1):
        print(f"Processing {idx}/{len(processed_images)}: {img.name}")
        t_start = time.time()
        res = loaded_pipeline.run(str(img))
        t_end = time.time()
        inf_time = t_end - t_start
        inference_times.append(inf_time)
        
        status = res.get("status")
        if status == "completed":
            meta_path = Path(res.get("metadata_path"))
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            
            results.append({
                "image_name": img.name,
                "status": "SUCCESS",
                "metadata": meta,
                "inference_time": inf_time,
                "path": str(meta_path)
            })
        else:
            results.append({
                "image_name": img.name,
                "status": "FAILED",
                "issues": res.get("issues"),
                "inference_time": inf_time,
                "path": None
            })
            
    end_time = time.time()
    total_time = end_time - start_time
    
    sys_metrics = get_sys_metrics()
    
    stats = {
        "preprocessing": {
            "time": prep_time,
            "stats": prep_stats.__dict__ if hasattr(prep_stats, '__dict__') else str(prep_stats)
        },
        "semantic_analysis": {
            "total_time": total_time,
            "inference_times": inference_times,
            "results": results
        },
        "system": sys_metrics
    }
    
    stats_path = workspace_root / "validation_50_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        
    print(f"Validation completed. Stats saved to {stats_path}")

if __name__ == "__main__":
    main()
